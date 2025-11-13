#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate MIDI samples with MusDr metrics (H1, H4, GS).
- 動態從你的 dictionary.pkl 解析 BAR_EV / POS_EVS / PITCH_EVS
- 讀取資料夾下所有 .mid / .MID
- 依 MusDr side_utils 計算 H1 / H4 / GS 後輸出 CSV

用法：
  python eval_metrics.py \
    --dict_path results/hw3_task1_final/tok/dictionary.pkl \
    --output_file_path results/hw3_task1_final/samples \
    --out_csv results/hw3_task1_final/metrics_musdr.csv
"""
import os
import re
import sys
import argparse
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------
# 尋找 utils.py（你的自家 REMI 事件抽取工具）
# -----------------------------------------------------------
HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent, HERE.parent.parent):
    if (cand / "utils.py").exists():
        sys.path.insert(0, str(cand))
        break
import utils  # noqa: E402

# -----------------------------------------------------------
# 匯入 MusDr 的 side_utils
#   建議：git clone https://github.com/slSeanWU/MusDr.git
#   並 export PYTHONPATH 或在此加入搜尋路徑
# -----------------------------------------------------------
try:
    from musdr.side_utils import (
        get_bars_crop,
        get_pitch_histogram,
        compute_histogram_entropy,
        get_onset_xor_distance,
    )
except Exception:
    # 若未能 import，到常見位置找
    env_root = os.environ.get("MUSDR_ROOT", "")
    if env_root and (Path(env_root) / "musdr" / "side_utils.py").exists():
        sys.path.insert(0, env_root)
    elif (HERE / "MusDr" / "musdr" / "side_utils.py").exists():
        sys.path.insert(0, str(HERE / "MusDr"))
    else:
        print("[eval] 找不到 musdr.side_utils，請先：")
        print("  git clone https://github.com/slSeanWU/MusDr.git")
        print("  並加上：export PYTHONPATH=$(pwd)/MusDr:$PYTHONPATH")
        sys.exit(1)
    from musdr.side_utils import (  # type: ignore
        get_bars_crop,
        get_pitch_histogram,
        compute_histogram_entropy,
        get_onset_xor_distance,
    )

# -----------------------------------------------------------
# 解析 CLI
# -----------------------------------------------------------
def parse_opt():
    p = argparse.ArgumentParser()
    p.add_argument("--dict_path", required=True, help="你的 dictionary.pkl 路徑（event<->word）")
    p.add_argument("--output_file_path", required=True,
                   help="要評分的 .mid/.MID 所在資料夾（e.g., results/.../samples）")
    p.add_argument("--out_csv", default="metrics_musdr.csv", help="輸出 CSV 路徑")
    p.add_argument("--max_pairs", type=int, default=1000, help="GS 每首歌最多取多少 bar 對 (加速)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# -----------------------------------------------------------
# 從 dictionary.pkl 動態推 BAR/POS/PITCH
# -----------------------------------------------------------
def discover_vocab(dict_pkl: str):
    event2word, word2event = pickle.load(open(dict_pkl, "rb"))
    # Bar
    bar_key_candidates = [k for k in event2word.keys() if k.startswith("Bar_")]
    if "Bar_None" in event2word:
        bar_ev = event2word["Bar_None"]
    elif bar_key_candidates:
        # 任選第一個 Bar 類（通常只有一個）
        bar_ev = event2word[bar_key_candidates[0]]
    else:
        raise ValueError("字典裡找不到 Bar_* 事件鍵")

    # Position: Position_x/16
    pos_evs = []
    for k, v in event2word.items():
        if k.startswith("Position_"):
            # 取 x/16 的 x 做排序
            try:
                frac = k.split("_", 1)[1]         # "1/16"
                num = int(frac.split("/")[0])     # 1
            except Exception:
                num = 999999
            pos_evs.append((num, v))
    pos_evs = [v for _, v in sorted(pos_evs, key=lambda z: z[0])]
    if not pos_evs:
        raise ValueError("字典裡找不到 Position_* 事件鍵")

    # Pitch：Note On_XX
    pitch_evs = []
    for k, v in event2word.items():
        if k.startswith("Note On_"):
            try:
                nn = int(k.split("_", 1)[1])
            except Exception:
                nn = 999999
            pitch_evs.append((nn, v))
    pitch_evs = [v for _, v in sorted(pitch_evs, key=lambda z: z[0])]
    if not pitch_evs:
        # 有些字典使用 Pitch_* 或 Note-On_* 之類命名，可再擴充：
        for k, v in event2word.items():
            if re.match(r"(?i)^(Pitch|Note\-?On)_\d+$", k):
                try:
                    nn = int(k.split("_", 1)[1])
                except Exception:
                    nn = 999999
                pitch_evs.append((nn, v))
        pitch_evs = [v for _, v in sorted(pitch_evs, key=lambda z: z[0])]
        if not pitch_evs:
            raise ValueError("字典裡找不到 Note On_* / Pitch_* 類事件鍵")

    return event2word, word2event, bar_ev, pos_evs, pitch_evs

# -----------------------------------------------------------
# 解析 MIDI → events → word ids（用你的 utils）
# -----------------------------------------------------------
def extract_events(input_path: str):
    note_items, tempo_items = utils.read_items(input_path)
    note_items = utils.quantize_items(note_items)
    if len(note_items) == 0:
        return []
    max_time = int(note_items[-1].end)
    items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    return events

def to_word_ids(events, event2word):
    words = []
    for ev in events:
        k = f"{ev.name}_{ev.value}"
        if k in event2word:
            words.append(event2word[k])
        else:
            # OOV fallback：把 velocity map 到常見值（你之前的習慣）
            if ev.name == "Note Velocity" and "Note Velocity_21" in event2word:
                words.append(event2word["Note Velocity_21"])
            # 其他 OOV 直接略過
    return words

# -----------------------------------------------------------
# 指標計算（MusDr）
# -----------------------------------------------------------
def compute_piece_pitch_entropy(piece_ev_seq, window_size, bar_ev_id, pitch_evs, verbose=False):
    # 去掉尾端多餘 Bar
    if len(piece_ev_seq) and piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]
    n_bars = piece_ev_seq.count(bar_ev_id)
    if n_bars <= 0:
        return np.nan
    if window_size > n_bars:
        window_size = n_bars

    ents = []
    for st_bar in range(0, n_bars - window_size + 1):
        seg = get_bars_crop(piece_ev_seq, st_bar, st_bar + window_size - 1, bar_ev_id)
        pitch_hist = get_pitch_histogram(seg, pitch_evs=pitch_evs)
        if pitch_hist is None:
            continue
        ents.append(compute_histogram_entropy(pitch_hist))
    return float(np.mean(ents)) if ents else np.nan

def compute_piece_groove_similarity(piece_ev_seq, bar_ev_id, pos_evs, pitch_evs, max_pairs=1000):
    if len(piece_ev_seq) and piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]
    n_bars = piece_ev_seq.count(bar_ev_id)
    if n_bars <= 1:
        return np.nan

    # 取每小節
    bars = [get_bars_crop(piece_ev_seq, b, b, bar_ev_id) for b in range(n_bars)]

    # 取 bar pairs
    pairs = []
    for i in range(n_bars):
        for j in range(i + 1, n_bars):
            pairs.append((i, j))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(3407)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False)]

    sims = []
    for i, j in pairs:
        d = get_onset_xor_distance(bars[i], bars[j], bar_ev_id, pos_evs, pitch_evs=pitch_evs)
        sims.append(1.0 - d)
    return float(np.mean(sims)) if sims else np.nan

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    args = parse_opt()
    out_dir = Path(args.output_file_path)
    mids = sorted([p for g in ("*.mid", "*.MID") for p in out_dir.glob(g)])
    if not mids:
        print(f"[eval] 找不到 MIDI：{out_dir}")
        sys.exit(1)

    event2word, word2event, BAR_EV, POS_EVS, PITCH_EVS = discover_vocab(args.dict_path)
    if args.verbose:
        print(f"[eval] BAR_EV={BAR_EV}")
        print(f"[eval] POS_EVS len={len(POS_EVS)}; first/last={POS_EVS[0]}/{POS_EVS[-1]}")
        print(f"[eval] PITCH_EVS len={len(PITCH_EVS)}; first/last={PITCH_EVS[0]}/{PITCH_EVS[-1]}")

    rows = []
    for mp in tqdm(mids, desc="Scoring", ncols=100):
        evs = extract_events(str(mp))
        words = to_word_ids(evs, event2word)
        if not words:
            rows.append((mp.name, np.nan, np.nan, np.nan))
            continue
        h1 = compute_piece_pitch_entropy(words, 1, BAR_EV, PITCH_EVS)
        h4 = compute_piece_pitch_entropy(words, 4, BAR_EV, PITCH_EVS)
        gs = compute_piece_groove_similarity(words, BAR_EV, POS_EVS, PITCH_EVS, max_pairs=args.max_pairs)
        rows.append((mp.name, h1, h4, gs))

    df = pd.DataFrame(rows, columns=["piece_name", "H1", "H4", "GS"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[eval] wrote {args.out_csv}  | files={len(df)}")

if __name__ == "__main__":
    main()

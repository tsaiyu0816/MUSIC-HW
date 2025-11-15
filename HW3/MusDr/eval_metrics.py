import os
import re
import sys
import argparse
import pickle
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------
# 尋找 utils.py（你的 custom 事件抽取工具）
# -----------------------------------------------------------
HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent, HERE.parent.parent):
    if (cand / "utils.py").exists():
        sys.path.insert(0, str(cand))
        break
import utils  # noqa: E402

# -----------------------------------------------------------
# 可選 Miditok
# -----------------------------------------------------------
HAVE_MIDITOK = False
try:
    from miditok import TokenizerConfig, REMI
    try:
        from miditok import TokSequence  # >= 3.x
    except Exception:
        from miditok.utils import TokSequence
    HAVE_MIDITOK = True
except Exception:
    HAVE_MIDITOK = False

# -----------------------------------------------------------
# 匯入 MusDr 的 side_utils
# -----------------------------------------------------------
try:
    from musdr.side_utils import (
        get_bars_crop,
        get_pitch_histogram,
        compute_histogram_entropy,
        get_onset_xor_distance,
    )
except Exception:
    env_root = os.environ.get("MUSDR_ROOT", "")
    if env_root and (Path(env_root) / "musdr" / "side_utils.py").exists():
        sys.path.insert(0, env_root)
    elif (HERE / "MusDr" / "musdr" / "side_utils.py").exists():
        sys.path.insert(0, str(HERE / "MusDr"))
    else:
        print("[eval] 找不到 musdr.side_utils，請先安裝 MusDr 或設定 PYTHONPATH")
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
    p.add_argument("--tokenizer", choices=["custom", "remi", "remi_plus"], default="custom",
                   help="custom 需 .pkl；remi/remi_plus 為 miditok（不需 .pkl）")
    p.add_argument("--dict_path", default="", help="(custom only) dictionary.pkl")
    p.add_argument("--with_chord", type=int, default=0, help="miditok 時是否啟用 chords")
    p.add_argument("--output_file_path", required=True,
                   help="要評分的 .mid/.MID 所在資料夾")
    p.add_argument("--out_csv", default="metrics_musdr.csv", help="輸出 CSV 路徑")
    p.add_argument("--max_pairs", type=int, default=1000, help="GS 每首最多取多少 bar 對（加速用）")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# -----------------------------------------------------------
# custom: 從 dictionary.pkl 動態推 BAR/POS/PITCH
# -----------------------------------------------------------
def discover_vocab_custom(dict_pkl: str):
    event2word, word2event = pickle.load(open(dict_pkl, "rb"))

    # Bar
    if "Bar_None" in event2word:
        bar_ev = event2word["Bar_None"]
    else:
        cands = [k for k in event2word.keys() if k.startswith("Bar_")]
        if not cands:
            raise ValueError("字典裡找不到 Bar_* 事件鍵")
        bar_ev = event2word[cands[0]]

    # Position
    pos_evs = []
    for k, v in event2word.items():
        if k.startswith("Position_"):
            try:
                frac = k.split("_", 1)[1]         # "1/16"
                num = int(frac.split("/")[0])     # 1
            except Exception:
                num = 999999
            pos_evs.append((num, v))
    pos_evs = [v for _, v in sorted(pos_evs, key=lambda z: z[0])]
    if not pos_evs:
        raise ValueError("字典裡找不到 Position_* 事件鍵")

    # Pitch
    pitch_evs = []
    for k, v in event2word.items():
        if k.startswith("Note On_"):
            try:
                nn = int(k.split("_", 1)[1])
            except Exception:
                nn = 999999
            pitch_evs.append((nn, v))
    if not pitch_evs:
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
# miditok: 建 tokenizer + 找出 BAR/POS/PITCH 的 id 集
# -----------------------------------------------------------
def build_miditok(kind: str, with_chord: bool):
    if not HAVE_MIDITOK:
        raise SystemExit("[eval] 需要 pip install miditok symusic")
    cfg = TokenizerConfig(
        beat_res={(0, 4): 4},          # 1/16
        num_velocities=32,
        use_chords=bool(with_chord),
        use_rests=False,
        use_tempos=True,
        use_time_signatures=(kind == "remi_plus"),
        use_programs=(kind == "remi_plus"),
        one_token_stream_for_programs=(kind == "remi_plus"),
    )
    tok = REMI(cfg)
    return tok

def ids_to_tokens_safe(tokenizer, ids):
    try:
        return tokenizer.ids_to_tokens(ids, as_str=True)
    except Exception:
        out = []
        for i in ids:
            try:
                out.append(str(tokenizer[i]))
            except Exception:
                out.append(None)
        return out

def discover_vocab_miditok(tokenizer):
    # Bar（一般只有一個 Bar token）
    bar_ids = []
    names = ids_to_tokens_safe(tokenizer, list(range(len(tokenizer))))
    for i, t in enumerate(names):
        if isinstance(t, str):
            head = t.split("_", 1)[0].lower()
            if head == "bar" or "bar" in head:
                bar_ids.append(i)
    if not bar_ids:
        raise ValueError("[eval] miditok vocab 找不到 Bar 類 token")
    bar_ev = int(bar_ids[0])
    if len(bar_ids) > 1:
        print(f"[warn] 發現多個 Bar tokens，選用 {bar_ev}")

    # Position
    pos_pairs = []
    for i, t in enumerate(names):
        if isinstance(t, str) and t.lower().startswith("position_"):
            try:
                num = int(t.split("_", 1)[1].split("/")[0])
            except Exception:
                num = 999999
            pos_pairs.append((num, i))
    pos_evs = [i for _, i in sorted(pos_pairs, key=lambda z: z[0])]
    if not pos_evs:
        raise ValueError("[eval] miditok vocab 找不到 Position_* tokens")

    # Pitch（Pitch_* 或 Note-On_* 都接受）
    pitch_pairs = []
    for i, t in enumerate(names):
        if isinstance(t, str) and re.match(r"(?i)^(Pitch|Note[-_ ]?On)_\d+$", t):
            try:
                nn = int(t.split("_", 1)[1])
            except Exception:
                nn = 999999
            pitch_pairs.append((nn, i))
    pitch_evs = [i for _, i in sorted(pitch_pairs, key=lambda z: z[0])]
    if not pitch_evs:
        raise ValueError("[eval] miditok vocab 找不到 Pitch_* / Note-On_* tokens")

    return bar_ev, pos_evs, pitch_evs

# -----------------------------------------------------------
# 解析 MIDI → ids（custom or miditok）
# -----------------------------------------------------------
def events_to_word_ids(events, event2word):
    words = []
    for ev in events:
        k = f"{ev.name}_{ev.value}"
        if k in event2word:
            words.append(event2word[k])
        elif ev.name == "Note Velocity" and "Note Velocity_21" in event2word:
            words.append(event2word["Note Velocity_21"])  # fallback
    return words

def midi_to_ids_custom(midi_path: str, event2word):
    note_items, tempo_items = utils.read_items(midi_path)
    note_items = utils.quantize_items(note_items)
    if len(note_items) == 0:
        return []
    max_time = int(note_items[-1].end)
    items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    return events_to_word_ids(events, event2word)

def midi_to_ids_miditok(midi_path: str, tokenizer):
    from symusic import Score
    try:
        seq = tokenizer.encode(Score(midi_path))
    except Exception:
        seq = tokenizer.midi_to_tokens(Score(midi_path))
    if isinstance(seq, list):
        assert len(seq) == 1
        seq = seq[0]
    return list(seq.ids)

# -----------------------------------------------------------
# 指標計算（MusDr）
# -----------------------------------------------------------
def compute_piece_pitch_entropy(piece_ids, window_size, bar_ev_id, pitch_evs):
    if len(piece_ids) and piece_ids[-1] == bar_ev_id:
        piece_ids = piece_ids[:-1]
    n_bars = piece_ids.count(bar_ev_id)
    if n_bars <= 0:
        return np.nan
    window_size = min(window_size, n_bars)
    ents = []
    for st in range(0, n_bars - window_size + 1):
        seg = get_bars_crop(piece_ids, st, st + window_size - 1, bar_ev_id)
        hist = get_pitch_histogram(seg, pitch_evs=pitch_evs)
        if hist is None:
            continue
        ents.append(compute_histogram_entropy(hist))
    return float(np.mean(ents)) if ents else np.nan

def compute_piece_groove_similarity(piece_ids, bar_ev_id, pos_evs, pitch_evs, max_pairs=1000):
    if len(piece_ids) and piece_ids[-1] == bar_ev_id:
        piece_ids = piece_ids[:-1]
    n_bars = piece_ids.count(bar_ev_id)
    if n_bars <= 1:
        return np.nan

    bars = [get_bars_crop(piece_ids, b, b, bar_ev_id) for b in range(n_bars)]
    pairs = [(i, j) for i in range(n_bars) for j in range(i + 1, n_bars)]
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

    # 準備 tokenizer 與 vocab
    if args.tokenizer == "custom":
        if not args.dict_path:
            raise SystemExit("[eval] custom 模式需要 --dict_path")
        event2word, word2event, BAR_EV, POS_EVS, PITCH_EVS = discover_vocab_custom(args.dict_path)
        to_ids = lambda mp: midi_to_ids_custom(str(mp), event2word)
    else:
        kind = "remi_plus" if args.tokenizer == "remi_plus" else "remi"
        tokenizer = build_miditok(kind, bool(args.with_chord))
        BAR_EV, POS_EVS, PITCH_EVS = discover_vocab_miditok(tokenizer)
        to_ids = lambda mp: midi_to_ids_miditok(str(mp), tokenizer)

    if args.verbose:
        print(f"[eval] BAR_EV={BAR_EV}")
        print(f"[eval] POS_EVS len={len(POS_EVS)}  first/last={POS_EVS[0]}/{POS_EVS[-1]}")
        print(f"[eval] PITCH_EVS len={len(PITCH_EVS)} first/last={PITCH_EVS[0]}/{PITCH_EVS[-1]}")

    rows = []
    for mp in tqdm(mids, desc="Scoring", ncols=100):
        ids = to_ids(mp)
        if not ids:
            rows.append((mp.name, np.nan, np.nan, np.nan)); continue
        h1 = compute_piece_pitch_entropy(ids, 1, BAR_EV, PITCH_EVS)
        h4 = compute_piece_pitch_entropy(ids, 4, BAR_EV, PITCH_EVS)
        gs = compute_piece_groove_similarity(ids, BAR_EV, POS_EVS, PITCH_EVS, max_pairs=args.max_pairs)
        rows.append((mp.name, h1, h4, gs))

    df = pd.DataFrame(rows, columns=["piece_name", "H1", "H4", "GS"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[eval] wrote {args.out_csv}  | files={len(df)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task2 混合評分（miditok REMI+ 與 custom REMI、含/不含 chord 都可）
- 不需要 dictionary.pkl：直接由 MIDI 量化為 1/16 事件流，建立自訂 BAR / POS / PITCH ids
- 指標：GS_within、GS_to_prompt、PitchHistCos、H4_match、DensityMatch、RegisterMatch
- 輸出每檔分數與各 config 平均分，協助挑出 prompt_1 下的最佳生成

需求：
  pip install miditoolkit numpy pandas tqdm
  git clone https://github.com/slSeanWU/MusDr.git
  export PYTHONPATH=$PWD/MusDr:$PYTHONPATH
"""

import os, sys, argparse, math
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import miditoolkit

# ---- MusDr ----
try:
    from musdr.side_utils import (
        get_bars_crop,
        get_pitch_histogram,
        compute_histogram_entropy,
        get_onset_xor_distance,
    )
except Exception:
    print("[rank] 找不到 musdr.side_utils，請先：git clone https://github.com/slSeanWU/MusDr.git 並 export PYTHONPATH")
    sys.exit(1)

BAR_ID = 999_999
def build_pos_ids(fraction=16):
    return [10_000 + i for i in range(fraction)]
def build_pitch_ids():
    return [20_000 + p for p in range(128)]

def nearest_pos_idx(tick, bar_st, bar_ticks, fraction=16):
    flags = np.linspace(bar_st, bar_st + bar_ticks, fraction, endpoint=False, dtype=float)
    return int(np.argmin(np.abs(flags - tick)))

def ticks_per_bar(midi: miditoolkit.MidiFile):
    tpq = midi.ticks_per_beat
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[0]  # 取第一個拍號
        beats_per_bar = ts.numerator
        beat_unit = ts.denominator
    else:
        beats_per_bar, beat_unit = 4, 4
    return int(tpq * beats_per_bar * (4.0 / beat_unit))

def midi_to_words(midi_path: str, *, fraction=16, merge_tracks=True):
    """直接由 MIDI 建立通用事件流（含 BAR、Position、Pitch）。"""
    m = miditoolkit.MidiFile(midi_path)
    bar_ticks = ticks_per_bar(m)
    pos_ids = build_pos_ids(fraction)
    pitch_ids = build_pitch_ids()

    # 收集所有音符
    notes = []
    tracks = m.instruments if merge_tracks else (m.instruments[:1] if m.instruments else [])
    for inst in tracks:
        for n in inst.notes:
            notes.append((n.start, n.pitch))
    if not notes:
        return []

    notes.sort(key=lambda x: x[0])
    max_tick = max(s for s, _ in notes)
    # downbeats
    bars = list(range(0, int(max_tick + bar_ticks), bar_ticks))

    words = []
    i = 0  # note pointer
    for b in bars:
        words.append(BAR_ID)
        # 本小節範圍
        b_end = b + bar_ticks
        while i < len(notes) and notes[i][0] < b_end:
            st, pitch = notes[i]
            pos = nearest_pos_idx(st, b, bar_ticks, fraction=fraction)
            words.append(pos_ids[pos])
            if 0 <= pitch < 128:
                words.append(pitch_ids[pitch])
            i += 1
    return words

def split_bars(words):
    bars, cur = [], []
    for w in words:
        if w == BAR_ID:
            if cur: bars.append(cur); cur=[]
        else:
            cur.append(w)
    if cur: bars.append(cur)
    return bars

def drop_first_bars(words, n):
    if n <= 0: return words
    seen = 0; out = []
    for w in words:
        if w == BAR_ID:
            seen += 1
            if seen <= n:
                continue 
            else :
                None
        if seen > n: out.append(w)
    return out

def cosine_sim(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.sum()==0 or b.sum()==0: return 0.0
    return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def compute_gs_within(bars, POS, PITCH):
    if len(bars) <= 1: return np.nan
    sims = []
    for i in range(len(bars)):
        for j in range(i + 1, len(bars)):
            # 之前是：get_onset_xor_distance(bars[i], bars[j], ...)
            d = get_onset_xor_distance([BAR_ID] + bars[i],
                                       [BAR_ID] + bars[j],
                                       BAR_ID, POS, pitch_evs=PITCH)
            sims.append(1.0 - d)
    return float(np.mean(sims)) if sims else np.nan


def best_match_to_prompt(bars_cont, bars_prompt, POS, PITCH):
    if not bars_cont or not bars_prompt: return np.nan
    sims = []
    for bc in bars_cont:
        best = 0.0
        for bp in bars_prompt:
            # 之前是：get_onset_xor_distance(bc, bp, ...)
            d = get_onset_xor_distance([BAR_ID] + bc,
                                       [BAR_ID] + bp,
                                       BAR_ID, POS, pitch_evs=PITCH)
            best = max(best, 1.0 - d)
        sims.append(best)
    return float(np.mean(sims))


def note_density_per_bar(bars, pitch_set):
    return np.array([sum(1 for w in b if w in pitch_set) for b in bars], float)

def mean_pitch(words, PITCH):
    idxmap = {pid:i for i,pid in enumerate(PITCH)}
    seq = [idxmap[w] for w in words if w in idxmap]
    return float(np.mean(seq)) if seq else np.nan

def h_entropy(words, window_bars, PITCH):
    bars = split_bars(words)
    n = len(bars)
    if n==0: return np.nan
    window_bars = min(window_bars, n)
    ents=[]
    for st in range(0, n-window_bars+1):
        seg=[]
        for k in range(st, st+window_bars):
            seg.append(BAR_ID); seg.extend(bars[k])
        ph = get_pitch_histogram(seg, pitch_evs=PITCH)
        if ph is not None:
            ents.append(compute_histogram_entropy(ph))
    return float(np.mean(ents)) if ents else np.nan

def score_file(gen_mid, prompt_mid, *, fraction=16, skip_first_gen_bars=0):
    POS = build_pos_ids(fraction); PITCH = build_pitch_ids()
    g_words = midi_to_words(gen_mid, fraction=fraction)
    p_words = midi_to_words(prompt_mid, fraction=fraction)
    if not g_words or not p_words:
        return dict(score=np.nan, GS_to_prompt=np.nan, GS_within=np.nan,
                    PitchHistCos=np.nan, H4_cont=np.nan, H4_prompt=np.nan,
                    H4_match=np.nan, DensityMatch=np.nan, RegisterMatch=np.nan)

    # 只評「續寫」：若你的輸出檔包含 8bar prompt，請用 --skip_first_gen_bars 8
    if skip_first_gen_bars>0:
        # 將 g_words 按 bar 切並丟掉前 n 個
        bars = split_bars(g_words)
        if len(bars)>skip_first_gen_bars:
            g_words = []
            for b in bars[skip_first_gen_bars:]:
                g_words.append(BAR_ID); g_words.extend(b)

    bars_g = split_bars(g_words)
    bars_p = split_bars(p_words)[:8]

    GS_within = compute_gs_within(bars_g, POS, PITCH)
    GS_to_prompt = best_match_to_prompt(bars_g, bars_p, POS, PITCH)

    def pitch_hist(words):
        return get_pitch_histogram([BAR_ID]+words, pitch_evs=PITCH)
    ph_g = pitch_hist(g_words); ph_p = pitch_hist(p_words)
    PitchHistCos = cosine_sim(ph_g, ph_p) if (ph_g is not None and ph_p is not None) else np.nan

    H4_g = h_entropy(g_words, 4, PITCH); H4_p = h_entropy(p_words, 4, PITCH)
    H4_match = (0.0 if (np.isnan(H4_g) or np.isnan(H4_p))
                else max(0.0, 1.0 - abs(H4_g - H4_p)/max(1e-6, H4_p)))

    dens_g = note_density_per_bar(bars_g, set(PITCH))
    dens_p = note_density_per_bar(bars_p, set(PITCH))
    DensityMatch = (np.nan if (dens_g.size==0 or dens_p.size==0)
                    else max(0.0, 1.0 - abs(dens_g.mean()-dens_p.mean())/max(1.0, dens_p.mean())))

    m_g = mean_pitch(g_words, PITCH); m_p = mean_pitch(p_words, PITCH)
    RegisterMatch = (np.nan if (np.isnan(m_g) or np.isnan(m_p))
                     else max(0.0, 1.0 - abs(m_g-m_p)/48.0))

    def nz(x): 
        return 0.0 if (x is None or (isinstance(x,float) and np.isnan(x))) else float(x)

    score = (0.30*nz(GS_to_prompt) + 0.20*nz(GS_within) + 0.20*nz(PitchHistCos) +
             0.15*nz(DensityMatch) + 0.10*nz(RegisterMatch) + 0.05*nz(H4_match))

    return dict(score=score, GS_to_prompt=GS_to_prompt, GS_within=GS_within,
                PitchHistCos=PitchHistCos, H4_cont=H4_g, H4_prompt=H4_p,
                H4_match=H4_match, DensityMatch=DensityMatch, RegisterMatch=RegisterMatch)

def parse_gen_specs(specs):
    """
    支援格式：--gen cfgA:/path/to/dir  （名稱:資料夾）
    可重複多次。
    """
    out=[]
    for s in specs:
        if ":" not in s:
            raise SystemExit(f"--gen 需要 name:dir → 收到：{s}")
        name, d = s.split(":",1)
        out.append((name, Path(d)))
    return out

def main():
    ap = argparse.ArgumentParser("Task2 混合評分（不用 pkl）")
    ap.add_argument("--prompt", required=True, help="8-bar prompt MIDI")
    ap.add_argument("--gen", action="append", required=True,
                    help="格式 name:dir，可給多次（例如 cfgA:/path/A）")
    ap.add_argument("--out_csv", required=True, help="輸出 per-file CSV")
    ap.add_argument("--fraction", type=int, default=16, help="每小節等分（預設 16）")
    ap.add_argument("--skip_first_gen_bars", type=int, default=0,
                    help="若生成檔案含 8bar prompt，請設 8 只評後 24 bars")
    args = ap.parse_args()

    gen_specs = parse_gen_specs(args.gen)
    rows=[]
    for name, d in gen_specs:
        mids = sorted([p for ext in ("*.mid","*.MID") for p in d.glob(ext)])
        if not mids:
            print(f"[rank] {name} → {d} 沒有 MIDI，跳過")
            continue
        for mp in tqdm(mids, desc=f"Scoring {name}", ncols=100):
            m = score_file(str(mp), args.prompt, fraction=args.fraction,
                           skip_first_gen_bars=args.skip_first_gen_bars)
            rows.append({"config":name, "file":mp.name, **m})

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[rank] wrote per-file scores → {args.out_csv}")

    if not df.empty:
        ranking = (df.groupby("config")["score"].mean()
                   .reset_index().sort_values("score", ascending=False))
        print("\n=== Config ranking (prompt-specific) ===")
        for _, r in ranking.iterrows():
            print(f"{r['config']:>20s} : {r['score']:.4f}")
        print("========================================")

if __name__ == "__main__":
    main()

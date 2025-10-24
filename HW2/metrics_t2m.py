#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute metrics for rows: target,generated,prompt

Metrics:
  - CLAP (text↔audio & audio↔audio)
      * clap_t2a_target : prompt vs target audio        (你的 1)
      * clap_t2a_gen    : prompt vs generated audio     (你的 2)
      * clap_a2a        : generated vs target audio
  - Meta Audiobox Aesthetics (CE, CU, PC, PQ)  (預設算在 generated；可選 target/both)
  - Melody accuracy (one-hot chroma match, gen vs target)

Robust features:
  - 路徑自動補 base_root（相對路徑時）
  - 文本/音訊 embedding 簡易快取，省時
  - 失敗不崩潰、逐列印出進度
"""

import argparse, csv, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import librosa
import scipy.signal as signal
import torch

# ---------------- Globals (caches) ----------------
_CLAP = None
_AESTH = None
_AUDIO_EMB_CACHE: Dict[str, np.ndarray] = {}
_TEXT_EMB_CACHE: Dict[str, np.ndarray] = {}

# ---------------- CLAP ----------------
def _load_clap(device: str):
    global _CLAP
    if _CLAP is not None:
        return _CLAP
    import laion_clap
    m = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    m.load_ckpt()
    m.eval()
    _CLAP = m
    return _CLAP

@torch.no_grad()
def _clap_text_emb(text: str, device: str) -> np.ndarray:
    if text in _TEXT_EMB_CACHE:
        return _TEXT_EMB_CACHE[text]
    model = _load_clap(device)
    # 兼容不同版本 API
    if hasattr(model, "get_text_embedding"):
        e = model.get_text_embedding([text], use_tensor=False)
    elif hasattr(model, "get_text_embedding_from_data"):
        e = model.get_text_embedding_from_data([text], use_tensor=False)
    else:
        raise RuntimeError("laion_clap: no text embedding method found")
    v = np.squeeze(np.asarray(e, dtype=np.float32))
    v = v / (np.linalg.norm(v) + 1e-12)
    _TEXT_EMB_CACHE[text] = v
    return v

@torch.no_grad()
def _clap_audio_emb_from_wav(wav: np.ndarray, device: str) -> np.ndarray:
    key = f"len:{len(wav)}:hash:{hash(wav.tobytes()[:1_000_000])}"  # 粗略快取鍵
    if key in _AUDIO_EMB_CACHE:
        return _AUDIO_EMB_CACHE[key]
    model = _load_clap(device)
    e = model.get_audio_embedding_from_data(x=wav.reshape(1, -1), use_tensor=False)
    v = np.squeeze(np.asarray(e, dtype=np.float32))
    v = v / (np.linalg.norm(v) + 1e-12)
    _AUDIO_EMB_CACHE[key] = v
    return v

@torch.no_grad()
def _clap_audio_emb_from_path(path: str, device: str, sr: int) -> np.ndarray:
    # 為了穩定性讓所有檔案同一 sr；CLAP 內部也會處理
    y, _ = librosa.load(path, sr=sr, mono=True)
    return _clap_audio_emb_from_wav(y, device)

def clap_cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))

# ---------------- Aesthetics ----------------
def _load_audiobox():
    global _AESTH
    if _AESTH is not None:
        return _AESTH
    from audiobox_aesthetics.infer import initialize_predictor
    _AESTH = initialize_predictor()
    return _AESTH

@torch.no_grad()
def audiobox_scores(audio_path: str) -> Dict[str, float]:
    try:
        predictor = _load_audiobox()
        out = predictor.forward([{"path": audio_path}])
        r = out[0] if out else {}
        def f(k): 
            v = r.get(k, np.nan)
            try: return float(v)
            except: return np.nan
        return dict(CE=f("CE"), CU=f("CU"), PC=f("PC"), PQ=f("PQ"))
    except Exception as e:
        print(f"[audiobox] skip {audio_path}: {e}", file=sys.stderr)
        return dict(CE=np.nan, CU=np.nan, PC=np.nan, PQ=np.nan)

# ---------------- Melody similarity ----------------
def extract_melody_one_hot(
    audio_path: str,
    sr: int = 44100,
    cutoff: float = 261.2,   # C4 ~ 261.63 Hz
    win_length: int = 2048,
    hop_length: int = 256,
) -> np.ndarray:
    import torchaudio
    from torchaudio import transforms as T
    audio, in_sr = torchaudio.load(audio_path)  # (C,T)
    y = audio.mean(dim=0)
    if in_sr != sr:
        y = T.Resample(in_sr, sr)(y)
    y = y.numpy()

    nyq = 0.5 * sr
    b, a = signal.butter(2, cutoff / nyq, btype="high")
    y = signal.filtfilt(b, a, y)

    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=win_length, win_length=win_length, hop_length=hop_length
    )  # (12, F)
    idx = np.argmax(chroma, axis=0)
    onehot = np.zeros_like(chroma)
    onehot[idx, np.arange(chroma.shape[1])] = 1.0
    return onehot

def melody_accuracy(gen_path: str, tgt_path: str) -> float:
    g, t = extract_melody_one_hot(gen_path), extract_melody_one_hot(tgt_path)
    Tlen = min(g.shape[1], t.shape[1])
    if Tlen == 0:
        return float("nan")
    matches = ((g[:, :Tlen] == t[:, :Tlen]) & (g[:, :Tlen] == 1)).sum()
    return float(matches) / float(Tlen)

# ---------------- IO helpers ----------------
def _resolve(path_str: str, base_root: Optional[Path]) -> str:
    p = Path(path_str.strip().strip('"'))
    if p.is_absolute():
        return str(p)
    if base_root is not None:
        q = (base_root / p).resolve()
        return str(q)
    return str(p.resolve())

def read_triplets_csv(p: Path) -> List[Tuple[str, str, str]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        exp = {"target","generated","prompt"}
        if set(rdr.fieldnames or []) & exp != exp:
            raise SystemExit("CSV must have columns: target,generated,prompt")
        for row in rdr:
            rows.append((row["target"], row["generated"], row["prompt"]))
    return rows

def load_trim_pair(tgt_path: str, gen_path: str, sr: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    gen, _ = librosa.load(gen_path, sr=sr, mono=True)
    tgt, _ = librosa.load(tgt_path, sr=sr, mono=True)
    L = min(len(gen), len(tgt))
    if L == 0:
        return gen, tgt
    return gen[:L], tgt[:L]

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Compute T-G-P metrics")
    ap.add_argument("--pairs_csv", required=True, help="CSV with columns: target,generated,prompt")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--base_root", default="", help="Optional base folder to resolve relative paths")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--skip_aesthetics", action="store_true")
    ap.add_argument("--aesthetics_on", choices=["generated","target","both"], default="generated")
    args = ap.parse_args()

    base_root = Path(args.base_root).resolve() if args.base_root else None
    rows = read_triplets_csv(Path(args.pairs_csv))

    # warmups
    _ = _load_clap(args.device)
    if not args.skip_aesthetics:
        try:
            _ = _load_audiobox()
        except Exception as e:
            print("[WARN] audiobox init failed → skip:", e, file=sys.stderr)
            args.skip_aesthetics = True

    out_rows = []
    for i, (tgt_raw, gen_raw, prompt) in enumerate(rows, 1):
        tgt_path = _resolve(tgt_raw, base_root)
        gen_path = _resolve(gen_raw, base_root)

        if not Path(tgt_path).exists():
            print(f"[{i}] target not found: {tgt_path}", file=sys.stderr)
            continue
        if not Path(gen_path).exists():
            print(f"[{i}] generated not found: {gen_path}", file=sys.stderr)
            continue

        try:
            # --- CLAP: text→audio (target / generated) ---
            t_text = _clap_text_emb(prompt, device=args.device)
            t_tgt  = _clap_audio_emb_from_path(tgt_path, device=args.device, sr=args.sr)
            t_gen  = _clap_audio_emb_from_path(gen_path, device=args.device, sr=args.sr)
            clap_t2a_target = clap_cosine(t_text, t_tgt)   # (1)
            clap_t2a_gen    = clap_cosine(t_text, t_gen)   # (2)

            # --- CLAP: audio↔audio (gen vs tgt；長度對齊) ---
            gen_wav, tgt_wav = load_trim_pair(tgt_path, gen_path, sr=args.sr)
            v_g = _clap_audio_emb_from_wav(gen_wav, device=args.device)
            v_t = _clap_audio_emb_from_wav(tgt_wav, device=args.device)
            clap_a2a = clap_cosine(v_g, v_t)

            # --- Aesthetics ---
            CE_g=CU_g=PC_g=PQ_g=np.nan
            CE_t=CU_t=PC_t=PQ_t=np.nan
            if not args.skip_aesthetics:
                if args.aesthetics_on in ("generated","both"):
                    scg = audiobox_scores(gen_path)
                    CE_g, CU_g, PC_g, PQ_g = scg["CE"], scg["CU"], scg["PC"], scg["PQ"]
                if args.aesthetics_on in ("target","both"):
                    sct = audiobox_scores(tgt_path)
                    CE_t, CU_t, PC_t, PQ_t = sct["CE"], sct["CU"], sct["PC"], sct["PQ"]

            # --- Melody accuracy ---
            mel = melody_accuracy(gen_path, tgt_path)

            out_rows.append(dict(
                target=Path(tgt_path).name,
                generated=Path(gen_path).name,
                prompt=prompt,

                CLAP_t2a_target=f"{clap_t2a_target:.6f}",
                CLAP_t2a_gen=f"{clap_t2a_gen:.6f}",
                CLAP_a2a=f"{clap_a2a:.6f}",

                CE_gen=f"{CE_g:.6f}" if np.isfinite(CE_g) else "",
                CU_gen=f"{CU_g:.6f}" if np.isfinite(CU_g) else "",
                PC_gen=f"{PC_g:.6f}" if np.isfinite(PC_g) else "",
                PQ_gen=f"{PQ_g:.6f}" if np.isfinite(PQ_g) else "",

                CE_tgt=f"{CE_t:.6f}" if np.isfinite(CE_t) else "",
                CU_tgt=f"{CU_t:.6f}" if np.isfinite(CU_t) else "",
                PC_tgt=f"{PC_t:.6f}" if np.isfinite(PC_t) else "",
                PQ_tgt=f"{PQ_t:.6f}" if np.isfinite(PQ_t) else "",

                melody_acc=f"{mel:.6f}" if mel==mel else "",  # NaN 保留空白
            ))
            print(f"[{i}] {Path(tgt_path).name} ←→ {Path(gen_path).name} | "
                  f"t2a(tgt)={clap_t2a_target:.3f}  t2a(gen)={clap_t2a_gen:.3f}  "
                  f"a2a={clap_a2a:.3f}  melody={mel:.3f}")
        except Exception as e:
            print(f"[{i}] skip: {e}", file=sys.stderr)

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target","generated","prompt",
        "CLAP_t2a_target","CLAP_t2a_gen","CLAP_a2a",
        "CE_gen","CU_gen","PC_gen","PQ_gen",
        "CE_tgt","CU_tgt","PC_tgt","PQ_tgt",
        "melody_acc",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[DONE] saved → {out}")

if __name__ == "__main__":
    main()

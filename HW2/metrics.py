"""
Read a CSV with columns: target,best_ref,cosine
Find audio files by stem in target_dir / ref_dir,
then compute:
  1) CLAP cosine (target trimmed to best_ref length)
  2) Meta Audiobox Aesthetics (CE, CU, PC, PQ) on best_ref
  3) Melody accuracy (one-hot chroma match)

Output: CSV with all metrics appended.
"""

import argparse, csv, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import librosa
import scipy.signal as signal
import torch
import re, unicodedata, difflib


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

# ---------- helpers: indexing & lookup ----------
def _normalize_key(s: str) -> str:

    s = Path(s).stem
    s = unicodedata.normalize("NFKC", s)   
    s = s.lower()
    s = s.replace("×", "x")                
    s = s.replace("・", "")
    s = s.replace("／", "/")
    # 把空白、破折號、底線都視為同一類分隔
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return s

def index_audio(root: Path) -> Dict[str, Path]:
    """
    為資料夾內所有音檔建立多個別名→路徑的索引，優先 .wav。
    """
    root = Path(root)
    table: Dict[str, Path] = {}
    for ext in AUDIO_EXTS:
        for p in root.rglob(f"*{ext}"):
            stem_raw = p.stem
            aliases = {
                stem_raw,
                stem_raw.lower(),
                stem_raw.replace(" ", "_"),
                stem_raw.lower().replace(" ", "_").replace("-", "_"),
                _normalize_key(stem_raw),
            }
            for k in aliases:
                k = k.strip()
                if not k:
                    continue
                # 若重複鍵，偏好 .wav 取代其他格式
                if (k not in table) or (table[k].suffix != ".wav" and p.suffix == ".wav"):
                    table[k] = p.resolve()
    return table

def find_by_stem(table: Dict[str, Path], key: str) -> Optional[Path]:
    """
    依 CSV 給的 key 做多輪正規化查找；找不到就給相近建議。
    """
    candidates = [
        key,
        key.lower(),
        key.replace(" ", "_"),
        key.lower().replace(" ", "_").replace("-", "_"),
        _normalize_key(key),
    ]
    for c in candidates:
        if c in table:
            return table[c]

    # 找不到：提供最接近的三個建議（debug 用）
    close = difflib.get_close_matches(_normalize_key(key), list(table.keys()), n=3, cutoff=0.6)
    if close:
        print(f"[warn] not found: {key} → did you mean: {', '.join(close)} ?", file=sys.stderr)
    else:
        print(f"[warn] not found: {key}", file=sys.stderr)
    return None


# ---------- metric 1: CLAP ----------
_CLAP = None
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
def clap_cosine(gen_wav: np.ndarray, tgt_wav: np.ndarray, device: str) -> float:
    model = _load_clap(device)
    def emb(x: np.ndarray) -> np.ndarray:
        e = model.get_audio_embedding_from_data(x=x.reshape(1, -1), use_tensor=False)  # (1,512)
        v = np.squeeze(e).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)
    eg, et = emb(gen_wav), emb(tgt_wav)
    return float(np.dot(eg, et))

# ---------- metric 2: Meta Audiobox Aesthetics ----------
_AESTH = None
def _load_audiobox():
    global _AESTH
    if _AESTH is not None:
        return _AESTH
    from audiobox_aesthetics.infer import initialize_predictor
    _AESTH = initialize_predictor()
    return _AESTH

@torch.no_grad()
def audiobox_scores(audio_path: str) -> Dict[str, float]:
    predictor = _load_audiobox()
    out = predictor.forward([{"path": audio_path}])
    if not out: return dict(CE=np.nan, CU=np.nan, PC=np.nan, PQ=np.nan)
    r = out[0]
    return dict(
        CE=float(r.get("CE", np.nan)),
        CU=float(r.get("CU", np.nan)),
        PC=float(r.get("PC", np.nan)),
        PQ=float(r.get("PQ", np.nan)),
    )

# ---------- metric 3: Melody accuracy ----------
def extract_melody_one_hot(
    audio_path: str,
    sr: int = 44100,
    cutoff: float = 261.2,
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
    matches = ((g[:, :Tlen] == t[:, :Tlen]) & (g[:, :Tlen] == 1)).sum()
    return float(matches) / float(Tlen)

# ---------- IO ----------
def read_pairs_csv(p: Path) -> List[Tuple[str, str, Optional[str]]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            tgt = row.get("target")
            br  = row.get("best_ref")
            cos = row.get("cosine")
            if not tgt or not br:
                raise SystemExit("CSV must have columns: target,best_ref[,cosine]")
            rows.append((tgt, br, cos))
    return rows

def load_trim_pair(tgt_path: str, gen_path: str, sr: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    gen, _ = librosa.load(gen_path, sr=sr, mono=True)
    tgt, _ = librosa.load(tgt_path, sr=sr, mono=True)
    L = min(len(gen), len(tgt))
    return gen[:L], tgt[:L]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Compute metrics for target/best_ref pairs")
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--skip_audiobox", action="store_true")
    args = ap.parse_args()

    pairs = read_pairs_csv(Path(args.pairs_csv))

    tgt_index = index_audio(Path(args.target_dir))
    ref_index = index_audio(Path(args.ref_dir))
    
    # warmup
    _ = _load_clap(args.device)
    if not args.skip_audiobox:
        try:
            _ = _load_audiobox()
        except Exception as e:
            print("[WARN] audiobox init failed → skip:", e, file=sys.stderr)
            args.skip_audiobox = True

    out_rows = []
    for i, (tgt_key, ref_key, prev_cos) in enumerate(pairs, 1):
        tgt_p = find_by_stem(tgt_index, tgt_key)
        ref_p = find_by_stem(ref_index, ref_key)
        if tgt_p is None:
            print(f"[{i}] target not found: {tgt_key}", file=sys.stderr); continue
        if ref_p is None:
            print(f"[{i}] best_ref not found: {ref_key}", file=sys.stderr); continue

        try:
            # CLAP（把 target 修到和 best_ref 一樣長）
            gen_wav, tgt_wav = load_trim_pair(str(tgt_p), str(ref_p), sr=args.sr)
            clap = clap_cosine(gen_wav, tgt_wav, device=args.device)

            # Aesthetics（對 best_ref 算）
            CE=CU=PC=PQ=np.nan
            if not args.skip_audiobox:
                sc = audiobox_scores(str(ref_p))
                CE, CU, PC, PQ = sc["CE"], sc["CU"], sc["PC"], sc["PQ"]

            # Melody accuracy（兩音檔直接比）
            mel = melody_accuracy(str(ref_p), str(tgt_p))

            out_rows.append(dict(
                target=str(tgt_p.name),
                best_ref=str(ref_p.name),
                retrieval_cosine=prev_cos if prev_cos is not None else "",
                CLAP_cosine=f"{clap:.6f}",
                CE=f"{CE:.6f}" if np.isfinite(CE) else "",
                CU=f"{CU:.6f}" if np.isfinite(CU) else "",
                PC=f"{PC:.6f}" if np.isfinite(PC) else "",
                PQ=f"{PQ:.6f}" if np.isfinite(PQ) else "",
                melody_acc=f"{mel:.6f}",
            ))
            print(f"[{i}] {tgt_p.name}  ↔  {ref_p.name} | CLAP={clap:.3f}  melody={mel:.3f}")
        except Exception as e:
            print(f"[{i}] skip {tgt_key}–{ref_key}: {e}", file=sys.stderr)

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["target","best_ref","retrieval_cosine","CLAP_cosine","CE","CU","PC","PQ","melody_acc"]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in out_rows: w.writerow(r)
    print(f"[DONE] saved → {out}")

if __name__ == "__main__":
    main()

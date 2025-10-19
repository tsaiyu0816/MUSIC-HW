"""
Read a folder of audio files, compute latent with music2latent.EncoderDecoder,
and save one .npy per audio. No checkpoint needed.
"""
import argparse
from pathlib import Path
from typing import List
import numpy as np
import librosa

from music2latent import EncoderDecoder

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

def list_audio(p: Path) -> List[Path]:
    files: List[Path] = []
    for ext in AUDIO_EXTS:
        files += sorted(p.rglob(f"*{ext}"))
    return sorted({f.resolve() for f in files})

def to_vector(latent, pool: str = "mean") -> np.ndarray:
    """Pool time axis (last dim) then flatten → 1D float32."""
    if hasattr(latent, "detach"):
        arr = latent.detach().cpu().numpy()
    else:
        arr = np.asarray(latent)
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim == 1:
        return arr
    if pool == "mean":
        arr = arr.mean(axis=-1)  # pool last axis as time
    elif pool == "max":
        arr = arr.max(axis=-1)
    else:
        raise ValueError("pool must be 'mean' or 'max'")
    return arr.reshape(-1)

def main():
    ap = argparse.ArgumentParser("encode audio → latent (.npy)")
    ap.add_argument("--input_dir", required=True, type=str)
    ap.add_argument("--out_dir",   required=True, type=str)
    ap.add_argument("--sr", type=int, default=44100, help="resample rate for librosa")
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "max"],
                    help="temporal pooling on latent output")
    args = ap.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    files = list_audio(in_dir)
    if not files:
        raise SystemExit(f"No audio files in {in_dir}")

    # init encoder (no checkpoint needed)
    try:
        encdec = EncoderDecoder()
    except Exception as e:
        raise SystemExit(f"Failed to create EncoderDecoder: {e}")

    print(f"[data] {len(files)} files")
    for i, f in enumerate(files, 1):
        try:
            wv, sr = librosa.load(str(f), sr=args.sr, mono=True)
            latent = encdec.encode(wv)               # ← 直接 encode(waveform)
            vec = to_vector(latent, pool=args.pool)  # 轉固定長度向量
            np.save(out_dir / f"{f.stem}.npy", vec)
            print(f"  [{i}/{len(files)}] {f.name} → {vec.shape}")
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")

if __name__ == "__main__":
    main()

"""
Read a folder of audio files, compute latent with EncoderDecoder,
and save one .npy per audio. No checkpoint needed.
"""
import argparse
from pathlib import Path
from typing import List
import numpy as np
import librosa
import torch
from diffusers.models import AutoencoderOobleck

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

DS_FACTOR = 2048  # Oobleck 的總下採樣率

def pad_to_multiple(x: np.ndarray, m: int) -> np.ndarray:
    """x: (C, T) → 右側補 0 到 T 是 m 的倍數"""
    T = x.shape[-1]
    r = (-T) % m
    return x if r == 0 else np.pad(x, ((0,0),(0,r)), mode="constant")


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

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        subfolder="vae",              # 這個模型的 VAE 在 repo 的 vae/ 子資料夾
        torch_dtype=torch.float16     # 若用 CPU 改成 torch.float32
    ).to(device).eval()

    print(f"[data] {len(files)} files")
    for i, f in enumerate(files, 1):
        try:
            # 1) 讀單聲道 → 複製成雙聲道 (2, T)，VAE 是用雙聲道訓練的
            wv, _ = librosa.load(str(f), sr=args.sr, mono=True)
            audio = np.stack([wv, wv], axis=0).astype(np.float32)   # (2, T)

            # 2) 對齊長度到 2048 的整數倍，避免下採樣殘數
            audio = pad_to_multiple(audio, DS_FACTOR)

            xt = torch.from_numpy(audio).unsqueeze(0).to(device, dtype=vae.dtype)  # (1,2,T)

            # 4) Encode：得到 latent 時序 (1, 64, L)
            with torch.no_grad():
                enc_out = vae.encode(xt)               # 包含 latent_dist（mean / std）
                lat = enc_out.latent_dist.mean         # 或 .sample()，想 deterministic 就用 mean

            print("latent shape:", tuple(lat.shape))    

            vec = to_vector(lat, pool=args.pool)        
            np.save(out_dir / f"{f.stem}.npy", vec)

            print(f"  [{i}/{len(files)}] {f.name} → {vec.shape}")
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")


if __name__ == "__main__":
    main()

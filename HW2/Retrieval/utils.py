"""
Cosine similarity on saved .npy embeddings (encoder-agnostic).
"""
import argparse, csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def _load_emb_dir(d: str) -> Dict[str, np.ndarray]:
    p = Path(d); files = sorted(p.glob("*.npy"))
    if not files: raise SystemExit(f"No .npy in {d}")
    table = {}
    for f in files:
        z = np.load(f)
        if z.ndim > 1: z = z.reshape(-1)
        table[f.stem] = z.astype(np.float32)
    return table

def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n

def cosine_top1(targets: Dict[str, np.ndarray], refs: Dict[str, np.ndarray]) -> List[Tuple[str, str, float]]:
    tk = list(targets.keys()); rk = list(refs.keys())
    T = np.stack([targets[k] for k in tk], axis=0)
    R = np.stack([refs[k] for k in rk], axis=0)
    S = _l2norm(T) @ _l2norm(R).T
    idx = S.argmax(axis=1)
    return [(tk[i], rk[int(j)], float(S[i, int(j)])) for i, j in enumerate(idx)]

def save_top1_csv(rows: List[Tuple[str, str, float]], out_csv: str):
    p = Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["target", "best_ref", "cosine"])
        for t, r, s in rows: w.writerow([t, r, f"{s:.6f}"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--refs",    required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = cosine_top1(_load_emb_dir(args.targets), _load_emb_dir(args.refs))
    save_top1_csv(rows, args.out_csv)
    print(f"Saved → {args.out_csv}")

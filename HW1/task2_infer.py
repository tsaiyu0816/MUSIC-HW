# -*- coding: utf-8 -*-
# file: task2_infer.py
import os, json, argparse, tempfile
import numpy as np
from tqdm import tqdm
import math
import librosa

import torch
from torch.utils.data import DataLoader

from task2_dataload import (
    CFG, ChunkDatasetFixed, _precompute_features, aggregate_by_keys, _chunk_quality
)

from model import SCNN

@torch.no_grad()
def eval_chunks_logits(model, loader, device):
    model.eval()
    probs, keys = [], []
    for x, _, k in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p); keys.extend(k)
    return np.vstack(probs), keys

def list_audio(root):
    ok = {'.mp3'}
    paths = []
    for name in sorted(os.listdir(root))[:]:
        p = os.path.join(root, name)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in ok:
            paths.append(os.path.abspath(p))
    return paths

def build_index_unlabeled(json_path, cfg):
    # 不檢查類別資料夾，單純切片；label 一律給 0（inference 不用到 y）
    with open(json_path, "r", encoding="utf-8") as f:
        paths = json.load(f)
    seg_len = int(cfg.segment_sec * cfg.sr)
    step = max(1, int(round(seg_len * (1.0 - float(cfg.overlap)))))
    index = []
    for p in paths:
        try:
            dur = librosa.get_duration(path=p)
        except TypeError:
            dur = librosa.get_duration(filename=p)
        total = int(max(1, math.floor((dur * cfg.sr - seg_len) / step) + 1))
        key = os.path.splitext(os.path.basename(p))[0]
        for k in range(total):
            s = k * step
            e = s + seg_len
            index.append((p, s, e, 0, key))  # lab=0 佔位
    return index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="ckpt from training (task2_best.pt or task2_scnn.pt)")
    ap.add_argument("--test_root", type=str, required=True, help="folder containing 001.mp3 ~ NNN.mp3")
    ap.add_argument("--cache_dir", type=str, default="cache_task2_infer")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)  # 評估穩定起見，預設 0
    ap.add_argument("--vote_method", type=str, default="mean", choices=["mean","majority"])
    ap.add_argument("--out_json", type=str, default="infer_top3.json")
    ap.add_argument("--overlap", type=float, default=None,
                help="Chunk overlap ratio in [0, 1). If not set, use training cfg.")
    

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # === 1) 載入 checkpoint：拿 classes & cfg ===
    # === 1) 載入 checkpoint：拿 classes & cfg（安全載入）===
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        # 舊版 PyTorch 沒有 weights_only 參數，就退回原本行為
        ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["classes"]
    cfg_dict = ckpt["cfg"]
    # 用訓練時的設定，僅覆蓋 cache_dir
    cfg = CFG(**cfg_dict)
    cfg.cache_dir = args.cache_dir
    if args.overlap is not None:
        # 夾在 [0, 0.99] 比較安全，避免 step 變成 0
        cfg.overlap = max(0.0, min(0.99, float(args.overlap)))

    print(f"[INFER] segment_sec={cfg.segment_sec}, overlap={cfg.overlap:.2f}")

    # === 2) 把 test_root 轉成暫存 JSON（沿用 build_index）===
    audio_list = list_audio(args.test_root)
    if len(audio_list) == 0:
        raise RuntimeError(f"No audio found under: {args.test_root}")
    tmp_json = os.path.join(args.cache_dir, "infer_list.json")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(audio_list, f, ensure_ascii=False, indent=2)

    # === 3) 建 index → 預先算特徵（GPU 主進程跑 Demucs）===
    idx_te = build_index_unlabeled(tmp_json, cfg)
    _precompute_features(idx_te, cfg, device)

    # === 4) 組 DataLoader（split='val' 禁用 SpecAug；Dataset 在 CPU）===
    ds_te = ChunkDatasetFixed(idx_te, cfg, split="val", device=torch.device('cpu'))
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    
    # === 5) 載入模型 & 推論 ===
    n_classes = len(classes)
    model = SCNN(n_class=n_classes, n_mels=cfg.n_mels).to(device)
    model.load_state_dict(ckpt["model"])

    proba_te, keys_te = eval_chunks_logits(model, dl_te, device)
    # === 6) track-level 投票（和 val 完全一樣：aggregate_by_keys）===
    # test 沒有標籤，給一個 dummy 的 labels（不會影響投票）
    labels_dummy = np.zeros(proba_te.shape[0], dtype=np.int64)

    # 這裡 method 就用 args.vote_method（"mean"/"majority"），和 val 相同
    _, P_track, ukeys = aggregate_by_keys(
        proba=proba_te,
        labels=labels_dummy,
        keys=keys_te,
        n_classes=n_classes,
        method=args.vote_method
    )

    # 轉成每首歌的 top-3 類別
    results = {}
    for k, p in zip(ukeys, P_track):
        top3 = np.argsort(p)[-3:][::-1]
        results[k] = [classes[i] for i in top3]

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFER] wrote top-3 to: {args.out_json}")

if __name__ == "__main__":
    main()

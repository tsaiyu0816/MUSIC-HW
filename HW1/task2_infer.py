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
    CFG, ChunkDatasetFixed, _precompute_features, aggregate_by_keys,
    FeatureComputer, _separate_vocals_demucs_mem, _feat_cache_path
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
    for name in sorted(os.listdir(root))[:50]:
        p = os.path.join(root, name)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in ok:
            paths.append(os.path.abspath(p))
    return paths

def build_index_unlabeled(json_path, cfg):
    # 不檢查類別資料夾，單純切片；label 一律給 0（inference 不用到 y）
    with open(json_path, "r", encoding="utf-8") as f:
        paths = json.load(f)
    seg_len = int(cfg.segment_sec * cfg.sr)
    step = int(seg_len * (1.0 - cfg.overlap)) if cfg.overlap < 1.0 else 1
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
    ap.add_argument("--num_workers", type=int, default=4)  # 評估穩定起見，預設 0
    ap.add_argument("--vote_method", type=str, default="mean", choices=["mean","majority"])
    ap.add_argument("--out_json", type=str, default="infer_top3.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # === 1) 載入 checkpoint：拿 classes & cfg ===
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["classes"]
    cfg_dict = ckpt["cfg"]
    # 用訓練時的設定，僅覆蓋 cache_dir
    cfg = CFG(**cfg_dict)
    cfg.cache_dir = args.cache_dir

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

    # === (3.5) sanity check：確認 cache 真的存在、而且和“即時重算”一致 ===
    # 拿第一個 chunk 做對比（不會很慢）
    p0, s0, e0, _, _ = idx_te[0]
    c0 = _feat_cache_path(cfg.cache_dir, p0, s0, e0, cfg)
    if not os.path.exists(c0):
        raise RuntimeError(f"[SANITY] 缺少特徵快取：{c0}\n代表 _precompute_features 沒有成功寫出特徵（Demucs/特徵流程沒跑到）")

    F_cache = np.load(c0, allow_pickle=False)
    # 用跟訓練/val 完全相同的流程：Demucs → vocal mono → 取相同段落 → FeatureComputer.compute(apply_specaug=False)
    v0 = _separate_vocals_demucs_mem(p0, cfg, device)
    seg0 = v0[s0:e0]
    need = e0 - s0
    if seg0.shape[0] < need:
        seg0 = np.pad(seg0, (0, need - seg0.shape[0]))
    comp_dbg = FeatureComputer(cfg, device)
    F_now = comp_dbg.compute(seg0.astype(np.float32), apply_specaug=False)

    if F_cache.shape != F_now.shape:
        raise RuntimeError(f"[SANITY] 特徵 shape 不一致：cache={F_cache.shape}, now={F_now.shape}。多半是 cfg(如 sr/hop/n_fft/mels) 或 segment_sec/overlap 不一致。")
    mae = float(np.mean(np.abs(F_cache - F_now)))
    print(f"[SANITY] cache vs recompute 形狀一致 {F_cache.shape}，MAE={mae:.6f}（應該非常小）")

    # === 4) 組 DataLoader（split='val' 禁用 SpecAug；Dataset 在 CPU）===
    ds_te = ChunkDatasetFixed(idx_te, cfg, split="val", device=torch.device('cpu'))
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # === 5) 載入模型 & 推論 ===
    n_classes = len(classes)
    model = SCNN(n_classes=n_classes).to(device)
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

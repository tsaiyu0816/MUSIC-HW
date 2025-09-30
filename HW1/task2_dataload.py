# -*- coding: utf-8 -*-
# file: task2_dataload.py
import os, json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import hashlib
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# =========================
# Config
# =========================
@dataclass
class CFG:
    sr: int = 22050
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 128
    fmin: int = 40
    fmax: int = 8000
    segment_sec: int = 10               # 每段秒數（Task 2 用短片段 CNN，建議 8~15s）
    overlap: float = 0.0                # 切片重疊比例 (train/val/test 全切片)
    batch_size: int = 32
    num_workers: int = 4
    cache_dir: str = "cache_task2"
# =========================
# Utilities
# =========================
def _paths_from_json(json_path: str) -> List[str]:
    """把 JSON 裡的路徑解析成實體檔案路徑；支援絕對路徑與相對於 JSON 檔的相對路徑。"""
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    base = os.path.abspath(os.path.dirname(json_path))     # JSON 檔所在資料夾
    parent = os.path.abspath(os.path.join(base, os.pardir))  # JSON 上層資料夾（兼容舊結構）
    out, seen = [], set()

    for p in items:
        cands = []
        p_norm = os.path.normpath(p)
        # 1) 絕對路徑
        if os.path.isabs(p_norm):
            cands.append(p_norm)
        else:
            # 2) 相對於 JSON 資料夾
            cands.append(os.path.normpath(os.path.join(base, p_norm)))
            # 3) 相對於 JSON 上層（像 hw1/artist20/train.json + "./train_val/..."）
            cands.append(os.path.normpath(os.path.join(parent, p_norm)))

        found = None
        for c in cands:
            if os.path.isfile(c):
                found = c
                break
        if found and found not in seen:
            out.append(found); seen.add(found)

    return sorted(out)


def _label_from_path(path: str) -> str:
    """預設以倒數第三層資料夾當歌手標籤：.../<artist>/<album>/<track>"""
    parts = Path(path).parts
    if len(parts) >= 3:
        return parts[-3]
    elif len(parts) >= 2:
        return parts[-2]
    return "unknown"

def build_classes_from_json(train_json: str) -> List[str]:
    paths = _paths_from_json(train_json)
    classes = sorted({_label_from_path(p) for p in paths})
    return classes

def build_index(json_path: str, cfg: CFG, classes: List[str]) -> List[Tuple[str,int,int,int,str]]:
    """
    將 JSON 裡每首歌切成多段固定長度 chunk（可設定 overlap）。
    回傳 list: (path, start_sample, end_sample, label_idx, track_key)
    """
    paths = _paths_from_json(json_path)
    cls2i = {c:i for i,c in enumerate(classes)}
    seg_len = int(cfg.segment_sec * cfg.sr)
    step = int(seg_len * (1.0 - cfg.overlap)) if cfg.overlap < 1.0 else 1
    index = []
    for p in paths:
        lab = _label_from_path(p)
        if lab not in cls2i:
            continue
        try:
            dur = librosa.get_duration(path=p)
        except TypeError:
            dur = librosa.get_duration(filename=p)
        total = int(max(1, math.floor((dur * cfg.sr - seg_len) / step) + 1))
        for k in range(total):
            s = k * step
            e = s + seg_len
            key = os.path.splitext(os.path.basename(p))[0]  # 用檔名當 track key
            index.append((p, s, e, cls2i[lab], key))
    return index

# =========================
# Feature (log-mel) with voiced mask
# =========================
def _load_chunk(path: str, s: int, e: int, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    seg = y[s:e]
    if seg.shape[0] < (e - s):
        seg = np.pad(seg, (0, (e - s) - seg.shape[0]))
    m = np.max(np.abs(seg))
    if m > 0: seg = seg / m
    return seg.astype(np.float32)

def _voiced_mask(y: np.ndarray, cfg: CFG) -> np.ndarray:
    """用 HPSS + RMS 產生時間遮罩（保留有聲框）。"""
    y_h, _ = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y_h, frame_length=cfg.n_fft, hop_length=cfg.hop)[0]
    thr = np.percentile(rms, 30)
    return (rms > thr).astype(np.float32)  # shape: (T,)

def waveform_to_logmel(y: np.ndarray, cfg: CFG, use_mask: bool = True) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0
    )
    S = librosa.power_to_db(S, ref=np.max)  # (n_mels, T)
    if use_mask:
        m = _voiced_mask(y, cfg)                            # (T,)
        if m.shape[0] != S.shape[1]:
            # 長度對不齊時就不 mask，避免報錯
            pass
        else:
            S = S * m[None, :]
    # per-sample normalize
    mu, sd = S.mean(), S.std() + 1e-6
    S = (S - mu) / sd
    return S.astype(np.float32)

def _score_one_chunk(args):
    """多進程：計算單一 chunk 的品質分數，不存特徵。"""
    idx, (path, s, e, cfg) = args
    try:
        y, _ = librosa.load(path, sr=cfg.sr, mono=True)
        seg = y[s:e]
        need = e - s
        if seg.shape[0] < need:
            seg = np.pad(seg, (0, need - seg.shape[0]))
        m = np.max(np.abs(seg))
        if m > 0: seg = seg / m
        # 用你既有的品質評分
        y_h, y_p = librosa.effects.hpss(seg)
        rms = librosa.feature.rms(y=y_h, frame_length=cfg.n_fft, hop_length=cfg.hop)[0]
        thr = np.percentile(rms, 30)
        voiced_ratio = float((rms > thr).mean())
        snr = 10.0 * np.log10((np.mean(y_h**2) + 1e-10) / (np.mean(y_p**2) + 1e-10))
        S = np.abs(librosa.stft(seg, n_fft=cfg.n_fft, hop_length=cfg.hop))**2
        flat = float(librosa.feature.spectral_flatness(S=S).mean())
        snr_score = 1.0 / (1.0 + np.exp(-(snr - 3.0)))
        flat_score = 1.0 - np.clip(flat, 0.0, 1.0)
        score = 0.5 * voiced_ratio + 0.3 * snr_score + 0.2 * flat_score
        return idx, score, voiced_ratio, snr, flat
    except Exception:
        # 出錯就給最低分，讓它自然被淘汰
        return idx, -1.0, 0.0, -1e9, 1.0


def _compute_mel_to_cache(args):
    """多進程：把選中的 chunk 算好 log-mel 並存快取（若已存在則跳過）。"""
    path, s, e, cfg = args
    cpath = _mel_cache_path(cfg.cache_dir, path, s, e, cfg)
    if os.path.exists(cpath):
        return cpath
    try:
        y, _ = librosa.load(path, sr=cfg.sr, mono=True)
        seg = y[s:e]
        need = e - s
        if seg.shape[0] < need:
            seg = np.pad(seg, (0, need - seg.shape[0]))
        m = np.max(np.abs(seg))
        if m > 0: seg = seg / m
        mel = waveform_to_logmel(seg.astype(np.float32), cfg, use_mask=True)
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.save(cpath, mel)
        return cpath
    except Exception:
        return None


def _precompute_mels(index, cfg: CFG, workers: Optional[int] = None):
    """
    對「已選中」的 index（(path,s,e,lab,key) 列表）多進程預先把 log-mel 存進快取。
    """
    if workers is None:
        workers = max(1, (cpu_count() or 2) - 1)

    tasks = []
    for (p, s, e, _, _) in index:
        cpath = _mel_cache_path(cfg.cache_dir, p, s, e, cfg)
        if not os.path.exists(cpath):
            tasks.append((p, s, e, cfg))
    if not tasks:
        return 0

    with Pool(processes=workers) as pool:
        for _ in tqdm(pool.imap_unordered(_compute_mel_to_cache, tasks),
                      total=len(tasks), desc="Precompute selected mels", ncols=100):
            pass
    return len(tasks)


# =========================
# Datasets
# =========================
def _cfg_fp(cfg: CFG) -> str:
    return f"sr{cfg.sr}_nfft{cfg.n_fft}_hop{cfg.hop}_m{cfg.n_mels}_f{cfg.fmin}-{cfg.fmax}"

def _mel_cache_path(cache_dir: str, path: str, s: int, e: int, cfg: CFG) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5(f"{os.path.abspath(path)}|{s}|{e}|{_cfg_fp(cfg)}".encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"mel_{h}.npy")

class ChunkDatasetFixed(Dataset):
    """train/val/test 共用：使用固定切片 index，逐段回傳 (mel, label, key)。"""
    def __init__(self, index: List[Tuple[str,int,int,int,str]], cfg: CFG):
        self.index = index
        self.cfg = cfg

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        path, s, e, lab, key = self.index[i]
        # ----- 先嘗試讀 cache -----
        cpath = _mel_cache_path(self.cfg.cache_dir, path, s, e, self.cfg)
        if os.path.exists(cpath):
            mel = np.load(cpath)
        else:
            y = _load_chunk(path, s, e, self.cfg.sr)
            mel = waveform_to_logmel(y, self.cfg, use_mask=True)
            try:
                np.save(cpath, mel)
            except Exception:
                pass
        x = torch.from_numpy(mel).unsqueeze(0)  # (1, M, T)
        return x, torch.tensor(lab, dtype=torch.long), key
    
# =========================
# Filters
# =========================   
def _chunk_quality(y: np.ndarray, cfg: CFG):
    # 1) HPSS
    y_h, y_p = librosa.effects.hpss(y)
    # 2) Voiced ratio by RMS mask (>30% percentile)
    rms = librosa.feature.rms(y=y_h, frame_length=cfg.n_fft, hop_length=cfg.hop)[0]
    thr = np.percentile(rms, 30)
    voiced_ratio = float((rms > thr).mean())
    # 3) SNR proxy: harmonic / percussive
    snr = 10.0 * np.log10((np.mean(y_h**2) + 1e-10) / (np.mean(y_p**2) + 1e-10))
    # 4) Spectral flatness (越低越「有音色」)
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop))**2
    flat = float(librosa.feature.spectral_flatness(S=S).mean())
    # 合成一個分數（0~1）：voiced 為主、snr/flat 輔助
    snr_score = 1.0 / (1.0 + np.exp(-(snr - 3.0)))     # 把 snr 映到 0~1
    flat_score = 1.0 - np.clip(flat, 0.0, 1.0)         # 越小越好 → 分數越高
    score = 0.5 * voiced_ratio + 0.3 * snr_score + 0.2 * flat_score
    return score, voiced_ratio, snr, flat

def build_index_filtered(train_json: str, cfg: CFG, classes: List[str],
                         per_track_cap: int = 6,
                         min_voiced: float = 0.45,
                         min_snr_db: float = 2.0,
                         max_flatness: float = 0.95,
                         workers: Optional[int] = None) -> List[Tuple[str,int,int,int,str]]:
    """
    多進程版本：
      1) 產生所有 chunk
      2) 平行評分（tqdm: Score chunks）
      3) 依閾值過濾 + 每首取前 K 段（排名）
      4) 立刻把「選中的 chunk」平行算特徵存快取（tqdm: Precompute selected mels）
    """
    if workers is None:
        workers = max(1, (cpu_count() or 2) - 1)

    paths = _paths_from_json(train_json)
    cls2i = {c:i for i,c in enumerate(classes)}
    seg_len = int(cfg.segment_sec * cfg.sr)
    step = int(seg_len * (1.0 - cfg.overlap)) if cfg.overlap < 1.0 else 1

    # ---- 1) 列出所有 chunk ----
    all_chunks = []  # [(p, s, e, lab, key)]
    for p in paths:
        lab_name = _label_from_path(p)
        if lab_name not in cls2i:
            continue
        lab = cls2i[lab_name]
        try:
            dur = librosa.get_duration(path=p)
        except TypeError:
            dur = librosa.get_duration(filename=p)
        total = int(max(1, math.floor((dur * cfg.sr - seg_len) / step) + 1))
        key = os.path.splitext(os.path.basename(p))[0]
        for k in range(total):
            s = k * step
            e = s + seg_len
            all_chunks.append((p, s, e, lab, key))

    if not all_chunks:
        return []

    # ---- 2) 平行評分 ----
    jobs = [(i, (p, s, e, cfg)) for i, (p, s, e, _, _) in enumerate(all_chunks)]
    scores = [None] * len(all_chunks)  # 對應每個 chunk 的 (score, vr, snr, flat)

    with Pool(processes=workers) as pool:
        for i, sc, vr, snr, fl in tqdm(pool.imap_unordered(_score_one_chunk, jobs),
                                       total=len(jobs), desc="Score chunks", ncols=100):
            scores[i] = (sc, vr, snr, fl)

    # ---- 3) 過濾 + 排名（每首取前 K 段）----
    buckets = {}  # key -> list of (score, tuple)
    for (info, met) in zip(all_chunks, scores):
        if met is None: 
            continue
        sc, vr, snr, fl = met
        if (vr >= min_voiced) and (snr >= min_snr_db) and (fl <= max_flatness):
            p, s, e, lab, key = info
            buckets.setdefault(key, []).append((sc, (p, s, e, lab, key)))

    final_index = []
    for key, lst in buckets.items():
        lst.sort(key=lambda t: t[0], reverse=True)  # by score desc
        for _, rec in lst[:per_track_cap]:
            final_index.append(rec)

    # ---- 4) 立刻把「選中的」chunk 特徵存入快取（下一步 Dataset 直接 np.load）----
    _precompute_mels(final_index, cfg, workers=workers)

    return final_index


def make_loaders(train_json: str,
                 val_json: str,
                 cfg: CFG,
                 classes: Optional[List[str]] = None):
    """建 train/val 的 DataLoader 與各自的 index（投票要用 key）。"""
    if classes is None:
        classes = build_classes_from_json(train_json)

    idx_tr = build_index_filtered(
        train_json, cfg, classes,
        per_track_cap=3,        # ← 可調：每首最多 N 段
        min_voiced=0.50,        # ← 可調：有聲比例閾值
        min_snr_db=2.0,         # ← 可調：SNR 近似閾值
        max_flatness=0.95       # ← 可調：平坦度上限
    )
    idx_va = build_index(val_json,   cfg, classes)

    ds_tr = ChunkDatasetFixed(idx_tr, cfg)
    ds_va = ChunkDatasetFixed(idx_va, cfg)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)

    return dl_tr, dl_va, idx_tr, idx_va, classes

# =========================
# Voting helpers
# =========================
def aggregate_by_keys(proba: np.ndarray, labels: np.ndarray, keys: List[str], n_classes: int,
                      method: str = "mean") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    chunk 機率依 key 聚合為 track 機率。
    回傳：y_true_track, proba_track, uniq_keys
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, k in enumerate(keys):
        buckets[k].append(i)

    y_track, P_track, ukeys = [], [], []
    for k, idxs in buckets.items():
        ukeys.append(k)
        y_track.append(int(np.bincount(labels[idxs]).argmax()))
        P = proba[idxs]
        if method == "mean":
            p = P.mean(axis=0)
        elif method == "majority":
            votes = np.argmax(P, axis=1)
            cnts = np.bincount(votes, minlength=n_classes).astype(np.float32)
            p = cnts / (cnts.sum() + 1e-8)
        else:
            p = P.mean(axis=0)
        p = p / (p.sum() + 1e-12)
        P_track.append(p)
    return np.array(y_track), np.vstack(P_track), ukeys

def topk_from_proba(y_true: np.ndarray, proba: np.ndarray, k: int = 3) -> float:
    kk = min(k, proba.shape[1])
    topk = np.argsort(proba, axis=1)[:, -kk:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))

# -*- coding: utf-8 -*-
# file: dataload_audio.py
import os, glob, json, random, hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import librosa
from tqdm import tqdm


SEED = 42
random.seed(SEED); np.random.seed(SEED)

@dataclass
class CFG:
    sr: int = 22050
    n_fft: int = 2048
    hop: int = 512
    n_mfcc: int = 40
    segment_sec: int = 5

# ---------- 基本工具 ----------

def discover_classes(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def _scan_items(root: str, classes: List[str]) -> List[Tuple[str,int]]:
    cls2idx = {c:i for i,c in enumerate(classes)}
    items = []
    for c in classes:
        for p in glob.glob(os.path.join(root, c, "**", "*.*"), recursive=True):
            if os.path.splitext(p)[1].lower() in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
                items.append((p, cls2idx[c]))
    return items

def _label_from_path(path: str) -> str:
    """
    依常見結構 …/<artist>/<album>/<track>.ext → label=artist
    若層級不夠，退而求其次用檔案上兩層資料夾名稱。
    """
    parts = Path(path).parts
    if len(parts) >= 3:
        return parts[-3]
    elif len(parts) >= 2:
        return parts[-2]
    else:
        return "unknown"

def _paths_from_json(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        paths = json.load(f)
    # 去重 & 過濾不存在的檔案
    uniq = []
    seen = set()
    for p in paths:
        p = os.path.join("./hw1/artist20",p)
        p2 = os.path.normpath(p)
        if p2 not in seen and os.path.isfile(p2):
            uniq.append(p2); seen.add(p2)
    return uniq

# ---------- 音訊片段與特徵 ----------

def load_segment(path: str, cfg: CFG):
    y, sr = librosa.load(path, sr=cfg.sr, mono=True)
    seg_len = cfg.segment_sec * cfg.sr
    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)))
    else:
        start = np.random.randint(0, len(y) - seg_len + 1)
        y = y[start:start+seg_len]
    m = np.max(np.abs(y))
    if m > 0: y = y / m
    return y, sr

def _pool_mean_std(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X.mean(axis=1), X.std(axis=1)])

def extract_features(path: str, cfg: CFG) -> np.ndarray:
    y, sr = load_segment(path, cfg)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop)
    d1   = librosa.feature.delta(mfcc, order=1)
    d2   = librosa.feature.delta(mfcc, order=2)
    # chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    # zcr  = librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=cfg.hop)
    # cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    # bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    # roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    # rms  = librosa.feature.rms(y=y, frame_length=cfg.n_fft, hop_length=cfg.hop)

    feat = np.concatenate([
        _pool_mean_std(mfcc),
        _pool_mean_std(d1),
        _pool_mean_std(d2),
        # _pool_mean_std(chroma),
        # _pool_mean_std(zcr),
        # _pool_mean_std(cent),
        # _pool_mean_std(bw),
        # _pool_mean_std(roll),
        # _pool_mean_std(rms),
    ]).astype(np.float32)
    return feat

# def _pool_mean_std_masked(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
#     """只在 mask=True 的時間步做 mean/std；若全 False 則退回全段。"""
#     if X.shape[1] != mask.shape[0]:
#         # 安全退回（長度不對齊）
#         return np.concatenate([X.mean(axis=1), X.std(axis=1)])
#     Xv = X[:, mask]
#     if Xv.size == 0:
#         Xv = X
#     return np.concatenate([Xv.mean(axis=1), Xv.std(axis=1)])

# def extract_features(path: str, cfg: CFG) -> np.ndarray:
#     # 1) 讀一段固定秒數（你原本的 load_segment），再做 pre-emphasis & HPSS
#     y, sr = load_segment(path, cfg)

#     # pre-emphasis（讓高頻細節更清楚，人聲子音更醒目）
#     y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)

#     # HPSS 拿 harmonic 分量（降低打擊/伴奏）
#     y_h, _ = librosa.effects.hpss(y)

#     # 2) 做有聲遮罩：用 RMS 取前 70% 能量的 frame 視為 voiced
#     rms = librosa.feature.rms(y=y_h, frame_length=cfg.n_fft, hop_length=cfg.hop).reshape(-1)
#     thr = np.percentile(rms, 30)  # 丟掉能量最低的 30%
#     voiced = rms > thr

#     # 3) timbre 特徵：MFCC（丟 c0）+ Δ + Δ² + Spectral Contrast
#     n_mfcc = max(cfg.n_mfcc, 24)  # 歌手分類通常 >=24 比較穩，我建議 40
#     M = librosa.feature.mfcc(y=y_h, sr=sr, n_mfcc=n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop)
#     M = M[1:, :]  # 丟掉 MFCC0 能量項 → 更聚焦音色（避免與 RMS 重複資訊）
#     D1 = librosa.feature.delta(M, order=1)
#     D2 = librosa.feature.delta(M, order=2)

#     # spectral contrast（對共鳴峰/共鳴槽敏感，補 MFCC 的不足）
#     S_con = librosa.feature.spectral_contrast(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)

#     # 4) 只在有聲 frame 上做 pooling（避免前奏/間奏稀釋統計）
#     feat = np.concatenate([
#         _pool_mean_std_masked(M,   voiced),
#         _pool_mean_std_masked(D1,  voiced),
#         _pool_mean_std_masked(D2,  voiced),
#         _pool_mean_std_masked(S_con, voiced),
#         # 可選：再加一點頻譜形狀（通常增益不大）
#         _pool_mean_std_masked(librosa.feature.spectral_centroid(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop), voiced),
#         _pool_mean_std_masked(librosa.feature.spectral_bandwidth(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop), voiced),
#         _pool_mean_std_masked(librosa.feature.spectral_rolloff(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop), voiced),
#         # 不再加入 chroma、ZCR、RMS（已用 voiced 篩選 & 丟了 c0）
#     ]).astype(np.float32)

#     return feat

# ---------- 封裝：JSON 模式（建議） ----------

def build_chunk_index(json_path: str, cfg: CFG, classes: Optional[List[str]] = None):
    paths = _paths_from_json(json_path)
    if classes is None:
        classes = sorted({_label_from_path(p) for p in paths})
    c2i = {c: i for i, c in enumerate(classes)}

    seg_len = int(cfg.segment_sec * cfg.sr)
    index = []
    for p in paths:
        lab = _label_from_path(p)
        if lab not in c2i:
            continue
        # 以秒取得時長，再換算區段數
        try:
            dur = librosa.get_duration(path=p)
        except TypeError:
            dur = librosa.get_duration(filename=p)
        n_full = int(dur // cfg.segment_sec)
        for i in range(n_full):
            s = i * seg_len
            e = s + seg_len
            index.append((p, s, e, c2i[lab]))
    return index, classes

def build_Xy_json_with_meta(json_path: str, cfg: CFG, classes=None, cache_dir=None):
    """
    與你原本 build_Xy_json 相同，但多回傳 src_paths（len=N，對齊 X 的每一列），
    每個元素是該 chunk 來自的「原始音檔完整路徑」。
    """
    # 你原本 build_Xy_json 的邏輯：先 build_chunk_index(...) → extract_features(...)
    index, classes = build_chunk_index(json_path, cfg)  # [(path, s, e, lab), ...]
    X, y, src_paths = [], [], []
    paths = _paths_from_json(json_path)
    for p, s, e, lab in tqdm(index):
        seg, sr = load_segment(p, cfg)        # 你原本的切片函式
        feat = extract_features(p, cfg)  # 或 extract_features(p, cfg) 視你現有命名
        X.append(feat); y.append(lab); src_paths.append(p)
    tag = f"json::{Path(json_path).resolve()}::{len(paths)}"
    return _build_from_items(items, classes, cfg, cache_dir, cache_tag=tag), src_paths

def build_Xy_json(json_path: str, cfg: CFG, classes: Optional[List[str]] = None,
                  cache_dir: Optional[str] = None):
    paths = _paths_from_json(json_path)
    if classes is None:
        classes = sorted({ _label_from_path(p) for p in paths })
    items = [(p, classes.index(_label_from_path(p))) for p in paths]
    tag = f"json::{Path(json_path).resolve()}::{len(paths)}"
    return _build_from_items(items, classes, cfg, cache_dir, cache_tag=tag)
# ---------- 共同底層：可選快取 ----------

def _build_from_items(items, classes, cfg: CFG, cache_dir: Optional[str], cache_tag: str):
    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(f"{cache_tag}|sr{cfg.sr}|fft{cfg.n_fft}|hop{cfg.hop}|mfcc{cfg.n_mfcc}|seg{cfg.segment_sec}".encode()).hexdigest()[:16]
        cache_path = Path(cache_dir) / f"{h}.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            return data["X"], data["y"], list(data["classes"])
    X, y = [], []
    for p, lab in tqdm(items, desc="Extracting features"):
        try:
            X.append(extract_features(p, cfg)); y.append(lab)
        except Exception as e:
            print(f"[WARN] Skip {p}: {e}")
    X = np.stack(X); y = np.array(y)
    if cache_path:
        np.savez_compressed(cache_path, X=X, y=y, classes=np.array(classes, dtype=object))
    return X, y, classes


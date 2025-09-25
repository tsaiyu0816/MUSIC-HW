# -*- coding: utf-8 -*-
# file: dataload_audio.py
import os, glob, json, random, hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Iterable, Union
import numpy as np
import librosa
from tqdm import tqdm

# =========================
# Global
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# =========================
# Config
# =========================
@dataclass
class CFG:
    sr: int = 22050
    n_fft: int = 2048
    hop: int = 512
    n_mfcc: int = 40
    segment_sec: int = 5  # 單段秒數（訓練：隨機取段；測試：固定切片）

# =========================
# Utilities
# =========================
def discover_classes(root: str) -> List[str]:
    """
    假設資料夾結構：root/<artist>/**/<track>.<ext>
    回傳 artist 名稱（排序後）。
    """
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def _scan_items(root: str, classes: List[str]) -> List[Tuple[str, int]]:
    """走訪有標籤資料夾，收集 (路徑, 類別索引)"""
    cls2idx = {c: i for i, c in enumerate(classes)}
    items: List[Tuple[str, int]] = []
    for c in classes:
        for p in glob.glob(os.path.join(root, c, "**", "*.*"), recursive=True):
            if Path(p).suffix.lower() in AUDIO_EXTS:
                items.append((p, cls2idx[c]))
    return items

def _iter_audio_files(root: str) -> Iterable[str]:
    """遞迴列出所有支援格式的音訊檔（完整路徑）。"""
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if Path(fn).suffix.lower() in AUDIO_EXTS:
                yield os.path.join(dp, fn)

def _label_from_path(path: str) -> str:
    """
    依常見結構 …/<artist>/<album>/<track>.ext → label=artist
    若層級不夠，退一步用倒數第二層資料夾名稱。
    """
    parts = Path(path).parts
    if len(parts) >= 3:   # .../<artist>/<album>/<track>
        return parts[-3]
    elif len(parts) >= 2: # .../<artist>/<track>
        return parts[-2]
    else:
        return "unknown"

def _paths_from_json(json_path: str) -> List[str]:
    """
    讀入 JSON 清單。依你原本的邏輯，將相對路徑接到 ./hw1/artist20 下。
    如不需要，改掉那行 os.path.join(...) 即可。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        paths = json.load(f)
    uniq, seen = [], set()
    for p in paths:
        # 若 JSON 已是絕對/完整路徑，請改為：p2 = os.path.normpath(p)
        p_full = os.path.join("./hw1/artist20", p)
        p2 = os.path.normpath(p_full)
        if p2 not in seen and os.path.isfile(p2):
            uniq.append(p2); seen.add(p2)
    return uniq

# =========================
# Audio segment & features
# =========================
def _pool_mean_std(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    return np.concatenate([X.mean(axis=1), X.std(axis=1)])

def load_segment(path: str, cfg: CFG) -> Tuple[np.ndarray, int]:
    """
    讀整首，隨機取一段 cfg.segment_sec 秒，不足補零，幅度正規化。
    用於訓練抽樣（每首歌 1 段）。
    """
    y, sr = librosa.load(path, sr=cfg.sr, mono=True)
    seg_len = int(cfg.segment_sec * cfg.sr)
    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)))
    else:
        start = np.random.randint(0, len(y) - seg_len + 1)
        y = y[start:start + seg_len]
    m = np.max(np.abs(y))
    if m > 0:
        y = y / m
    return y.astype(np.float32), sr

def _load_chunk(path: str, start_sample: int, end_sample: int, sr: int) -> Tuple[np.ndarray, int]:
    """
    固定切片：以指定 sr 載入整首，取 [start_sample:end_sample)。
    不足補零，幅度正規化。用於驗證/測試的固定切段投票。
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    seg = y[start_sample:end_sample]
    need = end_sample - start_sample
    if seg.shape[0] < need:
        seg = np.pad(seg, (0, need - seg.shape[0]))
    m = np.max(np.abs(seg))
    if m > 0:
        seg = seg / m
    return seg.astype(np.float32), sr

def _pool_mean_std_masked(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """只在 mask=True 的 frame 上做 mean/std；若長度不齊或全 False 則退回全段。"""
    X = np.atleast_2d(X)
    if mask is None:
        return np.concatenate([X.mean(axis=1), X.std(axis=1)])
    m = np.asarray(mask).astype(bool).reshape(-1)
    if X.shape[1] != m.shape[0] or m.sum() == 0:
        return np.concatenate([X.mean(axis=1), X.std(axis=1)])
    Xv = X[:, m]
    return np.concatenate([Xv.mean(axis=1), Xv.std(axis=1)])

# def extract_features_from_array(y: np.ndarray, sr: int, cfg: CFG) -> np.ndarray:
#     """對 waveform 計算特徵（MFCC + Δ + Δ² → mean/std pool）。"""
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc,
#                                 n_fft=cfg.n_fft, hop_length=cfg.hop)
#     d1   = librosa.feature.delta(mfcc, order=1)
#     d2   = librosa.feature.delta(mfcc, order=2)
#     feat = np.concatenate([
#         _pool_mean_std(mfcc),
#         _pool_mean_std(d1),
#         _pool_mean_std(d2),
#     ]).astype(np.float32)
#     return feat

def extract_features_from_array(y: np.ndarray, sr: int, cfg: CFG) -> np.ndarray:
    """pre-emphasis + HPSS + voiced mask + MFCC(去c0)+Δ+Δ²+spectral contrast → masked mean/std。"""
    # pre-emphasis（凸顯高頻/子音）
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)
    # 取 harmonic（降低打擊/伴奏）
    y_h, _ = librosa.effects.hpss(y)

    # 有聲遮罩：丟掉能量最低 30% 的 frame
    rms = librosa.feature.rms(y=y_h, frame_length=cfg.n_fft, hop_length=cfg.hop).reshape(-1)
    thr = np.percentile(rms, 30)
    voiced = rms > thr

    # timbre：MFCC（去 c0）+ Δ + Δ²
    n_mfcc = max(cfg.n_mfcc, 24)
    M  = librosa.feature.mfcc(y=y_h, sr=sr, n_mfcc=n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop)
    M  = M[1:, :]  # 去掉能量項 c0
    D1 = librosa.feature.delta(M, order=1)
    D2 = librosa.feature.delta(M, order=2)

    # spectral contrast + 一些頻譜形狀（輔助）
    Scon = librosa.feature.spectral_contrast(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    Cent = librosa.feature.spectral_centroid(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    BW   = librosa.feature.spectral_bandwidth(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    Roff = librosa.feature.spectral_rolloff(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)

    feat = np.concatenate([
        _pool_mean_std_masked(M,   voiced),
        _pool_mean_std_masked(D1,  voiced),
        _pool_mean_std_masked(D2,  voiced),
        _pool_mean_std_masked(Scon, voiced),
        _pool_mean_std_masked(Cent, voiced),
        _pool_mean_std_masked(BW,   voiced),
        _pool_mean_std_masked(Roff, voiced),
    ]).astype(np.float32)
    return feat

def extract_features(path: str, cfg: CFG) -> np.ndarray:
    """
    路徑版本：隨機取一段 → 抽特徵（訓練/簡易模式）。
    """
    y, sr = load_segment(path, cfg)
    return extract_features_from_array(y, sr, cfg)

# =========================
# JSON mode (per-file sampling)
# =========================
def build_chunk_index(json_path: str, cfg: CFG, classes: Optional[List[str]] = None):
    """
    若你要用「固定切片」的 JSON 模式，可先用這個產生 [(path, s, e, lab), ...] 索引。
    目前 build_Xy_json_with_meta 採用 _build_from_items（每檔 1 筆，隨機段），
    若要改成固定切片版，可另外寫 _build_from_index。
    """
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

def _build_from_items(
    items: List[Tuple[str, int]],
    classes: List[str],
    cfg: CFG,
    cache_dir: Optional[str],
    cache_tag: str,
    *,
    return_meta: bool = False,
    feature_fn: Optional[Callable[[str], np.ndarray]] = None,
    key_fn: Optional[Callable[[str], str]] = None,
):
    """
    通用建構器：
      items      : List[(path, label_idx)]
      return_meta: True 時，額外回傳 src_paths（與 X 對齊）
      feature_fn : 預設用 extract_features(path, cfg)
      key_fn     : 預設回傳原始路徑字串
    **快取僅在 `feature_fn is None 且 return_meta=False` 時啟用**（與舊版相容）。
    """
    use_cache = (cache_dir is not None) and (feature_fn is None) and (not return_meta)

    if feature_fn is None:
        def feature_fn(p: str) -> np.ndarray:
            return extract_features(p, cfg)
    if key_fn is None:
        def key_fn(p: str) -> str:
            return p

    cache_path = None
    if use_cache:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(
            f"{cache_tag}|sr{cfg.sr}|fft{cfg.n_fft}|hop{cfg.hop}|mfcc{cfg.n_mfcc}|seg{cfg.segment_sec}".encode()
        ).hexdigest()[:16]
        cache_path = Path(cache_dir) / f"{h}.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            if return_meta:
                # 舊快取不含 meta；為維持相容，回傳 src_paths=None
                return data["X"], data["y"], list(data["classes"]), None
            return data["X"], data["y"], list(data["classes"])

    X, y, src_paths = [], [], []
    for p, lab in tqdm(items, desc="Extracting features"):
        try:
            X.append(feature_fn(p))
            y.append(lab)
            if return_meta:
                src_paths.append(key_fn(p))
        except Exception as e:
            print(f"[WARN] Skip {p}: {e}")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    if use_cache and cache_path:
        np.savez_compressed(cache_path, X=X, y=y, classes=np.array(classes, dtype=object))

    if return_meta:
        return X, y, classes, (np.array(src_paths) if src_paths else None)
    else:
        return X, y, classes

def build_Xy_json(json_path: str,
                  cfg: CFG,
                  classes: Optional[List[str]] = None,
                  cache_dir: Optional[str] = None):
    """
    每檔取 1 段（隨機），回傳 X, y, classes。
    """
    paths = _paths_from_json(json_path)
    if classes is None:
        classes = sorted({_label_from_path(p) for p in paths})
    items = [(p, classes.index(_label_from_path(p))) for p in paths]
    tag = f"json::{Path(json_path).resolve()}::{len(paths)}"
    return _build_from_items(items, classes, cfg, cache_dir, cache_tag=tag)

def build_Xy_json_with_meta(json_path: str,
                            cfg: CFG,
                            classes: Optional[List[str]] = None,
                            cache_dir: Optional[str] = None):
    """
    每檔取 1 段（隨機），回傳 X, y, classes, src_paths（與 X 對齊）。
    src_paths 預設為「完整路徑字串」，可改 key_fn 只保留檔名。
    """
    paths = _paths_from_json(json_path)
    if classes is None:
        classes = sorted({_label_from_path(p) for p in paths})
    items = [(p, classes.index(_label_from_path(p))) for p in paths]
    tag = f"json::{Path(json_path).resolve()}::{len(paths)}"
    return _build_from_items(
        items, classes, cfg, cache_dir, cache_tag=tag,
        return_meta=True,
        feature_fn=None,                      # 用預設 extract_features(path, cfg)
        key_fn=lambda p: p,                   # 若只要檔名：lambda p: os.path.basename(p)
    )

# =========================
# Folder mode (labeled)
# =========================
def build_Xy_dir_with_meta(root: str,
                           cfg: CFG,
                           classes: Optional[List[str]] = None,
                           cache_dir: Optional[str] = None):
    """
    從資料夾讀檔（有標籤結構 root/<artist>/**/<track>），
    每檔各取 1 段（隨機），回傳 X, y, classes, src_paths。
    若想做固定切片投票（多段），建議另外實作 for-index 版本。
    """
    files = sorted(list(_iter_audio_files(root)))
    if classes is None:
        classes = sorted({_label_from_path(p) for p in files})
    c2i = {c: i for i, c in enumerate(classes)}

    items = []
    for p in files:
        lab = _label_from_path(p)
        if lab in c2i:
            items.append((p, c2i[lab]))

    tag = f"dir::{Path(root).resolve()}::{len(items)}"
    # 用 _build_from_items，回傳 meta（src_paths）
    return _build_from_items(
        items, classes, cfg, cache_dir, cache_tag=tag,
        return_meta=True,
        feature_fn=None,           # 預設 extract_features(path, cfg)（隨機段）
        key_fn=lambda p: os.path.relpath(p, root)
    )

# =========================
# Test (flat, unlabeled)
# =========================
def build_X_dir_with_meta_unlabeled(root: str, cfg: CFG):
    """
    僅用於測試：讀取 root 底下「當層」的 .wav（不遞迴子資料夾），
    將每個檔案切成多個固定長度 chunk，輸出：
      X: (N, D)   每個 chunk 的特徵
      keys: (N,)  每個 chunk 對應的「檔名（不含副檔名亦可自行調整）」
    """
    files = sorted([
        os.path.join(root, fn) for fn in os.listdir(root)
        if fn.lower().endswith(".mp3") and os.path.isfile(os.path.join(root, fn))
    ])
    if len(files) == 0:
        raise FileNotFoundError(f"No .wav files found directly under: {root}")

    seg_len = int(cfg.segment_sec * cfg.sr)
    X, keys = [], []
    for p in tqdm(files, desc=f"Scan {Path(root).name} (test)"):
        # 以秒為單位切不重疊區段；太短至少切 1 段（會補零）
        try:
            dur = librosa.get_duration(path=p)
        except TypeError:
            dur = librosa.get_duration(filename=p)
        n_full = max(1, int(dur // cfg.segment_sec))

        for i in range(n_full):
            s = i * seg_len
            e = s + seg_len
            seg, sr = _load_chunk(p, s, e, cfg.sr)
            feat = extract_features_from_array(seg, sr, cfg)
            X.append(feat)
            # 以「檔名（不含副檔名）」作 key，例 001.wav -> "001"
            keys.append(os.path.splitext(os.path.basename(p))[0])

    return np.stack(X).astype(np.float32), np.array(keys)

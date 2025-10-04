import os, glob, json, random, hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Iterable, Union
import numpy as np
import librosa
from tqdm import tqdm
import multiprocessing as mp

# =========================
# Global
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)

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

def _process_file_fixed(args):
    """
    讀取「單一檔案」一次 → 依 segment_sec 切不重疊區段 → 對每段抽特徵
    回傳: (X_file: (n_seg, feat_dim), y_file: (n_seg,), src_paths: (n_seg,))
    """
    p, cfg, lab = args

    # 限制每個進程內的底層執行緒（防止超額搶核）；沒有 threadpoolctl 也沒關係
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        from contextlib import nullcontext as threadpool_limits

    with threadpool_limits(limits=1):
        y, _ = librosa.load(p, sr=cfg.sr, mono=True)

    seg_len = int(cfg.segment_sec * cfg.sr)
    # 至少切 1 段（不足補零）
    n_full = max(1, int(len(y) // seg_len))

    feats, labs, srcs = [], [], []
    for i in range(n_full):
        s = i * seg_len
        e = s + seg_len
        seg = y[s:e]
        if seg.shape[0] < seg_len:
            seg = np.pad(seg, (0, seg_len - seg.shape[0]))
        m = np.max(np.abs(seg))
        if m > 0:
            seg = seg / m

        # 特徵抽取（沿用你原本的函式）
        feat = extract_features_from_array(seg.astype(np.float32), cfg.sr, cfg)
        feats.append(feat); labs.append(lab); srcs.append(p)

    return (np.stack(feats).astype(np.float32),
            np.array(labs, dtype=np.int64),
            np.array(srcs, dtype=object))


# =========================
# Audio segment & features
# =========================
def _pool_mean_std(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    return np.concatenate([X.mean(axis=1), X.std(axis=1)])

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

def extract_features_from_array(y: np.ndarray, sr: int, cfg: CFG) -> np.ndarray:
    """對 waveform 計算特徵（MFCC + Δ + Δ² → mean/std pool）。"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc,
                                n_fft=cfg.n_fft, hop_length=cfg.hop)
    d1   = librosa.feature.delta(mfcc, order=1)
    d2   = librosa.feature.delta(mfcc, order=2)
    # chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    # zcr  = librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=cfg.hop)
    # cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    # roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
    # rms  = librosa.feature.rms(y=y, frame_length=cfg.n_fft, hop_length=cfg.hop)


    feat = np.concatenate([
        _pool_mean_std(mfcc),
        _pool_mean_std(d1),
        _pool_mean_std(d2),
        # _pool_mean_std(chroma),
        # _pool_mean_std(zcr),
        # _pool_mean_std(cent), # helpful
        _pool_mean_std(bw), # helpful
        # _pool_mean_std(roll), 
        # _pool_mean_std(rms),
    ]).astype(np.float32)
    return feat

# def extract_features_from_array(y: np.ndarray, sr: int, cfg: CFG) -> np.ndarray:
#     """pre-emphasis + HPSS + voiced mask + MFCC(去c0)+Δ+Δ²+spectral contrast → masked mean/std。"""
#     # pre-emphasis（凸顯高頻/子音）
#     y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)
#     # 取 harmonic（降低打擊/伴奏）
#     y_h, _ = librosa.effects.hpss(y)

#     # 有聲遮罩：丟掉能量最低 30% 的 frame
#     rms = librosa.feature.rms(y=y_h, frame_length=cfg.n_fft, hop_length=cfg.hop).reshape(-1)
#     thr = np.percentile(rms, 30)
#     voiced = rms > thr

#     # timbre：MFCC（去 c0）+ Δ + Δ²
#     n_mfcc = max(cfg.n_mfcc, 24)
#     M  = librosa.feature.mfcc(y=y_h, sr=sr, n_mfcc=n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop)
#     M  = M[1:, :]  # 去掉能量項 c0
#     D1 = librosa.feature.delta(M, order=1)
#     D2 = librosa.feature.delta(M, order=2)

#     # # spectral contrast + 一些頻譜形狀（輔助）
#     # Scon = librosa.feature.spectral_contrast(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
#     # Cent = librosa.feature.spectral_centroid(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
#     # BW   = librosa.feature.spectral_bandwidth(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)
#     # Roff = librosa.feature.spectral_rolloff(y=y_h, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop)

#     feat = np.concatenate([
#         _pool_mean_std_masked(M,   voiced),
#         _pool_mean_std_masked(D1,  voiced),
#         _pool_mean_std_masked(D2,  voiced),
#         # _pool_mean_std_masked(Scon, voiced),
#         # _pool_mean_std_masked(Cent, voiced),
#         # _pool_mean_std_masked(BW,   voiced),
#         # _pool_mean_std_masked(Roff, voiced),
#     ]).astype(np.float32)
#     return feat

# =========================
# JSON mode (per-file sampling)
# =========================
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

def build_Xy_json_fixedchunks_with_meta(json_path: str, cfg: CFG, classes=None, cache_dir=None, num_workers: int = None):
    """
    JSON 清單 → 每首歌切固定長度 chunk（不重疊）→ 抽特徵。
    現在支援 per-file 多進程：每首歌只 decode 一次（大幅加速）。
    """
    # --------- Cache 準備 ---------
    paths = _paths_from_json(json_path)
    if classes is None:
        classes = sorted({_label_from_path(p) for p in paths})
    c2i = {c: i for i, c in enumerate(classes)}

    # cache key
    cache_path = None
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        tag = f"json_fixed::{Path(json_path).resolve()}::{len(paths)}"
        h = hashlib.md5(
            f"{tag}|sr{cfg.sr}|fft{cfg.n_fft}|hop{cfg.hop}|mfcc{cfg.n_mfcc}|seg{cfg.segment_sec}".encode()
        ).hexdigest()[:16]
        cache_path = Path(cache_dir) / f"{h}.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            X   = data["X"].astype(np.float32)
            y   = data["y"].astype(np.int64)
            cls = list(data["classes"])
            src = data["src_paths"]
            return X, y, cls, src

    # --------- 建任務（以「檔案」為粒度）---------
    jobs = []
    for p in paths:
        lab_name = _label_from_path(p)
        if lab_name not in c2i:
            continue
        jobs.append((p, cfg, c2i[lab_name]))

    # workers 預設：實體核心數的一半（留點給系統）
    if num_workers is None:
        try:
            import psutil
            num_workers = max(1, psutil.cpu_count(logical=False) or os.cpu_count() or 1)
        except Exception:
            num_workers = max(1, (os.cpu_count() or 2) // 2)

    # 防止每個進程內 BLAS/Omp 再開很多線程（在啟 Pool 前設置）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # --------- 多進程跑 per-file 抽特徵 ---------
    X_list, y_list, src_list = [], [], []
    if num_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            for Xi, yi, si in tqdm(pool.imap_unordered(_process_file_fixed, jobs),
                                   total=len(jobs), desc="Extract (CPU multiproc, per-file)", ncols=100):
                X_list.append(Xi); y_list.append(yi); src_list.append(si)
    else:
        # 單進程 fallback
        for job in tqdm(jobs, desc="Extract (single process)", ncols=100):
            Xi, yi, si = _process_file_fixed(job)
            X_list.append(Xi); y_list.append(yi); src_list.append(si)

    # 拼接
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    src = np.concatenate(src_list, axis=0)

    # --------- 寫入快取 ---------
    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            classes=np.array(classes, dtype=object),
            src_paths=src,
        )

    return X, y, classes, src


# =========================
# Test 
# =========================
def build_X_dir_with_meta_unlabeled(root: str, cfg: CFG):

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

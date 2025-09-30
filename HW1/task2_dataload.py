# -*- coding: utf-8 -*-
# file: task2_dataload.py
import os, json, math, random, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==== optional libs ====
TRY_AUG = False
try:
    import audiomentations as A
    TRY_AUG = True
except Exception:
    TRY_AUG = False

TRY_TORCHAUDIO = False
try:
    import torchaudio
    TRY_TORCHAUDIO = True
except Exception:
    TRY_TORCHAUDIO = False

# reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# =========================
# Config
# =========================
@dataclass
class CFG:
    # feature
    sr: int = 22050
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 128
    fmin: int = 40
    fmax: int = 8000

    # chunking
    segment_sec: int = 10
    overlap: float = 0.0

    # dataloader
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # cache
    cache_dir: str = "cache_task2"
    train_use_cache: bool = True

    # augmentation
    aug_wave_prob: float = 0.5
    wave_aug_heavy_prob: float = 0.1
    pitch_semitones: Tuple[float, float] = (-2.0, 2.0)
    time_stretch_range: Tuple[float, float] = (0.90, 1.10)
    gain_db: Tuple[float, float] = (-6.0, 6.0)
    noise_dir: Optional[str] = None
    noise_snr_db: Tuple[float, float] = (10.0, 30.0)

    # SpecAugment
    aug_spec_prob: float = 0.5
    specaug_freq_mask_param: int = 16
    specaug_time_mask_param: int = 32
    specaug_num_masks: int = 2

    # HPSS source separator / mask
    sep_harmonic_weight: float = 1.0   # 1.0=只用harmonic；可與percussive加權混合
    sep_percussive_weight: float = 0.0
    rms_percentile: float = 30.0       # 以 harmonic 的 RMS 百分位做 voiced mask

# =========================
# Utilities
# =========================
def _paths_from_json(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    base = os.path.abspath(os.path.dirname(json_path))
    parent = os.path.abspath(os.path.join(base, os.pardir))
    out, seen = [], set()
    for p in items:
        p_norm = os.path.normpath(p)
        cands = [p_norm] if os.path.isabs(p_norm) else [
            os.path.normpath(os.path.join(base, p_norm)),
            os.path.normpath(os.path.join(parent, p_norm)),
        ]
        for c in cands:
            if os.path.isfile(c) and c not in seen:
                out.append(c); seen.add(c); break
    return sorted(out)

def _label_from_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 3: return parts[-3]
    if len(parts) >= 2: return parts[-2]
    return "unknown"

def build_classes_from_json(train_json: str) -> List[str]:
    return sorted({_label_from_path(p) for p in _paths_from_json(train_json)})

def build_index(json_path: str, cfg: CFG, classes: List[str]) -> List[Tuple[str,int,int,int,str]]:
    paths = _paths_from_json(json_path)
    cls2i = {c:i for i,c in enumerate(classes)}
    seg_len = int(cfg.segment_sec * cfg.sr)
    step = int(seg_len * (1.0 - cfg.overlap)) if cfg.overlap < 1.0 else 1
    index = []
    for p in paths:
        lab_name = _label_from_path(p)
        if lab_name not in cls2i: continue
        try: dur = librosa.get_duration(path=p)
        except TypeError: dur = librosa.get_duration(filename=p)
        total = int(max(1, math.floor((dur * cfg.sr - seg_len) / step) + 1))
        key = os.path.splitext(os.path.basename(p))[0]
        for k in range(total):
            s = k * step; e = s + seg_len
            index.append((p, s, e, cls2i[lab_name], key))
    return index

# =========================
# Augment (wave)
# =========================
def _make_gain_transform(cfg: CFG):
    if not TRY_AUG: return None
    for kw in [dict(min_gain_in_db=cfg.gain_db[0], max_gain_in_db=cfg.gain_db[1], p=0.5),
               dict(min_gain_db=cfg.gain_db[0], max_gain_db=cfg.gain_db[1], p=0.5),
               dict(p=0.5)]:
        try: return A.Gain(**kw)
        except TypeError: continue
    return None

def _make_shift_transform():
    if not TRY_AUG: return None
    for kw in [dict(min_fraction=-0.2, max_fraction=0.2, rollover=True, p=0.3),
               dict(min_fraction=-0.2, max_fraction=0.2, p=0.3),
               dict(shift_min_fraction=-0.2, shift_max_fraction=0.2, rollover=True, p=0.3),
               dict(p=0.3)]:
        try: return A.Shift(**kw)
        except TypeError: continue
    try: return A.TimeShift(p=0.3)
    except Exception: return None

_WAVE_AUG = None
def _get_wave_augmenter(cfg: CFG):
    global _WAVE_AUG
    if _WAVE_AUG is not None: return _WAVE_AUG
    if not TRY_AUG: return None
    tfms = [
        A.PitchShift(min_semitones=cfg.pitch_semitones[0], max_semitones=cfg.pitch_semitones[1], p=0.5),
        A.TimeStretch(min_rate=cfg.time_stretch_range[0], max_rate=cfg.time_stretch_range[1], p=0.5),
    ]
    g = _make_gain_transform(cfg)
    if g is not None: tfms.append(g)
    if cfg.noise_dir and os.path.isdir(cfg.noise_dir):
        tfms.append(A.AddBackgroundNoise(
            sounds_path=cfg.noise_dir,
            min_snr_in_db=cfg.noise_snr_db[0], max_snr_in_db=cfg.noise_snr_db[1], p=0.5
        ))
    sh = _make_shift_transform()
    if sh is not None: tfms.append(sh)
    _WAVE_AUG = A.Compose(tfms)
    return _WAVE_AUG

def _apply_specaugment(S: np.ndarray, cfg: CFG) -> np.ndarray:
    if not TRY_TORCHAUDIO: return S
    x = torch.tensor(S, dtype=torch.float32).unsqueeze(0)  # (1,M,T)
    fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=cfg.specaug_freq_mask_param)
    tm = torchaudio.transforms.TimeMasking(time_mask_param=cfg.specaug_time_mask_param)
    for _ in range(int(cfg.specaug_num_masks)):
        x = fm(x); x = tm(x)
    return x.squeeze(0).cpu().numpy()

# =========================
# Feature (HPSS + RMS mask)
# =========================
def _load_chunk(path: str, s: int, e: int, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    seg = y[s:e]
    need = e - s
    if seg.shape[0] < need: seg = np.pad(seg, (0, need - seg.shape[0]))
    m = np.max(np.abs(seg))
    if m > 0: seg = seg / m
    return seg.astype(np.float32)

def _separate_mix(y: np.ndarray, cfg: CFG):
    """HPSS 分離後依權重混合；回傳 y_mix, y_h, y_p"""
    y_h, y_p = librosa.effects.hpss(y)
    y_mix = cfg.sep_harmonic_weight * y_h + cfg.sep_percussive_weight * y_p
    return y_mix.astype(np.float32), y_h.astype(np.float32), y_p.astype(np.float32)

def _voiced_mask_from_harmonic(y_h: np.ndarray, cfg: CFG) -> np.ndarray:
    rms = librosa.feature.rms(y=y_h, frame_length=cfg.n_fft, hop_length=cfg.hop)[0]
    thr = np.percentile(rms, cfg.rms_percentile)
    return (rms > thr).astype(np.float32)  # shape (T,)

def waveform_to_logmel(y: np.ndarray, cfg: CFG) -> np.ndarray:
    y_mix, y_h, _ = _separate_mix(y, cfg)
    P = librosa.feature.melspectrogram(
        y=y_mix, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0
    )  # (M,T)
    m = _voiced_mask_from_harmonic(y_h, cfg)  # (T_frame,)
    if m.shape[0] != P.shape[1]:
        x_old = np.linspace(0, 1, num=m.shape[0])
        x_new = np.linspace(0, 1, num=P.shape[1])
        m = np.interp(x_new, x_old, m).astype(np.float32)
    P = P * m[None, :]
    S = librosa.power_to_db(P, ref=np.max)
    mu, sd = S.mean(), S.std() + 1e-6
    return ((S - mu) / sd).astype(np.float32)

def _compute_mel_chunk(path: str, s: int, e: int, cfg: CFG,
                       augment: bool = False, for_train: bool = False) -> np.ndarray:
    y = _load_chunk(path, s, e, cfg.sr)
    if for_train and augment and TRY_AUG and np.random.rand() < cfg.aug_wave_prob:
        aug = _get_wave_augmenter(cfg)
        if aug is not None:
            y = aug(samples=y, sample_rate=cfg.sr).astype(np.float32)
    S = waveform_to_logmel(y, cfg)
    if for_train and np.random.rand() < cfg.aug_spec_prob:
        S = _apply_specaugment(S, cfg)
    return S

# =========================
# Cache helpers
# =========================
def _cfg_fp(cfg: CFG) -> str:
    core = f"sr{cfg.sr}_nfft{cfg.n_fft}_hop{cfg.hop}_m{cfg.n_mels}_f{cfg.fmin}-{cfg.fmax}"
    sep  = f"_sep{cfg.sep_harmonic_weight:.2f}-{cfg.sep_percussive_weight:.2f}_rms{cfg.rms_percentile}"
    return core + sep

def _mel_cache_path(cache_dir: str, path: str, s: int, e: int, cfg: CFG) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5(f"{os.path.abspath(path)}|{s}|{e}|{_cfg_fp(cfg)}".encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"mel_{h}.npy")

def _compute_mel_to_cache(args):
    path, s, e, cfg = args
    cpath = _mel_cache_path(cfg.cache_dir, path, s, e, cfg)
    if os.path.exists(cpath): return cpath
    try:
        mel = _compute_mel_chunk(path, s, e, cfg, augment=False, for_train=False)
        np.save(cpath, mel)
        return cpath
    except Exception:
        return None

def _precompute_mels(index, cfg: CFG, workers: Optional[int] = None):
    if workers is None: workers = max(1, (cpu_count() or 2) - 1)
    tasks = []
    for (p, s, e, _, _) in index:
        cpath = _mel_cache_path(cfg.cache_dir, p, s, e, cfg)
        if not os.path.exists(cpath): tasks.append((p, s, e, cfg))
    if not tasks: return 0
    with Pool(processes=workers) as pool:
        for _ in tqdm(pool.imap_unordered(_compute_mel_to_cache, tasks),
                      total=len(tasks), desc="Precompute mels", ncols=100):
            pass
    return len(tasks)

# =========================
# Quality scoring (HPSS-based)
# =========================
def _score_one_chunk(job):
    """(i, p, s, e, cfg) -> i, score, voiced_ratio, snr_db, flat"""
    i, p, s, e, cfg = job
    try:
        y = _load_chunk(p, s, e, cfg.sr)
        y_h, y_p = librosa.effects.hpss(y)
        rms = librosa.feature.rms(y=y_h, frame_length=cfg.n_fft, hop_length=cfg.hop)[0]
        thr = np.percentile(rms, cfg.rms_percentile)
        voiced_ratio = float((rms > thr).mean())
        snr = 10.0 * np.log10((np.mean(y_h**2) + 1e-10) / (np.mean(y_p**2) + 1e-10))
        S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop))**2
        flat = float(librosa.feature.spectral_flatness(S=S).mean())
        snr_score  = 1.0 / (1.0 + np.exp(-(snr - 3.0)))
        flat_score = 1.0 - np.clip(flat, 0.0, 1.0)
        score = 0.5 * voiced_ratio + 0.3 * snr_score + 0.2 * flat_score
        return i, score, voiced_ratio, snr, flat
    except Exception:
        return i, -1.0, 0.0, -1e9, 1.0

def _index_cache_path(train_json: str, cfg: CFG, per_track_cap:int, min_voiced:float, min_snr_db:float, max_flatness:float) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    tag = f"{os.path.abspath(train_json)}|seg{cfg.segment_sec}|ov{cfg.overlap}|{_cfg_fp(cfg)}|cap{per_track_cap}|vr{min_voiced}|snr{min_snr_db}|flat{max_flatness}"
    h = hashlib.md5(tag.encode()).hexdigest()[:16]
    return os.path.join(cfg.cache_dir, f"index_sel_{h}.json")

def _save_index_cache(path: str, index: List[Tuple[str,int,int,int,str]]):
    data = [{"p": p, "s": int(s), "e": int(e), "lab": int(lab), "key": key} for (p,s,e,lab,key) in index]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def _load_index_cache(path: str):
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(d["p"], int(d["s"]), int(d["e"]), int(d["lab"]), d["key"]) for d in data]

def build_index_filtered(train_json: str, cfg: CFG, classes: List[str],
                         per_track_cap: int = 6, min_voiced: float = 0.45,
                         min_snr_db: float = 2.0, max_flatness: float = 0.95,
                         workers: Optional[int] = None) -> List[Tuple[str,int,int,int,str]]:
    # cache check
    idx_cache = _index_cache_path(train_json, cfg, per_track_cap, min_voiced, min_snr_db, max_flatness)
    cached = _load_index_cache(idx_cache)
    if cached: return cached

    if workers is None: workers = max(1, (cpu_count() or 2) - 1)

    # enumerate all chunks
    paths = _paths_from_json(train_json)
    cls2i = {c:i for i,c in enumerate(classes)}
    seg_len = int(cfg.segment_sec * cfg.sr)
    step = int(seg_len * (1.0 - cfg.overlap)) if cfg.overlap < 1.0 else 1

    all_chunks = []
    for p in paths:
        lab_name = _label_from_path(p)
        if lab_name not in cls2i: continue
        lab = cls2i[lab_name]
        try: dur = librosa.get_duration(path=p)
        except TypeError: dur = librosa.get_duration(filename=p)
        total = int(max(1, math.floor((dur * cfg.sr - seg_len) / step) + 1))
        key = os.path.splitext(os.path.basename(p))[0]
        for k in range(total):
            s = k * step; e = s + seg_len
            all_chunks.append((p, s, e, lab, key))

    if not all_chunks: return []

    # score chunks (MP)
    jobs = [(i, p, s, e, cfg) for i, (p, s, e, _, _) in enumerate(all_chunks)]
    scores = [None] * len(all_chunks)
    with Pool(processes=workers) as pool:
        for i, sc, vr, snr, fl in tqdm(pool.imap_unordered(_score_one_chunk, jobs),
                                       total=len(jobs), desc="Score chunks", ncols=100):
            scores[i] = (sc, vr, snr, fl)

    # filter + per-track top-K
    buckets = {}
    for info, met in zip(all_chunks, scores):
        if met is None: continue
        sc, vr, snr, fl = met
        ok = (vr >= min_voiced) and (snr >= min_snr_db) and (fl <= max_flatness)
        if ok:
            p, s, e, lab, key = info
            buckets.setdefault(key, []).append((sc, (p, s, e, lab, key)))

    final_index = []
    for key, lst in buckets.items():
        lst.sort(key=lambda t: t[0], reverse=True)
        final_index.extend([rec for _, rec in lst[:per_track_cap]])

    # precompute mel for selected chunks (fast later)
    _precompute_mels(final_index, cfg, workers=workers)
    _save_index_cache(idx_cache, final_index)
    return final_index

# =========================
# Dataset / Dataloaders
# =========================
class ChunkDatasetFixed(Dataset):
    """固定切片 index，回傳 (1,M,T) mel, label, key"""
    def __init__(self, index: List[Tuple[str,int,int,int,str]], cfg: CFG, split: str = "train"):
        self.index = index
        self.cfg = cfg
        self.split = split

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        p, s, e, lab, key = self.index[i]
        cpath = _mel_cache_path(self.cfg.cache_dir, p, s, e, self.cfg)

        if self.split == "train":
            heavy = (np.random.rand() < self.cfg.wave_aug_heavy_prob)
            if self.cfg.train_use_cache and os.path.exists(cpath) and not heavy:
                mel = np.load(cpath).astype(np.float32)
                if np.random.rand() < self.cfg.aug_spec_prob:
                    mel = _apply_specaugment(mel, self.cfg)
            else:
                mel = _compute_mel_chunk(p, s, e, self.cfg, augment=True, for_train=True)
        else:
            if os.path.exists(cpath):
                mel = np.load(cpath).astype(np.float32)
            else:
                mel = _compute_mel_chunk(p, s, e, self.cfg, augment=False, for_train=False)
                try: np.save(cpath, mel)
                except: pass

        x = torch.from_numpy(mel).unsqueeze(0)  # (1,M,T)
        return x, torch.tensor(lab, dtype=torch.long), key

def make_loaders(train_json: str, val_json: str, cfg: CFG, classes: Optional[List[str]] = None):
    if classes is None:
        classes = build_classes_from_json(train_json)

    idx_tr = build_index_filtered(
        train_json, cfg, classes,
        per_track_cap=10, min_voiced=0.30, min_snr_db=2.0, max_flatness=0.95
    )
    idx_va = build_index(val_json, cfg, classes)
    _precompute_mels(idx_va, cfg)

    ds_tr = ChunkDatasetFixed(idx_tr, cfg, split="train")
    ds_va = ChunkDatasetFixed(idx_va, cfg, split="val")

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return dl_tr, dl_va, idx_tr, idx_va, classes

# =========================
# Voting helpers
# =========================
def aggregate_by_keys(proba: np.ndarray, labels: np.ndarray, keys: List[str], n_classes: int,
                      method: str = "mean"):
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, k in enumerate(keys): buckets[k].append(i)

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

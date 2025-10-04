# -*- coding: utf-8 -*-
# file: task2_dataload.py
import os, json, math, random, hashlib, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable
from collections import OrderedDict, defaultdict

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, MFCC
import audiomentations as A

# ===== Demucs（GPU 版）=====
from demucs.pretrained import get_model as demucs_get_model
from demucs.apply import apply_model
import multiprocessing as mp

try:
    import torch_audiomentations as TA
    _HAS_TA = True
except Exception:
    _HAS_TA = False


warnings.filterwarnings("ignore", category=UserWarning)
# ==== 多進程初始化（共享 GPU 鎖）====
_GPU_LOCK = None  # 全域 GPU 鎖（確保 Demucs 單線程使用 GPU）

def _mp_init(gpu_lock):
    """Pool initializer：把 Manager() 的 Lock 帶進子進程。"""
    global _GPU_LOCK
    _GPU_LOCK = gpu_lock
# reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
try: torch.set_float32_matmul_precision("high")
except Exception: pass


# =========================
# Config
# =========================
@dataclass
class CFG:
    # audio / feature
    sr: int = 22050
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 128
    fmin: int = 40
    fmax: int = 8000
    mfcc_n: int = 40

    # segmentation
    segment_sec: int = 10
    overlap: float = 0.0

    # dataloader
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # cache（只存特徵；不存 stem）
    cache_dir: str = "cache_task2"

    # 片段挑選條件（在 vocal 上打分）
    per_track_cap: int = 15     # Top-K
    min_voiced: float = 0.10    # 0.20
    min_snr_db: float = 1.0     # 2.0
    max_flatness: float = 0.85  # 0.95

    # in-memory stem LRU（GB）
    stem_mem_gb: float = 4.0


# =========================
# 路徑 & 類別
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
            os.path.join(base, p_norm),
            os.path.join(parent, p_norm),
        ]
        for c in cands:
            c = os.path.normpath(c)
            if os.path.isfile(c) and c not in seen:
                out.append(c); seen.add(c)
                break
    return sorted(out)

def _label_from_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 3: return parts[-3]
    if len(parts) >= 2: return parts[-2]
    return "unknown"

def build_classes_from_json(train_json: str) -> List[str]:
    paths = _paths_from_json(train_json)
    return sorted({_label_from_path(p) for p in paths})[:]


# =========================
# Cache path helpers（特徵）
# =========================
def _cfg_fp(cfg: CFG) -> str:
    core = f"sr{cfg.sr}_nfft{cfg.n_fft}_hop{cfg.hop}_m{cfg.n_mels}_{cfg.fmin}-{cfg.fmax}"
    feat = f"mel_dd_mfcc{cfg.mfcc_n}_dd"
    return f"{core}_demucs_{feat}_specaug"

def _feat_cache_path(cache_dir: str, src_path: str, s: int, e: int, cfg: CFG) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5(f"{os.path.abspath(src_path)}|{s}|{e}|{_cfg_fp(cfg)}".encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"feat_{h}.npy")

def _index_cache_path(train_json: str, cfg: CFG) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    tag = f"{os.path.abspath(train_json)}|seg{cfg.segment_sec}|ov{cfg.overlap}|sr{cfg.sr}|demucs|{_cfg_fp(cfg)}|cap{cfg.per_track_cap}|vr{cfg.min_voiced}|snr{cfg.min_snr_db}|flat{cfg.max_flatness}"
    h = hashlib.md5(tag.encode()).hexdigest()[:16]
    return os.path.join(cfg.cache_dir, f"index_sel_{h}.json")

def _save_index_cache(path: str, index: List[Tuple[str,int,int,int,str]]):
    data = [{"p": p, "s": int(s), "e": int(e), "lab": int(lab), "key": key} for (p,s,e,lab,key) in index]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def _load_index_cache(path: str) -> Optional[List[Tuple[str,int,int,int,str]]]:
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(d["p"], int(d["s"]), int(d["e"]), int(d["lab"]), d["key"]) for d in data]


# =========================
# Demucs（GPU）+ in-memory LRU（不落地 stem）
# =========================
_DEMUCS = None
def _get_demucs(device: torch.device):
    global _DEMUCS
    if _DEMUCS is None:
        _DEMUCS = demucs_get_model('htdemucs').to(device)
        _DEMUCS.eval()
    return _DEMUCS

class _StemLRU:
    """把 vocal stem 放記憶體（float16）做 LRU；不寫磁碟。"""
    def __init__(self, max_gb: float = 4.0):
        self.max_bytes = int(max_gb * (1024**3))
        self.cur_bytes = 0
        self.od: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def _nbytes(self, arr: np.ndarray) -> int:
        try: return arr.nbytes
        except: return int(arr.size) * 2  # float16 fallback

    def get(self, key: str) -> Optional[np.ndarray]:
        if key not in self.od: return None
        arr = self.od.pop(key)
        self.od[key] = arr  # move to end (MRU)
        return arr.astype(np.float32, copy=False)

    def put(self, key: str, arr_f32: np.ndarray):
        arr16 = arr_f32.astype(np.float16)
        n = self._nbytes(arr16)
        # evict
        while self.cur_bytes + n > self.max_bytes and len(self.od) > 0:
            _, old = self.od.popitem(last=False)
            self.cur_bytes -= self._nbytes(old)
        self.od[key] = arr16
        self.cur_bytes += n

    def has(self, key: str) -> bool:
        return key in self.od

_STEM_MEM = _StemLRU()

def _stem_mem_key(src_path: str, cfg: CFG) -> str:
    return hashlib.md5(f"memstem|{os.path.abspath(src_path)}|demucs|{cfg.sr}".encode()).hexdigest()

def _separate_vocals_demucs_mem(src_path: str, cfg: CFG, device: torch.device) -> np.ndarray:
    """回傳 mono float32 vocal，不落地。使用 LRU 記憶體快取。"""
    key = _stem_mem_key(src_path, cfg)
    hit = _STEM_MEM.get(key)
    if hit is not None:
        return hit

    # 44.1k stereo
    wav, _ = librosa.load(src_path, sr=44100, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=0)  # (2,T)
    wav_t = torch.from_numpy(wav).float().to(device).unsqueeze(0)  # (1,2,T)

    model = _get_demucs(device)
    with torch.no_grad():
        if device.type == "cuda":
            ctx = torch.amp.autocast('cuda', dtype=torch.float16)
        else:
            class _NoCtx:
                def __enter__(self): return None
                def __exit__(self,*args): return False
            ctx = _NoCtx()
        with ctx:
            out = apply_model(model, wav_t, split=True, overlap=0.25, shifts=0, progress=False)

    voc_idx = model.sources.index("vocals")
    if isinstance(out, torch.Tensor):
        if out.dim() == 4:   # (B,S,C,T)
            vocals = out[0, voc_idx]       # (C,T)
        elif out.dim() == 3: # (S,C,T)
            vocals = out[voc_idx]
        else:
            raise RuntimeError(f"Unexpected Demucs output shape: {tuple(out.shape)}")
    else:
        raise RuntimeError("Unexpected Demucs output type.")

    v = vocals.mean(0).detach().cpu().numpy()  # mono @ 44.1k
    if 44100 != cfg.sr:
        v = librosa.resample(v, orig_sr=44100, target_sr=cfg.sr, res_type="kaiser_fast")

    _STEM_MEM.put(key, v.astype(np.float32))
    return v.astype(np.float32)


# =========================
# 特徵（GPU：Mel/MFCC + Δ/ΔΔ；SpecAug）
# =========================
class FeatureComputer:
    """回傳 ndarray (H, T)，H = n_mels*3 + mfcc_n*3"""
    def __init__(self, cfg: CFG, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.mel = MelSpectrogram(
            sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop,
            f_min=cfg.fmin, f_max=cfg.fmax, n_mels=cfg.n_mels, power=2.0
        ).to(device)
        self.to_db = AmplitudeToDB(stype="power").to(device)
        self.mfcc = MFCC(
            sample_rate=cfg.sr, n_mfcc=cfg.mfcc_n,
            melkwargs=dict(n_fft=cfg.n_fft, hop_length=cfg.hop,
                           f_min=cfg.fmin, f_max=cfg.fmax, n_mels=cfg.n_mels)
        ).to(device)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=16).to(device)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=32).to(device)

    def compute(self, y_np: np.ndarray, apply_specaug: bool) -> np.ndarray:
        x = torch.from_numpy(y_np).to(self.device).float().unsqueeze(0)  # (1, T)
        S = self.to_db(self.mel(x)).squeeze(0)                           # (M, T)
        dS1 = torchaudio.functional.compute_deltas(S)
        dS2 = torchaudio.functional.compute_deltas(dS1)

        MF = self.mfcc(x).squeeze(0)                                    # (mfcc_n, T)
        dM1 = torchaudio.functional.compute_deltas(MF)
        dM2 = torchaudio.functional.compute_deltas(dM1)

        F = torch.cat([S, dS1, dS2, MF, dM1, dM2], dim=0)               # (H, T)

        # voiced mask（在 waveform 上做 HPSS+RMS，再時間軸遮罩）
        m = _voiced_mask_np(y_np, self.cfg)
        if m.shape[0] == F.shape[1]:
            F = F * torch.from_numpy(m).to(self.device).float().unsqueeze(0)

        # z-norm
        mu, sd = F.mean(), F.std()
        F = (F - mu) / (sd + 1e-6)

        if apply_specaug:
            for _ in range(2):
                F = self.freq_mask(F.unsqueeze(0)).squeeze(0)
                F = self.time_mask(F.unsqueeze(0)).squeeze(0)

        return F.detach().cpu().numpy().astype(np.float32)

def _voiced_mask_np(y: np.ndarray, cfg: CFG) -> np.ndarray:
    y_h, _ = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y_h, frame_length=cfg.n_fft, hop_length=cfg.hop)[0]
    thr = np.percentile(rms, 30)
    return (rms > thr).astype(np.float32)  # (T_frames,)

# ===== Waveform Aug（若 LRU 有該曲 stem 才做，避免臨時跑 Demucs）=====
def _get_wave_augmenter(cfg: CFG):
    
    sr = int(cfg.sr)

    if _HAS_TA:
        # ---- torch-audiomentations 版本（支援 RandomApply）----
        tfms = [
            TA.RandomApply([TA.PolarityInversion()], p=0.8),
            TA.RandomApply([TA.Gain()], p=0.2),
            TA.RandomApply([TA.AddColoredNoise(min_snr_in_db=6.0, max_snr_in_db=12.0)], p=0.3),
            TA.RandomApply([TA.HighLowPass(sample_rate=sr)], p=0.8),
            TA.RandomApply([TA.PitchShift(sample_rate=sr)], p=0.4),
            TA.RandomApply([TA.Delay(sample_rate=sr)], p=0.5),
        ]
        # Reverb 並非每版都有，安全起見 try 一下
        try:
            tfms.append(TA.RandomApply([TA.Reverb(sample_rate=sr)], p=0.3))
        except Exception:
            pass

        return ("torch", TA.Compose(transforms=tfms, p=1.0, shuffle=True))

    else:
        # ---- audiomentations 版本（用 p 近似 RandomApply）----
        tfms = []
        from audiomentations import (
            PolarityInversion, AddGaussianSNR, Gain,
            HighPassFilter, LowPassFilter, Delay, PitchShift
        )

        tfms.append(PolarityInversion(p=0.8))
        tfms.append(AddGaussianSNR(min_snr_in_db=6.0, max_snr_in_db=12.0, p=0.3))
        tfms.append(Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.2))
        # 以 HighPass+LowPass 近似 HighLowPass
        tfms.append(HighPassFilter(min_cutoff_freq=80,  max_cutoff_freq=300, p=0.4))
        tfms.append(LowPassFilter( min_cutoff_freq=3000, max_cutoff_freq=8000, p=0.4))
        tfms.append(Delay(min_delay=0.01, max_delay=0.07, p=0.5))
        tfms.append(PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.4))
        # Reverb 不是每版都有；有就加
        try:
            from audiomentations import Reverb
            tfms.append(Reverb(p=0.3))
        except Exception:
            pass

        return ("numpy", A.Compose(tfms))


_WAVE_AUG_KIND = None
_WAVE_AUG      = None

def _ensure_wave_aug(cfg: CFG):
    global _WAVE_AUG_KIND, _WAVE_AUG
    if _WAVE_AUG is None:
        _WAVE_AUG_KIND, _WAVE_AUG = _get_wave_augmenter(cfg)

def _wave_augment(seg: np.ndarray, cfg: CFG) -> np.ndarray:
    """
    單聲道 1D numpy -> 套用波形增強 -> numpy
    會根據可用套件自動走 torch-audiomentations（優先）或 audiomentations。
    """
    _ensure_wave_aug(cfg)
    if _WAVE_AUG_KIND == "torch":
        # torch-audiomentations 需要 [B, C, T] 的 tensor
        x = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        with torch.no_grad():
            y = _WAVE_AUG(x, sample_rate=int(cfg.sr))
        return y.squeeze(0).squeeze(0).cpu().numpy()
    else:
        # audiomentations：直接 numpy
        return _WAVE_AUG(samples=seg.astype(np.float32), sample_rate=int(cfg.sr)).astype(np.float32)



# =========================
# Index & 片段挑選（每首跑一次 Demucs，不存 stem）
# =========================
def build_index(json_path: str, cfg: CFG, classes: List[str]) -> List[Tuple[str,int,int,int,str]]:
    paths = _paths_from_json(json_path)
    cls2i = {c:i for i,c in enumerate(classes)}
    seg_len = int(cfg.segment_sec * cfg.sr)
    step = int(seg_len * (1.0 - cfg.overlap)) if cfg.overlap < 1.0 else 1
    index = []
    for p in paths:
        lab = _label_from_path(p)
        if lab not in cls2i: continue
        try: dur = librosa.get_duration(path=p)
        except TypeError: dur = librosa.get_duration(filename=p)
        total = int(max(1, math.floor((dur * cfg.sr - seg_len) / step) + 1))
        key = os.path.splitext(os.path.basename(p))[0]
        for k in range(total):
            s = k * step; e = s + seg_len
            index.append((p, s, e, cls2i[lab], key))
    return index

def _chunk_quality(y: np.ndarray, cfg: CFG):
    y_h, y_p = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y_h, frame_length=cfg.n_fft, hop_length=cfg.hop)[0]
    thr = np.percentile(rms, 30)
    voiced_ratio = float((rms > thr).mean())
    snr = 10.0 * np.log10((np.mean(y_h**2) + 1e-10) / (np.mean(y_p**2) + 1e-10))
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop))**2
    flat = float(librosa.feature.spectral_flatness(S=S).mean())
    snr_score  = 1.0 / (1.0 + np.exp(-(snr - 3.0)))
    flat_score = 1.0 - np.clip(flat, 0.0, 1.0)
    score = 0.5 * voiced_ratio + 0.3 * snr_score + 0.2 * flat_score
    return score, voiced_ratio, snr, flat

def _filter_track_job(args):
    """
    一個工作處理一首歌：
    - 先拿 GPU 鎖跑 Demucs 取得該曲 vocal（避免多進程同時占 GPU）
    - 然後在 CPU 上對這首歌的所有 chunk 打分，挑 topK
    回傳：[(score, (p, s, e, lab, key)), ...] 只保留 topK
    """
    p, segs, cfg, device_str, per_track_cap = args
    device = torch.device(device_str)

    # 1) GPU 區段（受鎖保護）
    global _GPU_LOCK
    with _GPU_LOCK:
        v = _separate_vocals_demucs_mem(p, cfg, device)  # mono vocal @ cfg.sr

    # 2) CPU 區段（可並行）
    bucket = []
    for (s, e, lab, key) in segs:
        seg = v[s:e]
        need = e - s
        if seg.shape[0] < need:
            seg = np.pad(seg, (0, need - seg.shape[0]))
        sc, vr, snr, fl = _chunk_quality(seg, cfg)
        if (vr >= cfg.min_voiced) and (snr >= cfg.min_snr_db) and (fl <= cfg.max_flatness):
            bucket.append((sc, (p, s, e, lab, key)))

    if bucket:
        bucket.sort(key=lambda t: t[0], reverse=True)
        bucket = bucket[:per_track_cap]
        comp_cpu = FeatureComputer(cfg, torch.device('cpu'))
        for _, (pp, ss, ee, lab2, key2) in bucket:
            cpath = _feat_cache_path(cfg.cache_dir, pp, ss, ee, cfg)
            if os.path.exists(cpath):
                continue
            seg = v[ss:ee]
            need = ee - ss
            if seg.shape[0] < need:
                seg = np.pad(seg, (0, need - seg.shape[0]))
            F = comp_cpu.compute(seg, apply_specaug=False)
            np.save(cpath, F)
    return bucket  # list of (score, record)

def build_index_filtered(train_json: str, cfg: CFG, classes: List[str]) -> List[Tuple[str,int,int,int,str]]:
    idx_cache = _index_cache_path(train_json, cfg)
    cached = _load_index_cache(idx_cache)
    if cached: 
        return cached

    # enumerate chunks & group by track
    all_chunks = []
    by_track = defaultdict(list)  # p -> [(s,e,lab,key), ...]
    cls2i = {c:i for i,c in enumerate(classes)}
    seg_len = int(cfg.segment_sec * cfg.sr)
    step = int(seg_len * (1.0 - cfg.overlap)) if cfg.overlap < 1.0 else 1

    for p in _paths_from_json(train_json):
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
            s = k * step; e = s + seg_len
            all_chunks.append((p, s, e, lab, key))
            by_track[p].append((s, e, lab, key))

    # 多進程：每首歌一個工作；用 Manager().Lock() 保證 Demucs 單線程跑 GPU
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    workers = 2
    ctx = mp.get_context("spawn")  # 安全的啟動方式
    mgr = ctx.Manager()
    gpu_lock = mgr.Lock()

    jobs = [(p, segs, cfg, device_str, cfg.per_track_cap) for p, segs in by_track.items()]
    final_index_scored = []  # 先收集 (score, record)

    with ctx.Pool(processes=workers, initializer=_mp_init, initargs=(gpu_lock,)) as pool:
        for bucket in tqdm(pool.imap_unordered(_filter_track_job, jobs),
                           total=len(jobs), desc="Filter chunks (MP: Demucs+score)", ncols=100):
            if bucket:
                final_index_scored.extend(bucket)

    # 這裡已經是 per-track topK 相加的結果（每首歌已在 worker 內截成 topK）
    final_index = [rec for _, rec in final_index_scored]

    _save_index_cache(idx_cache, final_index)
    return final_index



# =========================
# 特徵預算（按「每首歌」處理；重用 LRU 的 vocal）
# =========================
def _group_index_by_path(index: Iterable[Tuple[str,int,int,int,str]]) -> Dict[str, List[Tuple[int,int,int,str]]]:
    g: Dict[str, List[Tuple[int,int,int,str]]] = defaultdict(list)
    for p, s, e, lab, key in index:
        g[p].append((s, e, lab, key))
    return g

def _precompute_features(index, cfg: CFG, device: torch.device):
    comp = FeatureComputer(cfg, device)
    by_path = _group_index_by_path(index)

    # 先挑出「至少有一個片段尚未有快取」的歌曲，避免多做無用分離
    todo = []
    for p, segs in by_path.items():
        missing = False
        for (s, e, _, _) in segs:
            cpath = _feat_cache_path(cfg.cache_dir, p, s, e, cfg)
            if not os.path.exists(cpath):
                missing = True
                break
        if missing:
            todo.append((p, segs))

    for p, segs in tqdm(todo, desc=f"Precompute feats ({device.type})", ncols=100):
        # 只在真的有缺時，才做一次 Demucs
        v = _separate_vocals_demucs_mem(p, cfg, device)
        for (s, e, _, _) in segs:
            cpath = _feat_cache_path(cfg.cache_dir, p, s, e, cfg)
            if os.path.exists(cpath):
                continue
            seg = v[s:e]
            need = e - s
            if seg.shape[0] < need:
                seg = np.pad(seg, (0, need - seg.shape[0]))
            F = comp.compute(seg, apply_specaug=False)
            np.save(cpath, F)


# =========================
# Dataset（訓練：讀 cache + SpecAug；10% 波形增強僅在 LRU 命中時）
# =========================
class ChunkDatasetFixed(Dataset):
    """
    回傳 x, y, key
    x shape: (1, H, T) where H = n_mels*3 + mfcc_n*3
    """
    def __init__(self, index: List[Tuple[str,int,int,int,str]], cfg: CFG, split: str, device: torch.device):
        self.index = index
        self.cfg = cfg
        self.split = split  # "train"/"val"
        self.device = device
        self.comp = FeatureComputer(cfg, torch.device('cpu'))

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        path, s, e, lab, key = self.index[i]
        cpath = _feat_cache_path(self.cfg.cache_dir, path, s, e, self.cfg)

        # 90%：直接讀特徵 + SpecAug；10%：若 LRU 有該 stem 才做波形增強
        do_wave = (self.split == "train") and (random.random() < 0.15) and _STEM_MEM.has(_stem_mem_key(path, self.cfg))
        if do_wave:
            v = _STEM_MEM.get(_stem_mem_key(path, self.cfg))
            seg = v[s:e]
            need = e - s
            if seg.shape[0] < need:
                seg = np.pad(seg, (0, need - seg.shape[0]))
            # ★ 在這裡套用你指定的增強配方（會自動選 torch/audiomentations）
            seg = _wave_augment(seg, self.cfg)
            F = self.comp.compute(seg, apply_specaug=True)
        else:
            F = np.load(cpath, allow_pickle=False).astype(np.float32)
            if self.split == "train":
                F = _specaug_torch(F)


        x = torch.from_numpy(F).unsqueeze(0)  # (1, H, T)
        return x, torch.tensor(lab, dtype=torch.long), key

def _specaug_torch(F_np: np.ndarray) -> np.ndarray:
    x = torch.tensor(F_np, dtype=torch.float32).unsqueeze(0)
    fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)
    tm = torchaudio.transforms.TimeMasking(time_mask_param=32)
    for _ in range(2):
        x = fm(x); x = tm(x)
    return x.squeeze(0).detach().cpu().numpy().astype(np.float32)


# =========================
# 封裝：建 Loader（Demucs in-mem → 特徵快取 → DataLoader）
# =========================
def make_loaders(train_json: str, val_json: str, cfg: CFG, classes: Optional[List[str]] = None):
    if classes is None:
        classes = build_classes_from_json(train_json)

    # 產生 index（train 用 filtered、val 用全切）
    idx_tr = build_index_filtered(train_json, cfg, classes)
    idx_va = build_index(val_json, cfg, classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 一次性把 train/val 會用到的特徵都算好（按每首歌批次，重用 LRU 的 vocal）
    # _precompute_features(idx_tr, cfg, device)
    _precompute_features(idx_va, cfg, device)

    ds_tr = ChunkDatasetFixed(idx_tr, cfg, split="train", device=torch.device('cpu'))
    ds_va = ChunkDatasetFixed(idx_va, cfg, split="val", device=torch.device('cpu'))

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True, multiprocessing_context=mp.get_context("spawn"), persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, multiprocessing_context=mp.get_context("spawn"), persistent_workers=True)
    return dl_tr, dl_va, idx_tr, idx_va, classes


# =========================
# Voting helpers
# =========================
def aggregate_by_keys(proba: np.ndarray, labels: np.ndarray, keys: List[str], n_classes: int,
                      method: str = "mean") -> Tuple[np.ndarray, np.ndarray, List[str]]:
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

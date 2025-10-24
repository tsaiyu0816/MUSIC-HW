from __future__ import annotations
import argparse, os, sys, json, csv, math, time, random, re, subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import soundfile as sf
import importlib.util
import librosa  # 已在別處用到就忽略
try:
    import pretty_midi
    _HAS_PM = True
except Exception:
    _HAS_PM = False


# ----------------------
# Utils
# ----------------------
AUDIO_EXT = ".wav"

def _slug(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^\w\-]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:maxlen] or "untitled")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_json_items(p: Path) -> List[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 應該是 list[ {target,prompt,...} ]")
    return data

def _write_csv(rows: List[Tuple[str, str, str]], out_csv: Path) -> None:
    _ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target", "generated", "prompt"])
        for r in rows:
            w.writerow(list(r))

def _seed_everything(seed: Optional[int]):
    if seed is None: return
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# ====== Cond extraction (from target) ======
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _resample_to_len(x: np.ndarray, L: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros(L, dtype=np.float32)
    idx_src = np.linspace(0, len(x) - 1, num=len(x))
    idx_tgt = np.linspace(0, len(x) - 1, num=L)
    return np.interp(idx_tgt, idx_src, x).astype(np.float32)

def _extract_rhythm_clicks(y: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    """用 onset 偵測產生 click 軌（mono wav，與 target 同 sr/長度）"""
    need = int(sr * duration_s)
    hop = 512
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=False, units="frames")
    times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    clicks = librosa.clicks(times=times, sr=sr, length=need)
    return clicks.astype(np.float32)

def _extract_dynamics_noise(y: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    """用 RMS 生成一條噪音+包絡的動態軌（mono wav）"""
    need = int(sr * duration_s)
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop, center=True).flatten()
    env = _resample_to_len(rms / (np.max(rms) + 1e-8), need)
    noise = np.random.randn(need).astype(np.float32) * 0.2
    return (noise * env).astype(np.float32)

def _extract_melody_midi_from_audio(y: np.ndarray, sr: int, duration_s: float, out_mid: Path):
    """用 pyin 估 f0 → 量化 → 寫成 .mid（單音旋律）"""
    hop = 512
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr, hop_length=hop, center=True
    )
    t = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    valid = ~np.isnan(f0)
    if not np.any(valid):
        # 沒抓到旋律就寫一個空 MIDI
        pm = pretty_midi.PrettyMIDI()
        pm.instruments.append(pretty_midi.Instrument(program=0))
        pm.write(str(out_mid))
        return

    midi_vals = np.round(librosa.hz_to_midi(np.clip(f0, 1e-6, None))).astype(np.float32)
    # 把連續相同音高合併成 note
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    min_len = 0.08  # 最短音長（秒）
    start_idx = None
    cur_pitch = None
    for i in range(len(midi_vals)):
        if valid[i]:
            p = int(midi_vals[i])
            if start_idx is None:
                start_idx = i; cur_pitch = p
            elif p != cur_pitch:
                # 結束上一個 note
                t0, t1 = float(t[start_idx]), float(t[i])
                if t1 - t0 >= min_len:
                    inst.notes.append(pretty_midi.Note(velocity=90, pitch=cur_pitch, start=t0, end=t1))
                start_idx = i; cur_pitch = p
        else:
            if start_idx is not None:
                t0, t1 = float(t[start_idx]), float(t[i])
                if t1 - t0 >= min_len:
                    inst.notes.append(pretty_midi.Note(velocity=90, pitch=cur_pitch, start=t0, end=t1))
                start_idx = None; cur_pitch = None
    # 收尾
    if start_idx is not None:
        t0, t1 = float(t[start_idx]), float(min(t[-1], duration_s))
        if t1 - t0 >= min_len:
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(cur_pitch), start=t0, end=t1))
    pm.instruments.append(inst)
    pm.write(str(out_mid))

def build_conditions_from_target(
    target_path: str | Path,
    *,
    duration_s: float,
    sr: int,
    cond_dir: Path
) -> tuple[Path | None, Path | None, Path | None]:
    """
    從 target 音檔/MIDI 建出三個條件檔案：
      - rhythm_clicks.wav
      - dynamics_amp.wav
      - melody.mid（若 target 本身是 .mid，就直接複製那個）
    回傳 (rhythm_wav, dynamics_wav, melody_mid) 的路徑（可能為 None）
    """
    target_path = Path(target_path)
    _ensure_dir(cond_dir)

    if target_path.suffix.lower() in (".mid", ".midi"):
        # melody 直接用原 MIDI；rhythm/dynamics 這版先不做（可留空或之後加 drum 測算）
        melody_mid = cond_dir / "melody.mid"
        if _HAS_PM:
            # 簡單複製
            with open(target_path, "rb") as fsrc, open(melody_mid, "wb") as fdst:
                fdst.write(fsrc.read())
        else:
            raise RuntimeError("需要 pretty_midi 來處理 MIDI，請先安裝：pip install pretty_midi")
        return (None, None, melody_mid)

    # 音檔：先讀、裁切/補齊到 duration_s
    y, _ = librosa.load(str(target_path), sr=sr, mono=True)
    need = int(sr * duration_s)
    if len(y) < need:
        y = np.pad(y, (0, need - len(y)))
    else:
        y = y[:need]
    y = y.astype(np.float32)

    # 1) rhythm -> clicks.wav
    rhythm_wav = cond_dir / "rhythm_clicks.wav"
    clicks = _extract_rhythm_clicks(y, sr, duration_s)
    sf.write(str(rhythm_wav), clicks, sr)

    # 2) dynamics -> noise*envelope.wav
    dynamics_wav = cond_dir / "dynamics_amp.wav"
    dyn = _extract_dynamics_noise(y, sr, duration_s)
    sf.write(str(dynamics_wav), dyn, sr)

    # 3) melody -> .mid
    if not _HAS_PM:
        raise RuntimeError("需要 pretty_midi 來輸出旋律 MIDI，請先安裝：pip install pretty_midi")
    melody_mid = cond_dir / "melody.mid"
    _extract_melody_midi_from_audio(y, sr, duration_s, melody_mid)

    return (rhythm_wav, dynamics_wav, melody_mid)

# ----------------------
# Backend: MusicGen_melody (Transformers)
# ----------------------
from pathlib import Path
from typing import Tuple

def gen_with_musicgen_melody(
    prompt: str,
    out_path: Path,
    *,
    melody_path: str | Path,
    max_new_tokens: int = 512,   # 仍可由外部指定；內部會安全截斷
    device: str = "cuda",
) -> Tuple[Path, int]:
    import warnings, os
    import numpy as np
    import torch
    import librosa, soundfile as sf
    from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

    # ------------- 基本設定（穩定優先） -------------
    model_id = "facebook/musicgen-melody"
    use_cuda = (device == "cuda") and torch.cuda.is_available()
    torch_dtype = torch.float32                 # 避免半精度踩雷
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ------------- 載模型（禁用 flash-attn/SDPA） -------------
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenMelodyForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation="eager",           # 關閉 flash-attn/SDPA 路徑
    ).to("cuda" if use_cuda else "cpu")
    model.eval()

    # ------------- 讀旋律 → 32k/mono -------------
    sr_mel = 32000
    mel, _sr = librosa.load(str(melody_path), sr=sr_mel, mono=True)
    mel = mel.astype(np.float32)

    # ------------- 打包輸入 -------------
    inputs = processor(
        text=[prompt],
        audio=[mel],
        sampling_rate=sr_mel,
        return_tensors="pt",
        padding=True,
    )
    # 把所有 tensor 丟到裝置上
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    # 有些版本鍵名不同，盡量把浮點欄位對齊 dtype
    for k in ["input_features", "input_values", "audio_values"]:
        if k in inputs and torch.is_tensor(inputs[k]) and inputs[k].dtype != model.dtype:
            inputs[k] = inputs[k].to(model.dtype)

    # --- 推論前：決定每秒 tokens ---
    fps = getattr(getattr(model.config, "audio_encoder", None), "frame_rate", 50)
    # 想要幾秒？用 max_new_tokens 表示「目標秒數 × 50」也可以，但更直覺我們直接用秒數：
    target_seconds = 30  # 你要 12 秒就填 12；或把它改成函式參數
    target_tokens = int(target_seconds * fps)

    # 上個版本太保守了；現在只做一個很寬鬆的全域上限，避免 OOM（例如最多 60 秒）
    target_tokens = max(1, min(target_tokens, fps * 60))

    gen_kwargs = dict(
        # 關鍵：把 min_new_tokens 設成跟 max_new_tokens 一樣，避免過早遇到 EOS 提早結束
        max_new_tokens=target_tokens,
        min_new_tokens=target_tokens,
        do_sample=True,
        temperature=1.0,
        top_k=250,
        top_p=0.95,
        guidance_scale=3.0,
        # 不額外傳 eos_token_id，讓 min_new_tokens 生效去「撐住」長度
    )
    print(f"[musicgen] fps={fps}, target_seconds={target_seconds}, tokens={target_tokens}")


    # ------------- 生成（主要路徑） -------------
    def _generate(_model, _inputs, _kwargs):
        with torch.inference_mode():
            if _model.device.type == "cuda":
                # 關閉 autocast，避免被自動半精度
                with torch.cuda.amp.autocast(enabled=False):
                    return _model.generate(**_inputs, **_kwargs)
            else:
                return _model.generate(**_inputs, **_kwargs)

    try:
        audio = _generate(model, inputs, gen_kwargs)
    except RuntimeError as e:
        # 若是 device-side assert，一律做乾淨重試（CPU），拿更清楚的錯誤並不中斷批次
        msg = str(e)
        need_cpu_retry = (
            "device-side assert triggered" in msg
            or "indexSelectSmallIndex" in msg
            or "index out of range" in msg
        )
        if use_cuda and need_cpu_retry:
            warnings.warn("[musicgen-melody] CUDA 斷言觸發，改用 CPU 重試一次以獲得清楚錯誤訊息。")
            try:
                # 釋放 GPU
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass

            # CPU 重新載入 + 重跑（並把生成長度再縮一點）
            model = MusicgenMelodyForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            ).to("cpu").eval()

            inputs_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in inputs.items()}

            # 再保守一點
            gen_kwargs_cpu = dict(gen_kwargs)
            gen_kwargs_cpu["max_new_tokens"] = max(1, gen_kwargs["max_new_tokens"] // 2)

            audio = _generate(model, inputs_cpu, gen_kwargs_cpu)
        else:
            raise

    # ------------- 存檔 -------------
    sr_out = getattr(model.config, "sampling_rate", 32000)
    wav = audio[0].detach().cpu().to(torch.float32).numpy()
    # (C,T)->(T,C)
    if wav.ndim == 2 and wav.shape[0] <= 4 and wav.shape[0] < wav.shape[1]:
        wav = wav.T
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, samplerate=sr_out)
    return out_path, sr_out

# ----------------------
# Backend: MuseControlLite (subprocess)
# ----------------------
import os, sys  # ★ FIX: 遺漏的 import
import shlex
import json
import shutil
import subprocess
import textwrap
import re
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

def gen_with_musecontrol(
    prompt: str,
    out_path: Path,
    *,
    # 來源/執行
    muse_repo: Optional[Path] = None,
    python_exe: Optional[str] = None,
    cli_abs: Optional[str] = None,
    mc_arg: Optional[List[str]] = None,
    device: str = "cuda",

    # 生成參數
    duration: Optional[float] = None,
    seconds: float = 12.0,
    sample_rate: int = 44100,
    steps: Optional[int] = None,
    cond_fps: Optional[int] = None,
    seed: Optional[int] = 3407,

    # 條件
    target: Optional[str | Path] = None,
    melody_path: Optional[Path] = None,
    condition_type: Optional[List[str]] = None,

    # 其他
    env_extra: Optional[dict] = None,
    timeout_sec: int = 1800,
) -> Tuple[Path, int]:
    """
    穩定單條推論：
    - 強制 default.yaml 單一 audio + 單一 prompt（備份/還原）
    - melody_condition_audio 暫時只放本輪音檔（備份/還原）
    - 內建 sitecustomize 修補 & 關閉評分
    - 自動把 mp3/flac/ogg 轉成 WAV（若系統有 ffmpeg）
    """
    import soundfile as sf  # 用來讀回取樣率

    # ---- 秒數參數 ----
    sec = float(seconds if seconds is not None else (duration if duration is not None else 12.0))

    # ---- 路徑 ----
    repo = Path(muse_repo or os.environ.get("MUSECTRL_REPO", "")).expanduser().resolve()
    cli = Path(cli_abs).expanduser() if cli_abs else (repo / "MuseControlLite_inference.py")
    cfg_py = repo / "config_inference.py"
    if not cli.is_file() or not cfg_py.is_file():
        raise FileNotFoundError(
            f"[MuseControlLite] 找不到必要檔案：\n - {cli}\n - {cfg_py}\n請確認 muse_repo/cli_abs。"
        )
    py = python_exe or sys.executable
    mc_arg = list(mc_arg or [])

    # ---- condition_type ----
    if condition_type is None:
        condition_type = ["melody"] if (melody_path or target) else ["text"]

    # ---- 若提供 target，自動萃取條件檔 ----
    rhythm_wav = dynamics_wav = melody_mid = None
    if target:
        try:
            from txt2music import build_conditions_from_target
            cond_dir = (repo / "runs" / "tmp_conds"); cond_dir.mkdir(parents=True, exist_ok=True)
            rhythm_wav, dynamics_wav, melody_mid = build_conditions_from_target(
                target_path=target, duration_s=sec, sr=sample_rate, cond_dir=cond_dir
            )
            if ("melody" in condition_type) and (melody_path is None):
                if Path(target).suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                    melody_path = Path(target).expanduser().resolve()
        except Exception as e:
            print(f"[MuseControlLite][warn] 自動萃取條件失敗：{e}")

    # ---- 工具 ----
    def _scan_audio(root: Path) -> set[Path]:
        return set(p for p in root.rglob("*") if p.suffix.lower() in (".wav",".flac",".mp3",".ogg"))

    def _abs(p: Optional[Path | str]) -> Optional[str]:
        return None if p is None else str(Path(p).expanduser().resolve())

    def _ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    def _to_readable_wav(src: Path, mel_dir: Path, sr: int) -> Path:
        """若 src 為 mp3/flac/ogg，盡量轉成 WAV；否則複製/連結原檔。"""
        try:
            src = src.expanduser().resolve()
        except Exception:
            pass
        if src.suffix.lower() == ".wav":
            return src
        dst = mel_dir / (src.stem + ".wav")
        if _ffmpeg_available():
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(sr),
                     "-sample_fmt", "s16", str(dst)],
                    check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                if dst.exists():
                    return dst.resolve()
            except Exception as e:
                print(f"[mc-lite][warn] ffmpeg 轉檔失敗：{e}")
        # 兜底：直接複製原檔（讓後端嘗試解碼）
        fallback = mel_dir / src.name
        try:
            shutil.copy2(src, fallback)
            return fallback.resolve()
        except Exception:
            return src

    def _pick_audio_for_yaml() -> str:
        # 1) target 若是音訊就用它
        if target and Path(target).suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
            return _abs(target)  # type: ignore
        # 2) melody_path 若是音訊就用它
        if melody_path and Path(melody_path).suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
            return _abs(melody_path)  # type: ignore
        # 3) 試著在 <repo>/melody_condition_audio 找同名
        if target:
            stem = Path(target).stem
            mel_dir = repo / "melody_condition_audio"
            for ext in (".wav", ".mp3", ".flac", ".ogg"):
                cand = mel_dir / f"{stem}{ext}"
                if cand.exists():
                    return str(cand.resolve())
        raise FileNotFoundError(
            "[MuseControlLite] 本輪沒有可用的『音訊檔』做為 audio_files。"
            "若 target 是 .mid，請同名放一份 .wav 到 melody_condition_audio/。"
        )

    # ---- 準備輸出與暫存 ----
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = repo / "runs" / f"infer_{run_tag}"
    patch_dir = save_dir / "_py_patches"
    save_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    # ---- sitecustomize（修補 & 關閉評分 & melody 解析）----
    sitecustomize_code = textwrap.dedent("""
        # -*- coding: utf-8 -*-
        # sitecustomize for MuseControlLite subprocess

        # 0) NumPy 2.x alias compat
        try:
            import numpy as _np
            _alias_py = {"bool": bool, "int": int, "float": float, "complex": complex, "object": object, "str": str}
            for _name, _pytype in _alias_py.items():
                if not hasattr(_np, _name): setattr(_np, _name, _pytype)
            if not hasattr(_np, "long"): setattr(_np, "long", int)
            print("[mc-lite] sitecustomize: numpy alias compat applied")
        except Exception as _e:
            print("[mc-lite][warn] numpy alias compat failed:", _e)

        # 1) collections compat for madmom
        try:
            import collections as _c, collections.abc as _abc
            for _name in ("MutableMapping","MutableSequence","Mapping","Sequence","Iterable"):
                if not hasattr(_c, _name) and hasattr(_abc, _name):
                    setattr(_c, _name, getattr(_abc, _name))
            print("[mc-lite] sitecustomize: collections compat applied")
        except Exception as _e:
            print("[mc-lite][warn] collections compat failed:", _e)

        # 2) Diffusers .to('cuda') -> half precision default
        try:
            import torch, diffusers
            torch.set_grad_enabled(False)
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                try: torch.set_float32_matmul_precision("high")
                except Exception: pass
            except Exception: pass
            _orig_to = diffusers.pipelines.pipeline_utils.DiffusionPipeline.to
            def _safe_to(self, device=None, dtype=None):
                if str(device) == "cuda" and dtype is None:
                    try:
                        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
                    except Exception:
                        dtype = torch.float16
                return _orig_to(self, device=device, dtype=dtype)
            diffusers.pipelines.pipeline_utils.DiffusionPipeline.to = _safe_to
            print("[mc-lite] sitecustomize: pipeline.to → default dtype=bf16/fp16")
        except Exception as _e:
            print("[mc-lite][warn] diffusers half-precision patch failed:", _e)

        # 3) Global nn.Module.__call__ kw filter
        try:
            import torch, inspect
            _orig_mod_call = torch.nn.Module.__call__
            def _mod_call_shim(self, *args, **kwargs):
                try:
                    return _orig_mod_call(self, *args, **kwargs)
                except TypeError as e:
                    s = str(e)
                    if "unexpected keyword argument" in s:
                        try:
                            sig = inspect.signature(self.forward)
                            allowed = set(sig.parameters.keys())
                            filtered = {k: v for k, v in kwargs.items() if k in allowed}
                            if filtered != kwargs:
                                return _orig_mod_call(self, *args, **filtered)
                        except Exception:
                            pass
                    raise
            torch.nn.Module.__call__ = _mod_call_shim
            print("[mc-lite] sitecustomize: nn.Module.__call__ shim")
        except Exception as _e:
            print("[mc-lite][warn] nn.Module.__call__ shim failed:", _e)

        # 4) Disable evaluation hooks (robust)
        try:
            import builtins, sys
            _orig_import = builtins.__import__
            def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):
                mod = _orig_import(name, globals, locals, fromlist, level)
                try:
                    def _patch_eval_if_ready():
                        m = sys.modules.get("MuseControlLite_setup")
                        if m is not None:
                            def _skip_eval(*a, **k):
                                print("[mc-lite] evaluation disabled (patched via import hook).")
                                return 0.0, 0.0, 0.0
                            try:
                                m.evaluate_and_plot_results = _skip_eval
                            except Exception:
                                pass
                    _patch_eval_if_ready()
                except Exception:
                    pass
                return mod
            builtins.__import__ = _import_hook
            if "MuseControlLite_setup" in sys.modules:
                sys.modules["MuseControlLite_setup"].evaluate_and_plot_results = (lambda *a, **k: (0.0, 0.0, 0.0))
            print("[mc-lite] sitecustomize: evaluation will be disabled via import hook")
        except Exception as _e:
            print("[mc-lite][warn] cannot install evaluation hook:", _e)

        # 5) compute_melody_v2 path-resolve shim (audio only, no random fallback)
        try:
            import os, pathlib
            import utils.extract_conditions as _uec
            _orig_compute = _uec.compute_melody_v2

            AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")
            def _is_audio(p):
                return str(p).lower().endswith(AUDIO_EXTS)

            def _same_stem_in_mel_dir(p_in: pathlib.Path, cwd: pathlib.Path):
                stem = p_in.stem
                mel_dir = cwd / "melody_condition_audio"
                for ext in (p_in.suffix, *AUDIO_EXTS):
                    q = mel_dir / f"{stem}{ext}"
                    if q.exists():
                        return q.resolve()
                return None

            def _compute_shim(audio_file, *a, **k):
                from importlib import import_module
                cwd = pathlib.Path(os.getcwd())

                # 1) prefer config_inference.melody_path if it's an audio file
                cfg_mel = None
                try:
                    _cfg = import_module("config_inference")
                    _cand = getattr(_cfg, "melody_path", None)
                    if _cand and _is_audio(_cand) and pathlib.Path(_cand).exists():
                        cfg_mel = pathlib.Path(_cand).resolve()
                except Exception:
                    pass

                # 2) else try audio_file itself (must be audio)
                p_in = pathlib.Path(str(audio_file))
                if not p_in.is_absolute():
                    p_in = (cwd / p_in)

                p = cfg_mel if cfg_mel is not None else (p_in if p_in.exists() and _is_audio(p_in) else None)

                # 3) else try same-stem under melody_condition_audio/
                if p is None:
                    p = _same_stem_in_mel_dir(p_in, cwd)

                if p is None or not p.exists():
                    raise FileNotFoundError(f"[mc-lite] 無法解析可用的 melody 音訊：{audio_file}")

                print(f"[mc-lite] compute_melody_v2 using: {p}")
                try:
                    return _orig_compute(str(p), *a, **k)
                except Exception:
                    alt = p.with_suffix(".wav")
                    if alt != p and alt.exists():
                        print(f"[mc-lite] retry compute_melody_v2 with: {alt}")
                        return _orig_compute(str(alt), *a, **k)
                    raise

            _uec.compute_melody_v2 = _compute_shim
            print("[mc-lite] sitecustomize: compute_melody_v2 shim installed (audio only)")
        except Exception as _e:
            print("[mc-lite][warn] compute_melody_v2 shim failed:", _e)

        # 6) Make load_audio_file robust to list/tuple inputs
        try:
            import utils.stable_audio_dataset_utils as _sadu
            _orig_load_audio_file = _sadu.load_audio_file

            def _flatten1(x):
                if isinstance(x, (list, tuple)):
                    while isinstance(x, (list, tuple)) and len(x) > 0:
                        x = x[0]
                return x

            def _load_audio_file_shim(filename, *a, **k):
                fn = _flatten1(filename)
                if not isinstance(fn, (str, bytes,)):
                    raise TypeError(f"[mc-lite] load_audio_file got invalid type: {type(fn)} ({fn})")
                try:
                    print(f"[mc-lite] load_audio_file ← {fn}")
                except Exception:
                    pass
                return _orig_load_audio_file(fn, *a, **k)

            _sadu.load_audio_file = _load_audio_file_shim
            print("[mc-lite] sitecustomize: load_audio_file shim installed")
        except Exception as _e:
            print("[mc-lite][warn] cannot patch load_audio_file:", _e)
    """).strip()
    (patch_dir / "sitecustomize.py").write_text(sitecustomize_code, encoding="utf-8")

    # --- optional：若 target 是音訊，鏡射到 melody_condition_audio 方便尋徑，並盡量轉成 WAV
    audio_for_yaml = _pick_audio_for_yaml()
    try:
        src = Path(audio_for_yaml).expanduser().resolve()
    except Exception:
        src = None

    mel_dir = repo / "melody_condition_audio"
    mel_dir_backup = repo / f".melody_condition_audio.bak_{run_tag}"
    mel_dir_created = False
    try:
        if mel_dir.exists() and mel_dir.is_dir():
            mel_dir.rename(mel_dir_backup)
        mel_dir.mkdir(parents=True, exist_ok=True)
        mel_dir_created = True
        if src and src.exists():
            # 先嘗試「可讀 WAV」
            readable = _to_readable_wav(src, mel_dir, sample_rate)
            if readable.parent != mel_dir:
                # 若轉檔失敗仍為原路徑，就建立一個副本/連結到 mel_dir
                dst = mel_dir / readable.name
                try:
                    os.symlink(str(readable), str(dst))
                    readable = dst
                except Exception:
                    try:
                        shutil.copy2(readable, dst); readable = dst
                    except Exception:
                        pass
            # 若使用 melody 條件且尚未明確指定 melody_path，補上
            if ("melody" in condition_type) and (melody_path is None):
                melody_path = readable
    except Exception as e:
        print(f"[MuseControlLite][warn] 準備 melody_condition_audio（單檔模式）失敗：{e}")

    # ---- 覆蓋 config_inference.py（供 compute_melody_v2 讀取）----
    ov = {
        "condition_type": condition_type,
        "prompt": prompt,
        "seconds": sec,
        "sample_rate": int(sample_rate),
        "seed": int(seed) if seed is not None else None,
        "device": "cuda" if (device == "cuda") else "cpu",
        "save_dir": str(save_dir),
        "out_dir": str(save_dir),
        "output_dir": str(save_dir),
        "melody_path": _abs(melody_path) or (_abs(target) if target else None),
        "rhythm_path": _abs(rhythm_wav),
        "dynamics_path": _abs(dynamics_wav),
        "steps": int(steps) if steps is not None else None,
        "cond_fps": int(cond_fps) if cond_fps is not None else None,

        # ★ 關鍵：單一元素，避免空清單；後續再對齊長度
        "audio_files": [audio_for_yaml],
        "prompt_texts": [prompt],
        "prompts": [prompt],
    }

    def _q(s: str) -> str:
        return s.replace('\\', '\\\\').replace('"', '\\"')

    lines = [
        "# ==== appended by txt2music.py (DO NOT COMMIT) ====",
        "import json as _json",
        "try:\n    config\nexcept NameError:\n    config = {}",
    ]
    for k, v in ov.items():
        if v is None:
            continue
        if isinstance(v, str):
            s = _q(v)
            lines.append(f'{k} = "{s}"')
            lines.append(f'config["{k}"] = "{s}"')
        else:
            j = json.dumps(v, ensure_ascii=False)
            lines.append(f"{k} = {j}")
            lines.append(f'config["{k}"] = {j}')

    # ★★ 長度對齊與廣播（避免 IndexError）
    lines += [
        "",
        "def _align_lists(_cfg):",
        "    a = list(_cfg.get('audio_files', []))",
        "    p = list(_cfg.get('prompt_texts', [])) or list(_cfg.get('prompts', []))",
        "    if not a and _cfg.get('melody_path'):",
        "        a = [_cfg['melody_path']]",  # 不給空清單
        "    if len(a) == 1 and len(p) > 1:",
        "        a = a * len(p)",
        "    if len(p) == 1 and len(a) > 1:",
        "        p = p * len(a)",
        "    n = min(len(a), len(p)) if (a and p) else max(len(a), len(p))",
        "    a = a[:n]; p = p[:n]",
        "    _cfg['audio_files']  = a",
        "    _cfg['prompt_texts'] = p",
        "    _cfg['prompts']      = p",
        "    return _cfg",
        "config = _align_lists(config)",
        "",
        "# ---- disable evaluation to avoid madmom HMM crash ----",
        "try:",
        "    import MuseControlLite_setup as _msetup",
        "    def _skip_eval(*a, **k):",
        "        print('[mc-lite] evaluation disabled (patched).')",
        "        return 0.0, 0.0, 0.0",
        "    _msetup.evaluate_and_plot_results = _skip_eval",
        "except Exception as _e:",
        "    print('[mc-lite][warn] cannot patch evaluate_and_plot_results:', _e)",
        "# ==== end of override ====",
        "",
    ]
    patch_text = "\n".join(lines)

    cfg_backup = cfg_py.read_text(encoding="utf-8", errors="ignore")
    cfg_py.write_text(cfg_backup + "\n" + patch_text, encoding="utf-8")

    # ---- 「硬覆蓋」 default.yaml：單一 audio + 單一 prompt（跑完還原）----
    def _find_default_yaml() -> Path:
        if (repo / "default.yaml").is_file(): return (repo / "default.yaml")
        if (repo / "configs" / "default.yaml").is_file(): return (repo / "configs" / "default.yaml")
        for a in mc_arg:
            m = re.match(r"--cfg=(.+)", a.strip())
            if m:
                p = Path(m.group(1))
                return p if p.is_absolute() else (repo / p)
        raise FileNotFoundError("找不到 default.yaml；請把它放在 repo 根目錄或 configs/ 下。")

    base_yaml = _find_default_yaml()
    default_yaml_backup = base_yaml.with_suffix(".yaml.bak_" + run_tag)

    try:
        import yaml
        base_cfg = {}
        try:
            base_cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8")) or {}
        except Exception:
            base_cfg = {}

        # 強制長度 = 1（永不寫出空清單）
        base_cfg["audio_files"]   = [audio_for_yaml]
        base_cfg["prompt_texts"]  = [prompt]
        base_cfg["prompts"]       = [prompt]

        base_cfg["seconds"]       = sec
        base_cfg["sample_rate"]   = int(sample_rate)
        if seed is not None: base_cfg["seed"] = int(seed)
        if steps is not None: base_cfg["steps"] = int(steps)
        if cond_fps is not None: base_cfg["cond_fps"] = int(cond_fps)
        base_cfg["condition_type"] = condition_type
        base_cfg["save_dir"] = str(save_dir)
        base_cfg["out_dir"] = str(save_dir)
        base_cfg["output_dir"] = str(save_dir)

        shutil.copy2(base_yaml, default_yaml_backup)
        base_yaml.write_text(yaml.safe_dump(base_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"[mc-lite] default.yaml patched → 單一樣本模式（已備份 {default_yaml_backup.name}）")
    except Exception as e:
        print(f"[mc-lite][warn] 解析/覆蓋 default.yaml 失敗，改用最小 YAML：{e}")
        if not default_yaml_backup.exists():
            try:
                shutil.copy2(base_yaml, default_yaml_backup)
            except Exception:
                pass

        def _yq(s: str) -> str:
            return json.dumps(str(s), ensure_ascii=False)

        min_yaml = (
            "audio_files:\n  - " + _yq(audio_for_yaml) + "\n"
            + "prompt_texts:\n  - " + _yq(prompt) + "\n"
            + "prompts:\n  - " + _yq(prompt) + "\n"
        )
        base_yaml.write_text(min_yaml, encoding="utf-8")

    # ---- CLI 參數：移除原本的 --cfg，避免又去合併其它清單 ----
    mc_arg = [a for a in mc_arg if not a.strip().startswith("--cfg=")]

    # ---- 執行 ----
    before = _scan_audio(repo)
    try:
        env = os.environ.copy()
        if device != "cuda":
            env["CUDA_VISIBLE_DEVICES"] = ""
        if env_extra:
            env.update(env_extra)

        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(patch_dir) + (os.pathsep + old_pp if old_pp else "")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

        cmd = [py, str(cli)]
        for a in mc_arg:
            cmd.extend(shlex.split(a))

        print(f"[MuseControlLite] run: {' '.join(cmd)} (cwd={repo})")
        completed = subprocess.run(
            cmd, cwd=str(repo), env=env, timeout=timeout_sec,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        print(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"MuseControlLite 推論失敗（returncode={completed.returncode}）。")
    finally:
        # 還原 config_inference.py
        try:
            cfg_py.write_text(cfg_backup, encoding="utf-8")
        except Exception as e:
            print(f"[MuseControlLite][警告] 還原 {cfg_py.name} 失敗：{e}")

        # 還原 default.yaml
        try:
            if default_yaml_backup.exists():
                shutil.move(str(default_yaml_backup), str(base_yaml))
                print(f"[mc-lite] default.yaml restored ← {base_yaml}")
        except Exception as e:
            print(f"[MuseControlLite][警告] 還原 default.yaml 失敗：{e}")

        # 還原 melody_condition_audio
        try:
            if mel_dir_created and mel_dir.exists():
                shutil.rmtree(mel_dir, ignore_errors=True)
            if mel_dir_backup.exists():
                mel_dir_backup.rename(mel_dir)
                print("[mc-lite] melody_condition_audio restored.")
        except Exception as e:
            print(f"[MuseControlLite][警告] 還原 melody_condition_audio 失敗：{e}")

    # ---- 取出新音檔 ----
    def _pick_latest(paths: List[Path]) -> Optional[Path]:
        return max(paths, key=lambda p: p.stat().st_mtime) if paths else None
    new_from_save = [p for p in save_dir.rglob("*") if p.suffix.lower() in (".wav",".flac",".mp3",".ogg")]
    picked = _pick_latest(new_from_save)
    if picked is None:
        after = _scan_audio(repo)
        created = sorted(list(after - before), key=lambda p: p.stat().st_mtime, reverse=True)
        picked = created[0] if created else None
    if picked is None:
        raise FileNotFoundError("找不到 MuseControlLite 生成的音檔；請檢查日誌。")

    # ---- 複製到 out_path ----
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(picked, out_path)
    try:
        sr_out = sf.info(str(out_path)).samplerate or sample_rate
    except Exception:
        sr_out = sample_rate
    return out_path, int(sr_out)

# ----------------------
# Backend: MusicGen_style(Transformers)
# ----------------------
def gen_with_musicgen_style(
    prompt: str,
    out_path: Path,
    *,
    style_path: str | Path,      # 風格參考音檔（建議 1.5–4.5 秒）
    duration: int = 8,           # 生成長度（秒）
    device: str = "cuda",
    cfg_coef: float = 3.0,       # 推 style conditioning 力道（官方建議起手式）
    cfg_coef_2: float = 5.0,     # 推 text conditioning 力道
) -> Tuple[Path, int]:
    """
    MusicGen-Style（AudioCraft）推論：文字 + 短風格音訊 → 音樂
    """
    import torch, torchaudio, numpy as np, soundfile as sf
    from audiocraft.models import MusicGen

    use_cuda = torch.cuda.is_available() and device == "cuda"
    dev = "cuda" if use_cuda else "cpu"

    # 1) 載模型（style）
    model = MusicGen.get_pretrained("style", device=dev)
    # 生成參數（官方模型卡建議 cfg 係數）
    model.set_generation_params(
        duration=int(duration),
        use_sampling=True, top_k=250, top_p=0.95, temperature=1.0,
        cfg_coef=float(cfg_coef)
    )

    # 2) 載入風格音檔
    wav, sr = torchaudio.load(str(style_path))   # [C, T]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # 3) 生成（AudioCraft 對 style 也是用 generate_with_chroma 這個入口）
    out_list = model.generate_with_chroma([prompt], wav, sr)  # list of [C, T] tensors
    one = out_list[0].cpu().numpy()  # (C, T)
    if one.ndim == 2:
        one = one.T  # (T, C)

    # 4) 寫檔
    sr_out = model.sample_rate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.clip(one, -1, 1).astype(np.float32), sr_out)
    return out_path, sr_out

# ----------------------
# Backend: Jasco
# ----------------------
from pathlib import Path
from typing import Tuple, Optional

def gen_with_jasco(
    mix_wav_path: str | Path,
    prompt: str,
    out_path: str | Path,
    *,
    device: str = "cuda",
) -> Tuple[Path, int]:
    """
    讀 full-mix .wav，自動估和弦/鼓/旋律，呼叫 JASCO 生成 10s。
    若拿不到 melody 版 checkpoint（或沒偵測到旋律），自動退回 chords-only 版。
    修正點：可靠取得 chord_to_index_mapping.pkl。
    """
    import os, warnings, torch, librosa
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    from audiocraft.models import JASCO

    # ---------------------- helpers ----------------------
    def _find_chords_map() -> str:
        """環境變數 -> 打包資產 -> HF 下載 三段式尋找"""
        p = os.environ.get("JASCO_CHORDS_MAP")
        if p and os.path.isfile(p):
            return p

        # ① 打包資產：優先找 audiocraft.assets
        try:
            from importlib.resources import files as ires_files
            for pkg in ("audiocraft.assets", "audiocraft"):
                try:
                    f = ires_files(pkg).joinpath("chord_to_index_mapping.pkl")
                    if f.is_file():
                        return str(f)
                except Exception:
                    pass
        except Exception:
            pass

        # ② HF 下載當作備援
        try:
            from huggingface_hub import hf_hub_download
            # 這個檔案隨 JASCO 模型卡提供；任選一個 repo 下載
            return hf_hub_download(
                repo_id="facebook/jasco-chords-drums-400M",
                filename="chord_to_index_mapping.pkl",
                local_dir=os.path.expanduser("~/.cache/jasco"),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise RuntimeError(
                "找不到 chord_to_index_mapping.pkl，請設定環境變數 JASCO_CHORDS_MAP 指向該檔案。"
            ) from e

    def _estimate_chords_major_only(wav: np.ndarray, sr: int,
                                    hop_sec: float = 2.0, seg_sec: float = 10.0):
        """每 2 秒輸出 1 個大三和弦（簡易/穩定）。"""
        harm, _ = librosa.effects.hpss(wav)
        hop = max(1, int(hop_sec * sr))
        chroma = librosa.feature.chroma_cqt(y=harm, sr=sr, hop_length=hop, n_chroma=12)
        majors = []
        for r in range(12):
            t = np.zeros(12, dtype=np.float32)
            t[r % 12] = 1.0; t[(r + 4) % 12] = 0.9; t[(r + 7) % 12] = 0.9
            t /= (np.linalg.norm(t) + 1e-12)
            majors.append(t)
        majors = np.stack(majors, 0)
        names = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]

        out = []
        for i in range(chroma.shape[1]):
            t0 = float(i * hop_sec)
            if t0 >= seg_sec:
                break
            v = chroma[:, i]
            if np.allclose(v, 0):
                continue
            v = v / (np.linalg.norm(v) + 1e-12)
            root = int(np.argmax(majors @ v))
            out.append((names[root], t0))
        if not out:
            out = [("C", 0.0)]
        if out[0][1] > 0.0:
            out = [(out[0][0], 0.0)] + out
        dedup = [out[0]]
        for (name, t0) in out[1:]:
            if name != dedup[-1][0]:
                dedup.append((name, t0))
        return dedup

    def _melody_salience_from_audio(wav: np.ndarray, sr: int,
                                    bins: int = 53, fps: float = 50.0, seg_sec: float = 10.0):
        """pyin -> midi -> [bins, T] one-hot；失敗回 None。"""
        hop = max(1, int(round(sr / fps)))
        need = int(seg_sec * sr)
        if wav.shape[-1] < need:
            wav = np.pad(wav, (0, need - wav.shape[-1]))
        else:
            wav = wav[:need]
        try:
            f0, _, _ = librosa.pyin(
                wav,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("E6"),
                sr=sr, hop_length=hop, center=True,
            )
        except Exception:
            return None
        if f0 is None:
            return None

        midi = librosa.hz_to_midi(f0)      # [T]
        valid = np.isfinite(midi)
        if not np.any(valid):
            return None
        T = midi.shape[0]
        sal = np.zeros((bins, T), dtype=np.float32)
        min_midi, max_midi = 36, 36 + bins - 1
        midi_clip = np.clip(midi[valid], min_midi, max_midi)
        idxs = np.round(midi_clip - min_midi).astype(np.int32)   # 0..bins-1
        t_idx = np.nonzero(valid)[0]
        sal[idxs, t_idx] = 1.0
        return torch.from_numpy(sal)

    # ---------------------- I/O 與前處理 ----------------------
    SR = 32000
    SEG_SEC = 60.0
    y, _ = librosa.load(str(mix_wav_path), sr=SR, mono=True)
    need = int(SEG_SEC * SR)
    y = (np.pad(y, (0, max(0, need - y.shape[-1])))[:need]).astype(np.float32)

    chords = _estimate_chords_major_only(y, SR, hop_sec=2.0, seg_sec=SEG_SEC)
    melody_sal = _melody_salience_from_audio(y, SR, bins=53, fps=50.0, seg_sec=SEG_SEC)

    # ---------------------- 載模型 ----------------------
    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    chords_map = _find_chords_map()

    model_id = "facebook/jasco-chords-drums-melody-400M" if melody_sal is not None else "facebook/jasco-chords-drums-400M"
    try:
        model = JASCO.get_pretrained(model_id, device=dev, chords_mapping_path=chords_map)
    except AssertionError:
        # 若 melody 版 gated 或資產缺失，退回 chords 版
        print("[JASCO] melody checkpoint unavailable, fallback to chords-only.")
        model_id = "facebook/jasco-chords-drums-400M"
        model = JASCO.get_pretrained(model_id, device=dev, chords_mapping_path=chords_map)

    # 參數：文字引導為 0、整體 CFG 適中
    model.set_generation_params(cfg_coef_all=5.0, cfg_coef_txt=0.0)

    # ---------------------- 生成 ----------------------
    with torch.no_grad():
        kwargs = dict(descriptions=[prompt], chords=chords, progress=True)
        if (model_id.endswith("melody-400M") and melody_sal is not None):
            kwargs.update(dict(
                melody_salience_matrix=melody_sal,  # [53, T]
                segment_duration=SEG_SEC,
                frame_rate=50.0,
                melody_bins=53,
            ))
        out = model.generate_music(**kwargs)

    wav_out = out.cpu().squeeze(0).numpy().T  # (T, C)
    wav_out = np.clip(wav_out, -1.0, 1.0).astype(np.float32)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav_out, samplerate=model.sample_rate)
    return out_path, model.sample_rate

# ----------------------
# Backend: cocomulla
# ----------------------
def gen_with_cocomulla(prompt: str, out_path: Path, **kw) -> Tuple[Path, int]:
    raise NotImplementedError("Coco-mulla 後端尚未接；可用 gen_with_musecontrol 的方式包一層子行程。")

BACKENDS = {
    "musicgen_melody": gen_with_musicgen_melody,
    "musecontrol": gen_with_musecontrol,
    "musicgen_style": gen_with_musicgen_style,
    "jasco": gen_with_jasco,
    "coco_mulla": gen_with_cocomulla,
}

# ----------------------
# Main
# ----------------------
def parse_kv_list(kvs: List[str]) -> Dict[str, Any]:
    """
    將 --model_kv key=value 支援成多次出現的參數。
    例如：--model_kv duration=12 --model_kv steps=40
    會得到 {"duration":12, "steps":40}
    """
    out = {}
    for kv in kvs:
        if "=" not in kv: continue
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if re.fullmatch(r"-?\d+", v):
            v = int(v)
        elif re.fullmatch(r"-?\d+\.\d*", v):
            v = float(v)
        elif v.lower() in ("true","false"):
            v = (v.lower()=="true")
        out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser("Batch Text-to-Music (multi-backend)")
    ap.add_argument("--json_in", required=True, type=str, help="JSON 檔（含 prompt 欄位）")
    ap.add_argument("--backend", required=True, choices=list(BACKENDS.keys()))
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)

    # 通用
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 筆（0=全部）")

    # MusicGen 參數
    ap.add_argument("--model_id", type=str, default="facebook/musicgen-melody")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=250)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--dtype", type=str, default=None, help="例如 float16 / bfloat16 / float32")

    # MuseControlLite 參數
    ap.add_argument("--muse_repo", type=str, default=os.environ.get("MUSECTRL_REPO", ""))
    ap.add_argument("--python_exe", type=str, default=None)
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--mc_arg", action="append", default=[], help="額外傳給 MuseControl 的原始參數，重複使用，如 --mc_arg --cfg=xxx.yaml")

    # 任意補參數（全部會塞進對應 backend 的 **kwargs）
    ap.add_argument("--model_kv", action="append", default=[], help="任意 key=value，重複使用")

    args = ap.parse_args()
    _seed_everything(args.seed)

    items = _read_json_items(Path(args.json_in))
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    out_dir = Path(args.out_dir); _ensure_dir(out_dir)
    rows: List[Tuple[str, str, str]] = []

    # 挑後端
    runner = BACKENDS[args.backend]
    extra = parse_kv_list(args.model_kv)

    # 針對後端整理 kwargs
    if args.backend == "musicgen_melody":
        base_kw = dict(
            max_new_tokens=args.max_new_tokens,
            device=args.device
        )
    elif args.backend == "musecontrol":
        repo = Path(args.muse_repo).expanduser().resolve() if args.muse_repo else None
        cli_abs = str(repo / "MuseControlLite_inference.py") if repo else None
        base_kw = dict(
            muse_repo=repo,                 # ★ 傳進去，供函式找 config/default.yaml
            python_exe=args.python_exe,
            seconds=float(args.duration),   # ★ 用 seconds，而不是 duration（函式優先讀 seconds）
            steps=args.steps,
            seed=args.seed,
            cli_abs=cli_abs,
            mc_arg=args.mc_arg,
            device=args.device,
            # sample_rate / cond_fps 有需要再從 args 補
        )

    elif args.backend == "musicgen_style":
        base_kw = dict(duration=60, cfg_coef=3.0, cfg_coef_2=5.0)
    elif args.backend == "jasco":
        base_kw = dict(device=args.device)
    else:
        base_kw = {}

    base_kw.update(extra)

    # 開跑
    for i, it in enumerate(items, 1):
        tgt = str(it.get("target", ""))
        prompt = str(it.get("prompt", ""))
        if not prompt:
            print(f"[skip] #{i} no prompt")
            continue

        # 以原 target 檔名 + 模型名 做輸出檔名
        stem = _slug(Path(tgt).stem) if tgt else f"idx{i:04d}"
        out_path = out_dir / f"{stem}.{args.backend}{AUDIO_EXT}"

        try:
            if args.backend == "musicgen_melody" :
                gen_path, sr = runner(prompt, out_path, melody_path = tgt, **base_kw)
            elif args.backend == "musicgen_style":
                gen_path, sr = runner(prompt, out_path, style_path=tgt, **base_kw)
            elif args.backend == "jasco":
                mix_wav_path=tgt
                gen_path, sr = runner(mix_wav_path, prompt, out_path, **base_kw)
            elif args.backend == "musecontrol":
                call_kw = base_kw.copy()
                call_kw["target"] = tgt  # ★ 只給 target，其餘由函式自動萃取
                # 保險：避免 extra/model_kv 夾帶干擾 MuseControl 的鍵
                for k in ("melody_path", "style_path", "audio_files", "prompt_texts", "prompts", "cfg"):
                    call_kw.pop(k, None)
                gen_path, sr = runner(prompt, out_path, **call_kw)

            else:
                gen_path, sr = runner(prompt, out_path, **base_kw)
            print(f"[{i}/{len(items)}] ok → {gen_path.name} (sr={sr})")
            rows.append((tgt, str(gen_path), prompt))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[{i}/{len(items)}] FAIL {stem}\n----- TRACEBACK -----\n{tb}\n---------------------")

    _write_csv(rows, Path(args.out_csv))
    print(f"[DONE] CSV → {args.out_csv}  | audios → {out_dir}")

if __name__ == "__main__":
    main()

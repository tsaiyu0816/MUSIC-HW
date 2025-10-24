"""
Caption target music (audio -> text prompt) with selectable ALM.

implemented :
- clap  : LAION-CLAP based tagger (genre/mood/instrument + tempo)
- audioflamingo3
- Qwen
- lp-musiccaps

Outputs:
  - JSON: [{"target": "<abs path>", "prompt": "<caption>", "alm": "<backend>"}...]
  - CSV : target,prompt,alm
"""
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
import csv
import numpy as np
import librosa
import soundfile as sf
from utils import *
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

# ------------------------------
# Backend: CLAP tagger (ready-to-run)
# ------------------------------
CLAP_CANDIDATES = {
    "genre": [
        "pop", "rock", "hip hop", "jazz", "funk", "blues", "latin", "reggaeton",
        "country", "electronic", "techno", "house", "trance", "ambient", "classical",
        "lofi", "metal", "r&b", "soul"
    ],
    "mood": [
        "happy", "sad", "energetic", "calm", "romantic", "dark", "epic",
        "groovy", "uplifting", "melancholic", "aggressive", "relaxed", "mysterious"
    ],
    "instrument": [
        "piano", "electric guitar", "acoustic guitar", "bass guitar", "drums",
        "strings", "synth", "brass", "saxophone", "violin", "flute", "vocals",
        "female vocals", "male vocals", "choir"
    ],
}

def _clap_load(device="cpu"):
    try:
        import laion_clap
    except Exception as e:
        raise SystemExit(
            "[CLAP] pip install laion-clap 失敗或未安裝；請先安裝再重試。\n"
            f"原始錯誤: {e}"
        )
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    model.load_ckpt()
    model.eval()
    return model

def _clap_embed_audio(model, wav: np.ndarray) -> np.ndarray:
    emb = model.get_audio_embedding_from_data(x=wav.reshape(1, -1), use_tensor=False)  # (1,512)
    v = np.squeeze(emb).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

def _clap_embed_text(model, texts: List[str]) -> np.ndarray:
    emb = model.get_text_embedding(texts, use_tensor=False)  # (N,512)
    V = emb.astype(np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V

def _topk_labels(model, audio_vec: np.ndarray, group: str, k: int) -> List[str]:
    labels = CLAP_CANDIDATES[group]
    T = _clap_embed_text(model, labels)
    sims = T @ audio_vec  # (N,)
    idx = np.argsort(-sims)[:k]
    return [labels[i] for i in idx]

def _estimate_bpm(y: np.ndarray, sr: int) -> int:
    try:
        tempo = librosa.beat.tempo(y=y, sr=sr, aggregate="median")
        return int(round(float(tempo)))
    except Exception:
        return 0

def caption_with_clap(path: Path, model, sr: int, topk_genre=1, topk_mood=2, topk_inst=2) -> str:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    avec = _clap_embed_audio(model, y)
    genres = _topk_labels(model, avec, "genre", topk_genre)
    moods  = _topk_labels(model, avec, "mood",  topk_mood)
    insts  = _topk_labels(model, avec, "instrument", topk_inst)
    bpm = _estimate_bpm(y, sr)
    parts = []
    if genres: parts.append(f"{', '.join(genres)} track")
    if moods:  parts.append(f"{', '.join(moods)} mood")
    if insts:  parts.append(f"featuring {', '.join(insts)}")
    if bpm > 0: parts.append(f"around {bpm} BPM")
    parts.append("high quality, coherent arrangement")
    return ", ".join(parts) + "."

# ------------------------------
# audioflamingo3
# ------------------------------
import os, sys, subprocess
from pathlib import Path
def caption_with_audioflamingo3(path: Path,
                                sr: int,
                                *,
                                af3_root: Path | None = None,
                                model_base: str = "nvidia/audio-flamingo-3",
                                conv_mode: str = "auto",
                                question: str = "請詳細描述這段音樂的風格、樂器、節奏、拍號、速度與情緒。",
                                gpu: str = "0",
                                use_4bit: bool = True,
                                device_map: str = "auto",
                                offline: bool = False,
                                python_exe: str | None = None) -> str:
    # 1) 解析路徑
    if af3_root is None:
        af3_root = Path(__file__).resolve().parents[1]
    infer_py = af3_root/ "audio-flamingo" / "llava" / "cli" / "infer_audio.py"
    if not infer_py.exists():
        raise FileNotFoundError(f"找不到 AF3 infer 腳本: {infer_py}")

   
    cmd = [
        python_exe or sys.executable,  
        str(infer_py),
        "--model-base", model_base,
        "--conv-mode", conv_mode,
        "--text", question,
        "--media", str(path),
    ]
    if use_4bit:
        cmd += ["--load-4bit", "--device-map", device_map]

    # 3) 環境變數（指定 GPU、可選離線）
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    # 4) 執行（如 4bit 參數不被支援，自動去掉重跑）
    def _run(_cmd):
        return subprocess.run(_cmd, capture_output=True, text=True, env=env)

    result = _run(cmd)
    if result.returncode != 0 and "unrecognized arguments" in (result.stderr or "") and use_4bit:
        # 去掉 4bit 後重試一次
        cmd2 = [c for c in cmd if c not in ("--load-4bit", "--device-map", device_map)]
        result = _run(cmd2)

    if result.returncode != 0:
        raise RuntimeError(
            f"[AF3] 命令失敗（code={result.returncode})\nCMD: {' '.join(cmd)}\nSTDERR:\n{result.stderr}"
        )

    # 5) 取最後一行非空輸出當 caption
    lines = (result.stdout or "").strip().splitlines()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return ""

# ------------------------------
# Qwen
# ------------------------------
import inspect
def _pick_loud_window(y: np.ndarray, sr: int, target_sec: int = 20) -> np.ndarray:
    """從整段音訊中擷取能量最高的 target_sec 片段，讓模型專注代表性內容。"""
    win = target_sec * sr
    if y.shape[0] <= win:
        return y
    hop = sr // 2  # 每 0.5s 滑動一次
    best_s, best_sum = 0, -1.0
    s = 0
    while s + win <= y.shape[0]:
        seg = y[s:s+win]
        val = float(np.sum(np.abs(seg)))
        if val > best_sum:
            best_sum, best_s = val, s
        s += hop
    return y[best_s:best_s+win]

def _build_prompts(lang: str = "zh", style: str = "paragraph", bullets: int = 8):
    system = (
            "You are a music captioning assistant. 以你『實際聽到的聲音』為依據，"
            "給出具體、細節豐富、貼近音訊的描述。可以合理估計（如「約」「大約」「可能」），"
            "盡量使用可觀察到的線索（音色、節奏、律動、技巧、空間感、段落變化等）。"
        )
    # 1) 段落模式（最適合「盡可能完整描述」）
    if style == "paragraph":
        if lang == "zh":
            user = (
                "請用中文完整描述這段音樂，重點包含：曲風與可能的影響、主旋律/和聲與聲部分工、主要與次要樂器與演奏技巧、"
                "節奏/拍號與律動特徵、速度（可寫約略 BPM 或範圍）、段落/結構變化與過門、情緒與可能場景、"
                "混音/空間感/動態與音色。可寫成 1–2 個段落，必要時舉出關鍵聲響或時間點作為例子。最後加上一句總結。"
            )
        else:
            user = (
                "Write a complete description in English, covering: genre & possible influences, "
                "melody/harmony & part roles, primary/secondary instruments and playing techniques, "
                "rhythm/meter & groove traits, tempo (approx BPM or a range), section/structure changes and fills, "
                "mood & plausible use cases, mixing/spatial/dynamics/timbre notes. Use 1–2 paragraphs; "
                "cite notable sounds or timestamps if helpful. End with a one-sentence summary."
            )
        return system, user

    # 2) 條列模式（結構化但不嚴格；每點允許 1–2 句）
    if style == "bullets":
        if lang == "zh":
            user = (
                f"請用 {bullets} 點條列（每點 1–2 句）完整描述："
                "曲風/影響、主旋律/和聲與聲部分工、主要/次要樂器與技巧、節奏/拍號與律動、速度（約略 BPM 或範圍）、"
                "段落/結構變化與過門、情緒/場景、混音/空間/動態與音色、值得注意的聲響或時間點；最後加一行總結。"
            )
        else:
            user = (
                f"Provide {bullets} bullet points (1–2 sentences each) to fully describe: "
                "genre/influences, melody/harmony & part roles, primary/secondary instruments & techniques, "
                "rhythm/meter & groove, tempo (approx BPM or range), section/structure changes & fills, "
                "mood/scene, mixing/spatial/dynamics/timbre, notable sounds or timestamps; then a one-line summary."
            )
        return system, user
    # 預設回 paragraph
    return _build_prompts(lang=lang, style="paragraph", bullets=bullets)

def load_qwen_model(
    model_id: str = "Qwen/Qwen2-Audio-7B-Instruct",
    gpu_mem_gib: int = 6,
    offload_folder: str = "./offload_qwen2_audio",
    attn_impl: str = "sdpa",          # 或 "flash_attention_2"（若你的環境有裝 flash-attn）
    four_bit: bool = True
):
    import os, torch
    from transformers import Qwen2AudioProcessor, Qwen2AudioForConditionalGeneration
    from transformers import BitsAndBytesConfig

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

    use_cuda = torch.cuda.is_available()
    cc_major = torch.cuda.get_device_capability()[0] if use_cuda else 0
    compute_dtype = torch.bfloat16 if (use_cuda and cc_major >= 8) else torch.float16

    processor = Qwen2AudioProcessor.from_pretrained(model_id)

    max_mem = ({0: f"{gpu_mem_gib}GiB", "cpu": "48GiB"} if use_cuda else {"cpu": "64GiB"})

    kwargs = dict(
        device_map="auto",
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        max_memory=max_mem,
        offload_folder=offload_folder,
    )

    if four_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = compute_dtype

    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kwargs)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            # 退 CPU-only（會慢，但一定跑得動）
            kwargs["device_map"] = {"": "cpu"}
            kwargs.pop("max_memory", None)
            model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kwargs)
        else:
            raise

    # 簡單列印（可關）
    try:
        dev = next(model.parameters()).device
        print(f"[Qwen] loaded on {dev}, 4bit={four_bit}, attn={attn_impl}, max_mem={max_mem}")
    except Exception:
        pass

    return processor, model

def caption_with_qwen_one(
    path: Path,
    processor,
    model,
    *,
    lang: str = "zh",
    style: str = "bullets",      # "bullets" | "paragraph" | "json"
    bullets: int = 8,
    max_seconds: int = 30,       # 建議 12~20 秒，較聚焦且省顯存
    max_new_tokens: int = 120,
    min_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.90,
):
    """
    對單首音檔做音訊→文字 Caption。
    - 取能量最高片段（max_seconds）
    - 動態支援 processor.__call__(audios=...) / (audio=...)；若都沒有則 fallback 手動合併
    - 可選輸出風格：條列/段落/JSON（JSON 模式會嘗試校驗與正規化）
    """
    import torch, librosa

    # 1) 載音訊並取代表性片段
    sr_model = processor.feature_extractor.sampling_rate
    y, _ = librosa.load(str(path), sr=sr_model, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    y = _pick_loud_window(y, sr_model, target_sec=max_seconds)

    # 2) 構建 ChatML 對話
    system_text, user_text = _build_prompts(lang=lang, style=style, bullets=bullets)
    convo = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": str(path)},   # 真正波形透過 processor 傳
            {"type": "text",  "text": user_text},
        ]},
    ]
    text = processor.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)

    # 3) 構建 inputs（穩健處理 audios/audio 差異；必要時手動合併）
    try:
        call_params = set(inspect.signature(processor.__call__).parameters)
    except (ValueError, TypeError, AttributeError):
        call_params = set()
    if "audios" in call_params:
        inputs = processor(text=text, audios=y, sampling_rate=sr_model, return_tensors="pt", padding=True)
    elif "audio" in call_params:
        inputs = processor(text=text, audio=y, sampling_rate=sr_model, return_tensors="pt", padding=True)
    else:
        a = processor.feature_extractor(y, sampling_rate=sr_model, return_tensors="pt")
        t = processor.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {**t, **a}
        # 名稱對齊：有些版本用 audio_values，有些是 input_features
        if "audio_values" in inputs and "input_features" not in inputs:
            inputs["input_features"] = inputs.pop("audio_values")

    # 4) 把整包 inputs 搬到 model 的裝置（避免 CPU/CUDA 混用）
    device = next(model.parameters()).device
    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(device)

    # 5) 生成（拉長 + 去重）
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,   # 盡量寫到這長度；遇到 EOS 仍可能提前結束
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
        use_cache=True,
    )

    # 6) 解碼（只取新產生的部分）
    ctx_len = inputs["input_ids"].size(1)
    gen_only = gen_ids[:, ctx_len:]
    text_out = processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    # 7) JSON 模式：嘗試解析與簡單正規化
    if style == "json":
        try:
            obj = json.loads(text_out)
            # tempo 正規化
            if "tempo_bpm_range" in obj and isinstance(obj["tempo_bpm_range"], list) and len(obj["tempo_bpm_range"]) == 2:
                low, high = obj["tempo_bpm_range"]
                def _to_num(x):
                    if isinstance(x, (int, float)): return x
                    if isinstance(x, str):
                        x = x.replace(",", "").strip()
                        return float(x) if x else None
                    return None
                obj["tempo_bpm_range"] = [_to_num(low), _to_num(high)]
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return text_out

    return text_out

# ------------------------------
# LP-MusicCaps (audio -> caption)
# ------------------------------
def caption_with_lpmusiccaps(path: Path, sr: int) -> str:
    import os, sys
    import numpy as np
    import torch, librosa

    repo_dir = os.getenv("LPMC_REPO")
    ckpt_path = os.getenv("LPMC_CKPT")

    if not repo_dir or not os.path.isdir(repo_dir):
        raise SystemExit("請先設定 LPMC_REPO 指向 clone 下來的 lp-music-caps 目錄。")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise SystemExit("請先設定 LPMC_CKPT 指向 transfer.pth 權重。")

    # 讓 Python 能找到本地模組
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # 嘗試多個常見匯入路徑（repo / HF Space 結構）
    BartCaptionModel = None
    try:
        from lpmc.music_captioning.model.bart import BartCaptionModel as _B
        BartCaptionModel = _B
    except Exception:
        try:
            from model.bart import BartCaptionModel as _B
            BartCaptionModel = _B
        except Exception:
            try:
                from music_captioning.model.bart import BartCaptionModel as _B
                BartCaptionModel = _B
            except Exception as e:
                raise SystemExit(
                    "找不到 BartCaptionModel。請確認 LPMC_REPO 目錄裡包含 "
                    "`lpmc/music_captioning/model/bart.py` 或 `model/bart.py`。"
                ) from e

    # 懶載入（避免每首歌都重載模型）
    global _LPMC_MODEL, _LPMC_DEVICE, _LPMC_SR, _LPMC_CHUNK
    try:
        _LPMC_MODEL
    except NameError:
        _LPMC_MODEL = None

    if _LPMC_MODEL is None:
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        model = BartCaptionModel(max_length=128)
        model.load_state_dict(state, strict=False)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

            # === 這一段是關鍵補丁：覆蓋舊版的 generate ===
        import inspect
        from transformers.modeling_outputs import BaseModelOutput, ModelOutput
        import torch as _torch

        def _find_audio_encoder(m):
            # 嘗試幾個常見名字：依你的 repo 如有不同可改
            for name in ("encode_audio", "forward_encoder", "encode"):
                fn = getattr(m, name, None)
                if callable(fn):
                    return fn
            raise RuntimeError("找不到音訊 encoder 方法（encode_audio/forward_encoder/encode）。")

        _audio_encode = _find_audio_encoder(model)

        def _to_encoder_outputs(x):
            # 統一包成 BaseModelOutput，讓 HF 的 generate 能安全展開
            if isinstance(x, (BaseModelOutput, ModelOutput)):
                return x
            if isinstance(x, dict):
                if "last_hidden_state" in x:
                    return BaseModelOutput(**x)
                if "encoder_last_hidden_state" in x:
                    return BaseModelOutput(last_hidden_state=x["encoder_last_hidden_state"])
            if isinstance(x, (list, tuple)):
                return BaseModelOutput(last_hidden_state=x[0])
            if _torch.is_tensor(x):
                return BaseModelOutput(last_hidden_state=x)
            raise TypeError(f"不支援的 encoder_outputs 型別：{type(x)}")

        def _patched_generate(self, samples, num_beams=5, max_length=128):
            
            dev = next(self.parameters()).device

            # 1) 取 encoder 輸出並包裝
            enc_raw = _audio_encode(samples)
            enc_out = _to_encoder_outputs(enc_raw)

            # 2) 準備 decoder 起始 token
            bos_id = self.bart.config.decoder_start_token_id or self.bart.config.bos_token_id \
                    or getattr(self.bart.generation_config, "decoder_start_token_id", None) \
                    or getattr(self.bart.generation_config, "bos_token_id", None)
            if bos_id is None:
                raise RuntimeError("BART config 缺少 decoder_start_token_id / bos_token_id。")

            input_ids = _torch.full((samples.size(0), 1), bos_id, dtype=_torch.long, device=dev)

            # 3) 生成（新版 HF 會自動處理 beam 展開）
            seq = self.bart.generate(
                input_ids=input_ids,
                encoder_outputs=enc_out,
                num_beams=int(num_beams),
                max_length=int(max_length),
                use_cache=True,
            )
            # 4) 解碼成文字（維持原函式回傳 list[str]）
            texts = self.tokenizer.batch_decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return [t.strip() for t in texts]

        # 綁到模型實例（覆蓋原本的 generate）
        model.generate = _patched_generate.__get__(model, type(model))
        # === 補丁結束 ===


        _LPMC_MODEL = model
        _LPMC_DEVICE = device
        _LPMC_SR = 16000
        _LPMC_CHUNK = 10 * _LPMC_SR  # 10s
        print(f"[LPMC] loaded on {_LPMC_DEVICE} | ckpt={ckpt_path}")

    # 讀音檔 → 16k/mono → 10s 分段
    y, _ = librosa.load(str(path), sr=_LPMC_SR, mono=True)
    if y.ndim > 1:
        y = y.mean(-1)
    n = y.shape[-1]
    if n <= _LPMC_CHUNK:
        buf = np.zeros(_LPMC_CHUNK, dtype=np.float32)
        buf[:n] = y.astype(np.float32)
        chunks = buf[None, :]
    else:
        full = (n // _LPMC_CHUNK) * _LPMC_CHUNK
        chunks = np.stack(np.split(y[:full].astype(np.float32), full // _LPMC_CHUNK), axis=0)

    audio = torch.from_numpy(chunks).to(_LPMC_DEVICE)

    # 逐段生成
    with torch.no_grad():
        seg_texts = _LPMC_MODEL.generate(samples=audio, num_beams=5)

    # 串回一段文字（附時碼）
    out = []
    for i, t in enumerate(seg_texts):
        st, ed = i * 10, (i + 1) * 10
        out.append(f"[{st:02d}:00-{ed:02d}:00] {t}")
    return " ".join(out).strip()

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser("Target music -> text caption (ALM selectable)")
    ap.add_argument("--input_dir", required=True, type=str)
    ap.add_argument("--alm", required=True, choices=["clap", "audioflamingo3", "qwen", "lp-musiccaps"])
    ap.add_argument("--out_json", required=True, type=str)
    ap.add_argument("--out_csv",  required=True, type=str)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--device", type=str, default="cuda")
    # CLAP-specific options
    ap.add_argument("--genre_k", type=int, default=1)
    ap.add_argument("--mood_k",  type=int, default=2)
    ap.add_argument("--inst_k",  type=int, default=2)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    files = [p for p in list_audio(in_dir)]
    if not files:
        raise SystemExit(f"No audio files in {in_dir}")

    items: List[Dict[str, Any]] = []

    if args.alm == "clap":
        model = _clap_load(device=args.device)
        for i, p in enumerate(files, 1):
            try:
                text = caption_with_clap(p, model, sr=args.sr,
                                         topk_genre=args.genre_k, topk_mood=args.mood_k, topk_inst=args.inst_k)
                items.append({"target": str(p), "prompt": text, "alm": "clap"})
                print(f"[{i}/{len(files)}] {p.name} → {text}")
            except Exception as e:
                print(f"[skip] {p.name}: {e}")

    elif args.alm == "audioflamingo3":
        for i, p in enumerate(files, 1):
            try:
                text = caption_with_audioflamingo3(
                    p, sr=args.sr,
                    gpu="0",
                    use_4bit=True,         # 開 4bit
                    device_map="auto",    
                    offline=False          # 若模型都在快取可開；否則設 False
                )
                items.append({"target": str(p), "prompt": text, "alm": "audioflamingo3"})
                print(f"[{i}/{len(files)}] {p.name} → {text}")
            except Exception as e:
                print(f"[skip] {p.name}: {e}")

    elif args.alm == "qwen":  # 或你用的 'owen' 命名
        processor, model = load_qwen_model(
            model_id="Qwen/Qwen2-Audio-7B-Instruct",
            gpu_mem_gib=6,
            offload_folder="/home/tsmc-tsai/tmp/offload_qwen2_audio",  # 放本機 SSD
            attn_impl="sdpa",
            four_bit=True
        )
        for i, p in enumerate(files, 1):
            try:
                text = caption_with_qwen_one(
                    p, processor, model,
                    lang="zh", style="paragraph",
                    max_seconds=20, max_new_tokens=192, min_new_tokens=120
                )
                # 之後你可直接 json.loads(text) 抽欄位寫 CSV

                items.append({"target": str(p), "prompt": text, "alm": "qwen"})
                print(f"[{i}/{len(files)}] {p.name} → {text}")
            except Exception as e:
                print(f"[skip] {p.name}: {e}")

    elif args.alm == "lp-musiccaps":
        for i, p in enumerate(files, 1):
            text = caption_with_lpmusiccaps(p, sr=args.sr)
            items.append({"target": str(p), "prompt": text, "alm": "lp-musiccaps"})

    # save
    save_outputs(items, args.out_json, args.out_csv)
    print(f"[DONE] Saved prompts → {args.out_json} , {args.out_csv}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8
# generate.py — 只需 checkpoint(.pt) + dictionary.pkl，就能批次生成 N 首 MIDI 並自動渲染 WAV

import argparse, os, sys, pickle, random, shutil, subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Config

# ----- path shim for utils.py -----
HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent, HERE.parent.parent):
    if (cand / "utils.py").exists():
        sys.path.insert(0, str(cand))
        break
import utils  # noqa: E402


# ---------- helpers ----------
def render_midi_to_wav(midi_path: str, wav_path: str, sf2_path: str, sr: int = 22050) -> bool:
    """優先 fluidsynth CLI；不行再用 midi2audio.FluidSynth。成功回傳 True。"""
    if shutil.which("fluidsynth"):
        cmd = ["fluidsynth", "-ni", "-T", "wav", "-F", str(wav_path), "-r", str(sr), sf2_path, midi_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            pass
    try:
        from midi2audio import FluidSynth  # lazy import
        FluidSynth(sound_font=sf2_path, sample_rate=sr).midi_to_audio(midi_path, wav_path)
        return True
    except Exception:
        return False


def temp_topk_sample(logits_np, temperature=1.0, top_k=0) -> int:
    z = logits_np.astype(np.float64) / max(1e-8, float(temperature))
    if top_k and 0 < top_k < z.shape[0]:
        idx = np.argpartition(-z, top_k)[:top_k]
        vals = z[idx]
        p = np.exp(vals - vals.max()); p /= p.sum()
        return int(np.random.choice(idx, p=p))
    z -= z.max(); p = np.exp(z); p /= p.sum()
    return int(np.random.choice(len(z), p=p))


def load_dictionary(dict_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """相容 (e2w, w2e) 或 (w2e, e2w) 或單一 dict 格式。"""
    obj = pickle.load(open(dict_path, "rb"))
    if isinstance(obj, tuple) and len(obj) == 2:
        a, b = obj
        if isinstance(a, dict) and isinstance(b, dict):
            # 判別誰是 e2w：鍵是 str 的通常是 e2w
            if all(isinstance(k, str) for k in a.keys()):
                e2w, w2e = a, b
            else:
                e2w, w2e = b, a
            return e2w, w2e
    if isinstance(obj, dict):
        # 只有 e2w 時，合成 w2e
        if all(isinstance(k, str) for k in obj.keys()):
            e2w = obj
            w2e = {v: k for k, v in e2w.items()}
            return e2w, w2e
        # 只有 w2e 時，合成 e2w
        if all(isinstance(k, int) for k in obj.keys()):
            w2e = obj
            e2w = {v: k for k, v in w2e.items()}
            return e2w, w2e
    raise SystemExit(f"[generate] 不支援的 dictionary.pkl 格式：{dict_path}")


def _extract_state_dict(sd_raw):
    """
    相容多種保存格式：
    - {"model": state_dict, "config": {...}}
    - {"state_dict": state_dict, ...}
    - 直接就是 state_dict
    並去掉常見前綴：'module.'、'model.'、'gpt.'、'gpt2.'
    """
    # 1) 取出裡面的 state_dict
    sd = sd_raw
    if isinstance(sd_raw, dict):
        for k in ("model", "state_dict", "weights", "params"):
            if k in sd_raw and isinstance(sd_raw[k], dict):
                sd = sd_raw[k]
                break
    if not isinstance(sd, dict):
        raise SystemExit("[generate] checkpoint 不是可讀的 state_dict")

    # 2) 去前綴
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "gpt.", "gpt2."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v
    return cleaned

def build_model(vocab_size: int, ckpt_path: Path, device: str) -> GPT2LMHeadModel:
    sd_raw = torch.load(ckpt_path, map_location="cpu")

    # 嘗試讀出 config（若有）
    cfg_dict = {}
    if isinstance(sd_raw, dict):
        # 可能存在 "config" 或 "gpt_cfg" 之類的欄位
        for k in ("config", "gpt_cfg", "cfg"):
            if k in sd_raw and isinstance(sd_raw[k], dict):
                cfg_dict = dict(sd_raw[k])
                break

    # vocab_size 至少要正確
    cfg_dict.setdefault("vocab_size", vocab_size)
    cfg = GPT2Config(**cfg_dict)

    # 取乾淨的 state_dict
    state_dict = _extract_state_dict(sd_raw)

    model = GPT2LMHeadModel(cfg)
    # 多數情況嚴格載入就會成功；若你曾改動 head tying 等，也可改成 strict=False
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}（顯示前幾個） -> {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}（顯示前幾個） -> {unexpected[:5]}")
    model.to(device).eval()
    return model


def init_seq_from_prompt(e2w: Dict[str, int], prompt_midi: Optional[str]) -> List[int]:
    """若有 --prompt_midi，用它轉事件；否則用 Bar/Pos/Tempo 起手。"""
    if prompt_midi:
        note_items, tempo_items = utils.read_items(prompt_midi)
        note_items = utils.quantize_items(note_items)
        max_time = int(note_items[-1].end) if len(note_items) > 0 else 0
        groups = utils.group_items(tempo_items + note_items, max_time)
        events = utils.item2event(groups)
        keys = [f"{ev.name}_{ev.value}" for ev in events][:1024]
        seq = [e2w[k] for k in keys if k in e2w]
        if seq:
            return seq
    # fallback 起手
    seq = []
    if "Bar_None" in e2w:
        seq.append(e2w["Bar_None"])
    pos = e2w.get("Position_1/16", None)
    if pos is not None:
        seq.append(pos)
    tcls = [v for k, v in e2w.items() if k.startswith("Tempo Class_")]
    tval = [v for k, v in e2w.items() if k.startswith("Tempo Value_")]
    if tcls:
        seq.append(int(np.random.choice(tcls)))
    if tval:
        seq.append(int(np.random.choice(tval)))
    return seq


def generate_one(
    model: GPT2LMHeadModel,
    start_tokens: List[int],
    bars: int,
    e2w: Dict[str, int],
    temperature: float,
    top_k: int,
    device: str,
    max_factor: int = 2048,
) -> List[int]:
    """custom tokenizer：以 'Bar_None' 精準計數；同時設上限避免卡住。"""
    out_tokens = list(start_tokens)
    bars_done = 0
    max_steps = max(1024, bars * max_factor)

    with torch.inference_mode():
        for _ in range(max_steps):
            x = torch.tensor(out_tokens[-1024:], dtype=torch.long, device=device)[None, :]
            logits = model(x).logits[0, -1].detach().cpu().numpy()
            nxt = temp_topk_sample(logits, temperature=temperature, top_k=top_k)
            out_tokens.append(int(nxt))

            if nxt == e2w.get("Bar_None", -1):
                bars_done += 1
                if bars_done >= bars:
                    break
    return out_tokens


def save_midi(e2w: Dict[str, int], tokens: List[int], out_mid: Path):
    word2event = {i: k for k, i in e2w.items()}
    utils.write_midi(words=tokens, word2event=word2event, output_path=str(out_mid), prompt_path=None)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Batch MIDI/WAV generator from checkpoint + dictionary.pkl")
    ap.add_argument("--ckpt", required=True, help="模型 checkpoint 檔（.pt）")
    ap.add_argument("--dict_path", required=True, help="dictionary.pkl（e2w/w2e）")
    ap.add_argument("--out_dir", required=True, help="輸出資料夾")
    ap.add_argument("--num", type=int, default=20, help="生成幾首（預設 20）")
    ap.add_argument("--bars", type=int, default=32, help="目標 bars（依 Bar_None 計數）")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--prompt_midi", default="", help="以此 MIDI 當 prompt 起手（可選）")
    ap.add_argument("--sf2", type=str, default="./assets/sf2/GM.sf2", help="SoundFont 路徑（渲染 WAV 用）")
    ap.add_argument("--sr", type=int, default=22050, help="WAV 取樣率")
    ap.add_argument("--prefix", type=str, default="sample", help="輸出檔名前綴（預設 sample）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # RNG
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

    # dict & vocab
    e2w, w2e = load_dictionary(Path(args.dict_path))
    vocab_size = max(e2w.values()) + 1

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(vocab_size=vocab_size, ckpt_path=Path(args.ckpt), device=device)

    # prompt
    prompt_midi = args.prompt_midi if args.prompt_midi else None

    total_ok = 0
    for i in range(args.num):
        # 每首不同亂數
        seed_i = args.seed + i
        np.random.seed(seed_i); random.seed(seed_i); torch.manual_seed(seed_i)

        start_tokens = init_seq_from_prompt(e2w, prompt_midi)
        tokens = generate_one(
            model=model,
            start_tokens=start_tokens,
            bars=args.bars,
            e2w=e2w,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            max_factor=2048,
        )

        out_mid = out_dir / f"{args.prefix}_{i:04d}.mid"
        try:
            save_midi(e2w, tokens, out_mid)
            print(f"[gen] wrote {out_mid}")
        except Exception as e:
            print(f"[gen] 寫入 MIDI 失敗：{out_mid} — {e}")
            continue

        # render wav
        if os.path.isfile(args.sf2):
            out_wav = out_mid.with_suffix(".wav")
            ok = render_midi_to_wav(str(out_mid), str(out_wav), args.sf2, sr=args.sr)
            print(f"[gen] rendered {out_wav}" if ok else f"[gen] render failed → {out_wav}")
        else:
            print(f"[gen] skip render: sf2 not found → {args.sf2}")

        total_ok += 1

    print(f"[done] 成功輸出 {total_ok}/{args.num} 首到 {out_dir}")


if __name__ == "__main__":
    main()

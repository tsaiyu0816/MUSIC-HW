#!/usr/bin/env python
# Task2: Continuation generation (8 bars prompt -> +24 bars continuation)
# - 支援 tokenizer: custom | miditok_remi | miditok_remi_plus
# - 讀取 Task1 的 GPT-2 checkpoint（.pkl）
# - 對 3 個預設設定 (A/B/C) 逐首生成，產出 MIDI（與可選 WAV）
# 只依賴：utils.py（必需），miditok/symusic（選用，如果你用 miditok）

import os, glob, re, csv, argparse, pickle, warnings, shutil, subprocess, random
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from pathlib import Path
from tqdm import tqdm

import utils  

HAVE_MIDITOK = False
try:
    from miditok import TokenizerConfig, REMI
    try:
        from miditok import TokSequence
    except Exception:
        from miditok.utils import TokSequence
    from symusic import Score
    HAVE_MIDITOK = True
except Exception:
    HAVE_MIDITOK = False

X_LEN = 1024  # 和訓練一致

# ---------------- Argparse ----------------
def parse_opt():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', choices=['custom','miditok_remi','miditok_remi_plus'],
                   default='custom')
    p.add_argument('--dict_path', type=str, default='./dictionary.pkl',
                   help='custom 模式需要；若不存在可自動建立（掃 prompt 三首）')

    p.add_argument('--ckpt', type=str, required=True,
                   help='Task1 訓練出的 checkpoint（.pkl）')
    p.add_argument('--prompt_dir', type=str, required=True,
                   help='放 3 首 8 bars 的資料夾')
    p.add_argument('--out_dir', type=str, default='./results/task2_out')

    p.add_argument('--bars_prompt', type=int, default=8,
                   help='slides 指定 8 bars，保留可調')
    p.add_argument('--bars_cont', type=int, default=24,
                   help='需要接續的 bars 數')

    # 三組預設推論組合（可改）
    p.add_argument('--configs', type=str, default='A,B,C',
                   help='要跑哪些預設組合，逗號分隔（A/B/C）')

    # 其他推論細節
    p.add_argument('--with_chord', type=int, default=0,
                   help='custom/miditok 是否含和絃事件/開啟 chords')
    p.add_argument('--tempo_bpm_min', type=int, default=None,
                   help='miditok 範圍下限（避免太慢）')
    p.add_argument('--tempo_bpm_max', type=int, default=None,
                   help='miditok 範圍上限（避免太快）')

    # 渲染
    p.add_argument('--render_wav', type=int, default=1)
    p.add_argument('--sf2', type=str, default='./assets/sf2/GM.sf2')
    p.add_argument('--sr', type=int, default=22050)
    p.add_argument('--render_method', choices=['auto','fluidsynth','midi2audio'],
                   default='auto')

    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=3407)

    return p.parse_args()

# ---------------- Small utils ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_midis(d):
    pats = [os.path.join(d, ext) for ext in ('*.mid','*.MID','*.midi','*.MIDI')]
    out = []
    for p in pats: out += glob.glob(p)
    return sorted(out)

def render_midi_to_wav(midi_path: str, wav_path: str, sf2_path: str, sr: int = 22050, method: str = 'auto'):
    if method in ('auto','fluidsynth'):
        if shutil.which('fluidsynth'):
            cmd = ['fluidsynth','-ni','-T','wav','-F',wav_path,'-r',str(sr),sf2_path,midi_path]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            except subprocess.CalledProcessError:
                pass
    if method in ('auto','midi2audio'):
        try:
            from midi2audio import FluidSynth
            FluidSynth(sound_font=sf2_path, sample_rate=sr).midi_to_audio(midi_path, wav_path)
            return True
        except Exception:
            pass
    return False

def swap_ext(path, ext):
    r,_ = os.path.splitext(path)
    return r + ext

# ---------- Sampling helpers ----------
def softmax(z):
    z = z - z.max()
    p = np.exp(z)
    return p / np.clip(p.sum(), 1e-12, None)

def sample_with_constraints(
    logits, *, temperature=1.0, topk=0, topp=None,
    banned_ids=None, rep_penalty_ids=None, rep_penalty=1.0
):
    """單步取樣：支援 top-k、top-p、重複懲罰與 ban。"""
    z = logits.astype(np.float64) / max(1e-8, float(temperature))

    # 重複懲罰（把最近出現過的 token logit 拉低）
    if rep_penalty_ids and rep_penalty > 1.0:
        z[np.array(list(set(rep_penalty_ids)), dtype=np.int64)] /= rep_penalty

    # ban
    if banned_ids:
        z[np.array(list(set(banned_ids)), dtype=np.int64)] = -1e9

    # top-k
    if topk and 0 < topk < z.shape[0]:
        idx = np.argpartition(-z, topk)[:topk]
        z2 = np.full_like(z, -1e9)
        z2[idx] = z[idx]
        z = z2

    p = softmax(z)

    # top-p (nucleus)
    if topp is not None and 0.0 < topp < 1.0:
        idx_sorted = np.argsort(-p)
        cumsum = np.cumsum(p[idx_sorted])
        keep = idx_sorted[cumsum <= topp]
        if keep.size == 0:
            keep = idx_sorted[:1]
        p2 = np.zeros_like(p)
        p2[keep] = p[keep]
        s = p2.sum()
        if s > 0:
            p = p2 / s

    return int(np.random.choice(len(p), p=p))

def build_ngram_ban(seq, n):
    """HuggingFace 類似的 no-repeat-ngram：回傳給定序列目前 prefix 的被禁止 next ids 集合。"""
    if n <= 1 or len(seq) < n-1:
        return set()
    mapping = {}
    for i in range(len(seq) - n + 1):
        prefix = tuple(seq[i:i+n-1])
        nxt = seq[i+n-1]
        mapping.setdefault(prefix, set()).add(int(nxt))
    prefix = tuple(seq[-(n-1):])
    return mapping.get(prefix, set())

# ---------- Miditok helpers ----------
def _safe_ids_to_tokens(tokenizer, ids):
    try: return tokenizer.ids_to_tokens(ids, as_str=True)
    except Exception: pass
    toks = []
    for i in ids:
        try: toks.append(str(tokenizer[i]))
        except Exception: toks.append(None)
    return toks

def build_miditok_bar_checker(tokenizer):
    bar_ids = set()
    try:
        for i, t in enumerate(_safe_ids_to_tokens(tokenizer, list(range(len(tokenizer))))):
            if isinstance(t, str):
                h = t.split('_',1)[0].lower()
                if h == 'bar' or 'bar' in h:
                    bar_ids.add(i)
    except Exception: pass
    def is_bar(tok_id: int) -> bool:
        if tok_id in bar_ids: return True
        t = _safe_ids_to_tokens(tokenizer, [tok_id])[0]
        if isinstance(t, str):
            h = t.split('_',1)[0].lower()
            return h == 'bar' or 'bar' in h
        return False
    return is_bar

def build_miditok_tempo_guard(tokenizer, bpm_min=None, bpm_max=None):
    if bpm_min is None and bpm_max is None:
        return lambda _id: True
    def parse_bpm(s: str):
        m = re.findall(r'\d+', s)
        if not m: return None
        return int(m[-1])
    def guard(tok_id: int) -> bool:
        t = _safe_ids_to_tokens(tokenizer, [tok_id])[0]
        if not isinstance(t, str): return True
        if 'tempo' not in t.lower(): return True
        bpm = parse_bpm(t)
        if bpm is None: return True
        if bpm_min is not None and bpm < bpm_min: return False
        if bpm_max is not None and bpm > bpm_max: return False
        return True
    return guard

# ---------- Dictionary (custom) ----------
def build_dictionary_if_needed(dict_path, midi_paths, with_chord: bool):
    if os.path.isfile(dict_path):
        return
    print(f"[dict] build from {len(midi_paths)} prompts (with_chord={with_chord}) ...")
    vocab = {}
    for p in midi_paths:
        notes, tempos = utils.read_items(p)
        notes = utils.quantize_items(notes)
        if not notes: continue
        max_t = int(notes[-1].end)
        items = tempos + (utils.extract_chords(notes) if with_chord else []) + notes
        groups = utils.group_items(items, max_t)
        events = utils.item2event(groups)
        for ev in events:
            k = f"{ev.name}_{ev.value}"
            if k not in vocab: vocab[k] = len(vocab)
    # safety
    for k in ["Bar_None","Position_1/16"]:
        if k not in vocab: vocab[k] = len(vocab)
    e2w = vocab; w2e = {i:k for k,i in e2w.items()}
    with open(dict_path,'wb') as f:
        pickle.dump((e2w, w2e), f)
    print(f"[dict] wrote {dict_path} | vocab={len(e2w)}")

# ---------- Encode prompt (first N bars) ----------
def encode_custom_first_n_bars(midipath, n_bars, e2w, with_chord=False):
    notes, tempos = utils.read_items(midipath)
    notes = utils.quantize_items(notes)
    if not notes: return []
    max_t = int(notes[-1].end)
    items = tempos + (utils.extract_chords(notes) if with_chord else []) + notes
    groups = utils.group_items(items, max_t)
    events = utils.item2event(groups)

    ids = []
    bars = 0
    for ev in events:
        k = f"{ev.name}_{ev.value}"
        if k in e2w:
            ids.append(int(e2w[k]))
        elif ev.name == 'Note Velocity' and 'Note Velocity_21' in e2w:
            ids.append(int(e2w['Note Velocity_21']))
        if ev.name == 'Bar':
            bars += 1
            if bars >= n_bars:
                break
    return ids

def encode_miditok_first_n_bars(midipath, n_bars, kind, use_chords, tempo_bpm_min, tempo_bpm_max):
    cfg = TokenizerConfig(
        beat_res={(0,4):4}, num_velocities=32,
        use_chords=bool(use_chords),
        use_rests=False, use_tempos=True,
        use_time_signatures=(kind=='remi_plus'),
        use_programs=(kind=='remi_plus'),
        one_token_stream_for_programs=(kind=='remi_plus'),
    )
    tok = REMI(cfg)
    try:
        seq = tok.encode(Score(midipath))
    except Exception:
        seq = tok.midi_to_tokens(Score(midipath))
    if isinstance(seq, list):
        assert len(seq)==1
        seq = seq[0]
    ids = list(map(int, seq.ids))
    is_bar = build_miditok_bar_checker(tok)

    # keep first n_bars
    kept, bars = [], 0
    for t in ids:
        kept.append(int(t))
        if is_bar(int(t)):
            bars += 1
            if bars >= n_bars:
                break

    tempo_guard = build_miditok_tempo_guard(tok, tempo_bpm_min, tempo_bpm_max)
    ctx = {'tokenizer': tok, 'is_bar': is_bar, 'tempo_guard': tempo_guard}
    return kept, ctx

# ---------- Model ----------
class GPT2Wrapper(nn.Module):
    def __init__(self, vocab_size, n_layer=12, n_head=12, n_embd=768, dropout=0.1, max_len=1024):
        super().__init__()
        assert n_embd % n_head == 0
        cfg = GPT2Config(
            vocab_size=vocab_size, n_positions=max_len, n_ctx=max_len,
            n_embd=n_embd, n_layer=n_layer, n_head=n_head,
            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
            bos_token_id=None, eos_token_id=None,
        )
        self.model = GPT2LMHeadModel(cfg)
    def forward(self, x):  # x: [B,T]
        return self.model(input_ids=x).logits

# ---------- Preset configs (三組) ----------
# 你可以按需要改動
CONFIGS = {
    'A': dict(temperature=0.9, topk=32,   topp=0.95, rep_penalty=1.10, no_repeat_ngram=4, recent_k=120),
    'B': dict(temperature=0.9, topk=32,  topp=0.90, rep_penalty=1.00, no_repeat_ngram=4, recent_k=100),
    'C': dict(temperature=0.9, topk=32,  topp=0.85, rep_penalty=1.05, no_repeat_ngram=4, recent_k=200),
}

# ---------- Generate continuation ----------
@torch.no_grad()
def continue_bars(
    model, device, *,
    seed_ids, bars_to_add, tokenizer_mode, dict_or_ctx,
    cfg  # one of CONFIGS values
):
    model.eval()
    ids = list(seed_ids)

    # helpers
    if tokenizer_mode == 'custom':
        e2w = dict_or_ctx  # event2word
        bar_token = e2w.get('Bar_None', -1)
        is_bar = lambda tid: (tid == bar_token)
        tempo_guard = lambda _id: True
    else:
        tok_ctx = dict_or_ctx
        is_bar = tok_ctx['is_bar']
        tempo_guard = tok_ctx['tempo_guard']

    added = 0
    hard_cap = bars_to_add * X_LEN * 3  # 避免死循環
    while added < bars_to_add and len(ids) < hard_cap:
        x = torch.tensor([ids[-X_LEN:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].detach().cpu().numpy()

        # ngram ban
        banned = set()
        if cfg.get('no_repeat_ngram', 0) and cfg['no_repeat_ngram'] > 1:
            banned |= build_ngram_ban(ids, cfg['no_repeat_ngram'])

        # repetition penalty
        recent = ids[-int(cfg.get('recent_k', 100)):] if cfg.get('recent_k', 0) > 0 else []

        # tempo guard（只在 miditok 會生效）
        tries = 0
        while True:
            nxt = sample_with_constraints(
                logits,
                temperature=cfg['temperature'],
                topk=cfg['topk'],
                topp=cfg['topp'],
                banned_ids=banned,
                rep_penalty_ids=recent,
                rep_penalty=cfg['rep_penalty'],
            )
            if tempo_guard(nxt): break
            tries += 1
            if tries >= 10: break

        ids.append(int(nxt))
        if (is_bar(int(nxt))):
            added += 1
    return ids

# ---------- Save MIDI ----------
def save_custom_midi(ids, out_mid, dict_path, prompt_path=None):
    e2w, w2e = pickle.load(open(dict_path,'rb'))
    utils.write_midi(words=ids, word2event={i:k for k,i in e2w.items()},
                     output_path=out_mid, prompt_path=prompt_path or None)

def save_miditok_midi(ids, out_mid, tok):
    seq = TokSequence(ids=list(map(int, ids)))
    tok.complete_sequence(seq)
    try:
        m = tok.decode(seq)
        if hasattr(m,'save_midi'): m.save_midi(str(out_mid))
        elif hasattr(m,'dump_midi'): m.dump_midi(str(out_mid))
        elif hasattr(m,'to_pretty_midi'):
            pm = m.to_pretty_midi(); pm.write(str(out_mid))
        else:
            pm = tok.tokens_to_midi(seq); pm.write(str(out_mid))
    except Exception:
        pm = tok.tokens_to_midi(seq); pm.write(str(out_mid))

# ---------- Main logic ----------
def main():
    opt = parse_opt()
    torch.manual_seed(opt.seed); np.random.seed(opt.seed); random.seed(opt.seed)
    device = torch.device(opt.device)

    ensure_dir(opt.out_dir)
    prompt_list = list_midis(opt.prompt_dir)
    if len(prompt_list) == 0:
        raise SystemExit(f'No MIDI found under: {opt.prompt_dir}')

    # tokenizer & vocab
    if opt.tokenizer == 'custom':
        # 若沒有字典，用 prompt 自動建一份
        build_dictionary_if_needed(opt.dict_path, prompt_list, with_chord=bool(opt.with_chord))
        e2w, _ = pickle.load(open(opt.dict_path,'rb'))
        vocab_size = int(max(e2w.values())) + 1
        miditok_ctx = None
    else:
        if not HAVE_MIDITOK:
            raise SystemExit("miditok / symusic 未安裝，請改用 --tokenizer custom 或安裝套件")
        kind = 'remi_plus' if opt.tokenizer == 'miditok_remi_plus' else 'remi'
        cfg = TokenizerConfig(
            beat_res={(0,4):4}, num_velocities=32,
            use_chords=bool(opt.with_chord),
            use_rests=False, use_tempos=True,
            use_time_signatures=(kind=='remi_plus'),
            use_programs=(kind=='remi_plus'),
            one_token_stream_for_programs=(kind=='remi_plus'),
        )
        tokenizer = REMI(cfg)
        vocab_size = len(tokenizer)

    # 構 model 並載 checkpoint
    model = GPT2Wrapper(
        vocab_size=vocab_size, n_layer=12, n_head=12, n_embd=768,
        dropout=0.1, max_len=X_LEN
    ).to(device)
    ckpt = torch.load(opt.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # 解析 config 列表
    cfg_ids = [c.strip() for c in opt.configs.split(',') if c.strip()]
    for c in cfg_ids:
        if c not in CONFIGS:
            raise SystemExit(f'Unknown config id: {c}. Valid: {list(CONFIGS.keys())}')

    # CSV 紀錄
    csv_path = os.path.join(opt.out_dir, 'runs.csv')
    write_header = not os.path.isfile(csv_path)
    csv_f = open(csv_path, 'a', newline='')
    csv_w = csv.writer(csv_f)
    if write_header:
        csv_w.writerow(['song','config','midi','wav','temperature','topk','topp','rep_penalty','no_repeat_ngram','recent_k'])

    # 逐歌、逐 config 生成
    for i, midi_in in enumerate(prompt_list, 1):
        song_dir = os.path.join(opt.out_dir, f"song_{i:01d}")
        ensure_dir(song_dir)

        # 先把 prompt 的首 n bars 轉成 token 作為 seed
        if opt.tokenizer == 'custom':
            e2w, _ = pickle.load(open(opt.dict_path,'rb'))
            seed_ids = encode_custom_first_n_bars(midi_in, opt.bars_prompt, e2w, with_chord=bool(opt.with_chord))
            dict_or_ctx = e2w
        else:
            kind = 'remi_plus' if opt.tokenizer == 'miditok_remi_plus' else 'remi'
            seed_ids, miditok_ctx = encode_miditok_first_n_bars(
                midi_in, opt.bars_prompt, kind, bool(opt.with_chord),
                opt.tempo_bpm_min, opt.tempo_bpm_max
            )
            dict_or_ctx = miditok_ctx

        if len(seed_ids) == 0:
            print(f"[skip] cannot parse tokens from {midi_in}")
            continue

        for cfg_id in cfg_ids:
            cfg = CONFIGS[cfg_id]

            # 生成 +24 bars
            full_ids = continue_bars(
                model, device,
                seed_ids=seed_ids, bars_to_add=opt.bars_cont,
                tokenizer_mode=('custom' if opt.tokenizer=='custom' else 'miditok'),
                dict_or_ctx=dict_or_ctx, cfg=cfg
            )

            # 存 MIDI
            out_mid = os.path.join(song_dir, f'cfg{cfg_id}.mid')
            if opt.tokenizer == 'custom':
                save_custom_midi(full_ids, out_mid, opt.dict_path, prompt_path=None)
            else:
                save_miditok_midi(full_ids, out_mid, dict_or_ctx['tokenizer'])
            print(f'[gen] {Path(out_mid).name}  bars={opt.bars_prompt}+{opt.bars_cont}')

            # WAV（可選）
            out_wav = ''
            if opt.render_wav and os.path.isfile(opt.sf2):
                out_wav = swap_ext(out_mid, '.wav')
                ok = render_midi_to_wav(out_mid, out_wav, opt.sf2, sr=opt.sr, method=opt.render_method)
                if ok: print(f'      rendered {Path(out_wav).name}')
                else:  print(f'      render failed (check fluidsynth/midi2audio)')

            # CSV 記錄
            csv_w.writerow([
                f'song_{i}', f'cfg{cfg_id}', out_mid, out_wav,
                cfg['temperature'], cfg['topk'], cfg['topp'],
                cfg['rep_penalty'], cfg['no_repeat_ngram'], cfg['recent_k']
            ])
            csv_f.flush()

    csv_f.close()
    print(f'[done] outputs → {opt.out_dir}')
    print(f'      log → {csv_path}')


if __name__ == '__main__':
    main()

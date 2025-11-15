#!/usr/bin/env python
# main.py — GPT-2 (12-layer) + tqdm + per-epoch sampling + loss logging/plot + MIDI->WAV
# 新增：
# - --tokenizer {custom, miditok_remi, miditok_remi_plus}
# - --with_chord (custom 模式抽 Chord 事件；miditok 也可開 use_chords)
# - Miditok 模式：以「Bar token」計數，保證剛好 N bars
# - --tempo_bpm_min / --tempo_bpm_max 夾 BPM（miditok）
# - custom 模式若找不到字典，會自動掃描 --midi_glob 建 dictionary.pkl
# - 新增 --stride（hop size）。預設 stride=block（不重疊）；減小可重疊切段
# - 將 multiprocessing 的 worker 移到頂層，避免 pickling 問題
# - block 長度 = --max_len（動態），不再寫死 1024

import os, glob, pickle, argparse, warnings, re, random, subprocess, shutil, sys
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
import multiprocessing as mp

# ---- plotting (no GUI needed) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 自家工具（custom tokenizer）
import utils

# ===== 可選 Miditok =====
HAVE_MIDITOK = False
try:
    from miditok import TokenizerConfig, REMI
    try:
        from miditok import TokSequence  # miditok >= 3.x
    except Exception:
        from miditok.utils import TokSequence
    HAVE_MIDITOK = True
except Exception:
    HAVE_MIDITOK = False

# 由 main() 在解析參數後設定（等於 --max_len）
BLOCK_LEN = 1024


# ------------------------- Args -------------------------
def parse_opt():
    p = argparse.ArgumentParser()
    # I/O
    p.add_argument('--tokenizer', choices=['custom', 'miditok_remi', 'miditok_remi_plus'],
                   default='custom', help='自家字典/REMI，或 Miditok 的 REMI/REMI+')
    p.add_argument('--dict_path', type=str, default='./dictionary.pkl',
                   help='(custom) event<->word 字典 pkl；不存在會自動建立')
    p.add_argument('--midi_glob', type=str, default='./Pop1K7/midi_analyzed/*.mid',
                   help='訓練檔案路徑或 glob（或給資料夾）')
    p.add_argument('--ckp_folder', type=str, default='./checkpoints', help='checkpoint 輸出資料夾')
    p.add_argument('--log_dir', type=str, default='./logs', help='loss CSV / 圖表輸出資料夾')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=3407)

    # Train
    p.add_argument('--mode', choices=['train', 'test'], default='train')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--resume', type=str, default='', help='載入現有 checkpoint 繼續訓練')

    # 模型
    p.add_argument('--n_layer', type=int, default=12)
    p.add_argument('--n_head', type=int, default=12)
    p.add_argument('--n_embd', type=int, default=768)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max_len', type=int, default=1024, help='block/位置嵌入長度（訓練序列長度）')

    # 生成 / 取樣
    p.add_argument('--model_ckpt', type=str, default='', help='(test) 推論用 checkpoint')
    p.add_argument('--out_midi', type=str, default='./results/sample.mid')
    p.add_argument('--n_bars', type=int, default=32, help='(test) 生成 bars 數')
    p.add_argument('--temperature', type=float, default=1.2)
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--prompt_midi', type=str, default='', help='選填：以此 MIDI 當 prompt')

    # 每 epoch 取樣
    p.add_argument('--sample_every', type=int, default=10)
    p.add_argument('--sample_bars', type=int, default=32)
    p.add_argument('--sample_out', type=str, default='./results')
    p.add_argument('--sample_temperature', type=float, default=None)
    p.add_argument('--sample_topk', type=int, default=None)

    # loss 紀錄 / 圖
    p.add_argument('--log_steps', type=int, default=1)
    p.add_argument('--plot_every', type=int, default=1)
    p.add_argument('--smooth_window', type=int, default=5)

    # 渲染
    p.add_argument('--render_wav', type=int, default=1)
    p.add_argument('--sf2', type=str, default='./assets/sf2/GM.sf2')
    p.add_argument('--sr', type=int, default=22050)
    p.add_argument('--render_method', choices=['auto','fluidsynth','midi2audio'], default='auto')

    # 自家 REMI 選項
    p.add_argument('--with_chord', type=int, default=0, help='(custom) 抽和絃事件並納入字典')

    # Miditok 選項
    p.add_argument('--tempo_bpm_min', type=int, default=None, help='(miditok) 最低 BPM 允許')
    p.add_argument('--tempo_bpm_max', type=int, default=None, help='(miditok) 最高 BPM 允許')

    # 資料切段
    p.add_argument('--workers', type=int, default=8, help='並行處理 MIDI 的進程數（>=2 啟用多進程）')
    p.add_argument('--stride', type=int, default=None,
                   help='切段 hop size（預設=block，非重疊）。例：--max_len 1024 --stride 256 → 75% 重疊')

    return p.parse_args()


# ------------------------- helpers -------------------------
def collect_midi_paths(path_or_glob: str):
    """支援：直接給資料夾 / 或 glob。"""
    if os.path.isdir(path_or_glob):
        pats = [os.path.join(path_or_glob, ext) for ext in ('*.mid','*.MID','*.midi','*.MIDI')]
        files = []
        for p in pats:
            files.extend(glob.glob(p))
        return sorted(files)
    files = glob.glob(path_or_glob)
    if files:
        return sorted(files)
    base = os.path.dirname(path_or_glob) or '.'
    if os.path.isdir(base):
        return collect_midi_paths(base)
    return []

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_losses(losses_epoch, png_path, smooth_window=5):
    ensure_dir(os.path.dirname(png_path) or '.')
    x = np.arange(1, len(losses_epoch)+1)
    plt.figure(figsize=(7,4))
    plt.plot(x, losses_epoch, label='epoch loss')
    if smooth_window and smooth_window > 1 and len(losses_epoch) >= smooth_window:
        k = np.ones(smooth_window) / smooth_window
        sm = np.convolve(losses_epoch, k, mode='valid')
        xs = np.arange(smooth_window, len(losses_epoch)+1)
        plt.plot(xs, sm, linestyle='--', label=f'moving avg ({smooth_window})')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Training loss')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

def swap_ext(path, new_ext):
    root, _ = os.path.splitext(path)
    return root + new_ext

def render_midi_to_wav(midi_path: str, wav_path: str, sf2_path: str, sr: int = 22050, method: str = 'auto'):
    # 1) fluidsynth CLI
    if method in ('auto','fluidsynth'):
        if shutil.which('fluidsynth'):
            cmd = ['fluidsynth','-ni','-T','wav','-F',wav_path,'-r',str(sr),sf2_path,midi_path]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            except subprocess.CalledProcessError as e:
                print(f'[render] fluidsynth failed: {e}')
    # 2) midi2audio
    if method in ('auto','midi2audio'):
        try:
            from midi2audio import FluidSynth
            FluidSynth(sound_font=sf2_path, sample_rate=sr).midi_to_audio(midi_path, wav_path)
            return True
        except Exception as e:
            print(f'[render] midi2audio failed: {e}')
    return False

def temperature_sampling(logits, temperature=1.0, topk=0):
    z = logits.astype(np.float64) / max(1e-8, float(temperature))
    if topk and 0 < topk < z.shape[0]:
        idxs = np.argpartition(-z, topk)[:topk]
        vals = z[idxs]
        p = np.exp(vals - vals.max()); p /= p.sum()
        return int(np.random.choice(idxs, p=p))
    z -= z.max(); p = np.exp(z); p /= p.sum()
    return int(np.random.choice(len(z), p=p))


# ------------------------- Miditok helpers -------------------------
def _safe_ids_to_tokens(tokenizer, ids):
    try:
        return tokenizer.ids_to_tokens(ids, as_str=True)
    except Exception:
        pass
    toks = []
    for i in ids:
        try: toks.append(str(tokenizer[i]))
        except Exception: toks.append(None)
    return toks

def build_miditok_bar_checker(tokenizer):
    bar_ids = set()
    try:
        ids = list(range(len(tokenizer)))
        for i, t in enumerate(_safe_ids_to_tokens(tokenizer, ids)):
            if isinstance(t, str):
                head = t.split('_',1)[0].lower()
                if head == 'bar' or 'bar' in head:
                    bar_ids.add(i)
    except Exception:
        pass
    def is_bar(tok_id: int) -> bool:
        if tok_id in bar_ids: return True
        t = _safe_ids_to_tokens(tokenizer, [tok_id])[0]
        if isinstance(t, str):
            head = t.split('_',1)[0].lower()
            return (head == 'bar' or 'bar' in head)
        return False
    return is_bar, bar_ids

def build_miditok_tempo_guard(tokenizer, bpm_min=None, bpm_max=None):
    if bpm_min is None and bpm_max is None:
        return lambda _id: True
    def parse_bpm(token_str: str):
        m = re.findall(r'\d+', token_str)
        if not m: return None
        try: return int(m[-1])
        except: return None
    def is_ok(tok_id: int) -> bool:
        t = _safe_ids_to_tokens(tokenizer, [tok_id])[0]
        if not isinstance(t, str): return True
        if 'tempo' not in t.lower(): return True
        bpm = parse_bpm(t)
        if bpm is None: return True
        if bpm_min is not None and bpm < bpm_min: return False
        if bpm_max is not None and bpm > bpm_max: return False
        return True
    return is_ok


# ------------------------- 字典（custom） -------------------------
def build_dictionary_if_needed(dict_path: str, midi_paths, with_chord: bool):
    """custom 模式：若 dict 不存在，掃描 MIDI 建 event2word/word2event 並存 pkl。"""
    if os.path.isfile(dict_path):
        return
    print(f"[dict] building dictionary from {len(midi_paths)} MIDIs (with_chord={with_chord}) ...")
    vocab = {}
    for p in midi_paths:
        note_items, tempo_items = utils.read_items(p)
        note_items = utils.quantize_items(note_items)
        if not note_items:
            continue
        max_time = int(note_items[-1].end)
        items = tempo_items + (utils.extract_chords(note_items) if with_chord else []) + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        for ev in events:
            k = f"{ev.name}_{ev.value}"
            if k not in vocab:
                vocab[k] = len(vocab)
    # 關鍵 token 保底
    for must in ["Bar_None", "Position_1/16"]:
        if must not in vocab:
            vocab[must] = len(vocab)
    e2w = vocab
    w2e = {i:k for k,i in e2w.items()}
    with open(dict_path, "wb") as f:
        pickle.dump((e2w, w2e), f)
    print(f"[dict] wrote {dict_path} | vocab={len(e2w)}")


# ------------------------- multiprocessing workers (top-level, picklable) -------------------------
def worker_custom_build(args):
    """(path, block, stride, with_chord, e2w) -> ndarray[n_seg, block+1] or None"""
    p, block, stride, with_chord, e2w = args
    try:
        note_items, tempo_items = utils.read_items(p)
        note_items = utils.quantize_items(note_items)
        if not note_items:
            return None
        max_time = int(note_items[-1].end)
        items = tempo_items + (utils.extract_chords(note_items) if with_chord else []) + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        words = []
        for ev in events:
            k = f'{ev.name}_{ev.value}'
            if k in e2w:
                words.append(int(e2w[k]))
            elif ev.name == 'Note Velocity' and 'Note Velocity_21' in e2w:
                words.append(int(e2w['Note Velocity_21']))
        ids = np.asarray(words, dtype=np.int64)
        if len(ids) < block + 1:
            return None
        step = stride if (stride and stride > 0 and stride <= block) else block
        out = []
        for i in range(0, len(ids) - block - 1, step):
            out.append(ids[i:i + block + 1])
        if not out:
            return None
        return np.stack(out, axis=0)
    except Exception:
        return None


def worker_miditok_build(args):
    """(path, cfg_kwargs, block, stride, kind) -> ndarray[n_seg, block+1] or None"""
    p, cfg_kwargs, block, stride, kind = args
    try:
        from symusic import Score
        cfg = TokenizerConfig(**cfg_kwargs)
        tok = REMI(cfg)
        try:
            seq = tok.encode(Score(p))
        except Exception:
            seq = tok.midi_to_tokens(Score(p))
        if isinstance(seq, list):
            assert len(seq) == 1
            seq = seq[0]
        ids = np.asarray(seq.ids, dtype=np.int64)
        if len(ids) < block + 1:
            return None
        step = stride if (stride and stride > 0 and stride <= block) else block
        out = []
        for i in range(0, len(ids) - block - 1, step):
            out.append(ids[i:i + block + 1])
        if not out:
            return None
        return np.stack(out, axis=0)
    except Exception:
        return None


# ------------------------- Dataset -------------------------
class PairDataset(Dataset):
    """回傳 (x, y)；支援 custom 或 miditok。"""
    def __init__(self, midi_paths, *, mode, dict_path=None, with_chord=False,
                 miditok_kind=None, block_len=1024, stride=None, workers=1):
        self.block = int(block_len)
        self.stride = int(stride) if (stride is not None and stride > 0) else self.block
        self.samples = []   # (np.array ids)
        self.mode = mode
        self.with_chord = with_chord
        self.miditok_kind = miditok_kind
        self.workers = max(1, int(workers))

        if mode == 'custom':
            assert dict_path and os.path.isfile(dict_path), "custom 模式需提供 --dict_path（或由 main 自動建立）"
            self.event2word, self.word2event = pickle.load(open(dict_path, 'rb'))
            self._build_custom(midi_paths)
        else:
            assert HAVE_MIDITOK, "miditok 模式需要已安裝 miditok"
            self._build_miditok(midi_paths, miditok_kind)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        a = self.samples[i]
        x = a[:-1]; y = a[1:]
        return torch.from_numpy(x).long(), torch.from_numpy(y).long()

    # ---- custom（多進程）----
    def _build_custom(self, midi_paths):
        args_iter = [(p, self.block, self.stride, bool(self.with_chord), self.event2word) for p in midi_paths]
        arrs = []
        if self.workers >= 2:
            with mp.get_context("spawn").Pool(processes=self.workers) as pool:
                for segs in tqdm(pool.imap_unordered(worker_custom_build, args_iter),
                                 total=len(midi_paths), desc="[build] custom", ncols=100):
                    if segs is not None:
                        arrs.append(segs)
        else:
            for a in tqdm(args_iter, total=len(midi_paths), desc="[build] custom", ncols=100):
                segs = worker_custom_build(a)
                if segs is not None:
                    arrs.append(segs)
        if arrs:
            self.samples = [x for x in np.concatenate(arrs, axis=0)]
        else:
            self.samples = []
        self.samples = [np.asarray(s, dtype=np.int64) for s in self.samples]

    # ---- miditok（多進程）----
    def _build_miditok(self, midi_paths, kind):
        cfg_kwargs = dict(
            beat_res={(0, 4): 4},            # 1/16
            num_velocities=32,
            use_chords=bool(self.with_chord),
            use_rests=False,                 # 關閉 rest（避免 resolution 衝突）
            use_tempos=True,
            use_time_signatures=(kind == 'remi_plus'),
            use_programs=(kind == 'remi_plus'),
            one_token_stream_for_programs=(kind == 'remi_plus'),
        )
        args_iter = [(p, cfg_kwargs, self.block, self.stride, kind) for p in midi_paths]

        arrs = []
        if self.workers >= 2:
            with mp.get_context("spawn").Pool(processes=self.workers) as pool:
                for segs in tqdm(pool.imap_unordered(worker_miditok_build, args_iter),
                                 total=len(midi_paths), desc="[build] miditok", ncols=100):
                    if segs is not None:
                        arrs.append(segs)
        else:
            for a in tqdm(args_iter, total=len(midi_paths), desc="[build] miditok", ncols=100):
                segs = worker_miditok_build(a)
                if segs is not None:
                    arrs.append(segs)

        if arrs:
            self.samples = [x for x in np.concatenate(arrs, axis=0)]
        else:
            self.samples = []
        self.samples = [np.asarray(s, dtype=np.int64) for s in self.samples]


# ------------------------- Model -------------------------
class GPT2Wrapper(nn.Module):
    def __init__(self, vocab_size, n_layer=12, n_head=12, n_embd=768, dropout=0.1, max_len=1024):
        super().__init__()
        assert n_embd % n_head == 0
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_len,
            n_ctx=max_len,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            bos_token_id=None, eos_token_id=None,
        )
        self.model = GPT2LMHeadModel(cfg)
    def forward(self, x):
        return self.model(input_ids=x).logits  # [B,T,V]


# ------------------------- Sampling -------------------------
@torch.no_grad()
def generate_sample(model, device, *, bars, temperature, topk, out_path, prompt_midi,
                    mode, dict_path, with_chord, miditok_ctx):
    """
    miditok_ctx = {
      'tokenizer': tokenizer or None,
      'is_bar': callable or None,
      'tempo_guard': callable or None
    }
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    model.eval()

    if mode == 'custom':
        e2w, _ = pickle.load(open(dict_path, 'rb'))
        # seed
        if prompt_midi:
            # 用 custom 解析 prompt
            note_items, tempo_items = utils.read_items(prompt_midi)
            note_items = utils.quantize_items(note_items)
            words = []
            if len(note_items) > 0:
                max_time = int(note_items[-1].end)
                items = tempo_items + (utils.extract_chords(note_items) if with_chord else []) + note_items
                groups = utils.group_items(items, max_time)
                events = utils.item2event(groups)
                for ev in events[:BLOCK_LEN]:
                    k = f'{ev.name}_{ev.value}'
                    if k in e2w:
                        words.append(int(e2w[k]))
        else:
            words = [e2w['Bar_None']]
            pos = e2w.get('Position_1/16', None)
            if pos is not None: words.append(pos)
            tcls = [v for k,v in e2w.items() if k.startswith('Tempo Class_')]
            tval = [v for k,v in e2w.items() if k.startswith('Tempo Value_')]
            if tcls: words.append(int(np.random.choice(tcls)))
            if tval: words.append(int(np.random.choice(tval)))

        cur = 0
        while cur < bars:
            x_np = np.array(words[-BLOCK_LEN:], dtype=np.int64)[None,:]
            x = torch.from_numpy(x_np).long().to(device)
            logits = model(x)[0,-1].detach().cpu().numpy()
            nxt = temperature_sampling(logits, temperature, topk)
            words.append(int(nxt))
            if int(nxt) == e2w.get('Bar_None', -1):
                cur += 1
        # write
        utils.write_midi(words=words, word2event={i:k for k,i in e2w.items()},
                         output_path=out_path, prompt_path=prompt_midi or None)
        return out_path

    # ---- miditok ----
    tokenizer = miditok_ctx['tokenizer']
    is_bar = miditok_ctx['is_bar']
    tempo_guard = miditok_ctx['tempo_guard']

    # seed：從空或從 prompt（用 miditok 解析也可，但保持簡單）
    words = []

    cur = 0
    hard_cap = bars * BLOCK_LEN * 2  # 安全上限避免死循環
    while cur < bars and len(words) < hard_cap:
        x_np = np.array(words[-BLOCK_LEN:], dtype=np.int64)[None,:] if len(words) else np.zeros((1,1),dtype=np.int64)
        x = torch.from_numpy(x_np).long().to(device)
        logits = model(x)[0,-1].detach().cpu().numpy()

        tries = 0
        while True:
            nxt = temperature_sampling(logits, temperature, topk)
            if tempo_guard(nxt): break
            tries += 1
            if tries >= 10: break
        words.append(int(nxt))
        if is_bar(int(nxt)):
            cur += 1

    # decode to MIDI
    seq_obj = TokSequence(ids=words)
    tokenizer.complete_sequence(seq_obj)
    try:
        midi_obj = tokenizer.decode(seq_obj)
        if hasattr(midi_obj, "save_midi"):
            midi_obj.save_midi(str(out_path))
        elif hasattr(midi_obj, "dump_midi"):
            midi_obj.dump_midi(str(out_path))
        elif hasattr(midi_obj, "to_pretty_midi"):
            pm = midi_obj.to_pretty_midi(); pm.write(str(out_path))
        else:
            pm = tokenizer.tokens_to_midi(seq_obj); pm.write(str(out_path))
    except Exception:
        pm = tokenizer.tokens_to_midi(seq_obj); pm.write(str(out_path))
    return out_path


# ------------------------- Train / Test -------------------------
def train(opt):
    device = torch.device(opt.device)
    print(f'Training on {device}')
    ensure_dir(opt.ckp_folder); ensure_dir(opt.log_dir)

    # 準備資料
    train_list = collect_midi_paths(opt.midi_glob)
    print('train list len =', len(train_list))
    if len(train_list) == 0:
        raise FileNotFoundError(f'No MIDI found: {opt.midi_glob}')

    # tokenizer & vocab
    if opt.tokenizer == 'custom':
        # 確保字典存在
        build_dictionary_if_needed(opt.dict_path, train_list, with_chord=bool(opt.with_chord))
        e2w, _ = pickle.load(open(opt.dict_path, 'rb'))
        vocab_size = int(max(e2w.values())) + 1
        ds = PairDataset(
            train_list, mode='custom', dict_path=opt.dict_path, with_chord=bool(opt.with_chord),
            block_len=BLOCK_LEN, stride=opt.stride, workers=opt.workers
        )
        miditok_ctx = None
    else:
        assert HAVE_MIDITOK, "需要 pip install miditok symusic"
        kind = 'remi_plus' if opt.tokenizer == 'miditok_remi_plus' else 'remi'
        # 這裡只建 tokenizer 取得 vocab_size 與 helper（真正 encode 在 worker 內重建 tokenizer）
        cfg = TokenizerConfig(
            beat_res={(0,4):4},
            num_velocities=32,
            use_chords=bool(opt.with_chord),
            use_rests=False,
            use_tempos=True,
            use_time_signatures=(kind=='remi_plus'),
            use_programs=(kind=='remi_plus'),
            one_token_stream_for_programs=(kind=='remi_plus'),
        )
        tokenizer = REMI(cfg)
        vocab_size = len(tokenizer)
        is_bar, _ = build_miditok_bar_checker(tokenizer)
        tempo_guard = build_miditok_tempo_guard(tokenizer, opt.tempo_bpm_min, opt.tempo_bpm_max)
        ds = PairDataset(
            train_list, mode='miditok', miditok_kind=kind, with_chord=bool(opt.with_chord),
            block_len=BLOCK_LEN, stride=opt.stride, workers=opt.workers
        )
        miditok_ctx = {'tokenizer': tokenizer, 'is_bar': is_bar, 'tempo_guard': tempo_guard}

    if len(ds) == 0:
        raise ValueError('Dataset parsed 0 segments，請調小 --max_len 或設定 --stride（如 256），或檢查 tokenizer 是否匹配。')

    dl_workers = min(4, max(0, int(opt.workers) // 2))
    dl = DataLoader(
        ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=dl_workers,
        pin_memory=(opt.device == 'cuda'),
        persistent_workers=True if dl_workers > 0 else False
    )

    model = GPT2Wrapper(
        vocab_size=vocab_size,
        n_layer=opt.n_layer, n_head=opt.n_head, n_embd=opt.n_embd,
        dropout=opt.dropout, max_len=BLOCK_LEN
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # resume
    start_epoch = 1
    if opt.resume and os.path.isfile(opt.resume):
        ckpt = torch.load(opt.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt: optim.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        print(f'[resume] from {opt.resume} @ epoch {start_epoch}')

    # CSV
    epoch_csv = os.path.join(opt.log_dir, 'training_loss_epoch.csv')
    step_csv  = os.path.join(opt.log_dir, 'training_loss_steps.csv')
    if not os.path.exists(epoch_csv):
        with open(epoch_csv, 'w') as f: f.write('epoch,loss\n')
    if opt.log_steps and not os.path.exists(step_csv):
        with open(step_csv, 'w') as f: f.write('epoch,global_step,step_in_epoch,loss\n')

    losses_epoch = []
    steps_per_epoch = len(dl)
    print('Model ready. Start training. | vocab =', vocab_size)

    for epoch in range(start_epoch, opt.epochs+1):
        model.train(); epoch_losses = []
        pbar = tqdm(dl, desc=f'Epoch {epoch}/{opt.epochs}', ncols=100)
        for step_idx, (x,y) in enumerate(pbar, 1):
            x = x.to(device); y = y.to(device)
            logits = model(x)                   # [B,T,V]
            loss = nn.CrossEntropyLoss()(logits.permute(0,2,1), y)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            l = float(loss.detach().cpu()); epoch_losses.append(l)
            pbar.set_postfix(loss=f'{l:.4f}')

            if opt.log_steps:
                g = (epoch-1)*steps_per_epoch + step_idx
                with open(step_csv,'a') as f: f.write(f'{epoch},{g},{step_idx},{l}\n')

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        losses_epoch.append(mean_loss)
        print(f'>>> Epoch: {epoch}, Loss: {mean_loss:.5f}')
        with open(epoch_csv,'a') as f: f.write(f'{epoch},{mean_loss}\n')
        np.save(os.path.join(opt.ckp_folder,'training_loss.npy'), np.array(losses_epoch,dtype=np.float32))

        if opt.plot_every > 0 and (epoch % opt.plot_every == 0):
            plot_losses(losses_epoch, os.path.join(opt.log_dir,'loss_curve.png'), opt.smooth_window)

        # 存 ckpt（也在 sample_every 時刻存一份）
        if (opt.sample_every > 0 and (epoch % opt.sample_every == 0)) or (epoch == opt.epochs):
            ckpt_path = os.path.join(opt.ckp_folder, f'epoch_{epoch:03d}.pkl')
            torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':optim.state_dict(),'loss':mean_loss}, ckpt_path)

        # 每 sample_every 取樣
        if opt.sample_every > 0 and (epoch % opt.sample_every == 0):
            ensure_dir(opt.sample_out)
            out_mid = os.path.join(opt.sample_out, f'epoch_{epoch:03d}_sample.mid')
            temp = opt.sample_temperature if opt.sample_temperature is not None else opt.temperature
            tk   = opt.sample_topk if opt.sample_topk is not None else opt.topk
            try:
                generate_sample(
                    model, device,
                    bars=opt.sample_bars, temperature=temp, topk=tk,
                    out_path=out_mid, prompt_midi=opt.prompt_midi,
                    mode=('custom' if opt.tokenizer=='custom' else 'miditok'),
                    dict_path=opt.dict_path, with_chord=bool(opt.with_chord),
                    miditok_ctx=miditok_ctx
                )
                print(f'[preview] wrote {out_mid}')
                if opt.render_wav and os.path.isfile(opt.sf2):
                    out_wav = swap_ext(out_mid, '.wav')
                    ok = render_midi_to_wav(out_mid, out_wav, opt.sf2, sr=opt.sr, method=opt.render_method)
                    print(f'[preview] {"rendered" if ok else "render failed"} {out_wav}')
                elif opt.render_wav:
                    print(f'[preview] skip render: sf2 not found -> {opt.sf2}')
            except Exception as e:
                print(f'[preview] failed: {e}')

def test(opt):
    device = torch.device(opt.device)
    ensure_dir(os.path.dirname(opt.out_midi) or '.')
    # 構 tokenizer & vocab
    if opt.tokenizer == 'custom':
        # 確保字典存在（允許 test 時自動建）
        test_list = collect_midi_paths(opt.midi_glob) if opt.midi_glob else []
        build_dictionary_if_needed(opt.dict_path, test_list, with_chord=bool(opt.with_chord))
        e2w, _ = pickle.load(open(opt.dict_path, 'rb'))
        vocab_size = int(max(e2w.values())) + 1
        miditok_ctx = None
    else:
        assert HAVE_MIDITOK, "需要 pip install miditok symusic"
        kind = 'remi_plus' if opt.tokenizer == 'miditok_remi_plus' else 'remi'
        cfg = TokenizerConfig(
            beat_res={(0,4):4}, num_velocities=32, use_chords=bool(opt.with_chord),
            use_rests=False, use_tempos=True,
            use_time_signatures=(kind=='remi_plus'),
            use_programs=(kind=='remi_plus'),
            one_token_stream_for_programs=(kind=='remi_plus'),
        )
        tokenizer = REMI(cfg)
        vocab_size = len(tokenizer)
        is_bar, _ = build_miditok_bar_checker(tokenizer)
        tempo_guard = build_miditok_tempo_guard(tokenizer, opt.tempo_bpm_min, opt.tempo_bpm_max)
        miditok_ctx = {'tokenizer': tokenizer, 'is_bar': is_bar, 'tempo_guard': tempo_guard}

    model = GPT2Wrapper(vocab_size=vocab_size, n_layer=opt.n_layer, n_head=opt.n_head,
                        n_embd=opt.n_embd, dropout=opt.dropout, max_len=BLOCK_LEN).to(device)
    ckpt = torch.load(opt.model_ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])

    out_mid = opt.out_midi
    temp = opt.temperature; tk = opt.topk; bars = opt.n_bars
    generate_sample(
        model, device,
        bars=bars, temperature=temp, topk=tk,
        out_path=out_mid, prompt_midi=opt.prompt_midi,
        mode=('custom' if opt.tokenizer=='custom' else 'miditok'),
        dict_path=opt.dict_path, with_chord=bool(opt.with_chord),
        miditok_ctx=miditok_ctx
    )
    print(f'[gen] wrote {out_mid}')
    if opt.render_wav and os.path.isfile(opt.sf2):
        out_wav = swap_ext(out_mid, '.wav')
        ok = render_midi_to_wav(out_mid, out_wav, opt.sf2, sr=opt.sr, method=opt.render_method)
        print(f'[gen] {"rendered" if ok else "render failed"} {out_wav}')
    elif opt.render_wav:
        print(f'[gen] skip render: sf2 not found -> {opt.sf2}')


# ------------------------- Main -------------------------
def main():
    global BLOCK_LEN
    opt = parse_opt()
    BLOCK_LEN = int(opt.max_len)  # 以 --max_len 作為 block 長度
    if BLOCK_LEN <= 0:
        raise ValueError('--max_len 需為正整數')

    torch.manual_seed(opt.seed); np.random.seed(opt.seed); random.seed(opt.seed)

    if opt.mode == 'train':
        train(opt)
    else:
        assert opt.model_ckpt, '--model_ckpt 必填（測試/生成模式）'
        test(opt)

if __name__ == '__main__':
    main()

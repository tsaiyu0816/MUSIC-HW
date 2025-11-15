# music_HW3 – GPT-2 for Symbolic Music (Pop1K7)

Generate **32-bar** MIDI phrases with a GPT-2–style language model trained on the Pop1K7 dataset (1747 MIDI files).

- **Task-1**: Train an autoregressive symbolic-music language model and generate 32-bar phrases.
- **Task-2**: Use the first **8 bars** as a conditioning prefix and let the Task-1 model continue for **24 bars** under multiple inference configs (A/B/C).

---

## Tokenizers

This repo supports:

- `custom` — REMI-style events defined in `utils.py`, stored in `dictionary.pkl`  
- `miditok_remi` — Miditok REMI  
- `miditok_remi_plus` — Miditok REMI+

> **Important:** the tokenizer and chord setting used for **training** must be reused at **inference**:
> - `--tokenizer {custom, miditok_remi, miditok_remi_plus}`
> - `--with_chord {0,1}`

---

## 1) Data & Assets
Download the Pop1K7 dataset and a General MIDI soundfont.

> If you already have them, place files to the shown paths and skip.

```bash
set -euo pipefail

mkdir -p Pop1K7 assets/sf2 tmp_dl
cd tmp_dl

# --- Pop1K7 dataset (Zenodo, official record) ---
if [ ! -f Pop1K7.zip ]; then
  echo "[get] Pop1K7.zip"
  curl -fL "https://zenodo.org/records/13167761/files/Pop1K7.zip?download=1" -o Pop1K7.zip \
  || wget -O Pop1K7.zip "https://zenodo.org/records/13167761/files/Pop1K7.zip?download=1"
fi
echo "[unzip] Pop1K7.zip -> ../Pop1K7"
unzip -oq Pop1K7.zip -d ../Pop1K7

# --- FluidR3 GM SoundFont (primary: KeyMusician S3 ZIP) ---
if [ ! -f FluidR3_GM.zip ]; then
  echo "[get] FluidR3_GM.zip"
  curl -fL "https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip" -o FluidR3_GM.zip \
  || wget -O FluidR3_GM.zip "https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip" || true
fi

if [ -f FluidR3_GM.zip ]; then
  echo "[unzip] FluidR3_GM.zip"
  unzip -oq FluidR3_GM.zip -d FluidR3_GM || true
  SF2_PATH="$(find FluidR3_GM -type f -iname '*.sf2' -print -quit || true)"
  if [ -n "${SF2_PATH:-}" ]; then
    cp -f "$SF2_PATH" ../assets/sf2/GM.sf2
  fi
fi

# --- Fallback: direct .sf2 (SourceForge) if ZIP route failed ---
if [ ! -f ../assets/sf2/GM.sf2 ]; then
  echo "[fallback] FluidR3_GM.sf2 direct download"
  curl -fL "https://sourceforge.net/projects/pianobooster/files/pianobooster/1.0.0/FluidR3_GM.sf2/download" -o ../assets/sf2/GM.sf2 \
  || wget -O ../assets/sf2/GM.sf2 "https://sourceforge.net/projects/pianobooster/files/pianobooster/1.0.0/FluidR3_GM.sf2/download"
fi

cd ..
[ -f assets/sf2/GM.sf2 ] && sha256sum assets/sf2/GM.sf2 | awk '{print $1}' > assets/sf2/sf2.sha256

echo
echo "[ok] Dataset path : $(realpath Pop1K7)"
echo "[ok] SoundFont    : $(realpath assets/sf2/GM.sf2)"
echo "[ok] SF2 checksum : $(realpath assets/sf2/sf2.sha256)"
```

Expected layout:

```text
Pop1K7/
└─ midi_analyzed/
   └─ */*.mid      # 1747 MIDI files

assets/
└─ sf2/
   └─ GM.sf2       # General MIDI SoundFont
```

---

## 2) Environment

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## 3) One-click pipeline

The entire Task-1 + Task-2 pipeline can be run via:

```bash
bash run.sh
```

The script performs:

1. Tokenization (e.g., Miditok REMI+ with chords)  
2. GPT-2 training (`max_len = 1024`, optional `--stride` for overlapping segments)  
3. Generate **20** 32-bar samples (`--temperature 0.9 --top_k 32`)  
4. Evaluate the 20 songs with **MusDr** metrics  
5. Run Task-2 continuation (8→24 bars) under configs A/B/C

To run another tokenizer setting (e.g., `miditok_remi_plus` vs `custom`), open `run.sh` and **uncomment** the corresponding block.

---

## 3.2 Train model

### 3.2.1 custom tokenizer (REMI + chord)

```bash
python main.py \
  --mode train \
  --tokenizer custom \
  --with_chord 1 \
  --dict_path ./results/hw3_task1_final/dictionary.pkl \
  --midi_glob "./Pop1K7/Pop1K7/midi_analyzed/*/*.mid" \
  --ckp_folder ./results/hw3_task1_final/ckpt \
  --log_dir   ./results/hw3_task1_final/logs \
  --epochs 100 --batch_size 8 --lr 3e-4 \
  --n_layer 12 --n_head 12 --n_embd 768 --max_len 1024 \
  --sample_every 10 --sample_bars 32 \
  --sample_out ./results/hw3_task1_final/samples \
  --render_wav 0 --workers 8
```

- Remove chords: `--with_chord 0`  
- Render preview `.wav` while training: set `--render_wav 1` (requires `assets/sf2/GM.sf2`)

### 3.2.2 Miditok REMI / REMI+

```bash
python main.py \
  --mode train \
  --tokenizer miditok_remi_plus \
  --with_chord 1 \
  --midi_glob "./Pop1K7/Pop1K7/midi_analyzed/*/*.mid" \
  --ckp_folder ./results/hw3_task1_final/ckpt \
  --log_dir   ./results/hw3_task1_final/logs \
  --epochs 100 \
  --batch_size 8 \
  --lr 3e-4 \
  --n_layer 12 --n_head 12 --n_embd 768 \
  --max_len 1024 \
  --stride 860 \
  --workers 8 \
  --sample_every 10 \
  --sample_bars 32 \
  --sample_out ./results/hw3_task1_final/samples \
  --render_wav 0
```

- `--tokenizer` can be `miditok_remi` or `miditok_remi_plus`  
- `--stride` < `--max_len` gives overlapping segments (more data per MIDI)

---

## 3.3 Task-1 Inference (32 bars)

Generate **20** samples with a trained checkpoint.

```bash
python generate.py \
  --ckpt ./results/hw3_task1_final/ckpt/epoch_040.pkl \
  --tokenizer remi_plus --with_chord 1 \
  --out_dir ./results/hw3_task1_final/generate_20 \
  --num 20 --bars 32 --temperature 0.9 --top_k 32 \
  --sf2 ./assets/sf2/GM.sf2 --sr 22050
```

- Ensure the checkpoint was trained with **REMI+** (since `--tokenizer remi_plus`) and `--with_chord 1`.  

---

## 3.4 Metrics for Task-1 (MusDr)

Evaluate the generated 20 songs using **MusDr** metrics.

### 3.4.1 custom tokenizer (dictionary-based)

```bash
python ./MusDr/eval_metrics.py \
  --dict_path ./results/hw3_task1_final/dictionary.pkl \
  --output_file_path ./results/hw3_task1_final/generate_20 \
  --out_csv ./results/hw3_task1_final/generate_20/metrics_musdr.csv \
  --max_pairs 1000 --verbose
```

### 3.4.2 miditok tokenizer (REMI+)

```bash
python ./MusDr/eval_metrics.py \
  --tokenizer remi_plus --with_chord 1 \
  --output_file_path ./results/hw3_task1_final/generate_20 \
  --out_csv ./results/hw3_task1_final/generate_20/metrics_musdr.csv \
  --max_pairs 1000 --verbose
```

Output CSV columns (examples):

- `piece_name` — MIDI filename  
- `H1`, `H4` — pitch entropy over 1-bar / 4-bar windows  
- `GS` — groove similarity  
- (optionally) other derived stats

These averages are used to compare:

- `REMI` vs `REMI+`  
- with vs without `--with_chord`  
- different hyper-parameters (`temperature`, `top_k`, etc.)

---

## 3.5 Task-2：Conditional continuation（8 → 24 bars）

For Task-2, we are given 3 prompt songs in `./prompt_song`:

```text
prompt_song/
├─ song_1.mid
├─ song_2.mid
└─ song_3.mid
```

We use the first **8 bars** of each song as the **conditioning prefix**, then generate the next **24 bars** using the Task-1 model under three inference configs: **A**, **B**, **C**.

```bash
python task2_generate.py \
  --tokenizer miditok_remi_plus \
  --ckpt ./results/hw3_task1_final/ckpt/epoch_030.pkl \
  --prompt_dir ./prompt_song \
  --out_dir results/hw3_task2/results_wchord_remi_plus_30/ \
  --bars_prompt 8 --bars_cont 24 \
  --with_chord 1 \
  --tempo_bpm_min 70 --tempo_bpm_max 180 \
  --configs A,B,C \
  --render_wav 1 --sf2 ./assets/sf2/GM.sf2
```

Notes:

- `--bars_prompt 8` — use first 8 bars of each prompt as prefix  
- `--bars_cont 24` — generate 24 bars of continuation and then stop  
- `--configs A,B,C` — three inference strategies (e.g., different `temperature`, `top_k`, sampling tricks)  
- For each prompt, outputs:
  - `cfgA.mid`, `cfgB.mid`, `cfgC.mid` (+ `.wav`) in the corresponding subfolder  

We then:

1. Use **MusDr Task-2** ranking scripts (e.g., `task2_rank.py`) to compute scores such as:
   - groove similarity to prompt (`GS_to_prompt`)
   - within-continuation groove (`GS_within`)
   - pitch histogram cosine (`PitchHistCos`)
   - register / density matching, etc.
2. Combine metrics with **subjective listening** to select the best config (or Top-2) for each prompt.

---

## 4) Repository Structure (simplified)

```text
music_HW3/
├─ Pop1K7/
│  └─ midi_analyzed/**.mid
├─ assets/
│  └─ sf2/GM.sf2
├─ MusDr/
│  ├─ eval_metrics.py
│  ├─ task2_rank.py
│  └─ musdr/side_utils.py
├─ results/
│  └─ hw3_task1_final/
│     ├─ ckpt/           # GPT-2 checkpoints
│     ├─ logs/           # loss logs + curves
│     ├─ samples/        # per-epoch preview MIDIs/WAVs
│     └─ generate_20/    # Task-1 generations + MusDr CSV
├─ prompt_song/
│  ├─ song_1.mid
│  ├─ song_2.mid
│  └─ song_3.mid
├─ main.py               # train / test GPT-2 (custom + miditok)
├─ generate.py           # batch generation for custom tokenizer
├─ task2_generate.py     # Task-2 continuation generator
├─ utils.py              # custom REMI tokenizer utilities
├─ run.sh                # one-click pipeline
└─ requirements.txt
```

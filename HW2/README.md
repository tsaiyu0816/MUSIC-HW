# DeepMIR HW2
_Controllable Text-to-Music & Retrieval project._

---

## Overview
This repository contains two main pipelines:

1) **Retrieval** — Encode **target** and **reference** audio with multiple encoders, retrieve the most similar reference via **cosine similarity**, then compute metrics and compare encoders.

2) **Controllable Text-to-Music (T2M)** — Use LLMs to create **audio captions** for each target, feed **text + optional audio features** into T2M models to generate music, then compute metrics and compare both LLMs and T2M backends.

---

## Environment
Install all Python dependencies:
```bash
pip install -r requirements.txt
```

---

## Data Layout (expected)
```
data/
├─ targets/        # target audio (.wav; consistent sample rate & channels recommended)
└─ references/     # reference audio for retrieval
```

---

## Retrieval
Retrieve the most similar **reference** for each **target** using embedding **cosine similarity**; use the retrieved clip as the proxy output and compute metrics to compare encoder quality.

**Encoders (under `Retrieval/`):** `CLAP`, `MuQ`, `Music2latent`, `Stable_audio`  
**Note:** `Stable_audio` was trained in **stereo** → use **2-channel** audio at inference.

### 1) Run retrieval (Top-1)
```bash
bash Retrieval/CLAP.sh
bash Retrieval/MuQ.sh
bash Retrieval/Music2latent.sh
bash Retrieval/Stable_audio.sh
```
**Output:** CSV files like `./runs/top1_clap.csv` (similar for other encoders).

### 2) Compute retrieval metrics (use Step-1 CSVs)
```bash
bash run_retrieval_metrics.sh
```
**Output:** per-encoder metrics, e.g., `./runs/metrics_clap.csv`.

### 3) Compare encoders & visualize
```bash
bash plot_compare.sh --dir ./runs
```
- Use `--dir` to point to the folder containing all `metrics_*.csv` you want to compare.  
- **Output:** comparison figures in `./runs/compare/`.

---

## Controllable Text-to-Music (T2M)
Use LLMs to caption each target audio; feed **text** and optional **audio features** to T2M backends; then compute metrics and compare across LLMs/models.

**LLMs for captioning:** `AF3 (Audio-Flamingo-3)`, `Qwen-audio`, `IP-MusicCaps`, `CLAP captions`  
**Backends:**
- `musicgen` (facebook/musicgen-small) — text only (~40 s)
- `musicgen_melody` — text + melody (**float32**, ~30 s)
- `musicgen_style` — style (extracted from target) + text (~60 s)
- `jasco` (jasco-chords-drums-melody-400M) — auto chords + melody + text (~10 s)
- `MuseControlLite` — text + rhythm + dynamics + melody (~47 s; default params)

> Example: generations using **AF3** prompts with backends (`jasco`, `musecontrollite`, `musicgen`, `musicgen_melody`, `musicgen_style`) can be stored under `Result_gen_af3/`.

### 1) Caption targets
```bash
bash ./t2m/caption_target.sh
```
- The script contains multiple LLM blocks — **uncomment** the one you want.  
- Use `--alm` to choose the LLM: `af3 | qwen | lpmusiccaps | clap`.  
- **Output:** in `./t2m/`, both **.json** and **.csv** (filenames follow the `--alm` value), one caption per target audio.

### 2) Generate music from captions (and optional features)
```bash
bash ./t2m/txt2music.sh
```
- The script contains several T2M backends — **uncomment** the one you want:
  - `musicgen | musicgen_melody | musicgen_style | jasco`
- Use `--backend` to select the T2M model.  
- **Output example (musicgen_melody):**
  - Audio: `./t2m/t2m_musicgen_melody/`
  - CSV (paths + prompts): `./t2m/t2m_musicgen_melody/t2m_musicgen_melody.csv`

#### MuseControlLite (separate runner)
```bash
cd MuseControlLite
pip install -r requirements.txt
bash batch_run.sh
```
- Use `--cond` to add controls such as `melody_mono`, `rhythm`, `dynamics`.  
- **Output:** audio and CSV under `./mc_out/`.

### 3) Evaluate T2M metrics (use Step-2 CSVs)
```bash
bash run_t2m_metrics.sh
```
- `--aesthetics_on generated|target` — choose whether to score **generated** or **target** audio quality.  
- `--out_csv <path>` — set the output CSV path.  
- **Output:** a metrics CSV (e.g., CLAP similarities, **CE/CU/PC/PQ**, `melody_acc`, etc.).

### 4) Compare T2M models & visualize
```bash
bash plot_compare.sh --dir <folder_with_metrics_csvs>
```
- Use `--dir` to point to the folder containing the metrics CSVs you want to compare.  
- **Output:** figures saved under `<folder>/compare/`.

---

## Repository Structure (simplified)
```
HW2/
├─ Retrieval/
│  ├─ CLAP.sh / MuQ.sh / Music2latent.sh / Stable_audio.sh
│  └─ (writes runs/top1_*.csv, runs/metrics_*.csv, runs/compare/*)
├─ t2m/
│  ├─ caption_target.sh      # Step 1: captioning
│  ├─ txt2music.sh           # Step 2: generation
│  ├─ t2m_*                  # per-backend outputs (audio + csv)
│  └─ prompts_*.json         # prompt sources
├─ MuseControlLite/          # optional backend (uses batch_run.sh)
├─ run_retrieval_metrics.sh  # retrieval metrics
├─ run_t2m_metrics.sh        # T2M metrics
├─ plot_compare.sh           # comparison & visualization
└─ requirements.txt
```

# HW1: Singer Classification
### Dataset:
- 20 artists, 120 albums, 1413 tracks
- Train / Validation / Test
- Album-level split : 4 / 1 / 1
- Tracks number: 949 / 231 / 233
```
artist20/
├─ train_val/                 
│  ├─ singer_1/
│  │  ├─ album_1/
│  │  │  ├─ song_1.mp3
│  │  │  ├─ song_2.mp3
│  │  │  └─ ...
│  │  ├─ album_2/
│  │  │  └─ ...
│  │  └─ ...
│  ├─ singer_2/
│  │  └─ ...
│  └─ singer_20/
│     └─ ...
└─ test/                     
   ├─ 001.mp3
   ├─ 002.mp3
   ├─ ...
   └─ 233.mp3
```

## Environment Setup
```bash
pip install -r requirements.txt
```
## Task1 : Train ML model
 **save validation metrics** and **test predictions** automatically
```bash
bash task1_run.sh
```
or
```bash
python task1_train_main.py \
  --train_json ./hw1/artist20/train.json \
  --val_json   ./hw1/artist20/val.json \
  --test_root  ./hw1/artist20/test \
  --clf svm  --segment_sec 15\
  --report_dir reports_svm_15 --ckpt_path checkpoints_svm_15/model_svm_15.pkl
```
### Run your scripts with configurable arguments
### Dataset paths
- `--train_json` *(str, default: `./hw1/artist20/train.json`)*  
  JSON list of **training** file paths with labels (your own format).
- `--val_json` *(str, default: `./hw1/artist20/val.json`)*  
  JSON list of **validation** file paths with labels.
- `--test_root` *(str, default: `./hw1/artist20/test`)*  
  Folder that contains **unlabeled** flat `.wav` files for inference.

### Training
- `--clf` *(str, default: `svm`)*  
  Classifier to use. Choices: `svm | knn | rf | lr | xgb`.
- `--segment_sec` *(int, default: `30`)*  
  Audio segment length (seconds) used during feature extraction/inference voting.
- `--cache_dir` *(str, default: `cache`)*  
  Directory to store cached intermediate features to speed up reruns.
- `--report_dir` *(str, default: `reports`)*  
  Directory to save classification reports, confusion matrices, logs, etc.
- `--xgb_es_rounds` *(int, default: `50`)*  
  Early-stopping rounds for XGBoost; set `0` or negative to disable.  
  *Note:* Requires XGBoost installed; if unsupported in your setup, code falls back to no early stop.
- `--ckpt_path` *(str, default: `checkpoints/model.pkl`)*  
  Path to save the trained model checkpoint (`.pkl`, sklearn pipeline serialization).

## Task2 : Train DL model
```bash
bash task2_run.sh
```
or
```bash
python task2_train_main.py \
  --train_json ./hw1/artist20/train.json \
  --val_json   ./hw1/artist20/val.json \
  --lr 1e-4 --batch_size 16 --epochs 30 --segment_sec 15 --report_dir reports_task2
```
### Run your scripts with configurable arguments
### Dataset paths (required)
- `--train_json` *(str, required)*  
  JSON list of **training** file paths with labels.
- `--val_json` *(str, required)*  
  JSON list of **validation** file paths with labels.

### Segmentation & Voting
- `--segment_sec` *(int, default: `10`)*  
  Audio segment length (seconds) used to chunk clips.
- `--overlap` *(float, default: `0.0`)*  
  Overlap ratio between consecutive segments (`0.0`–`<1.0`).
- `--vote_method` *(str, default: `mean`)*  
  Clip-level aggregation of segment predictions. Choices: `mean | majority`.

### Training
- `--batch_size` *(int, default: `16`)*  
  Mini-batch size.
- `--epochs` *(int, default: `40`)*  
  Maximum training epochs.
- `--lr` *(float, default: `1e-3`)*  
  Initial learning rate.
- `--patience` *(int, default: `10`)*  
  Early-stopping patience (epochs without val improvement).

### I/O 
- `--report_dir` *(str, default: `reports_task2`)*  
  Directory to write logs/reports/plots.
- `--ckpt_path` *(str, default: `checkpoints/task2_scnn.pt`)*  
  Path to save the best model checkpoint (PyTorch)

## Task2 : Inference DL model
### After training is completed (assuming the best ckpt is in checkpoints/task2_best.pt)
```bash
bash task2_infer.sh
```
or
```bash
python task2_infer.py \
  --ckpt checkpoints_final/task2_best_0.94.pt \
  --test_root ./hw1/artist20/test \
  --out_json reports_task2/r13942126.json \
  --overlap 0.5
```
### Run your scripts with configurable arguments
### Inference arguments

- `--ckpt` *(str, required)*  
  Path to the trained checkpoint, e.g. `task2_best.pt` or `task2_scnn.pt`.

- `--test_root` *(str, required)*  
  Folder containing test audio files (e.g., `001.mp3` ~ `NNN.mp3`), flat layout.

- `--cache_dir` *(str, default: `cache_task2_infer`)*  
  Directory for caching intermediate features during inference.

- `--batch_size` *(int, default: `32`)*  
  Batch size used by the inference DataLoader.

- `--num_workers` *(int, default: `0`)*  
  Number of DataLoader workers. `0` is recommended for stable evaluation.

- `--vote_method` *(str, default: `mean`)*  
  Clip-level aggregation of segment predictions. Choices: `mean | majority`.

- `--out_json` *(str, default: `infer_top3.json`)*  
  Output JSON file path (e.g., per-file top-k predictions/probabilities).

- `--overlap` *(float, default: `None`)*  
  Segment overlap ratio in `[0, 1)`. If not set, uses the overlap from the training config.

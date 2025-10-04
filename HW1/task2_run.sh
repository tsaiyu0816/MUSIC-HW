export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python task2_train_main.py \
  --train_json ./hw1/artist20/train.json \
  --val_json   ./hw1/artist20/val.json \
  --lr 1e-4 --batch_size 16 --epochs 30 --segment_sec 15 --report_dir reports_task2
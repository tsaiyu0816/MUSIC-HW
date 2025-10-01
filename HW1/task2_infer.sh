# 訓練完成後（假設 best ckpt 在 checkpoints/task2_best.pt）
python task2_infer.py \
  --ckpt checkpoints/task2_best_94.pt \
  --test_root ./hw1/artist20/test \
  --out_json infer_top3.json \
  --sanity_val_json ./hw1/artist20/val.json

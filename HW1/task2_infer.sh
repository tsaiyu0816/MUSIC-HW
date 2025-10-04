# 訓練完成後（假設 best ckpt 在 checkpoints/task2_best.pt）
python task2_infer.py \
  --ckpt checkpoints_final/task2_best_0.94.pt \
  --test_root ./hw1/artist20/test \
  --out_json reports_task2/r13942126.json \
  --overlap 0.5
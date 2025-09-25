python task1_train_main.py \
  --train_json ./hw1/artist20/train.json \
  --val_json   ./hw1/artist20/val.json \
  --test_root  ./hw1/artist20/test \
  --clf xgb --xgb_es_rounds 50 \
  --segment_sec 30

python task1_train_main.py \
  --train_json ./hw1/artist20/train.json \
  --val_json   ./hw1/artist20/val.json \
  --test_root  ./hw1/artist20/test \
  --clf svm  --segment_sec 15\
  --report_dir reports_svm_15 --ckpt_path checkpoints_svm_15/model_svm_15.pkl

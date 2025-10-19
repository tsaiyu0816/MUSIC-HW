# 1) 跑 targets
python ./Retrieval/MuQ.py \
  --input_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
  --out_dir ./emb/muq_targets

# 2) 跑 references
python ./Retrieval/MuQ.py \
  --input_dir ./home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s \
  --out_dir ./emb/muq_refs

# 3) 算相似度（每個 target 最像的 ref）
python ./Retrieval/utils.py \
  --targets ./emb/muq_targets \
  --refs ./emb/muq_refs \
  --out_csv ./runs/top1_muq.csv

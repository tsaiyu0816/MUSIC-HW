# 1) 跑 targets
python ./Retrieval/CLAP.py \
  --input_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
  --out_dir ./emb/clap_targets

# 2) 跑 references
python ./Retrieval/CLAP.py \
  --input_dir ./home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s \
  --out_dir ./emb/clap_refs

# 3) 算相似度（每個 target 最像的 ref）
python ./Retrieval/utils.py \
  --targets ./emb/clap_targets \
  --refs ./emb/clap_refs \
  --out_csv ./runs/top1_clap.csv

# 1) 跑 targets
python ./Retrieval/Music2latent.py \
  --input_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
  --out_dir ./emb/m2l_targets

# 2) 跑 references
python ./Retrieval/Music2latent.py \
  --input_dir ./home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s \
  --out_dir ./emb/m2l_refs

# 3) 算相似度（每個 target 最像的 ref）
python ./Retrieval/utils.py \
  --targets ./emb/m2l_targets \
  --refs ./emb/m2l_refs \
  --out_csv ./runs/top1_m2l.csv

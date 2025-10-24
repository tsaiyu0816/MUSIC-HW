python metrics.py \
  --pairs_csv ./runs/top1_stable_audio.csv \
  --target_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
  --ref_dir    ./home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s \
  --out_csv    ./runs/metrics_stable_audio.csv \
  --device cuda

python metrics.py \
  --pairs_csv ./runs/top1_m2l.csv \
  --target_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
  --ref_dir    ./home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s \
  --out_csv    ./runs/metrics_m2l.csv \
  --device cuda

python metrics.py \
  --pairs_csv ./runs/top1_clap.csv \
  --target_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
  --ref_dir    ./home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s \
  --out_csv    ./runs/metrics_clap.csv \
  --device cuda

python metrics.py \
  --pairs_csv ./runs/top1_muq.csv \
  --target_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
  --ref_dir    ./home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s \
  --out_csv    ./runs/metrics_muq.csv \
  --device cuda
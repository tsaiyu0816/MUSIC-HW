
python metrics_t2m.py \
  --pairs_csv /home/tsmc-tsai/music_hw/HW2/MuseControlLite/mc_out/batch_result_qwen_1.csv \
  --out_csv results_qwen_1/metrics_MuseControlLite.csv \
  --base_root /home/tsmc-tsai/music_hw/HW2 \
  --device cuda \
  --sr 44100 \
  --aesthetics_on generated     # 可選: target / both

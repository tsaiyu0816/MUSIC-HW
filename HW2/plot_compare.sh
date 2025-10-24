# # 1) 指定一批檔案
# python plot_t2m_metrics.py \
#   --csvs /path/metrics_jasco.csv /path/metrics_musicgen.csv /path/metrics_other.csv \
#   --out plots

# 2) 或掃描資料夾下所有 CSV（檔名須以 metrics_ 開頭）
python plot_compare.py --dir /home/tsmc-tsai/music_hw/HW2/runs --out /home/tsmc-tsai/music_hw/HW2/runs/compare

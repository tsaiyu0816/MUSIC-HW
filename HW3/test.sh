# python generate.py \
#   --ckpt /home/tsmc-tsai/music_hw/HW3/results_wchord_custom/hw3_task1_final/ckpt/epoch_040.pkl \
#   --tokenizer custom --with_chord 1\
#   --dict_path /home/tsmc-tsai/music_hw/HW3/results_wchord_custom/hw3_task1_final/dictionary.pkl \
#   --out_dir ./results/hw3_task1_final/generate_20_10 \
#   --num 20 --bars 32 --temperature 0.9 --top_k 32 \
#   --sf2 ./assets/sf2/GM.sf2 --sr 22050

# # REMI+
# python generate.py \
#   --ckpt /home/tsmc-tsai/music_hw/HW3/results_remi_plus/hw3_task1_final/ckpt/epoch_040.pkl \
#   --tokenizer remi_plus --with_chord 1 \
#   --out_dir ./results/hw3_task1_final/generate_20_12 \
#   --num 20 --bars 32 --temperature 0.9 --top_k 32 \
#   --sf2 ./assets/sf2/GM.sf2 --sr 22050

# python ./MusDr/eval_metrics.py \
#   --tokenizer custom --with_chord 1\
#   --dict_path /home/tsmc-tsai/music_hw/HW3/results_wchord_custom/hw3_task1_final/dictionary.pkl \
#   --output_file_path ./results/hw3_task1_final/generate_20_10 \
#   --out_csv ./results/hw3_task1_final/generate_20_10/metrics_musdr.csv \
#   --max_pairs 1000 --verbose

# # REMI+
# python ./MusDr/eval_metrics.py \
#   --tokenizer remi_plus --with_chord 1 \
#   --output_file_path ./results/hw3_task1_final/generate_20_12 \
#   --out_csv ./results/hw3_task1_final/generate_20_12/metrics_musdr.csv \
#   --max_pairs 1000 --verbose

# python mean_metrics.py --csv ./results/hw3_task1_final/generate_20_12/metrics_musdr.csv \
#  --columns H1,H4,GS \
#  --out ./results/hw3_task1_final/generate_20_12/mean_metrics.csv



# config setup default => use --configs
# CONFIGS = {
#     'A': dict(temperature=0.9, topk=32,   topp=0.95, rep_penalty=1.10, no_repeat_ngram=4, recent_k=120),
#     'B': dict(temperature=0.9, topk=32,  topp=0.90, rep_penalty=1.00, no_repeat_ngram=4, recent_k=100),
#     'C': dict(temperature=0.9, topk=32,  topp=0.85, rep_penalty=1.05, no_repeat_ngram=4, recent_k=200),
# }

# python task2_generate.py \
#   --tokenizer custom \
#   --dict_path /home/tsmc-tsai/music_hw/HW3/results_wchord_custom/hw3_task1_final/dictionary.pkl \
#   --ckpt /home/tsmc-tsai/music_hw/HW3/results_wchord_custom/hw3_task1_final/ckpt/epoch_030.pkl \
#   --prompt_dir ./prompt_song \
#   --out_dir results/hw3_task2/results_wchord_custom_30/ \
#   --bars_prompt 8 --bars_cont 24 \
#   --with_chord 1 \
#   --configs A,B,C \
#   --render_wav 1 --sf2 ./assets/sf2/GM.sf2

# # REMI+
python task2_generate.py \
  --tokenizer miditok_remi_plus \
  --ckpt /home/tsmc-tsai/music_hw/HW3/results_remi_plus/hw3_task1_final/ckpt/epoch_030.pkl \
  --prompt_dir ./prompt_song \
  --out_dir results/hw3_task2/results_wchord_remi_plus_30/ \
  --bars_prompt 8 --bars_cont 24 \
  --with_chord 1 \
  --tempo_bpm_min 70 --tempo_bpm_max 180 \
  --configs A,B,C \
  --render_wav 1 --sf2 ./assets/sf2/GM.sf2
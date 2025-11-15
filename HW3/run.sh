########################
#task1
########################
########################
#train
########################
python main.py \
  --mode train \
  --tokenizer custom \
  --with_chord 1 \
  --dict_path ./results/hw3_task1_final/dictionary.pkl \
  --midi_glob "./Pop1K7/midi_analyzed/*/*.mid" \
  --ckp_folder ./results/hw3_task1_final/ckpt \
  --log_dir   ./results/hw3_task1_final/logs \
  --epochs 100 --batch_size 8 --lr 3e-4 \
  --n_layer 12 --n_head 12 --n_embd 768 --max_len 1024 \
  --sample_every 10 --sample_bars 32 \
  --sample_out ./results/hw3_task1_final/samples \
  --render_wav 0 --workers 8

# #REMI+
# python main.py \
#   --mode train \
#   --tokenizer miditok_remi_plus \
#   --with_chord 1 \
#   --midi_glob "./Pop1K7/midi_analyzed/*/*.mid" \
#   --ckp_folder ./results/hw3_task1_final/ckpt \
#   --log_dir   ./results/hw3_task1_final/logs \
#   --epochs 100 \
#   --batch_size 8 \
#   --lr 3e-4 \
#   --n_layer 12 --n_head 12 --n_embd 768 \
#   --max_len 1024 \
#   --stride 860 \
#   --workers 8 \
#   --sample_every 10 \
#   --sample_bars 32 \
#   --sample_out ./results/hw3_task1_final/samples \
#   --render_wav 0

########################
# generate and eval
########################

python generate.py \
  --ckpt ./results/hw3_task1_final/ckpt/epoch_040.pkl \
  --tokenizer custom --with_chord 1\
  --dict_path ./results/hw3_task1_final/dictionary.pkl \
  --out_dir ./results/hw3_task1_final/generate_20 \
  --num 20 --bars 32 --temperature 0.9 --top_k 32 \
  --sf2 ./assets/sf2/GM.sf2 --sr 22050

# # REMI+
# python generate.py \
#   --ckpt ./results/hw3_task1_final/ckpt/epoch_040.pkl \
#   --tokenizer remi_plus --with_chord 1 \
#   --out_dir ./results/hw3_task1_final/generate_20 \
#   --num 20 --bars 32 --temperature 0.9 --top_k 32 \
#   --sf2 ./assets/sf2/GM.sf2 --sr 22050

########################################################################

python ./MusDr/eval_metrics.py \
  --tokenizer custom --with_chord 1\
  --dict_path ./results/hw3_task1_final/dictionary.pkl \
  --output_file_path ./results/hw3_task1_final/generate_20 \
  --out_csv ./results/hw3_task1_final/generate_20/metrics_musdr.csv \
  --max_pairs 1000 --verbose

# # REMI+
# python ./MusDr/eval_metrics.py \
#   --tokenizer remi_plus --with_chord 1 \
#   --output_file_path ./results/hw3_task1_final/generate_20 \
#   --out_csv ./results/hw3_task1_final/generate_20/metrics_musdr.csv \
#   --max_pairs 1000 --verbose

########################################################################

python mean_metrics.py --csv ./results/hw3_task1_final/generate_20/metrics_musdr.csv \
 --columns H1,H4,GS \
 --out ./results_wchord_custom/hw3_task1_final/generate_20/mean_metrics.csv

########################
#task2
########################
# config setup default => use --configs
# 'A': dict(temperature=0.9, topk=32,   topp=0.95, rep_penalty=1.10, no_repeat_ngram=4, recent_k=120),
# 'B': dict(temperature=0.9, topk=32,  topp=0.90, rep_penalty=1.00, no_repeat_ngram=4, recent_k=100),
# 'C': dict(temperature=0.9, topk=32,  topp=0.85, rep_penalty=1.05, no_repeat_ngram=4, recent_k=200),


python task2_generate.py \
  --tokenizer custom \
  --dict_path ./results/hw3_task1_final/dictionary.pkl \
  --ckpt ./results/hw3_task1_final/ckpt/epoch_030.pkl \
  --prompt_dir ./prompt_song \
  --out_dir results/hw3_task2/results_wchord_custom_30/ \
  --bars_prompt 8 --bars_cont 24 \
  --with_chord 1 \
  --configs A,B,C \
  --render_wav 1 --sf2 ./assets/sf2/GM.sf2

# # REMI+
# python task2_generate.py \
#   --tokenizer miditok_remi_plus \
#   --ckpt ./results/hw3_task1_final/ckpt/epoch_030.pkl \
#   --prompt_dir ./prompt_song \
#   --out_dir results/hw3_task2/results_wchord_remi_plus_30/ \
#   --bars_prompt 8 --bars_cont 24 \
#   --with_chord 1 \
#   --tempo_bpm_min 70 --tempo_bpm_max 180 \
#   --configs A,B,C \
#   --render_wav 1 --sf2 ./assets/sf2/GM.sf2

########################################################################

# # select best song
# python ./MusDr/task2_rank.py \
#   --prompt ./prompt_song/song_1.mid \
#   --gen remiplus:./results/hw3_task2/results_wchord_remi_plus_30/song_1 \
#   --gen chd:./results/hw3_task2/results_wchord_custom_30/song_1 \
#   --gen wochd:./results/hw3_task2/results_wochord_custom_40/song_1 \
#   --out_csv ./results/hw3_task2/task2_scores_song1.csv

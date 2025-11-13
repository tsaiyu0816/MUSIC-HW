python main.py \
  --mode train \
  --tokenizer custom \
  --with_chord 1 \
  --dict_path results/hw3_task1_final/dictionary.pkl \
  --midi_glob "../HW3/Pop1K7/midi_analyzed/*/*.mid" \
  --ckp_folder results/hw3_task1_final/ckpt \
  --log_dir   results/hw3_task1_final/logs \
  --epochs 100 --batch_size 8 --lr 3e-4 \
  --n_layer 12 --n_head 12 --n_embd 768 --max_len 1024 \
  --sample_every 10 --sample_bars 32 \
  --sample_out results/hw3_task1_final/samples \
  --render_wav 0 --workers 8


# python main.py \
#   --mode test \
#   --tokenizer custom \
#   --with_chord 1 \
#   --dict_path results/hw3_task1_final/dictionary.pkl \
#   --model_ckpt results/hw3_task1_final/ckpt/epoch_100.pkl \
#   --out_midi   results/hw3_task1_final/gen/ep100_32bars.mid \
#   --n_bars 32 --temperature 1.0 --topk 0 \
#   --render_wav 0 --workers 8

# python main.py \
#   --mode train \
#   --tokenizer miditok_remi_plus \
#   --with_chord 1 \
#   --midi_glob "../HW3/Pop1K7/midi_analyzed/*/*.mid" \
#   --ckp_folder results/hw3_miditok/ckpt \
#   --log_dir   results/hw3_miditok/logs \
#   --epochs 50 --batch_size 8 --lr 3e-4 \
#   --sample_every 10 --sample_bars 32 \
#   --sample_out results/hw3_miditok/samples \
#   --tempo_bpm_min 90 --tempo_bpm_max 160 \
#   --render_wav 0 --workers 8


# python generate.py \
#   --ckpt /home/tsmc-tsai/music_hw/sample_code/Result_wochord/checkpoints/epoch_100.pkl \
#   --dict_path ./basic_event_dictionary.pkl \
#   --out_dir results/hw3_train/samples \
#   --num 20 --bars 32 --temperature 0.9 --top_k 32 \
#   --sf2 assets/sf2/GM.sf2 --sr 22050


# python eval_metrics.py \
#   --dict_path ../basic_event_dictionary.pkl \
#   --output_file_path ../Result_wochord/hw3_train/samples \
#   --out_csv ../Result_wochord/hw3_train/samples/metrics_musdr.csv \
#   --max_pairs 1000 --verbose

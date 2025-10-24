# # use clap
# python ./t2m/caption_target.py \
#   --input_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
#   --alm clap \
#   --out_json ./t2m/prompts_clap.json \
#   --out_csv  ./t2m/prompts_clap.csv \
#   --device cuda \
#   --genre_k 1 --mood_k 2 --inst_k 2

# # use ALM: audioflamingo3
# python ./t2m/caption_target.py \
#   --input_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
#   --alm audioflamingo3 \
#   --out_json ./t2m/prompts_af3.json \
#   --out_csv  ./t2m/prompts_af3.csv

# # use ALM: Qwen-audio
python ./t2m/caption_target.py \
  --input_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
  --alm qwen \
  --out_json ./t2m/prompts_qwen_3.json \
  --out_csv  ./t2m/prompts_qwen_3.csv


# use ALM: lp-musiccaps
# export LPMC_REPO=/home/tsmc-tsai/repos/lp-music-caps
# export LPMC_CKPT=/home/tsmc-tsai/models/lpmc/transfer.pth
# python ./t2m/caption_target.py \
#   --input_dir ./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s \
#   --alm lp-musiccaps \
#   --out_json ./t2m/prompts_lp-musiccaps_1.json \
#   --out_csv  ./t2m/prompts_lp-musiccaps_1.csv
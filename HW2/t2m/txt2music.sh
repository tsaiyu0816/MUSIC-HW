# 讀前一步 AF3/Qwen 產生的 JSON（有 prompt）

# # musicgen_melody
python t2m/txt2music.py \
  --json_in ./t2m/prompts_clap.json \
  --backend musicgen_melody \
  --out_dir ./t2m/t2m_musicgen_melody \
  --out_csv ./t2m/t2m_musicgen_melody/t2m_musicgen_melody.csv \
  --max_new_tokens 2048 \
  
# # musicgen_style
# python t2m/txt2music.py \
#   --json_in ./t2m/prompts_af3.json \
#   --backend musicgen_style \
#   --out_dir ./t2m/t2m_musicgen_style \
#   --out_csv ./t2m/t2m_musicgen_style.csv \
#   --max_new_tokens 2048 \

# #jasco
# python - <<'PY'
# import os, pickle
# maj = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
# mapping = {name:i for i,name in enumerate(maj)}
# mapping["N"] = len(mapping)  # no-chord
# out = "/home/tsmc-tsai/models/jasco/chord_to_index_mapping.pkl"
# os.makedirs(os.path.dirname(out), exist_ok=True)
# with open(out, "wb") as f:
#     pickle.dump(mapping, f)
# print("Wrote:", out, " | size:", len(mapping), "| keys:", sorted(mapping.keys()))
# PY
# export JASCO_CHORDS_MAP="/home/tsmc-tsai/models/jasco/chord_to_index_mapping.pkl"
# python t2m/txt2music.py \
#   --json_in ./t2m/prompts_af3.json \
#   --backend jasco \
#   --out_dir ./t2m/t2m_jasco_1 \
#   --out_csv ./t2m/t2m_jasco.csv \

# #musecontrolite
# export MUSECTRL_REPO=./MuseControlLite
# PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
# python t2m/txt2music.py \
#   --json_in ./t2m/prompts_af3.json \
#   --backend musecontrol \
#   --out_dir ./t2m/t2m_musecontrol \
#   --out_csv ./t2m/t2m_musecontrol.csv \
#   --duration 12 --device cuda \
#   --mc_arg="--cfg=default.yaml"

from pathlib import Path
from typing import List, Dict, Any
import json
import csv
import soundfile as sf
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

def list_audio(p: Path) -> List[Path]:
    files: List[Path] = []
    for ext in AUDIO_EXTS:
        files += list(p.rglob(f"*{ext}"))
    return sorted({f.resolve() for f in files})

def save_outputs(items: List[Dict[str, Any]], out_json: str, out_csv: str):
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    # JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    # CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target", "prompt", "alm"])
        for it in items:
            w.writerow([it["target"], it["prompt"], it["alm"]])
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read many CSVs with the same columns (e.g., produced by your evaluation).
File name matters: we extract the method label from 'metrics_<label>.csv'.
We average across ALL rows (= all targets) per CSV for each metric:
  CLAP_t2a_target, CLAP_t2a_gen, CLAP_a2a, CE_gen, CU_gen, PC_gen, PQ_gen, melody_acc
Then we create 8 comparison bar charts (one per metric), highlighting 'jasco'.

Outputs:
- aggregated_means.csv (rows = methods, cols = metrics)
- One PNG per metric inside --out directory

Usage:
  python plot_t2m_metrics.py --csvs metrics_jasco.csv metrics_musicgen.csv --out plots
  python plot_t2m_metrics.py --dir /some/dir --out plots
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

METRICS = [
    "CLAP_t2a_target",
    "CLAP_t2a_gen",
    "CLAP_a2a",
    "CE_gen",
    "CU_gen",
    "PC_gen",
    "PQ_gen",
    "melody_acc",
    "CLAP_cosine",
    "CE",
    "CU",
    "PC",
    "PQ",
]

TITLES = {
    "CLAP_t2a_target": "CLAP (Text→Target)",
    "CLAP_t2a_gen": "CLAP (Text→Generated)",
    "CLAP_a2a": "CLAP (Audio→Audio)",
    "CE_gen": "Content Enjoyment",
    "CU_gen": "CLIP (CU) (Generated)",
    "PC_gen": "Production Complexity (Generated)",
    "PQ_gen": "Perceptual Quality (Generated)",
    "melody_acc": "Melody Accuracy",
}

def extract_label(path: Path) -> str:
    """Get method label from file name, e.g. metrics_jasco.csv -> jasco."""
    stem = path.stem  # metrics_jasco
    m = re.match(r"metrics[_-](.+)$", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return stem

def read_and_mean(csv_path: Path) -> pd.Series:
    """Read a CSV and return the mean of the METRICS columns (numeric-only)."""
    # Robust read: skip bad lines, keep UTF-8 BOM safe
    df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
    # Force numeric for our metrics; non-numeric -> NaN
    for col in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # Missing column -> all NaN
            df[col] = np.nan
    means = df[METRICS].mean(skipna=True)  # average across rows (targets)
    means["__n_rows__"] = len(df)
    return means

def make_bar(metric: str, summary: pd.DataFrame, out_dir: Path, dpi: int = 150,
             sort: str = "desc"):
    """Draw a comparison bar chart for a single metric across methods."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = summary[metric].copy()
    # Sorting
    if sort.lower() == "desc":
        data = data.sort_values(ascending=False)
    elif sort.lower() == "asc":
        data = data.sort_values(ascending=True)
    # else: keep original order

    labels = data.index.tolist()
    values = data.values.astype(float)

    # Colors: default; highlight any label containing 'jasco'
    colors = ["C0"] * len(labels)
    for i, lab in enumerate(labels):
        if "jasco" in lab.lower():
            colors[i] = "C3"  # highlight

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(TITLES.get(metric, metric))
    ax.set_ylabel("Mean over targets")
    ax.set_xlabel("Method (from file name)")
    ax.set_xticklabels(labels, rotation=20, ha="right")

    # Annotate values on top of bars
    ymax = np.nanmax(values) if len(values) else 1.0
    for b, v in zip(bars, values):
        if np.isnan(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
    fig.tight_layout()
    out_path = out_dir / f"{metric}_comparison.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="*", help="List of CSV files (metrics_*.csv).")
    ap.add_argument("--dir", type=str, default=None, help="Directory to scan for CSVs (metrics_*.csv).")
    ap.add_argument("--out", type=str, required=True, help="Output directory for plots/summary.")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    ap.add_argument("--sort", type=str, default="desc",
                    choices=["desc", "asc", "none"], help="Sort bars by value.")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = []
    if args.csvs:
        csv_paths.extend([Path(p).expanduser().resolve() for p in args.csvs])
    if args.dir:
        d = Path(args.dir).expanduser().resolve()
        csv_paths.extend(sorted(d.glob("metrics_*.csv")))
        csv_paths.extend(sorted(d.glob("metrics_*.CSV")))

    # de-dup while preserving order
    seen = set()
    unique_paths = []
    for p in csv_paths:
        if p.exists() and p.suffix.lower() == ".csv" and str(p) not in seen:
            unique_paths.append(p)
            seen.add(str(p))

    if not unique_paths:
        raise SystemExit("No CSVs found. Use --csvs or --dir with files named like metrics_<label>.csv")

    # Build summary: rows = method label, cols = METRICS
    records = []
    index = []
    counts = []
    for p in unique_paths:
        label = extract_label(p)
        s = read_and_mean(p)
        records.append(s[METRICS].values)
        index.append(label)
        counts.append(int(s.get("__n_rows__", 0)))

    summary = pd.DataFrame(records, index=index, columns=METRICS)
    summary.index.name = "method"
    summary["n_rows"] = counts  # number of targets aggregated (for reference)

    # Save aggregated table
    summary_path = out_dir / "aggregated_means.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")
    print(f"[OK] Wrote summary: {summary_path}")

    # One plot per metric
    for metric in METRICS:
        out_path = make_bar(metric, summary, out_dir, dpi=args.dpi, sort=args.sort)
        print(f"[OK] Wrote plot: {out_path}")

if __name__ == "__main__":
    main()

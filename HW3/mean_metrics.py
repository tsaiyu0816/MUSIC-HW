#!/usr/bin/env python3
# mean_csv.py — read CSV(s) and compute column-wise means
import argparse, csv, glob, math, os, sys
from collections import defaultdict

def is_number(x: str) -> bool:
    try:
        v = float(x)
        # treat NaN/Inf as non-numeric for averaging
        return math.isfinite(v)
    except Exception:
        return False

def read_header(path):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if row and not row[0].startswith("#"):
                return [c.strip() for c in row]
    return []

def compute_means_one(path, cols=None, skip_cols=("piece_name", "name")):
    header = read_header(path)
    if not header:
        raise SystemExit(f"[err] empty or invalid CSV: {path}")

    # decide columns: all numeric columns except those in skip_cols
    if cols is None:
        cols = [c for c in header if c not in skip_cols]

    idx = [header.index(c) for c in cols if c in header]
    keep_cols = [header[i] for i in idx]
    sums = defaultdict(float)
    counts = defaultdict(int)

    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        # skip header
        first = True
        for row in r:
            if first:
                first = False
                continue
            if not row or row[0].startswith("#"):
                continue
            for h, i in zip(keep_cols, idx):
                if i < len(row) and is_number(row[i].strip()):
                    v = float(row[i])
                    sums[h] += v
                    counts[h] += 1

    means = {}
    for h in keep_cols:
        n = counts[h]
        means[h] = (sums[h] / n) if n > 0 else float("nan")
    return means

def merge_means(list_of_means):
    # compute overall means by aggregating sums/counts again
    sums = defaultdict(float)
    counts = defaultdict(int)
    for m in list_of_means:
        for k, v in m.items():
            if not math.isfinite(v):
                continue
            sums[k] += v
            counts[k] += 1
    overall = {}
    for k in sorted(sums.keys()):
        n = counts[k]
        overall[k] = (sums[k] / n) if n > 0 else float("nan")
    return overall

def main():
    ap = argparse.ArgumentParser(description="Compute column-wise means from CSV.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="single CSV file to read")
    g.add_argument("--glob", help="glob pattern for multiple CSVs (e.g., 'results/*.csv')")
    ap.add_argument("--columns", help="comma-separated column names; default=all numeric except piece_name")
    ap.add_argument("--round", type=int, default=6, help="decimal places to print")
    ap.add_argument("--out", help="optional output CSV for the means")
    args = ap.parse_args()

    files = []
    if args.csv:
        files = [args.csv]
    else:
        files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit("[err] no CSV matched.")

    cols = [c.strip() for c in args.columns.split(",")] if args.columns else None

    all_results = []
    for fp in files:
        means = compute_means_one(fp, cols=cols)
        all_results.append((fp, means))

    # print per-file
    for fp, means in all_results:
        print(f"\n=== {fp} ===")
        for k in sorted(means.keys()):
            v = means[k]
            s = "nan" if not math.isfinite(v) else f"{v:.{args.round}f}"
            print(f"{k}: {s}")

    # print overall when multiple files
    if len(all_results) > 1:
        overall = {}
        # Aggregate by recomputing sums / counts per value cell-wise across files:
        # here we use simple mean of means (same weight per file). If you want
        # sample-count-weighted average, compute in compute_means_one with raw sums.
        overall = merge_means([m for _, m in all_results])
        print("\n=== OVERALL (mean of file means) ===")
        for k in sorted(overall.keys()):
            v = overall[k]
            s = "nan" if not math.isfinite(v) else f"{v:.{args.round}f}"
            print(f"{k}: {s}")

    # optional CSV out
    if args.out:
        # If single file → one row; multi-files → overall row + per-file rows
        import csv as _csv
        keys = sorted(all_results[0][1].keys()) if all_results else []
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["source"] + keys)
            for fp, means in all_results:
                row = [os.path.basename(fp)]
                for k in keys:
                    v = means.get(k, float("nan"))
                    row.append("" if not math.isfinite(v) else f"{v:.{args.round}f}")
                w.writerow(row)
            if len(all_results) > 1:
                overall = merge_means([m for _, m in all_results])
                row = ["OVERALL"]
                for k in keys:
                    v = overall.get(k, float("nan"))
                    row.append("" if not math.isfinite(v) else f"{v:.{args.round}f}")
                w.writerow(row)
        print(f"\n[ok] wrote {args.out}")

if __name__ == "__main__":
    main()

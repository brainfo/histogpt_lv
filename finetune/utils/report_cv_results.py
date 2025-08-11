#!/usr/bin/env python3
"""
Generate publication-ready summary tables from CV results JSON produced by
finetune/train_mil_classifier.py.

Outputs:
- Markdown summary table with mean ± SD (and 95% CI approximations)
- Optional CSV summary
- Optional per-fold table (Markdown/CSV)

Usage examples:
  python -m finetune.utils.report_cv_results \
    --json /abs/path/to/cv_results_YYYYMMDD_HHMMSS.json \
    --out_md ./cv_summary.md --out_csv ./cv_summary.csv --per_fold_md ./cv_folds.md

Notes:
- 95% CI is estimated as mean ± 1.96 × (SD / sqrt(k)), where k = #folds
  (for AUROC in publications, prefer DeLong or bootstrap if possible).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


PRIMARY_METRICS: List[Tuple[str, str]] = [
    ("val_auroc", "AUROC"),
    ("val_f1_macro", "F1 (macro)"),
    ("val_balanced_acc", "Balanced accuracy"),
    ("val_acc", "Accuracy"),
    ("val_loss", "Loss"),
]


def compute_stats(values: List[float]) -> Dict[str, float]:
    array = np.array(values, dtype=float)
    mean = float(np.nanmean(array))
    sd = float(np.nanstd(array, ddof=0))
    n = int(np.sum(~np.isnan(array)))
    se = sd / np.sqrt(n) if n > 0 else float("nan")
    ci_low = mean - 1.96 * se if n > 0 else float("nan")
    ci_high = mean + 1.96 * se if n > 0 else float("nan")
    return {"mean": mean, "sd": sd, "n": n, "ci_low": ci_low, "ci_high": ci_high}


def format_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def format_ci(low: float, high: float, digits: int = 3) -> str:
    if np.isnan(low) or np.isnan(high):
        return "—"
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def build_summary_table(cv_results: List[Dict[str, Any]], metrics: List[Tuple[str, str]]) -> List[List[str]]:
    header = ["Metric", "Mean ± SD", "95% CI (approx)"]
    rows: List[List[str]] = [header]

    for key, label in metrics:
        values = [float(fold.get(key, np.nan)) for fold in cv_results]
        stats = compute_stats(values)
        rows.append([
            label,
            format_mean_sd(stats["mean"], stats["sd"]),
            format_ci(stats["ci_low"], stats["ci_high"]),
        ])

    return rows


def build_per_fold_table(cv_results: List[Dict[str, Any]], metrics: List[Tuple[str, str]]) -> List[List[str]]:
    header = ["Fold"] + [label for _, label in metrics]
    rows: List[List[str]] = [header]

    for idx, fold in enumerate(cv_results):
        row: List[str] = [str(idx + 1)]
        for key, _ in metrics:
            value = fold.get(key, np.nan)
            row.append("" if np.isnan(value) else f"{float(value):.3f}")
        rows.append(row)

    return rows


def write_markdown_table(rows: List[List[str]], out_path: Path) -> None:
    if not rows:
        return
    # Add separator row after header
    header = rows[0]
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: List[List[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Report tables from CV results JSON")
    parser.add_argument("--json", required=True, help="Path to cv_results_*.json")
    parser.add_argument("--out_md", default=None, help="Path to write summary Markdown table")
    parser.add_argument("--out_csv", default=None, help="Path to write summary CSV table")
    parser.add_argument("--per_fold_md", default=None, help="Path to write per-fold Markdown table")
    parser.add_argument("--per_fold_csv", default=None, help="Path to write per-fold CSV table")
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    cv_results = data.get("cv_results", [])
    if not cv_results:
        raise SystemExit("No cv_results found in JSON")

    summary_rows = build_summary_table(cv_results, PRIMARY_METRICS)
    per_fold_rows = build_per_fold_table(cv_results, [m for m in PRIMARY_METRICS if m[0] != "val_loss"])  # omit loss in per-fold table if desired

    # Default to printing summary Markdown to stdout if no outputs specified
    if not any([args.out_md, args.out_csv, args.per_fold_md, args.per_fold_csv]):
        for line in ["| "+" | ".join(r)+" |" for r in summary_rows[:1]]:
            print(line)
        print("| " + " | ".join(["---"] * len(summary_rows[0])) + " |")
        for r in summary_rows[1:]:
            print("| " + " | ".join(r) + " |")
        return

    if args.out_md:
        write_markdown_table(summary_rows, Path(args.out_md))
    if args.out_csv:
        write_csv(summary_rows, Path(args.out_csv))
    if args.per_fold_md:
        write_markdown_table(per_fold_rows, Path(args.per_fold_md))
    if args.per_fold_csv:
        write_csv(per_fold_rows, Path(args.per_fold_csv))

    print("Summary tables written.")


if __name__ == "__main__":
    main()



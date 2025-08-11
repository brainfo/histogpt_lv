#!/usr/bin/env python3
"""
Plot charts from cross-validation results JSON produced by finetune/train_mil_classifier.py

Usage:
  python -m finetune.utils.plot_cv_results \
    --json /abs/path/to/cv_results_YYYYMMDD_HHMMSS.json \
    --outdir ./plots --format png --dpi 150

Generates bar charts for key metrics per fold and saves them to the output directory.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl


PRIMARY_METRICS = [
    ("val_auroc", "Validation AUROC"),
    ("val_f1_macro", "Validation F1 (macro)"),
    ("val_balanced_acc", "Validation Balanced Accuracy"),
    ("val_acc", "Validation Accuracy"),
]

PER_CLASS_KEYS = [
    ("val_f1_bcc", "F1 BCC"),
    ("val_f1_scc", "F1 SCC"),
    ("val_precision_bcc", "Precision BCC"),
    ("val_precision_scc", "Precision SCC"),
    ("val_recall_bcc", "Recall BCC"),
    ("val_recall_scc", "Recall SCC"),
]


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _annotate_bars(ax) -> None:
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2.0, height),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')


def plot_metric_per_fold(cv_results: List[Dict[str, Any]], key: str, title: str, outpath: Path) -> None:
    values = [float(fold.get(key, float('nan'))) for fold in cv_results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([f"Fold {i+1}" for i in range(len(values))], values, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Score")
    _annotate_bars(ax)

    mean_val = sum(values) / len(values) if values else float('nan')
    sd_val = float(np.std(values)) if values else float('nan')
    # Visualize mean ± SD as a shaded band
    if not np.isnan(mean_val) and not np.isnan(sd_val):
        lower = max(0.0, mean_val - sd_val)
        upper = min(1.05, mean_val + sd_val)
        ax.fill_between([-0.5, len(values)-0.5], lower, upper, color="#C5E0B4", alpha=0.35, label=f"Mean ± SD: {mean_val:.3f} ± {sd_val:.3f}")
    ax.axhline(mean_val, color="#55A868", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.3f}")
    # Place legend outside plotting area, anchored at x=1.05
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    fig.tight_layout()
    fig.savefig(outpath, dpi=plt.rcParams.get("savefig.dpi", 150), bbox_inches="tight")
    plt.close(fig)


def plot_per_class_grouped(cv_results: List[Dict[str, Any]], outpath: Path) -> None:
    labels = [f"Fold {i+1}" for i in range(len(cv_results))]
    width = 0.12
    x = list(range(len(labels)))

    series = []
    for key, label in PER_CLASS_KEYS:
        series.append(([float(fold.get(key, float('nan'))) for fold in cv_results], label))

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (values, label) in enumerate(series):
        offsets = [xi + (idx - (len(series) - 1) / 2) * width for xi in x]
        ax.bar(offsets, values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class metrics per fold")
    ax.set_ylabel("Score")
    # Place legend outside plotting area, anchored at x=1.05
    ax.legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(outpath, dpi=plt.rcParams.get("savefig.dpi", 150), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot charts from CV results JSON")
    parser.add_argument("--json", required=True, help="Path to cv_results_*.json")
    parser.add_argument("--outdir", default="./plots", help="Directory to write charts")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Image format")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI")
    parser.add_argument("--style", default="~/reference/general.mplstyle", help="Path to Matplotlib style file (.mplstyle)")
    args = parser.parse_args()

    # Apply style if available
    style_path = Path(args.style).expanduser()
    if style_path.exists():
        try:
            mpl.style.use(str(style_path))
        except Exception:
            pass

    plt.rcParams["savefig.dpi"] = args.dpi

    json_path = Path(args.json)
    outdir = Path(args.outdir)
    _ensure_outdir(outdir)

    with open(json_path, "r") as f:
        data = json.load(f)

    cv_results = data.get("cv_results", [])
    if not cv_results:
        raise SystemExit("No cv_results found in JSON")

    # Plot primary metrics per fold
    for key, title in PRIMARY_METRICS:
        outpath = outdir / f"{key}_per_fold.{args.format}"
        plot_metric_per_fold(cv_results, key, title, outpath)

    # Plot grouped per-class metrics
    outpath = outdir / f"per_class_metrics_per_fold.{args.format}"
    plot_per_class_grouped(cv_results, outpath)

    print(f"Charts written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()




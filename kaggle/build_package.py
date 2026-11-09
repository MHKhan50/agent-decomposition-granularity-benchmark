#!/usr/bin/env python3
"""Build the Kaggle upload folder and helper files from finished results."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE = "Clinic Task Decomposition: Small Local LLMs"
SUBTITLE = "1, 2, 4 or 8 sub-calls on English and Roman Urdu clinic forms"
NAVY, ROYAL, PAPER = "#0f1932", "#2b5ce6", "#f4f7fd"
RESULT_FILES = ("runs.jsonl", "per_run.csv", "results.json", "run_meta.json", "chart_field_accuracy.png", "chart_time_per_form.png")

FIELDS_TEMPLATE = """# Kaggle dataset fields

**Title** (paste exactly)

{title}

**Subtitle**

{subtitle}

**License**: CC BY 4.0

**Tags**: health, nlp, artificial intelligence, languages

**Cover image**: cover.png (1280 x 640)

**Description**

How does the number of sub-calls change what a small local language model gets right on a multi-field form?

Each of the {n_items} invented clinic cases is written in English and in Roman Urdu with the same facts. A form of 8 fields is filled with 1, 2, 4 or 8 separate model calls. Half of the forms have independent fields. The other half form a billing chain where later fields use earlier ones, so a sub-call receives the answers produced before it.

All cases are invented from a fixed seed. No real patient data is included.

**Files**

- clinic_tasks.jsonl: cases, both language versions and known answers
- runs.jsonl: one row per model call, with prompt size, raw reply, parsed reply, tokens and seconds
- per_run.csv: one row per finished form with accuracy and cost
- results.json: aggregated values with bootstrap 95 percent intervals
- run_meta.json: model digests, Ollama version and options
- chart_field_accuracy.png and chart_time_per_form.png

Code: https://github.com/MHKhan50/agent-decomposition-granularity-benchmark
"""


def _source(text: str) -> List[str]:
    return text.strip("\n").splitlines(keepends=True)


def notebook() -> Dict[str, object]:
    def markdown(text: str) -> Dict[str, object]:
        return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}

    def code(text: str) -> Dict[str, object]:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(text)}

    return {
        "cells": [
            markdown("# Clinic task decomposition: starter\n\nLoads the per-form results and compares field accuracy across the number of sub-calls."),
            code(
                "import glob\nimport os\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
                "matches = glob.glob('/kaggle/input/**/per_run.csv', recursive=True) or glob.glob('**/per_run.csv', recursive=True)\n"
                "assert matches, 'per_run.csv not found; attach the dataset first'\n"
                "runs = pd.read_csv(matches[0])\nprint(os.path.dirname(matches[0]), runs.shape)\nruns.head()"
            ),
            code(
                "summary = (runs.groupby(['model', 'language', 'dependency', 'granularity'])\n"
                "           .agg(items=('item_id', 'nunique'), field_accuracy=('field_accuracy', 'mean'),\n"
                "                all_correct=('all_correct', 'mean'), seconds=('seconds_total', 'mean'))\n"
                "           .round(3).reset_index())\nsummary"
            ),
            code(
                "fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)\n"
                "for ax, dependency in zip(axes, ['independent', 'chained']):\n"
                "    part = runs[runs.dependency == dependency]\n"
                "    for (model, language), group in part.groupby(['model', 'language']):\n"
                "        line = group.groupby('granularity').field_accuracy.mean()\n"
                "        ax.plot([str(g) for g in line.index], line.values, marker='o', label=f'{model}, {language}')\n"
                "    ax.set_title(f'{dependency} fields')\n    ax.set_xlabel('Sub-calls per form')\n    ax.set_ylim(0, 1.04)\n"
                "axes[0].set_ylabel('Fields correct (share)')\naxes[0].legend(fontsize=8)\nplt.tight_layout()\nplt.show()"
            ),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def draw_cover(path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=NAVY)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 640)
    ax.axis("off")
    ax.text(80, 520, "Clinic Task Decomposition", color=PAPER, fontsize=44, fontweight="bold", va="center")
    ax.text(80, 450, "Small local LLMs, English and Roman Urdu", color=PAPER, fontsize=22, va="center")
    for row, pieces in enumerate((1, 2, 4, 8)):
        y = 310 - row * 70
        width = (1120 - (pieces - 1) * 10) / pieces
        for piece in range(pieces):
            ax.add_patch(Rectangle((80 + piece * (width + 10), y), width, 46, facecolor=ROYAL if piece % 2 == 0 else PAPER, alpha=0.95 if piece % 2 == 0 else 0.85))
    fig.savefig(path, dpi=100, facecolor=NAVY)
    plt.close(fig)


def write_helpers(out_dir: str, n_items: int) -> None:
    helpers = os.path.join(out_dir, "2-helpers")
    os.makedirs(helpers, exist_ok=True)
    with open(os.path.join(helpers, "KAGGLE_FIELDS.md"), "w", encoding="utf-8") as handle:
        handle.write(FIELDS_TEMPLATE.format(title=TITLE, subtitle=SUBTITLE, n_items=n_items))
    with open(os.path.join(helpers, "starter_notebook.ipynb"), "w", encoding="utf-8") as handle:
        json.dump(notebook(), handle, indent=1)
        handle.write("\n")
    draw_cover(os.path.join(helpers, "cover.png"))


def build(results_dir: str, tasks_path: str, out_dir: str, helpers_only: bool = False) -> None:
    if len(TITLE) > 50 or len(TITLE) < 6 or len(SUBTITLE) > 80:
        raise ValueError("Kaggle title or subtitle has the wrong length")
    with open(tasks_path, "r", encoding="utf-8") as handle:
        n_items = sum(1 for line in handle if line.strip())
    write_helpers(out_dir, n_items)
    if helpers_only:
        return
    missing = [name for name in RESULT_FILES if not os.path.exists(os.path.join(results_dir, name))]
    if missing:
        raise SystemExit("Finish run.py and analysis.py first. Missing: %s" % ", ".join(missing))
    upload = os.path.join(out_dir, "1-upload-these")
    os.makedirs(upload, exist_ok=True)
    shutil.copy(tasks_path, os.path.join(upload, "clinic_tasks.jsonl"))
    for name in RESULT_FILES:
        shutil.copy(os.path.join(results_dir, name), os.path.join(upload, name))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--tasks", default="tasks/clinic_tasks.jsonl")
    parser.add_argument("--out", default=HERE)
    parser.add_argument("--helpers-only", action="store_true")
    args = parser.parse_args()
    build(args.results, args.tasks, args.out, args.helpers_only)
    print("Kaggle package written to %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

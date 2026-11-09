#!/usr/bin/env python3
"""Turn results/runs.jsonl into tables, charts and results.json. Needs no model."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

from adgb import __version__
from adgb.clinic_tasks import FIELDS_BY_DEPENDENCY, LANGUAGES, load_tasks
from adgb.decomposition import GRANULARITY_LEVELS
from adgb.scoring import score_form

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
PER_RUN_FIELDS = [
    "model", "language", "dependency", "item_id", "granularity", "fields_correct", "n_fields",
    "field_accuracy", "all_correct", "step_correct", "step_total", "parse_failures", "calls",
    "prompt_chars", "output_tokens", "seconds_total", "seconds_ideal_parallel",
]
Row = Dict[str, object]
ROYAL, ORANGE, NAVY = "#2b5ce6", "#c8551b", "#0f1932"
MODEL_STYLE = [(ROYAL, "o"), (ORANGE, "s")]


def load_calls(path: str) -> List[Row]:
    latest: Dict[str, Row] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                latest[str(record["key"])] = record
    return list(latest.values())


def build_runs(calls: Sequence[Row], tasks: Sequence[Mapping[str, object]]) -> Tuple[List[Row], int]:
    """Join sub-calls into scored forms; unfinished forms are counted and left out."""
    by_id = {str(task["item_id"]): task for task in tasks}
    grouped: Dict[Tuple[str, str, str, int], List[Row]] = defaultdict(list)
    for call in calls:
        grouped[(str(call["model"]), str(call["language"]), str(call["item_id"]), int(call["granularity"]))].append(call)  # type: ignore[arg-type]
    runs: List[Row] = []
    incomplete = 0
    for (model, language, item_id, granularity), parts in sorted(grouped.items()):
        if sorted(int(p["chunk_index"]) for p in parts) != list(range(granularity)):  # type: ignore[arg-type]
            incomplete += 1
            continue
        task = by_id[item_id]
        dependency = str(task["dependency"])
        fields = list(FIELDS_BY_DEPENDENCY[dependency])
        answers: Dict[str, object] = {}
        for part in parts:
            parsed = part["parsed"] or {}
            for name in part["fields"]:  # type: ignore[attr-defined]
                if name in parsed:  # type: ignore[operator]
                    answers[name] = parsed[name]  # type: ignore[index]
        score = score_form(fields, answers, task["truth"])  # type: ignore[arg-type]
        seconds = [float(p["seconds"]) for p in parts]  # type: ignore[arg-type]
        runs.append({
            "model": model, "language": language, "dependency": dependency, "item_id": item_id,
            "granularity": granularity, "fields_correct": score["fields_correct"],
            "n_fields": score["n_fields"],
            "field_accuracy": float(score["fields_correct"]) / float(score["n_fields"]),  # type: ignore[arg-type]
            "all_correct": score["all_correct"], "step_correct": score["step_correct"],
            "step_total": score["step_total"],
            "parse_failures": sum(1 for p in parts if p["parsed"] is None),
            "calls": len(parts), "prompt_chars": sum(int(p["prompt_chars"]) for p in parts),  # type: ignore[arg-type]
            "output_tokens": sum(int(p["output_tokens"] or 0) for p in parts),  # type: ignore[arg-type]
            "seconds_total": round(sum(seconds), 4),
            # Independent sub-calls could run side by side; chained ones must wait for each other.
            "seconds_ideal_parallel": round(max(seconds) if dependency == "independent" else sum(seconds), 4),
        })
    return runs, incomplete


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """95 percent percentile interval from resampling items with replacement."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


def _summary(values: Sequence[float]) -> Dict[str, float]:
    low, high = bootstrap_ci(values)
    return {"mean": round(mean(values), 4), "ci_low": round(low, 4), "ci_high": round(high, 4)}


def cell_table(runs: Sequence[Row]) -> List[Row]:
    groups: Dict[Tuple[str, str, str, int], List[Row]] = defaultdict(list)
    for run in runs:
        groups[(str(run["model"]), str(run["language"]), str(run["dependency"]), int(run["granularity"]))].append(run)  # type: ignore[arg-type]
    table: List[Row] = []
    for (model, language, dependency, granularity), rows in sorted(groups.items()):
        step_total = sum(int(r["step_total"]) for r in rows)  # type: ignore[arg-type]
        table.append({
            "model": model, "language": language, "dependency": dependency, "granularity": granularity,
            "n_items": len(rows),
            "field_accuracy": _summary([float(r["field_accuracy"]) for r in rows]),  # type: ignore[arg-type]
            "all_correct": _summary([float(r["all_correct"]) for r in rows]),  # type: ignore[arg-type]
            "computed_step_accuracy": round(sum(int(r["step_correct"]) for r in rows) / step_total, 4) if step_total else None,  # type: ignore[arg-type]
            "parse_failures": sum(int(r["parse_failures"]) for r in rows),  # type: ignore[arg-type]
            "mean_output_tokens": round(mean([float(r["output_tokens"]) for r in rows]), 1),  # type: ignore[arg-type]
            "mean_prompt_chars": round(mean([float(r["prompt_chars"]) for r in rows]), 1),  # type: ignore[arg-type]
            "mean_seconds_total": round(mean([float(r["seconds_total"]) for r in rows]), 3),  # type: ignore[arg-type]
            "mean_seconds_ideal_parallel": round(mean([float(r["seconds_ideal_parallel"]) for r in rows]), 3),  # type: ignore[arg-type]
        })
    return table


def _lookup(runs: Sequence[Row]) -> Dict[Tuple[str, str, str, int], Row]:
    return {(str(r["model"]), str(r["language"]), str(r["item_id"]), int(r["granularity"])): r for r in runs}  # type: ignore[arg-type]


def paired_vs_single(runs: Sequence[Row]) -> List[Row]:
    """Field accuracy at g minus accuracy at one call, on the same items."""
    index = _lookup(runs)
    groups: Dict[Tuple[str, str, str, int], List[float]] = defaultdict(list)
    for (model, language, item_id, granularity), run in index.items():
        base = index.get((model, language, item_id, 1))
        if granularity != 1 and base is not None:
            groups[(model, language, str(run["dependency"]), granularity)].append(
                float(run["field_accuracy"]) - float(base["field_accuracy"]))  # type: ignore[arg-type]
    return [
        {"model": k[0], "language": k[1], "dependency": k[2], "granularity": k[3], "n_items": len(v), "difference": _summary(v)}
        for k, v in sorted(groups.items())
    ]


def language_gap(runs: Sequence[Row]) -> List[Row]:
    """Roman Urdu accuracy minus English accuracy on the same items."""
    index = _lookup(runs)
    groups: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for (model, language, item_id, granularity), run in index.items():
        english = index.get((model, "english", item_id, granularity))
        if language == "roman_urdu" and english is not None:
            groups[(model, str(run["dependency"]), granularity)].append(
                float(run["field_accuracy"]) - float(english["field_accuracy"]))  # type: ignore[arg-type]
    return [
        {"model": k[0], "dependency": k[1], "granularity": k[2], "n_items": len(v), "difference": _summary(v)}
        for k, v in sorted(groups.items())
    ]


def _style_axis(ax: object) -> None:
    ax.grid(axis="y", color="#dfe4ee", linewidth=0.8)  # type: ignore[attr-defined]
    ax.set_axisbelow(True)  # type: ignore[attr-defined]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)  # type: ignore[attr-defined]
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9aa3b5")  # type: ignore[attr-defined]


def _plot_series(ax: object, table: Sequence[Row], model: str, index: int, metric: str) -> None:
    rows = sorted((r for r in table if r["model"] == model), key=lambda r: int(r["granularity"]))  # type: ignore[arg-type]
    if not rows:
        return
    colour, marker = MODEL_STYLE[index % len(MODEL_STYLE)]
    x = [GRANULARITY_LEVELS.index(int(r["granularity"])) + (index - 0.5) * 0.06 for r in rows]  # type: ignore[arg-type]
    if metric == "field_accuracy":
        y = [float(r[metric]["mean"]) for r in rows]  # type: ignore[index,arg-type]
        lower = [max(0.0, float(r[metric]["mean"]) - float(r[metric]["ci_low"])) for r in rows]  # type: ignore[index,arg-type]
        upper = [max(0.0, float(r[metric]["ci_high"]) - float(r[metric]["mean"])) for r in rows]  # type: ignore[index,arg-type]
        ax.errorbar(x, y, yerr=[lower, upper], color=colour, marker=marker, markersize=7, linewidth=2, capsize=3, label=model)  # type: ignore[attr-defined]
    else:
        ax.plot(x, [float(r[metric]) for r in rows], color=colour, marker=marker, markersize=7, linewidth=2, label=model)  # type: ignore[attr-defined,arg-type]


def make_charts(table: Sequence[Row], out_dir: str) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    models = sorted({str(r["model"]) for r in table})
    dependencies = ("independent", "chained")
    positions = list(range(len(GRANULARITY_LEVELS)))
    names = {"english": "English", "roman_urdu": "Roman Urdu"}

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True, sharey=True)
    for row, language in enumerate(LANGUAGES):
        for column, dependency in enumerate(dependencies):
            ax = axes[row][column]
            cell = [r for r in table if r["language"] == language and r["dependency"] == dependency]
            for index, model in enumerate(models):
                _plot_series(ax, cell, model, index, "field_accuracy")
            ax.set_title("%s, %s fields" % (names[language], dependency), fontsize=11, color=NAVY, loc="left")
            ax.set_ylim(0, 1.04)
            ax.set_xticks(positions)
            ax.set_xticklabels([str(g) for g in GRANULARITY_LEVELS])
            _style_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel("Sub-calls per form")
    for row in axes:
        row[0].set_ylabel("Fields correct (share)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    accuracy_path = os.path.join(out_dir, "chart_field_accuracy.png")
    fig.savefig(accuracy_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for column, dependency in enumerate(dependencies):
        ax = axes[column]
        groups: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for row_data in table:
            if row_data["dependency"] == dependency:
                groups[(str(row_data["model"]), int(row_data["granularity"]))].append(float(row_data["mean_seconds_total"]))  # type: ignore[arg-type]
        pooled: List[Row] = [{"model": k[0], "granularity": k[1], "mean_seconds_total": mean(v)} for k, v in groups.items()]
        for index, model in enumerate(models):
            _plot_series(ax, pooled, model, index, "mean_seconds_total")
        ax.set_title("%s fields" % dependency.capitalize(), fontsize=11, color=NAVY, loc="left")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(g) for g in GRANULARITY_LEVELS])
        ax.set_xlabel("Sub-calls per form")
        ax.set_ylim(bottom=0)
        _style_axis(ax)
    axes[0].set_ylabel("Mean seconds per form")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    time_path = os.path.join(out_dir, "chart_time_per_form.png")
    fig.savefig(time_path, dpi=160)
    plt.close(fig)
    return [accuracy_path, time_path]


def print_tables(table: Sequence[Row], paired: Sequence[Row], gaps: Sequence[Row]) -> None:
    print("\nField accuracy, all-correct forms and cost")
    print("model        language    dependency   g  n  field_acc [95% CI]      all_ok  step_acc parse_fail  out_tok  sec")
    for r in table:
        fa, ok = r["field_accuracy"], r["all_correct"]
        step = "-" if r["computed_step_accuracy"] is None else "%.3f" % float(r["computed_step_accuracy"])  # type: ignore[arg-type]
        print("%-12s %-11s %-12s %d %2d  %.3f [%.3f, %.3f]   %.3f   %-7s %-10d %-8.1f %.2f" % (
            r["model"], r["language"], r["dependency"], r["granularity"], r["n_items"],
            fa["mean"], fa["ci_low"], fa["ci_high"], ok["mean"], step, r["parse_failures"],  # type: ignore[index]
            r["mean_output_tokens"], r["mean_seconds_total"]))
    print("\nChange in field accuracy compared with one call (same items)")
    for r in paired:
        d = r["difference"]
        print("%-12s %-11s %-12s g=%d  %+.3f [%+.3f, %+.3f]" % (r["model"], r["language"], r["dependency"], r["granularity"], d["mean"], d["ci_low"], d["ci_high"]))  # type: ignore[index]
    print("\nRoman Urdu minus English field accuracy (same items)")
    for r in gaps:
        d = r["difference"]
        print("%-12s %-12s g=%d  %+.3f [%+.3f, %+.3f]" % (r["model"], r["dependency"], r["granularity"], d["mean"], d["ci_low"], d["ci_high"]))  # type: ignore[index]


def analyse(results_dir: str, tasks_path: str) -> Dict[str, object]:
    tasks = load_tasks(tasks_path)
    calls = load_calls(os.path.join(results_dir, "runs.jsonl"))
    runs, incomplete = build_runs(calls, tasks)
    if not runs:
        raise SystemExit("No finished forms in %s yet." % results_dir)
    with open(os.path.join(results_dir, "per_run.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_RUN_FIELDS)
        writer.writeheader()
        writer.writerows(runs)
    table = cell_table(runs)
    paired = paired_vs_single(runs)
    gaps = language_gap(runs)
    models = sorted({str(r["model"]) for r in runs})
    meta_path = os.path.join(results_dir, "run_meta.json")
    meta: object = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    results: Dict[str, object] = {
        "benchmark": "adgb-clinic-decomposition", "version": __version__, "run_meta": meta,
        "coverage": {
            "sub_calls": len(calls), "finished_forms": len(runs), "unfinished_forms": incomplete,
            "expected_forms": len(models) * len(tasks) * len(LANGUAGES) * len(GRANULARITY_LEVELS),
            "models": models, "items": len(tasks),
        },
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "unit": "item", "interval": "95 percent percentile"},
        "cells": table, "paired_vs_single_call": paired, "roman_urdu_minus_english": gaps,
    }
    with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    make_charts(table, results_dir)
    print_tables(table, paired, gaps)
    coverage = results["coverage"]
    print("\nFinished forms: %d of %d expected. Files written to %s" % (coverage["finished_forms"], coverage["expected_forms"], results_dir))  # type: ignore[index]
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--tasks", default="tasks/clinic_tasks.jsonl")
    args = parser.parse_args()
    analyse(args.results, args.tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())

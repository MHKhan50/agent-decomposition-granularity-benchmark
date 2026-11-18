#!/usr/bin/env python3
"""Rerun the chained (billing) cases with a clarified fee description.

In v0.2.0 the field read "fee per session in rupees in the current course". Asked for it alone,
llama3.1:8b often returned the fee after the discount. This rerun changes only that description,
to say "before any discount", and repeats every chained case for both models, both note
languages and all four granularities (960 sub-calls). Results go to results_clarified_fee/,
so the v0.2.0 results are untouched. The run resumes if it is interrupted.

    python3 rerun_clarified_fee.py
    python3 analysis.py --results results_clarified_fee --tasks results_clarified_fee/tasks_chained.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

from adgb import clinic_tasks
from adgb.clinic_tasks import load_tasks, write_tasks
from adgb.llm import OllamaClient, OllamaUnavailable
from adgb.runner import DEFAULT_MODELS, RunStore, preflight, run_benchmark, write_run_meta

CLARIFIED = "fee per session in rupees in the current course, before any discount, as a number"


def clarify_fee() -> None:
    """Replace the chained fee description in place; every other field stays as in v0.2.0."""
    fields = tuple(
        replace(spec, description=CLARIFIED) if spec.name == "fee_per_session" else spec
        for spec in clinic_tasks.FIELDS_BY_DEPENDENCY["chained"]
    )
    clinic_tasks.FIELDS_BY_DEPENDENCY["chained"] = fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="tasks/clinic_tasks.jsonl")
    parser.add_argument("--out", default="results_clarified_fee")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args()

    clarify_fee()
    items = [item for item in load_tasks(args.tasks) if item["dependency"] == "chained"]
    os.makedirs(args.out, exist_ok=True)
    write_tasks(os.path.join(args.out, "tasks_chained.jsonl"), items)
    client = OllamaClient(base_url=args.host)
    store = RunStore(os.path.join(args.out, "runs.jsonl"))
    try:
        installed = client.installed_models()
        preflight(installed, args.models)
        write_run_meta(os.path.join(args.out, "run_meta.json"), client, args.models, installed, len(items))
        made = run_benchmark(client, items, store, models=args.models)
    except OllamaUnavailable as exc:
        print("Stopped: %s" % exc, file=sys.stderr)
        return 1
    print("Finished. New sub-calls: %d. Next: python3 analysis.py --results %s --tasks %s"
          % (made, args.out, os.path.join(args.out, "tasks_chained.jsonl")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run the clinic decomposition benchmark on local Ollama models."""
from __future__ import annotations

import argparse
import os
import sys

from adgb.clinic_tasks import load_tasks
from adgb.llm import OllamaClient, OllamaUnavailable
from adgb.runner import DEFAULT_MODELS, RunStore, preflight, run_benchmark, write_run_meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="tasks/clinic_tasks.jsonl")
    parser.add_argument("--out", default="results")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args()

    items = load_tasks(args.tasks)
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
    print("Finished. New sub-calls in this session: %d. Next: python analysis.py" % made)
    return 0


if __name__ == "__main__":
    sys.exit(main())

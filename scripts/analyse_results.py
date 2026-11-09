#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adgb.io import read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise ADGB policy metrics")
    parser.add_argument("--input", default="synthetic_results/episode_metrics.csv")
    args = parser.parse_args()
    rows = read_csv(args.input)
    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)
    print("policy,n,mean_regret,mean_makespan,mean_quality,deadline_success,mean_granularity,mean_merge_conflict")
    for policy in sorted(groups):
        items = groups[policy]
        n = len(items)
        mean = lambda key: sum(float(r[key]) for r in items) / n
        print("%s,%d,%.6f,%.6f,%.6f,%.4f,%.3f,%.4f" % (
            policy, n, mean("regret"), mean("makespan"), mean("quality_score"),
            mean("deadline_met"), mean("granularity"), mean("merge_conflict_rate")))


if __name__ == "__main__":
    main()

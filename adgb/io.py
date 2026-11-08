"""CSV helpers using only the Python standard library."""
import csv
import os
from typing import Dict, Iterable, List, Sequence

from .simulator import FEATURES, GRANULARITIES

BASE_FIELDS = ["seed", "episode_id"] + list(FEATURES) + ["execution_shock", "merge_shock", "oracle_granularity"]
DATA_FIELDS = BASE_FIELDS + ["objective_g%d" % g for g in GRANULARITIES]
RESULT_FIELDS = [
    "seed", "episode_id", "policy", "granularity", "objective", "oracle_objective",
    "regret", "makespan", "quality_score", "coordination_cost", "decomposition_overhead",
    "merge_conflict_rate", "deadline_met",
]


def write_csv(path: str, rows: Sequence[Dict], fields: Sequence[str]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str) -> List[Dict[str, float]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    converted: List[Dict[str, float]] = []
    for row in rows:
        out: Dict[str, float] = {}
        for key, value in row.items():
            if key == "policy":
                out[key] = value
            else:
                out[key] = float(value)
        converted.append(out)
    return converted

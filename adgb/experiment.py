"""Policy evaluation for decomposition-granularity selection."""
from __future__ import annotations

from typing import Dict, List, Sequence

from .model import TinyMLP
from .simulator import FEATURES, GRANULARITIES, evaluate_granularity


def heuristic_policy(row: Dict[str, float]) -> int:
    score = (
        0.34 * row["complexity"]
        + 0.24 * row["context_pressure"]
        + 0.22 * row["parallelism_availability"]
        + 0.12 * row["spec_ambiguity"]
        - 0.22 * row["merge_sensitivity"]
        - 0.18 * row["tool_setup_cost"]
        - 0.16 * row["dependency_density"]
    )
    if score < 0.10:
        return 1
    if score < 0.27:
        return 2
    if score < 0.46:
        return 4
    return 8


def evaluate_policies(rows: Sequence[Dict[str, float]], model: TinyMLP) -> List[Dict[str, float]]:
    policies = {
        "fixed_coarse": lambda r: 1,
        "fixed_2way": lambda r: 2,
        "fixed_4way": lambda r: 4,
        "fixed_fine": lambda r: 8,
        "adaptive_heuristic": heuristic_policy,
        "neural_mlp": model.predict,
        "oracle_reference": lambda r: int(r["oracle_granularity"]),
    }
    results: List[Dict[str, float]] = []
    for row in rows:
        oracle = evaluate_granularity(row, int(row["oracle_granularity"]))
        for name, choose in policies.items():
            granularity = int(choose(row))
            metrics = evaluate_granularity(row, granularity)
            result = {
                "seed": int(row["seed"]),
                "episode_id": int(row["episode_id"]),
                "policy": name,
                "granularity": granularity,
                "objective": metrics["objective"],
                "oracle_objective": oracle["objective"],
                "regret": max(0.0, metrics["objective"] - oracle["objective"]),
                "makespan": metrics["makespan"],
                "quality_score": metrics["quality_score"],
                "coordination_cost": metrics["coordination_cost"],
                "decomposition_overhead": metrics["decomposition_overhead"],
                "merge_conflict_rate": metrics["merge_conflict_rate"],
                "deadline_met": int(metrics["deadline_met"]),
            }
            results.append(result)
    return results

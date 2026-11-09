"""Synthetic workload generator and decomposition-cost simulator."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

FEATURES = (
    "complexity",
    "dependency_density",
    "heterogeneity",
    "context_pressure",
    "merge_sensitivity",
    "parallelism_availability",
    "deadline_pressure",
    "tool_setup_cost",
    "spec_ambiguity",
)
GRANULARITIES = (1, 2, 4, 8)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def make_episode(rng: random.Random, seed: int, episode_id: int) -> Dict[str, float]:
    """Create one task profile; hidden shocks are not exposed to the learner."""
    row: Dict[str, float] = {
        "seed": seed,
        "episode_id": episode_id,
        "complexity": rng.random(),
        "dependency_density": rng.random(),
        "heterogeneity": rng.random(),
        "context_pressure": rng.random(),
        "merge_sensitivity": rng.random(),
        "parallelism_availability": rng.random(),
        "deadline_pressure": rng.random(),
        "tool_setup_cost": rng.random(),
        "spec_ambiguity": rng.random(),
        "execution_shock": rng.uniform(-0.08, 0.08),
        "merge_shock": rng.uniform(-0.06, 0.06),
    }
    return row


def evaluate_granularity(row: Dict[str, float], granularity: int) -> Dict[str, float]:
    """Evaluate a decomposition choice using a transparent synthetic simulator.

    The objective is a benchmark design choice, not a claim about real-world
    wall-clock time, token prices, or agent quality.
    """
    if granularity not in GRANULARITIES:
        raise ValueError("granularity must be one of %s" % (GRANULARITIES,))

    x = {name: float(row[name]) for name in FEATURES}
    work = 70.0 + 260.0 * x["complexity"]
    effective_workers = min(float(granularity), 1.0 + 7.0 * x["parallelism_availability"])
    effective_workers *= max(0.28, 1.0 - 0.72 * x["dependency_density"])
    execution_time = (work / max(1.0, effective_workers)) * (
        1.0 + 0.16 * x["heterogeneity"] + float(row.get("execution_shock", 0.0))
    )

    decomposition_overhead = (granularity - 1) * (
        1.0 + 4.0 * x["tool_setup_cost"] + 1.5 * x["dependency_density"]
    )
    coordination_cost = (granularity - 1) * (
        0.8 + 2.5 * x["dependency_density"] + 1.5 * x["tool_setup_cost"]
    )
    split_fraction = (granularity - 1) / 7.0
    merge_conflict = clamp(
        x["merge_sensitivity"] * split_fraction * (0.25 + 0.75 * x["heterogeneity"])
        + float(row.get("merge_shock", 0.0)) * split_fraction
    )

    fragmentation_loss = (
        0.10 * x["complexity"] * x["heterogeneity"] * (1.0 - 1.0 / granularity)
        + 0.07 * x["spec_ambiguity"] * (1.0 - 1.0 / granularity)
    )
    context_gain = 0.06 * x["context_pressure"] * x["complexity"] * (1.0 - 1.0 / granularity)
    quality = clamp(1.0 - fragmentation_loss - 0.10 * merge_conflict + context_gain, 0.0, 1.0)

    makespan = (
        execution_time
        + decomposition_overhead
        + coordination_cost
        + merge_conflict * work * 0.08
    )
    deadline = work * (0.78 + 0.45 * (1.0 - x["deadline_pressure"]))
    deadline_violation = max(0.0, makespan - deadline) / max(1.0, deadline)
    normalized_coordination = (coordination_cost + decomposition_overhead) / work
    objective = (
        0.50 * (makespan / work)
        + 0.25 * (1.0 - quality)
        + 0.10 * normalized_coordination
        + 0.15 * deadline_violation
    )

    return {
        "granularity": float(granularity),
        "work_units": work,
        "execution_time": execution_time,
        "decomposition_overhead": decomposition_overhead,
        "coordination_cost": coordination_cost,
        "merge_conflict_rate": merge_conflict,
        "quality_score": quality,
        "makespan": makespan,
        "deadline": deadline,
        "deadline_met": 1.0 if makespan <= deadline else 0.0,
        "objective": objective,
    }


def label_episode(row: Dict[str, float]) -> Dict[str, float]:
    """Attach oracle label and per-choice objective values for dataset auditing."""
    scores = {g: evaluate_granularity(row, g) for g in GRANULARITIES}
    best = min(GRANULARITIES, key=lambda g: (scores[g]["objective"], g))
    out = dict(row)
    out["oracle_granularity"] = best
    for g in GRANULARITIES:
        out["objective_g%d" % g] = scores[g]["objective"]
    return out


def generate_dataset(seeds: Sequence[int], episodes_per_seed: int) -> List[Dict[str, float]]:
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be >= 1")
    rows: List[Dict[str, float]] = []
    for seed in seeds:
        rng = random.Random(seed)
        for episode_id in range(episodes_per_seed):
            rows.append(label_episode(make_episode(rng, seed, episode_id)))
    return rows

# Experimental Protocol

This file describes the synthetic simulator. The local LLM run is described in the README.


## Objective

Evaluate policies that choose an initial task-decomposition granularity under controlled variation in task complexity, dependency density, context pressure, parallelism availability, tool setup cost, specification ambiguity, and merge sensitivity.

## Hypothesis

A small neural classifier trained on synthetic task profiles may reduce mean quality-adjusted objective regret compared with fixed-granularity and hand-designed heuristic policies.

## Design

- Candidate granularities: 1, 2, 4, and 8 subtasks.
- Training seeds: 17, 29, 43.
- Test seeds: 71, 101.
- Default episodes per seed: 600 training; 300 test.
- MLP: one tanh hidden layer, 12 units, softmax output, cross-entropy updates, 30 epochs, initialization seed 123.
- No LLM calls, network calls, or hardware acceleration.
- The test-set oracle label is used only for reference scoring and label-agreement reporting; the MLP receives only the nine observable features.

## Procedure

1. Generate the training and held-out test CSV files with disjoint seeds.
2. Train the MLP only on `train.csv`.
3. Evaluate all policies on the same held-out test episodes and hidden simulator shocks.
4. Aggregate per-episode metrics by policy.
5. Compare mean regret, makespan, quality, deadline success, coordination cost, decomposition overhead, and merge-conflict rate.

## Objective function

For each episode and granularity, the simulator computes:

`J = 0.50 * (makespan / work) + 0.25 * (1 - quality) + 0.10 * ((coordination + decomposition_overhead) / work) + 0.15 * deadline_violation_fraction`

Lower is better. The weights are design parameters. The per-episode oracle chooses the candidate granularity with the smallest realized `J`.

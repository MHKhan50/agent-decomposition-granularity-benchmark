# Agent Decomposition Granularity Benchmark (ADGB)

**Version:** 0.1.0  
**Release date:** 8 November 2026  
**Research theme:** Agentic AI, task decomposition, neural decision policies  
**Runtime:** Python 3.9+; standard library only

## Research problem

An agentic system often has to decide not only *which* agent should perform a subtask, but also **how finely to split a task before delegation**. Coarse decomposition can leave complex work trapped in oversized subtasks and underuse parallel workers. Fine decomposition can expose parallelism and reduce per-agent context pressure, but it can also increase tool setup, coordination, integration, and merge-conflict costs.

ADGB is a reproducible synthetic benchmark for studying this **decomposition-granularity decision**. It compares fixed split sizes, a transparent heuristic, and a small neural classifier that selects among 1, 2, 4, and 8 subtasks.

### Research question

Can a lightweight neural policy select task-decomposition granularity with lower quality-adjusted execution cost and lower regret than fixed and heuristic policies when dependency density, context pressure, parallelism availability, setup cost, and merge sensitivity vary?

### Scope and distinction

This is not a general DAG scheduler, tool-permission benchmark, agent-memory benchmark, or retry/recovery benchmark. The decision under study is the *initial granularity of decomposition*. The simulator holds the task profile fixed and evaluates the consequences of four split levels. It does not implement real LLM agents or execute external tools.

Prior work already studies graph-based planning and parallel multi-agent execution. In particular, Plan-over-Graph studies generating parallel execution plans over task graphs, while DynTaskMAS studies dynamic task-graph orchestration and asynchronous execution. ADGB should therefore be treated as a focused, complementary experimental testbed—not as the first work on parallel agent planning, and not as a claim of established novelty.

## Included files

- `adgb/simulator.py` — deterministic synthetic workload generation and transparent cost/quality simulation.
- `adgb/model.py` — one-hidden-layer MLP implemented in pure Python.
- `adgb/experiment.py` — fixed, heuristic, neural, and oracle-reference policies.
- `adgb/io.py` — CSV serialization helpers.
- `scripts/generate_dataset.py` — train/test dataset generation.
- `scripts/run_experiment.py` — neural training and policy evaluation.
- `scripts/analyse_results.py` — aggregate policy metrics.
- `tests/` — unit tests for generation, simulator bounds, and model behavior.
- `docs/` — experimental protocol, data dictionary, limitations, and citation metadata.

## Quick start (macOS)

From the extracted project directory:

```bash
python3 --version
python3 -m unittest discover -s tests -v
python3 scripts/generate_dataset.py --train-episodes 600 --test-episodes 300 --train-seeds 17 29 43 --test-seeds 71 101 --out data
python3 scripts/run_experiment.py --data data --epochs 30 --hidden 12 --seed 123 --out results
python3 scripts/analyse_results.py --input results/episode_metrics.csv
```

No `pip install` is required. For a faster smoke run:

```bash
python3 scripts/generate_dataset.py --train-episodes 80 --test-episodes 40 --train-seeds 17 29 --test-seeds 71 --out smoke_data
python3 scripts/run_experiment.py --data smoke_data --epochs 5 --hidden 8 --out smoke_results
python3 scripts/analyse_results.py --input smoke_results/episode_metrics.csv
```

## Policies

1. `fixed_coarse` — one subtask.
2. `fixed_2way` — two subtasks.
3. `fixed_4way` — four subtasks.
4. `fixed_fine` — eight subtasks.
5. `adaptive_heuristic` — hand-designed rule using task profile features.
6. `neural_mlp` — MLP predicts the oracle-optimal split class from observable features.
7. `oracle_reference` — selects the best split using the simulator's realized objective; this is an unattainable reference, not a deployable policy.

## Main outcomes

- **Objective / regret:** lower is better. Regret is policy objective minus per-episode oracle objective.
- **Makespan:** simulated normalized work-time units; lower is better.
- **Quality score:** synthetic task-quality proxy in [0, 1]; higher is better.
- **Deadline success:** fraction of episodes meeting the synthetic deadline.
- **Coordination cost / decomposition overhead:** lower is better, all else equal.
- **Merge-conflict rate:** synthetic integration-risk proxy in [0, 1].
- **Granularity:** chosen number of subtasks.

## Reproducibility

All dataset seeds, model initialization, feature definitions, and simulator equations are explicit. The training seeds (17, 29, 43) are separate from the held-out test seeds (71, 101) by default. Do not tune the model on test outcomes. Report the exact command, Python version, seed lists, episode counts, and model hyperparameters with any results.

## Responsible interpretation

The benchmark uses synthetic task profiles and a hand-specified objective. It can test whether a policy learns this simulator's trade-offs; it cannot establish real-world agent productivity, actual token savings, live latency, correctness, safety, or generalization to deployed agent teams. Any paper should label results as synthetic and should add external workloads or live-system validation before making deployment claims.

## License

MIT. See `LICENSE`.

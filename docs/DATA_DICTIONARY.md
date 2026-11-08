# Data Dictionary — ADGB v0.1.0

The generated `train.csv` and `test.csv` contain one row per synthetic task episode. Values for the nine observable features are sampled independently from Uniform[0, 1]. Hidden shocks affect the simulated outcome but are not input features to the MLP.

| Field | Type | Definition | Used by MLP? |
|---|---|---|---|
| `seed` | integer | Random seed used for the episode stream. | No |
| `episode_id` | integer | Zero-based episode index within a seed. | No |
| `complexity` | float | Normalized task-workload complexity. | Yes |
| `dependency_density` | float | Normalized degree of precedence coupling; high values limit parallel execution. | Yes |
| `heterogeneity` | float | Normalized variation among subtasks. | Yes |
| `context_pressure` | float | Normalized pressure from large per-agent task context. | Yes |
| `merge_sensitivity` | float | Normalized sensitivity to integration/merge errors. | Yes |
| `parallelism_availability` | float | Normalized availability of concurrent workers. | Yes |
| `deadline_pressure` | float | Normalized tightness of the episode deadline. | Yes |
| `tool_setup_cost` | float | Normalized per-subtask setup/dispatch overhead. | Yes |
| `spec_ambiguity` | float | Normalized ambiguity in the task specification. | Yes |
| `execution_shock` | float | Hidden execution-time perturbation sampled from Uniform[-0.08, 0.08]. | No |
| `merge_shock` | float | Hidden merge-risk perturbation sampled from Uniform[-0.06, 0.06]. | No |
| `oracle_granularity` | integer | Best of {1, 2, 4, 8} under the realized simulator objective. Training target; also used for oracle reference. | Target only |
| `objective_g1` | float | Realized objective if split into one subtask. | No |
| `objective_g2` | float | Realized objective if split into two subtasks. | No |
| `objective_g4` | float | Realized objective if split into four subtasks. | No |
| `objective_g8` | float | Realized objective if split into eight subtasks. | No |

The generated `results/episode_metrics.csv` contains one row per policy per test episode:

| Field | Definition |
|---|---|
| `policy` | Policy identifier. |
| `granularity` | Number of subtasks selected by that policy. |
| `objective` | Quality-adjusted simulator objective for the selected granularity. |
| `oracle_objective` | Best realized objective among the four candidate granularities. |
| `regret` | Non-negative difference between policy and oracle objective. |
| `makespan` | Simulated completion-time proxy. |
| `quality_score` | Synthetic output-quality proxy in [0, 1]. |
| `coordination_cost` | Simulated coordination overhead. |
| `decomposition_overhead` | Simulated decomposition/setup overhead. |
| `merge_conflict_rate` | Synthetic integration-risk proxy in [0, 1]. |
| `deadline_met` | 1 if simulated makespan meets the episode deadline, otherwise 0. |

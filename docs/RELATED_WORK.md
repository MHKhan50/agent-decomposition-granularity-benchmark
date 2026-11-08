# Related Work and Research Positioning

The topic is active; ADGB must not be presented as the first work on agent parallelism or task-graph scheduling.

1. **Plan-over-Graph: Towards Parallelable LLM Agent Schedule** (Zhang et al., 2025). Studies graph-based planning and parallel execution schedules for LLM agents. arXiv:2502.14563. https://arxiv.org/abs/2502.14563
2. **DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems** (Yu, Ding, & Sato, 2025). Studies dynamic task graphs and asynchronous parallel execution in multi-agent LLM systems. ICAPS 2025, DOI: 10.1609/icaps.v35i1.36130. https://doi.org/10.1609/icaps.v35i1.36130

ADGB's experimental focus is the choice of *initial decomposition granularity* under a controlled trade-off among context pressure, parallelism, dependencies, setup/coordination overhead, and merge sensitivity. The v0.1.0 simulator is intentionally small and synthetic. A defensible follow-on study should compare against graph-level schedulers and validate on external task traces.

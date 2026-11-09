# Related Work

Each entry below was checked against its arXiv or publisher page.

1. Khot, T., Trivedi, H., Finlayson, M., Fu, Y., Richardson, K., Clark, P., and Sabharwal, A. (2023). *Decomposed Prompting: A Modular Approach for Solving Complex Tasks.* ICLR 2023. https://arxiv.org/abs/2210.02406
2. Prasad, A., Koller, A., Hartmann, M., Clark, P., Sabharwal, A., Bansal, M., and Khot, T. (2023). *ADaPT: As-Needed Decomposition and Planning with Language Models.* https://arxiv.org/abs/2311.05772
3. Zhang, S., Ma, X., Cao, Z., Zhang, Z., and Zhao, H. (2025). *Plan-over-Graph: Towards Parallelable LLM Agent Schedule.* https://arxiv.org/abs/2502.14563
4. Yu, J., Ding, Y., and Sato, H. (2025). *DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems.* ICAPS 2025. https://doi.org/10.1609/icaps.v35i1.36130

Decomposed Prompting breaks tasks into sub-tasks handled by dedicated prompts. ADaPT decomposes only when the executor fails, up to a preset depth. Plan-over-Graph and DynTaskMAS study task graphs and parallel execution. ADGB is narrower: it holds one form fixed and measures how the number of sub-calls changes accuracy and cost for small local models, in English and Roman Urdu.

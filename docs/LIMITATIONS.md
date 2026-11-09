# Limitations

## Local LLM run

1. The clinic cases are invented from templates. They are not real consultations and are shorter and cleaner than real notes.
2. There are 16 cases per dependency type, so confidence intervals are wide.
3. Only two small models were used, on one machine, through Ollama's JSON mode.
4. Roman Urdu spelling varies in practice; the notes use one spelling for each word.
5. Instructions and keys are in English. A fully Roman Urdu prompt may behave differently.
6. Fields are split into equal consecutive groups. Other groupings may give different results.
7. Merging is done by code, so errors from a model-based merge step are not measured.
8. Ideal parallel time is derived from sequential timings. Sub-calls were not run at the same time.
9. Timings depend on the machine and on what else it was doing.

## Synthetic simulator

1. All tasks and costs come from a hand-written simulator; no real agent traces are included.
2. Objective weights are design choices, and other weights can change which split is best.
3. Dependencies are a single number; no real task graph is built or executed.
4. Quality is an equation, not a judged output.
5. Work units are normalised quantities, not tokens, money or wall-clock time.
6. The oracle sees hidden shocks and is an unattainable reference.

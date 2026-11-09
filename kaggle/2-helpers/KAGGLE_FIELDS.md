# Kaggle dataset fields

**Title** (paste exactly)

Clinic Task Decomposition: Small Local LLMs

**Subtitle**

1, 2, 4 or 8 sub-calls on English and Roman Urdu clinic forms

**License**: CC BY 4.0

**Tags**: health, nlp, artificial intelligence, languages

**Cover image**: cover.png (1280 x 640)

**Description**

How does the number of sub-calls change what a small local language model gets right on a multi-field form?

Each of the 32 invented clinic cases is written in English and in Roman Urdu with the same facts. A form of 8 fields is filled with 1, 2, 4 or 8 separate model calls. Half of the forms have independent fields. The other half form a billing chain where later fields use earlier ones, so a sub-call receives the answers produced before it.

All cases are invented from a fixed seed. No real patient data is included.

**Files**

- clinic_tasks.jsonl: cases, both language versions and known answers
- runs.jsonl: one row per model call, with prompt size, raw reply, parsed reply, tokens and seconds
- per_run.csv: one row per finished form with accuracy and cost
- results.json: aggregated values with bootstrap 95 percent intervals
- run_meta.json: model digests, Ollama version and options
- chart_field_accuracy.png and chart_time_per_form.png

Code: https://github.com/MHKhan50/agent-decomposition-granularity-benchmark

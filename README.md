# Agent Decomposition Granularity Benchmark (ADGB)

How finely should an agent split a task before delegating it? ADGB studies this choice in two parts: a controlled simulator, and a real run of small local language models on clinic forms.

## Research question

When a small local LLM fills an 8-field clinic form, how does splitting the work into 1, 2, 4 or 8 sub-calls change field accuracy, whole-form correctness and cost, and does the answer differ between English and Roman Urdu, or between independent fields and fields that depend on earlier ones?

## Design of the local LLM run

| Factor | Levels |
|---|---|
| Sub-calls per form | 1, 2, 4, 8 |
| Models (Ollama) | `llama3.1:8b`, `qwen2.5:3b` |
| Language of the note | English, Roman Urdu (same facts, same case) |
| Field dependency | independent (8 extracted fields), chained (billing fields where later values use earlier ones) |
| Cases | 16 independent and 16 chained, in `tasks/clinic_tasks.jsonl` |
| Settings | temperature 0, seed 42, JSON output mode, `num_ctx` 2048 |

This gives 1,920 model calls. All cases are invented from a fixed seed in `adgb/clinic_tasks.py`. Notes also contain older or unrelated values, such as a relative's age or a closed course, which the model has to ignore. Instructions and JSON keys stay in English; only the note changes language.

Fields are split into equal consecutive groups and each group is one model call. Answers are merged by code, so no model takes part in merging. In chained forms, each sub-call receives the values produced by earlier sub-calls exactly as they were produced, so an early mistake can reach later fields. The field list comes before the note in every prompt, so Ollama's prompt cache does not make later sub-calls look cheaper.

Measures, per form:

- field accuracy and all-fields-correct
- computed-step accuracy: for chained forms, how often a computed field is right when its inputs are right
- JSON parse failures
- output tokens, prompt characters, and summed seconds
- ideal parallel seconds: the slowest sub-call for independent forms and the sum for chained forms. This is derived from the sequential run and was not measured on parallel hardware.

Confidence intervals are 95 percent percentile bootstrap intervals over items (2,000 resamples, seed 42). Comparisons against one call and between languages are paired on the same items.

## Run it

Needs Python 3.9 or later and [Ollama](https://ollama.com) with the two models pulled.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
caffeinate -i python run.py
python analysis.py
```

`caffeinate` is macOS only; elsewhere use `python run.py`. `run.py` prints progress and an estimated finish time and writes each call to `results/runs.jsonl`. If it stops, run the same command again: finished calls are skipped. `analysis.py` needs no model and writes `results/results.json`, `results/per_run.csv`, `results/chart_field_accuracy.png` and `results/chart_time_per_form.png`.

The tests run offline with a scripted fake model and include a full end-to-end mock run.

## Results

Run complete: 512 forms, 1,920 sub-calls, no unfinished forms and no JSON parse failures. Full numbers with intervals are in `results/results.json`.

**Field accuracy** (share of the 8 fields correct; 16 forms per cell)

| Model | Note | Fields | 1 call | 2 calls | 4 calls | 8 calls |
|---|---|---|---|---|---|---|
| llama3.1:8b | English | independent | 1.000 | 1.000 | 1.000 | 1.000 |
| llama3.1:8b | English | chained | 1.000 | 0.641 | 0.523 | 0.625 |
| llama3.1:8b | Roman Urdu | independent | 0.758 | 0.727 | 0.719 | 0.727 |
| llama3.1:8b | Roman Urdu | chained | 1.000 | 0.797 | 0.477 | 0.625 |
| qwen2.5:3b | English | independent | 0.875 | 0.875 | 0.875 | 0.875 |
| qwen2.5:3b | English | chained | 0.977 | 0.797 | 0.539 | 0.602 |
| qwen2.5:3b | Roman Urdu | independent | 0.727 | 0.727 | 0.680 | 0.641 |
| qwen2.5:3b | Roman Urdu | chained | 0.500 | 0.477 | 0.375 | 0.344 |

What the run shows:

- **Splitting independent fields cost little accuracy.** For English notes it changed nothing for either model; for Roman Urdu the paired differences against one call were between 0 and −8.6 points.
- **Splitting chained billing fields cost a lot.** Against one call, llama3.1:8b lost 35.9 to 47.7 points on English notes (paired 95% intervals all below zero), and qwen2.5:3b lost 18.0 to 43.8.
- **Most of that loss is one misread field.** When `fee_per_session` was asked without the other billing fields, llama3.1:8b often returned the fee after the discount: 35 of its 36 wrong English fee values in split forms equal fee × (100 − discount) / 100. With all fields in one call, where `gross_fee` and `discount_amount` are also requested, it made none of these errors. The field description ("fee per session in rupees in the current course") does not say before or after discount, so this is partly an underspecified field, and partly the model losing the context that settled it. Each wrong fee then carried into the gross fee, discount and payable amount.
- **Roman Urdu notes were harder on extraction, not on splitting.** Typical errors in one call were the joint ("knee" for a hip or ankle), the side, and the review day left untranslated (for example "mangal" for Tuesday).
- **qwen2.5:3b's 0.875 ceiling on English independent forms is the weekday field alone:** it wrote abbreviations such as "tue" where the prompt asked for one English weekday in lower case, and the scorer requires the full name.
- **Cost:** for llama3.1:8b on English independent forms, 8 calls took 18.5 s in total against 5.9 s for one call. If the 8 calls ran in parallel the slowest would take 4.2 s; that figure is derived from the sequential run, not measured.

The test of the fee explanation is a rerun with the field described as "before any discount"; that is planned and not part of these results.

## Synthetic simulator

A separate, standard-library-only simulator compares fixed splits, a hand-written heuristic and a small neural classifier that picks among 1, 2, 4 and 8 subtasks. Its costs come from equations in `adgb/simulator.py` and it calls no model. It shows whether a policy can learn the simulator's trade-offs and says nothing about real agents.

```bash
python scripts/generate_dataset.py
python scripts/run_experiment.py
python scripts/analyse_results.py
```

See `docs/EXPERIMENTAL_PROTOCOL.md` and `docs/DATA_DICTIONARY.md`.

## Repository layout

- `run.py`, `analysis.py`: local LLM run and its analysis
- `adgb/clinic_tasks.py`, `adgb/decomposition.py`, `adgb/scoring.py`, `adgb/llm.py`, `adgb/runner.py`: tasks, splitting, scoring, Ollama client, resumable runner
- `adgb/simulator.py`, `adgb/model.py`, `adgb/experiment.py`, `adgb/io.py`: synthetic simulator and classifier
- `scripts/`: synthetic experiment commands
- `kaggle/build_package.py`: builds the Kaggle upload folder from finished results
- `kaggle/2-helpers/KAGGLE_FIELDS.md`, `kaggle/2-helpers/starter_notebook.ipynb`: dataset page text and starter notebook
- `tests/`: offline tests
- `docs/`: protocol, data dictionary, limitations, related work

## Limitations

The cases are template-written and invented, 32 per model and language, so intervals are wide and the findings may not carry over to real clinic notes. Two small models on one Mac are not a general result. See `docs/LIMITATIONS.md` for the full list.

## Citation and licence

See `CITATION.cff`. Code is under the MIT licence (`LICENSE`).

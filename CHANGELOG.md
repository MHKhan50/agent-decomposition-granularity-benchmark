# Changelog

## 0.2.0 (2026-11-09)

- Added a local LLM experiment on clinic forms: 1, 2, 4 or 8 sub-calls, English and Roman Urdu, independent and chained fields, two Ollama models.
- Added a resumable runner, scoring, analysis with bootstrap intervals and charts, and a Kaggle package builder.
- Added pytest tests with a scripted fake model and an end-to-end mock run.
- Renamed the simulator's quality penalty from `coarse_loss` to `fragmentation_loss` because it grows with the number of splits. The equations are unchanged.
- Synthetic output now goes to `synthetic_data/` and `synthetic_results/`; `results/` holds the local LLM run.
- Moved `CITATION.cff` to the repository root and added ORCID.

## 0.1.0 (2026-11-08)

- Initial synthetic benchmark: four split levels, fixed, heuristic, neural and oracle policies, standard-library classifier and tests.

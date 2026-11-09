from __future__ import annotations

import ast
import csv
import glob
import json
import os
import re

import pytest

import analysis
from adgb.clinic_tasks import build_tasks, write_tasks
from adgb.runner import RunStore, run_benchmark
from fake_llm import ScriptedLLM
from kaggle import build_package

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    folder = tmp_path_factory.mktemp("run")
    items = build_tasks()
    tasks = str(folder / "tasks.jsonl")
    write_tasks(tasks, items)
    results = folder / "results"
    client = ScriptedLLM(items, wrong={("bad", "roman_urdu"): {"weeks"}}, broken_json={("broken", "english")})
    run_benchmark(client, items, RunStore(str(results / "runs.jsonl")), models=("good", "bad", "broken"), log=lambda m: None)
    (results / "run_meta.json").write_text("{}\n", encoding="utf-8")
    return analysis.analyse(str(results), tasks), str(results), tasks


def _cell(results, model, language, dependency, granularity):
    return next(c for c in results["cells"] if (c["model"], c["language"], c["dependency"], c["granularity"]) == (model, language, dependency, granularity))


def test_perfect_model_scores_one_with_no_failures(finished):
    results = finished[0]
    assert results["coverage"]["finished_forms"] == 3 * 32 * 2 * 4
    assert results["coverage"]["unfinished_forms"] == 0
    cell = _cell(results, "good", "english", "chained", 8)
    assert cell["field_accuracy"] == {"mean": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    assert cell["all_correct"]["mean"] == 1.0
    assert cell["parse_failures"] == 0
    assert cell["n_items"] == 16


def test_injected_mistake_in_independent_form_costs_exactly_one_field(finished):
    assert _cell(finished[0], "bad", "roman_urdu", "independent", 2)["field_accuracy"]["mean"] == 0.875


def test_injected_mistake_spreads_through_the_chain(finished):
    cell = _cell(finished[0], "bad", "roman_urdu", "chained", 8)
    assert cell["all_correct"]["mean"] == 0.0
    assert cell["field_accuracy"]["mean"] < 0.875


def test_broken_json_counts_as_parse_failure(finished):
    cell = _cell(finished[0], "broken", "english", "independent", 4)
    assert cell["field_accuracy"]["mean"] == 0.0
    assert cell["parse_failures"] == 16 * 4


def test_language_gap_is_measured_on_the_same_items(finished):
    gap = next(g for g in finished[0]["roman_urdu_minus_english"] if (g["model"], g["dependency"], g["granularity"]) == ("bad", "independent", 1))
    assert gap["difference"]["mean"] == -0.125


def test_paired_difference_is_zero_when_splitting_changes_nothing(finished):
    row = next(p for p in finished[0]["paired_vs_single_call"] if (p["model"], p["language"], p["dependency"], p["granularity"]) == ("good", "english", "independent", 8))
    assert row["difference"]["mean"] == 0.0


def test_output_files_exist(finished):
    results_dir = finished[1]
    for name in ("results.json", "per_run.csv", "chart_field_accuracy.png", "chart_time_per_form.png"):
        assert os.path.getsize(os.path.join(results_dir, name)) > 0
    with open(os.path.join(results_dir, "per_run.csv"), newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3 * 32 * 2 * 4


def test_bootstrap_interval_contains_the_mean():
    low, high = analysis.bootstrap_ci([0.0, 1.0, 1.0, 0.5])
    assert 0.0 <= low <= 0.625 <= high <= 1.0


def test_unfinished_form_is_left_out():
    items = build_tasks(cases_per_dependency=1)
    call = {"model": "m", "language": "english", "item_id": items[0]["item_id"], "granularity": 2, "chunk_index": 0}
    runs, incomplete = analysis.build_runs([call], items)
    assert runs == [] and incomplete == 1


def test_kaggle_package_cover_fields_and_notebook(finished, tmp_path, monkeypatch):
    from PIL import Image

    _, results_dir, tasks = finished
    out = tmp_path / "kaggle"
    build_package.build(results_dir, tasks, str(out))
    upload = sorted(os.listdir(out / "1-upload-these"))
    assert upload == sorted(list(build_package.RESULT_FILES) + ["clinic_tasks.jsonl"])
    assert Image.open(out / "2-helpers" / "cover.png").size == (1280, 640)
    assert 6 <= len(build_package.TITLE) <= 50 and len(build_package.SUBTITLE) <= 80

    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.chdir(out)
    with open(out / "2-helpers" / "starter_notebook.ipynb", encoding="utf-8") as handle:
        book = json.load(handle)
    assert book["nbformat"] == 4
    scope = {}
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), scope)
    assert scope["summary"]["items"].max() == 16


def test_kaggle_builder_refuses_missing_results(tmp_path):
    tasks = tmp_path / "t.jsonl"
    write_tasks(str(tasks), build_tasks(cases_per_dependency=1))
    with pytest.raises(SystemExit):
        build_package.build(str(tmp_path / "none"), str(tasks), str(tmp_path / "k"))


def _sources():
    patterns = ("*.py", "adgb/*.py", "scripts/*.py", "tests/*.py", "kaggle/*.py")
    return [p for pattern in patterns for p in glob.glob(os.path.join(ROOT, pattern))]


def test_python_files_parse_as_python_39():
    for path in _sources():
        with open(path, encoding="utf-8") as handle:
            ast.parse(handle.read(), filename=path, feature_version=(3, 9))


def test_readme_paths_exist():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as handle:
        text = handle.read()
    named = set(re.findall(r"`([\w./-]+\.(?:py|md|cff|jsonl|txt|ipynb|png))`", text))
    generated = {"results/runs.jsonl", "results/results.json", "results/per_run.csv", "results/run_meta.json", "results/chart_field_accuracy.png", "results/chart_time_per_form.png"}
    for name in named - generated:
        assert os.path.exists(os.path.join(ROOT, name)), name

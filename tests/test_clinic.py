from __future__ import annotations

import io
import json
import os
import urllib.error

import pytest

from adgb.clinic_tasks import CHAINED_FIELDS, FIELDS_BY_DEPENDENCY, INDEPENDENT_FIELDS, build_tasks, load_tasks
from adgb.decomposition import build_prompt, split_fields
from adgb.llm import OllamaClient, OllamaUnavailable
from adgb.runner import RunStore, count_calls, preflight, run_benchmark
from adgb.scoring import as_number, field_correct, parse_reply, score_form
from fake_llm import ScriptedLLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ("model-a", "model-b")


@pytest.fixture(scope="module")
def items():
    return build_tasks()


def test_tasks_are_reproducible_and_match_shipped_file(items):
    assert build_tasks() == items
    assert load_tasks(os.path.join(ROOT, "tasks", "clinic_tasks.jsonl")) == items
    assert len(items) == 32


def test_chained_answers_are_correct(items):
    for item in items:
        if item["dependency"] != "chained":
            continue
        t = item["truth"]
        assert t["total_sessions"] == t["sessions_per_week"] * t["weeks"]
        assert t["gross_fee"] == t["total_sessions"] * t["fee_per_session"]
        assert t["discount_amount"] * 100 == t["gross_fee"] * t["discount_percent"]
        assert t["payable"] == t["gross_fee"] - t["discount_amount"]


def test_notes_state_each_extracted_number_in_both_languages(items):
    for item in items:
        for spec in FIELDS_BY_DEPENDENCY[item["dependency"]]:
            if spec.kind == "number" and not spec.parents:
                for note in item["notes"].values():
                    assert str(item["truth"][spec.name]) in note


def test_roman_urdu_notes_do_not_contain_the_english_answer_words(items):
    for item in items:
        if item["dependency"] == "independent":
            note = item["notes"]["roman_urdu"].lower()
            for name in ("side", "body_part", "followup_day"):
                assert str(item["truth"][name]) not in note


@pytest.mark.parametrize("granularity,sizes", [(1, [8]), (2, [4, 4]), (4, [2, 2, 2, 2]), (8, [1] * 8)])
def test_split_keeps_order_and_covers_all_fields(granularity, sizes):
    chunks = split_fields(CHAINED_FIELDS, granularity)
    assert [len(c) for c in chunks] == sizes
    assert [s for c in chunks for s in c] == list(CHAINED_FIELDS)


def test_split_rejects_other_levels():
    with pytest.raises(ValueError):
        split_fields(CHAINED_FIELDS, 3)


def test_prompt_lists_fields_before_note_and_shows_known_only_when_given():
    chunk = split_fields(INDEPENDENT_FIELDS, 4)[0]
    plain = build_prompt("NOTE", chunk)
    assert plain.index("- age:") < plain.index("Clinic note:\nNOTE")
    assert "earlier steps" not in plain
    assert '"weeks": 3' in build_prompt("NOTE", chunk, {"weeks": 3})


def test_parse_reply_variants():
    assert parse_reply('{"a": 1}') == {"a": 1}
    assert parse_reply('Sure: {"a": 1} done') == {"a": 1}
    assert parse_reply("[1, 2]") is None
    assert parse_reply("nope") is None


def test_number_and_label_comparison():
    number, label = INDEPENDENT_FIELDS[0], INDEPENDENT_FIELDS[1]
    assert as_number("Rs 1,200") == 1200.0
    assert as_number(True) is None
    assert field_correct(number, "33 years", 33)
    assert not field_correct(number, None, 33)
    assert field_correct(label, " Left ", "left")
    assert not field_correct(label, "bayen", "left")


def test_computed_step_is_only_counted_when_inputs_are_right(items):
    item = next(i for i in items if i["dependency"] == "chained")
    answers = dict(item["truth"])
    answers["weeks"] = answers["weeks"] + 1
    score = score_form(list(CHAINED_FIELDS), answers, item["truth"])
    assert score["fields_correct"] == 7
    assert score["all_correct"] == 0
    assert score["step_total"] == 3  # total_sessions has a wrong input and is excluded
    assert score["step_correct"] == 3


def test_end_to_end_mock_run_and_resume(items, tmp_path):
    path = str(tmp_path / "runs.jsonl")
    client = ScriptedLLM(items)
    expected = count_calls(len(items), MODELS)
    assert expected == 1920
    assert run_benchmark(client, items, RunStore(path), models=MODELS, log=lambda m: None) == expected
    assert len(client.prompts) == expected
    again = ScriptedLLM(items)
    assert run_benchmark(again, items, RunStore(path), models=MODELS, log=lambda m: None) == 0
    assert again.prompts == []


def test_resume_after_cut_off_line_repeats_only_missing_calls(items, tmp_path):
    path = tmp_path / "runs.jsonl"
    subset = items[:1]
    run_benchmark(ScriptedLLM(subset), subset, RunStore(str(path)), models=MODELS[:1], log=lambda m: None)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-3]) + lines[-3][:20], encoding="utf-8")
    client = ScriptedLLM(subset)
    made = run_benchmark(client, subset, RunStore(str(path)), models=MODELS[:1], log=lambda m: None)
    assert made == 3
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(lines)
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_only_chained_sub_calls_receive_earlier_answers(items):
    ind = items[:1]
    client = ScriptedLLM(ind)
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        run_benchmark(client, ind, RunStore(folder + "/r.jsonl"), models=MODELS[:1], log=lambda m: None)
    assert not any("earlier steps" in p for p in client.prompts)


def test_wrong_early_answer_reaches_later_sub_calls(items, tmp_path):
    chained = [i for i in items if i["dependency"] == "chained"][:1]
    client = ScriptedLLM(chained, wrong={("model-a", "english"): {"weeks"}})
    store = RunStore(str(tmp_path / "r.jsonl"))
    run_benchmark(client, chained, store, models=MODELS[:1], log=lambda m: None)
    truth = chained[0]["truth"]
    record = store.records["model-a|english|%s|g8|c2" % chained[0]["item_id"]]
    assert record["known"]["weeks"] == truth["weeks"] + 1
    assert record["parsed"]["total_sessions"] == truth["sessions_per_week"] * (truth["weeks"] + 1)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_ollama_client_sends_fixed_options_and_reads_counts():
    seen = {}

    def opener(request, timeout):
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(json.dumps({"response": "{}", "prompt_eval_count": 7, "eval_count": 3, "total_duration": 2000000000}).encode("utf-8"))

    reply = OllamaClient(urlopen=opener).complete("m", "p")
    assert seen["body"]["options"]["temperature"] == 0.0
    assert seen["body"]["options"]["seed"] == 42
    assert seen["body"]["format"] == "json"
    assert (reply.prompt_tokens, reply.output_tokens, reply.seconds) == (7, 3, 2.0)


def test_ollama_connection_refused_is_a_clear_stop():
    def opener(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError())

    with pytest.raises(OllamaUnavailable):
        OllamaClient(urlopen=opener).installed_models()


def test_preflight_names_missing_models():
    with pytest.raises(OllamaUnavailable, match="ollama pull b"):
        preflight({"a": "x"}, ("a", "b"))

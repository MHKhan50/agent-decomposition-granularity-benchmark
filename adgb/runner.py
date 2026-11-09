"""Resumable runner: every sub-call is stored as soon as it finishes."""
from __future__ import annotations

import json
import os
import platform
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set

from . import __version__
from .clinic_tasks import FIELDS_BY_DEPENDENCY, LANGUAGES
from .decomposition import GRANULARITY_LEVELS, build_prompt, known_values_for, split_fields
from .llm import OllamaUnavailable
from .scoring import parse_reply

# The slower model goes first, so the first time estimate errs on the long side.
DEFAULT_MODELS = ("llama3.1:8b", "qwen2.5:3b")


def call_key(model: str, language: str, item_id: str, granularity: int, chunk_index: int) -> str:
    return "|".join((model, language, item_id, "g%d" % granularity, "c%d" % chunk_index))


class RunStore:
    """Append-only JSONL file; a line cut off by a power loss is removed on open."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.records: Dict[str, Dict[str, object]] = {}
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        if os.path.exists(path):
            self._load()

    def _load(self) -> None:
        with open(self.path, "rb") as handle:
            raw = handle.read()
        keep = raw.rfind(b"\n") + 1
        if keep != len(raw):
            with open(self.path, "r+b") as handle:
                handle.truncate(keep)
        for line in raw[:keep].decode("utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                self.records[str(record["key"])] = record

    def append(self, record: Dict[str, object]) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[str(record["key"])] = record

    def keys(self) -> Set[str]:
        return set(self.records)


def count_calls(n_items: int, models: Sequence[str], languages: Sequence[str] = LANGUAGES) -> int:
    return n_items * len(models) * len(languages) * sum(GRANULARITY_LEVELS)


def _clock(seconds: float) -> str:
    whole = int(max(0.0, seconds))
    return "%02d:%02d:%02d" % (whole // 3600, whole % 3600 // 60, whole % 60)


def run_benchmark(
    client: object,
    items: Sequence[Mapping[str, object]],
    store: RunStore,
    models: Sequence[str] = DEFAULT_MODELS,
    languages: Sequence[str] = LANGUAGES,
    log: Callable[[str], None] = print,
    log_every: int = 20,
) -> int:
    """Run all missing sub-calls and return how many were made in this session."""
    total = count_calls(len(items), models, languages)
    done_before = len(store.records)
    made = 0
    started = time.monotonic()
    log("Planned sub-calls: %d | already finished: %d" % (total, done_before))
    for model in models:
        for language in languages:
            for item in items:
                dependency = str(item["dependency"])
                fields = FIELDS_BY_DEPENDENCY[dependency]
                note = str(item["notes"][language])  # type: ignore[index]
                for granularity in GRANULARITY_LEVELS:
                    answers: List[Mapping[str, object]] = []
                    for chunk_index, chunk in enumerate(split_fields(fields, granularity)):
                        key = call_key(model, language, str(item["item_id"]), granularity, chunk_index)
                        if key in store.records:
                            answers.append(store.records[key].get("parsed") or {})  # type: ignore[arg-type]
                            continue
                        known: Optional[Dict[str, object]] = None
                        if dependency == "chained" and answers:
                            known = known_values_for(answers)
                        prompt = build_prompt(note, chunk, known)
                        reply = client.complete(model, prompt)  # type: ignore[attr-defined]
                        parsed = parse_reply(reply.text)
                        store.append({
                            "key": key, "model": model, "language": language,
                            "item_id": item["item_id"], "dependency": dependency,
                            "granularity": granularity, "chunk_index": chunk_index,
                            "fields": [spec.name for spec in chunk],
                            "known": known, "prompt_chars": len(prompt),
                            "response": reply.text, "parsed": parsed,
                            "prompt_tokens": reply.prompt_tokens,
                            "output_tokens": reply.output_tokens,
                            "seconds": round(reply.seconds, 4),
                        })
                        answers.append(parsed or {})
                        made += 1
                        finished = done_before + made
                        if made <= 3 or made % log_every == 0 or finished == total:
                            elapsed = time.monotonic() - started
                            eta = elapsed / made * (total - finished)
                            log("[%d/%d %.1f%%] %s %s %s g=%d | elapsed %s | ETA %s" % (
                                finished, total, 100.0 * finished / total, model, language,
                                item["item_id"], granularity, _clock(elapsed), _clock(eta)))
    return made


def write_run_meta(path: str, client: object, models: Sequence[str], installed: Mapping[str, str], n_items: int) -> None:
    meta = {
        "benchmark": "adgb-clinic-decomposition",
        "version": __version__,
        "models": {model: installed.get(model, "") for model in models},
        "ollama_version": client.version(),  # type: ignore[attr-defined]
        "options": client.options,  # type: ignore[attr-defined]
        "n_items": n_items,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")


def preflight(installed: Mapping[str, str], models: Sequence[str]) -> None:
    missing = [model for model in models if model not in installed]
    if missing:
        raise OllamaUnavailable("Missing models: %s. Run: %s" % (", ".join(missing), " && ".join("ollama pull %s" % m for m in missing)))

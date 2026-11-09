"""Scripted stand-in for Ollama; answers from the known truth and can inject mistakes."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from adgb.clinic_tasks import FIELDS_BY_DEPENDENCY
from adgb.llm import LLMReply

COMPUTE = {
    "total_sessions": lambda v: v["sessions_per_week"] * v["weeks"],
    "gross_fee": lambda v: v["total_sessions"] * v["fee_per_session"],
    "discount_amount": lambda v: v["gross_fee"] * v["discount_percent"] / 100,
    "payable": lambda v: v["gross_fee"] - v["discount_amount"],
}


class ScriptedLLM:
    """Wrong fields add 1 to a number or return "unknown" for a label."""

    def __init__(self, items: Sequence[Mapping[str, object]], wrong: Mapping[Tuple[str, str], Set[str]] = None, broken_json: Set[Tuple[str, str]] = None) -> None:
        self.by_note = {str(note): (item, language) for item in items for language, note in item["notes"].items()}  # type: ignore[attr-defined]
        self.wrong = wrong or {}
        self.broken_json = broken_json or set()
        self.prompts: List[str] = []

    def complete(self, model: str, prompt: str) -> LLMReply:
        self.prompts.append(prompt)
        note = prompt.split("Clinic note:\n", 1)[1]
        item, language = self.by_note[note]
        if (model, language) in self.broken_json:
            return LLMReply("not json at all", 50, 4, 0.5)
        fields = re.findall(r"^- (\w+):", prompt, flags=re.M)
        match = re.search(r"use them as given\): (\{.*\})", prompt)
        values: Dict[str, object] = dict(json.loads(match.group(1))) if match else {}
        truth = item["truth"]
        wrong = self.wrong.get((model, language), set())
        answer: Dict[str, object] = {}
        for name in fields:
            if name in COMPUTE and all(isinstance(values.get(p), (int, float)) for p in {"total_sessions": ["sessions_per_week", "weeks"], "gross_fee": ["total_sessions", "fee_per_session"], "discount_amount": ["gross_fee", "discount_percent"], "payable": ["gross_fee", "discount_amount"]}[name]):
                value = COMPUTE[name](values)
            else:
                value = truth[name]  # type: ignore[index]
            if name in wrong:
                value = value + 1 if isinstance(value, (int, float)) else "unknown"
            answer[name] = value
            values[name] = value
        return LLMReply(json.dumps(answer), 50, 10 * len(fields), 0.25 * len(fields))

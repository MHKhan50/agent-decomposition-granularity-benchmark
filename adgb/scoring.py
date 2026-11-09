"""Parse model replies and compare field values with the known answers."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Mapping, Optional

from .clinic_tasks import FieldSpec

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_reply(text: str) -> Optional[Dict[str, object]]:
    """Return the JSON object in a reply, or None when there is no valid object."""
    cleaned = text.strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def as_number(value: object) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER.search(str(value).replace(",", ""))
    return float(match.group()) if match else None


def field_correct(spec: FieldSpec, value: object, truth: object) -> bool:
    if spec.kind == "number":
        number = as_number(value)
        return number is not None and abs(number - float(truth)) < 1e-6  # type: ignore[arg-type]
    return value is not None and str(value).strip().lower() == str(truth).lower()


def score_form(fields: List[FieldSpec], answers: Mapping[str, object], truth: Mapping[str, object]) -> Dict[str, object]:
    """Score one finished form; a computed step counts only when its inputs were right."""
    correct = {spec.name: field_correct(spec, answers.get(spec.name), truth[spec.name]) for spec in fields}
    steps = [spec for spec in fields if spec.parents and all(correct[p] for p in spec.parents)]
    return {
        "fields_correct": sum(correct.values()),
        "n_fields": len(fields),
        "all_correct": int(all(correct.values())),
        "step_total": len(steps),
        "step_correct": sum(correct[spec.name] for spec in steps),
        "per_field": correct,
    }

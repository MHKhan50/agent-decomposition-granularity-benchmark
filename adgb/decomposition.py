"""Split a form into sub-calls and build the prompt for each sub-call."""
from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .clinic_tasks import FieldSpec

GRANULARITY_LEVELS: Tuple[int, ...] = (1, 2, 4, 8)


def split_fields(fields: Sequence[FieldSpec], granularity: int) -> List[Tuple[FieldSpec, ...]]:
    """Cut the form into equal consecutive groups, keeping the field order."""
    if granularity not in GRANULARITY_LEVELS:
        raise ValueError("granularity must be one of %s" % (GRANULARITY_LEVELS,))
    if len(fields) % granularity:
        raise ValueError("%d fields cannot be split evenly into %d" % (len(fields), granularity))
    size = len(fields) // granularity
    return [tuple(fields[i:i + size]) for i in range(0, len(fields), size)]


def build_prompt(note: str, chunk: Sequence[FieldSpec], known: Optional[Mapping[str, object]] = None) -> str:
    """Build one sub-call prompt.

    The field list comes before the note on purpose. Each sub-call then starts
    with different text, so the Ollama prompt cache cannot make later sub-calls
    look cheaper than the first one.
    """
    lines = ["Fill in these fields from the clinic note.", "", "Fields:"]
    lines += ["- %s: %s" % (spec.name, spec.description) for spec in chunk]
    lines += [
        "",
        "Rules: the note may also mention older or other information; use only the current "
        "information for this patient. Write numbers as plain digits. Return one JSON object "
        "with exactly the listed keys and nothing else.",
    ]
    if known:
        lines += [
            "",
            "Values worked out in earlier steps (use them as given): "
            + json.dumps(dict(known), ensure_ascii=False, sort_keys=True),
        ]
    lines += ["", "Clinic note:", note]
    return "\n".join(lines)


def known_values_for(previous_answers: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Merge the answers from earlier sub-calls; later calls see exactly what was produced."""
    merged: Dict[str, object] = {}
    for answer in previous_answers:
        merged.update(answer)
    return merged

"""Clinic form-filling tasks for the local LLM decomposition experiment.

Every case is invented from a fixed seed. No real patient information is used.
Each case is written in English and in Roman Urdu with identical facts, so the
two languages can be compared item by item.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

DEPENDENCIES: Tuple[str, str] = ("independent", "chained")
LANGUAGES: Tuple[str, str] = ("english", "roman_urdu")
TASK_SEED = 42
CASES_PER_DEPENDENCY = 16


@dataclass(frozen=True)
class FieldSpec:
    """One form field: how it is described, compared and what it relies on."""

    name: str
    description: str
    kind: str  # "number" or "label"
    parents: Tuple[str, ...] = ()


INDEPENDENT_FIELDS: Tuple[FieldSpec, ...] = (
    FieldSpec("age", "the patient's current age in years, as a number", "number"),
    FieldSpec("side", "the side that hurts today, exactly left or right", "label"),
    FieldSpec("body_part", "the painful joint, exactly knee, shoulder, hip or ankle", "label"),
    FieldSpec("pain_score", "today's pain score out of 10, as a number", "number"),
    FieldSpec("sessions_per_week", "planned sessions per week, as a number", "number"),
    FieldSpec("weeks", "planned number of weeks, as a number", "number"),
    FieldSpec("fee_per_session", "this patient's fee per session in rupees, as a number", "number"),
    FieldSpec("followup_day", "the day of the next review, one English weekday in lower case", "label"),
)

CHAINED_FIELDS: Tuple[FieldSpec, ...] = (
    FieldSpec("sessions_per_week", "sessions per week in the current course, as a number", "number"),
    FieldSpec("weeks", "number of weeks in the current course, as a number", "number"),
    FieldSpec("total_sessions", "sessions_per_week multiplied by weeks", "number", ("sessions_per_week", "weeks")),
    FieldSpec("fee_per_session", "fee per session in rupees in the current course, as a number", "number"),
    FieldSpec("gross_fee", "total_sessions multiplied by fee_per_session", "number", ("total_sessions", "fee_per_session")),
    FieldSpec("discount_percent", "discount percent of the current course, as a number", "number"),
    FieldSpec("discount_amount", "gross_fee multiplied by discount_percent divided by 100", "number", ("gross_fee", "discount_percent")),
    FieldSpec("payable", "gross_fee minus discount_amount", "number", ("gross_fee", "discount_amount")),
)

FIELDS_BY_DEPENDENCY: Dict[str, Tuple[FieldSpec, ...]] = {
    "independent": INDEPENDENT_FIELDS,
    "chained": CHAINED_FIELDS,
}

_SIDES = {"left": "bayen", "right": "dayen"}
_PARTS = {"knee": "ghutne", "shoulder": "kandhe", "hip": "koolhe", "ankle": "takhne"}
_DAYS = {
    "monday": "Peer", "tuesday": "Mangal", "wednesday": "Budh", "thursday": "Jumerat",
    "friday": "Jumma", "saturday": "Hafta", "sunday": "Itwar",
}

_INDEPENDENT_TEMPLATES: Tuple[Dict[str, str], ...] = (
    {
        "english": (
            "Patient {pid} is {age} years old and came in with pain in the {side} {part}. "
            "Pain today is {pain} out of 10. At the last visit the pain was {old_pain} out of 10 "
            "in the {other} {part}. Plan: {spw} sessions per week for {weeks} weeks. "
            "Each session costs {fee} rupees. The patient's brother, aged {rel_age}, paid "
            "{old_fee} rupees per session last year. The next review is on {day_en}; the previous "
            "review was on {old_day_en}."
        ),
        "roman_urdu": (
            "Mareez {pid} ki umar {age} saal hai aur {side_ru} {part_ru} mein dard ki shikayat hai. "
            "Aaj dard 10 mein se {pain} hai. Pichli visit par {other_ru} {part_ru} mein dard 10 mein "
            "se {old_pain} tha. Plan: hafte mein {spw} session, {weeks} hafton tak. Har session ki "
            "fees {fee} rupay hai. Mareez ke bhai, jin ki umar {rel_age} saal hai, ne pichle saal "
            "{old_fee} rupay per session diye the. Agla review {day_ru} ko hai; pichla review "
            "{old_day_ru} ko hua tha."
        ),
    },
    {
        "english": (
            "{pid}: {age}-year-old with {side} {part} pain, score {pain}/10 today (it was "
            "{old_pain}/10 in the {other} {part} at the last visit). The course of treatment is "
            "{weeks} weeks with {spw} sessions every week, {fee} rupees a session. Last year the "
            "patient's brother, then {rel_age}, was charged {old_fee} rupees a session. The review "
            "was previously on {old_day_en} and will next be on {day_en}."
        ),
        "roman_urdu": (
            "{pid}: umar {age} saal, {side_ru} {part_ru} mein dard, aaj ka score {pain}/10 (pichli "
            "visit par {other_ru} {part_ru} mein {old_pain}/10 tha). Ilaaj {weeks} hafte chalega, "
            "har hafte {spw} session, aik session ke {fee} rupay. Pichle saal mareez ke bhai, us "
            "waqt umar {rel_age} saal, se {old_fee} rupay per session liye gaye the. Review pehle "
            "{old_day_ru} ko tha, ab agla {day_ru} ko hoga."
        ),
    },
)

_CHAINED_TEMPLATES: Tuple[Dict[str, str], ...] = (
    {
        "english": (
            "Billing note for patient {pid}. The plan is {spw} sessions per week for {weeks} weeks "
            "at {fee} rupees per session. A discount of {disc} percent applies to this course. An "
            "earlier course, now closed, had {old_spw} sessions per week for {old_weeks} weeks at "
            "{old_fee} rupees per session with a {old_disc} percent discount."
        ),
        "roman_urdu": (
            "Mareez {pid} ka billing note. Plan hafte mein {spw} session, {weeks} hafton ke liye, "
            "{fee} rupay per session hai. Is course par {disc} feesad discount hai. Pichla course, "
            "jo ab band ho chuka hai, hafte mein {old_spw} session, {old_weeks} hafton ke liye, "
            "{old_fee} rupay per session aur {old_disc} feesad discount par tha."
        ),
    },
    {
        "english": (
            "{pid}, billing. Earlier closed course: {old_weeks} weeks, {old_spw} sessions a week, "
            "{old_fee} rupees each, {old_disc} percent off. Current course: {weeks} weeks, {spw} "
            "sessions a week, {fee} rupees each, {disc} percent off."
        ),
        "roman_urdu": (
            "{pid}, billing. Pichla band course: {old_weeks} hafte, hafte mein {old_spw} session, "
            "har session {old_fee} rupay, {old_disc} feesad discount. Maujooda course: {weeks} "
            "hafte, hafte mein {spw} session, har session {fee} rupay, {disc} feesad discount."
        ),
    },
)


def _different(rng: random.Random, choices: Sequence[int], avoid: int) -> int:
    return rng.choice([c for c in choices if c != avoid])


def _independent_case(rng: random.Random, index: int) -> Dict[str, object]:
    side = rng.choice(sorted(_SIDES))
    other = "left" if side == "right" else "right"
    part = rng.choice(sorted(_PARTS))
    day, old_day = rng.sample(sorted(_DAYS), 2)
    age = rng.randint(18, 79)
    pain = rng.randint(1, 9)
    fee = rng.randrange(500, 3100, 100)
    values = dict(
        pid="P-%04d" % rng.randint(1000, 9999), age=age, side=side, other=other, part=part,
        side_ru=_SIDES[side],
        other_ru=_SIDES[other], part_ru=_PARTS[part],
        pain=pain, old_pain=_different(rng, range(1, 10), pain),
        spw=rng.randint(1, 5), weeks=rng.randint(2, 12), fee=fee,
        rel_age=_different(rng, range(18, 80), age),
        old_fee=_different(rng, range(500, 3100, 100), fee),
        day_en=day.capitalize(), old_day_en=old_day.capitalize(),
        day_ru=_DAYS[day], old_day_ru=_DAYS[old_day],
    )
    template = _INDEPENDENT_TEMPLATES[index % len(_INDEPENDENT_TEMPLATES)]
    truth: Dict[str, object] = {
        "age": age, "side": side, "body_part": part, "pain_score": pain,
        "sessions_per_week": values["spw"], "weeks": values["weeks"],
        "fee_per_session": fee, "followup_day": day,
    }
    return {
        "item_id": "ind-%02d" % (index + 1), "dependency": "independent",
        "template": index % len(_INDEPENDENT_TEMPLATES), "truth": truth,
        "notes": {language: template[language].format(**values) for language in LANGUAGES},
    }


def _chained_case(rng: random.Random, index: int) -> Dict[str, object]:
    spw = rng.randint(1, 5)
    weeks = rng.randint(2, 12)
    fee = rng.randrange(500, 3100, 100)
    disc = rng.choice([10, 20, 25, 50])
    total = spw * weeks
    gross = total * fee
    amount = gross * disc // 100
    values = dict(
        pid="P-%04d" % rng.randint(1000, 9999), spw=spw, weeks=weeks, fee=fee, disc=disc,
        old_spw=_different(rng, range(1, 6), spw),
        old_weeks=_different(rng, range(2, 13), weeks),
        old_fee=_different(rng, range(500, 3100, 100), fee),
        old_disc=_different(rng, [10, 20, 25, 50], disc),
    )
    template = _CHAINED_TEMPLATES[index % len(_CHAINED_TEMPLATES)]
    truth: Dict[str, object] = {
        "sessions_per_week": spw, "weeks": weeks, "total_sessions": total, "fee_per_session": fee,
        "gross_fee": gross, "discount_percent": disc, "discount_amount": amount,
        "payable": gross - amount,
    }
    return {
        "item_id": "chn-%02d" % (index + 1), "dependency": "chained",
        "template": index % len(_CHAINED_TEMPLATES), "truth": truth,
        "notes": {language: template[language].format(**values) for language in LANGUAGES},
    }


def build_tasks(seed: int = TASK_SEED, cases_per_dependency: int = CASES_PER_DEPENDENCY) -> List[Dict[str, object]]:
    """Return all cases in a fixed order; the same seed always gives the same text."""
    rng = random.Random(seed)
    items = [_independent_case(rng, i) for i in range(cases_per_dependency)]
    items += [_chained_case(rng, i) for i in range(cases_per_dependency)]
    return items


def write_tasks(path: str, items: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def load_tasks(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

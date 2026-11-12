"""Assess declared review data; do not authenticate a person, clip, or timeline."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator

from core.budget import BudgetLedger
from core.json_data import unique_fields


MAX_SCORECARD_BYTES = 64 * 1024
SCHEMA_PATH = Path(__file__).with_name("scorecard.schema.json")
_VALIDATOR = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


class ScorecardError(ValueError):
    """Declared review data does not meet the scorecard contract."""


@dataclass(frozen=True)
class ScorecardAssessment:
    """A rubric result from declarations, without a global visual verdict."""

    rubric_verdict: str
    qualifying_failure_indexes: tuple[int, ...]


def _parse_integer(token: str) -> int:
    if len(token.removeprefix("-")) > 12:
        raise ScorecardError("Scorecard numbers must have at most twelve whole digits.")
    return int(token)


def _parse_decimal(token: str) -> Decimal:
    if len(token) > 32:
        raise ScorecardError("Scorecard numeric tokens must have at most 32 characters.")
    exponent = token.lower().partition("e")[2]
    if exponent and (len(exponent) > 3 or not -9 <= int(exponent) <= 11):
        raise ScorecardError("Scorecard numeric exponents exceed the allowed range.")
    value = Decimal(token)
    if not value.is_finite() or value.as_tuple().exponent < -9 or value.adjusted() > 11:
        raise ScorecardError("Scorecard numbers allow twelve whole and nine fractional digits.")
    return value


def _reject_constant(value):
    raise ScorecardError("Scorecards must use finite standard JSON numbers.")


def _check_utc(value: str) -> None:
    pattern = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|\+00:00)"
    if re.fullmatch(pattern, value) is None:
        raise ScorecardError("Review time must be a UTC timestamp ending in Z or +00:00.")
    datetime.fromisoformat(value.replace("Z", "+00:00"))


def _failure_indexes(failures: list[dict]) -> tuple[int, ...]:
    qualifying = []
    unoccluded_runs = {}
    for index, failure in enumerate(failures):
        start, end = Decimal(failure["startPTS"]), Decimal(failure["endPTS"])
        if start >= end:
            raise ScorecardError("Failure intervals must have an end after their start.")
        kind = failure["kind"]
        if kind != "catastrophic" and not failure["continuous"]:
            raise ScorecardError("Report ordinary artifacts as separate continuous runs.")
        if failure["unoccluded"]:
            unoccluded_runs.setdefault(kind, []).append((start, end))
        if kind == "catastrophic" or (failure["unoccluded"] and end - start >= Decimal("0.250")):
            qualifying.append(index)
    for intervals in unoccluded_runs.values():
        ordered = sorted(intervals)
        if any(later[0] <= earlier[1] for earlier, later in zip(ordered, ordered[1:])):
            raise ScorecardError("Same-kind unoccluded intervals overlap or touch; use one interval or an explicit gap.")
    return tuple(qualifying)


def assess_scorecard(contents: bytes | None, *, run_id: str, attempt_id: str) -> ScorecardAssessment:
    """Validate bounded JSON and assess the declared scores and artifact intervals.

    None or an otherwise valid unsigned draft returns NOT_RUN. A complete name,
    UTC time, and signature permit rubric grading, but do not prove a real human
    supplied them. Malformed or mismatched data raises ScorecardError. Require
    at least two full normal-speed plays and inspected source times zero to four.

    Intervals use declared native PTS and half-open [startPTS, endPTS) bounds.
    The end is the next unaffected frame boundary, so six 24 FPS frames can span
    exactly 0.250 seconds. This function does not establish a final display end
    or bind declarations to actual frame timestamps. Do not infer those from FPS
    or container duration. Separate runs with a positive gap are never summed.
    """
    try:
        BudgetLedger._run_id(run_id)
        BudgetLedger._attempt_id(attempt_id)
        if contents is None:
            return ScorecardAssessment("NOT_RUN", ())
        if type(contents) is not bytes or not 0 < len(contents) <= MAX_SCORECARD_BYTES:
            raise ScorecardError("Scorecard must contain 1 byte to 64 KiB of UTF-8 JSON.")
        # Input digits are bounded before arithmetic. Do not inherit caller traps
        # or precision; these finite subtractions need at most 22 digits.
        context = Context(prec=32, rounding=ROUND_HALF_EVEN, Emin=-99, Emax=99,
                          capitals=1, clamp=0, flags=[], traps=[])
        with localcontext(context):
            data = json.loads(contents.decode("utf-8"), object_pairs_hook=unique_fields,
                              parse_int=_parse_integer, parse_float=_parse_decimal, parse_constant=_reject_constant)
            _VALIDATOR.validate(data)
            if data["runId"] != run_id or data["attemptId"] != attempt_id:
                raise ScorecardError("Scorecard belongs to another run or attempt.")
            reviewed_at = data.get("reviewedAtUtc", "")
            if reviewed_at.strip():
                _check_utc(reviewed_at)
            failures = _failure_indexes(data["hardFailures"])
            if any(not data.get(field, "").strip() for field in ("reviewerName", "reviewedAtUtc", "signature")):
                return ScorecardAssessment("NOT_RUN", ())
            verdict = "FAIL" if failures or any(value < 3 for value in data["scores"].values()) else "PASS"
            return ScorecardAssessment(verdict, failures)
    except ScorecardError:
        raise
    except Exception:
        raise ScorecardError("Scorecard must match the declared review contract.") from None

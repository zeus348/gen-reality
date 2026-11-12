"""Check declared rubric rules; synthetic cards do not establish human review."""

from copy import deepcopy
from decimal import Inexact, ROUND_UP, localcontext
import json
import unittest
from unittest.mock import patch

from features.review import score


RUN = "review-run"
ATTEMPT = "att-" + "a" * 32


def card(**changes):
    return {"schemaVersion": "1.4", "runId": RUN, "attemptId": ATTEMPT,
            "reviewerName": "Fixture reviewer", "reviewedAtUtc": "2026-11-12T10:00:00Z",
            "signature": "Fixture sign-off", "normalSpeedPlays": 2, "inspectedSeconds": [0, 1, 2, 3, 4],
            "scores": {"identity": 3, "motion": 3, "scene": 3, "placement": 3},
            "hardFailures": [], "comment": "Synthetic rubric fixture, not a live visual review."} | changes


def event(start=0, end=0.25, kind="duplicate", **changes):
    return {"kind": kind, "startPTS": start, "endPTS": end,
            "reason": "Fixture artifact observation.", "continuous": True, "unoccluded": True} | changes


def encode(data):
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


class ScorecardTests(unittest.TestCase):
    def assess(self, data):
        return score.assess_scorecard(encode(data), run_id=RUN, attempt_id=ATTEMPT)

    def test_complete_declared_review_can_pass_with_two_or_more_plays(self):
        for plays, reviewed_at in ((2, "2026-11-12T10:00:00Z"), (3, "2026-11-12T10:00:00.123456+00:00")):
            with self.subTest(plays=plays):
                result = self.assess(card(normalSpeedPlays=plays, reviewedAtUtc=reviewed_at))
                self.assertEqual(result.rubric_verdict, "PASS")
                self.assertEqual(result.qualifying_failure_indexes, ())

    def test_each_dimension_must_pass_without_averaging(self):
        for dimension in ("identity", "motion", "scene", "placement"):
            ratings = {"identity": 5, "motion": 5, "scene": 5, "placement": 5} | {dimension: 2}
            with self.subTest(dimension=dimension):
                result = self.assess(card(scores=ratings))
                self.assertEqual(result.rubric_verdict, "FAIL")
                self.assertEqual(result.qualifying_failure_indexes, ())

    def test_absent_or_unsigned_valid_draft_is_not_run(self):
        result = score.assess_scorecard(None, run_id=RUN, attempt_id=ATTEMPT)
        self.assertEqual((result.rubric_verdict, result.qualifying_failure_indexes), ("NOT_RUN", ()))
        for field in ("reviewerName", "reviewedAtUtc", "signature"):
            for value in (None, "", "   "):
                data = card(hardFailures=[event()])
                if value is None:
                    del data[field]
                else:
                    data[field] = value
                with self.subTest(field=field, value=value):
                    result = self.assess(data)
                    self.assertEqual((result.rubric_verdict, result.qualifying_failure_indexes), ("NOT_RUN", ()))

    def test_scores_plays_inspected_times_and_required_observation_are_strict(self):
        cases = [card(normalSpeedPlays=value) for value in (0, 1, True, 2.0, "2")]
        cases += [card(inspectedSeconds=value) for value in
                  ([0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [0, 1, 1, 3, 4], [4, 3, 2, 1, 0],
                   [0, True, 2, 3, 4], [0, 1.0, 2, 3, 4], "0,1,2,3,4")]
        for value in (0, 6, True, 3.0, "3", None):
            data = card()
            data["scores"]["identity"] = value
            cases.append(data)
        for field in ("schemaVersion", "scores", "comment", "hardFailures"):
            data = card()
            del data[field]
            cases.append(data)
        data = card()
        del data["scores"]["scene"]
        cases.extend([data, card(comment=" "), card(scores=card()["scores"] | {"average": 3})])
        for index, data in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(score.ScorecardError):
                self.assess(data)

    def test_exact_native_pts_boundary_uses_half_open_intervals(self):
        for kind in ("duplicate", "missing_limb", "replaced_subject", "teleport_or_slip"):
            for end, verdict in ((1.249, "PASS"), (1.249999999, "PASS"), (1.250, "FAIL"), (1.250000001, "FAIL")):
                with self.subTest(kind=kind, end=end):
                    result = self.assess(card(hardFailures=[event(1, end, kind)]))
                    self.assertEqual(result.rubric_verdict, verdict)
                    self.assertEqual(result.qualifying_failure_indexes, (0,) if verdict == "FAIL" else ())
        # Six bad frames at 24 FPS occupy [0, 6/24), including the sixth frame's display time.
        six_frames = self.assess(card(hardFailures=[event(0, 0.250)]))
        self.assertEqual((six_frames.rubric_verdict, six_frames.qualifying_failure_indexes), ("FAIL", (0,)))

    def test_occluded_observations_and_justified_catastrophic_failures_are_distinct(self):
        observations = [event(0, 0.5, unoccluded=False),
                        event(1, 1.001, "catastrophic", continuous=False, unoccluded=False)]
        result = self.assess(card(hardFailures=observations[:1]))
        self.assertEqual((result.rubric_verdict, result.qualifying_failure_indexes), ("PASS", ()))
        result = self.assess(card(hardFailures=observations))
        self.assertEqual((result.rubric_verdict, result.qualifying_failure_indexes), ("FAIL", (1,)))
        with self.assertRaises(score.ScorecardError):
            self.assess(card(hardFailures=[event(1, 1.001, "catastrophic", reason=" ")]))

    def test_discontinuous_ordinary_artifacts_require_separate_runs(self):
        for kind in ("duplicate", "missing_limb", "replaced_subject", "teleport_or_slip"):
            with self.subTest(kind=kind), self.assertRaisesRegex(score.ScorecardError, "separate continuous runs"):
                self.assess(card(hardFailures=[event(0, 1, kind, continuous=False)]))

    def test_split_touching_or_overlapping_runs_cannot_evade_the_threshold(self):
        for observations in ([event(0, 0.125), event(0.125, 0.25)],
                             [event(0.1, 0.3), event(0, 0.2)]):
            with self.subTest(observations=observations), self.assertRaisesRegex(score.ScorecardError, "overlap or touch"):
                self.assess(card(hardFailures=observations))

    def test_runs_with_positive_gaps_are_never_added_together(self):
        result = self.assess(card(hardFailures=[event(0, 0.249), event(0.250, 0.499)]))
        self.assertEqual((result.rubric_verdict, result.qualifying_failure_indexes), ("PASS", ()))
        result = self.assess(card(hardFailures=[event(0, 0.2), event(0.1, 0.3, "missing_limb")]))
        self.assertEqual((result.rubric_verdict, result.qualifying_failure_indexes), ("PASS", ()))

    def test_invalid_interval_fields_and_order_are_errors(self):
        observations = [event(0, 0), event(1, 0.5), event(-0.1, 0.2), event("0", 0.25),
                        event(0, True), event(kind="unknown"), event(unoccluded=1), event(continuous="true"),
                        event(reason=""), event(extra="untrusted")]
        for observation in observations:
            with self.subTest(observation=observation), self.assertRaises(score.ScorecardError):
                self.assess(card(hardFailures=[observation]))

    def test_exact_arithmetic_ignores_ambient_decimal_precision_rounding_and_traps(self):
        template = encode(card(hardFailures=[event("START", "END")]))
        with localcontext() as caller:
            caller.prec = 2
            caller.rounding = ROUND_UP
            caller.Emin, caller.Emax = -2, 2
            for signal in caller.traps:
                caller.traps[signal] = True
            for end, verdict in ((b"999999999999.249999999", "PASS"), (b"999999999999.250000000", "FAIL")):
                contents = template.replace(b'"START"', b"999999999999.000000000").replace(b'"END"', end)
                result = score.assess_scorecard(contents, run_id=RUN, attempt_id=ATTEMPT)
                self.assertEqual(result.rubric_verdict, verdict)
            self.assertEqual((caller.prec, caller.rounding, caller.Emin, caller.Emax), (2, ROUND_UP, -2, 2))
            self.assertIs(caller.traps[Inexact], True)

    def test_nonfinite_and_unbounded_numeric_tokens_are_rejected(self):
        template = encode(card(hardFailures=[event(0, "END")]))
        tokens = ("NaN", "Infinity", "-Infinity", "1e999999999", "1e-999999999", "0.2499999999",
                  "1000000000000", "0." + "0" * 40 + "1")
        for token in tokens:
            contents = template.replace(b'"END"', token.encode())
            with self.subTest(token=token), self.assertRaises(score.ScorecardError):
                score.assess_scorecard(contents, run_id=RUN, attempt_id=ATTEMPT)

    def test_nonblank_review_time_must_be_an_actual_utc_timestamp(self):
        for value in ("2026-11-12", "2026-11-12T10:00:00", "2026-11-12T10:00:00+01:00",
                      "2026-02-30T10:00:00Z", "2026-11-12T25:00:00Z", " 2026-11-12T10:00:00Z",
                      "2026-11-12T10:00:00-00:00", 1000, None):
            with self.subTest(value=value), self.assertRaises(score.ScorecardError):
                self.assess(card(reviewedAtUtc=value))

    def test_mixed_ids_and_input_verdict_or_provenance_claims_are_rejected(self):
        cases = [card(runId="other-run"), card(attemptId="att-" + "b" * 32), card(schemaVersion="1.3")]
        for field, value in (("rubric_verdict", "PASS"), ("visualEffectDemonstrated", "PASS"),
                             ("automatedWorkflowComplete", "PASS"), ("humanVerified", True)):
            cases.append(card(**{field: value}))
        for data in cases:
            with self.subTest(data=data), self.assertRaises(score.ScorecardError):
                self.assess(data)
        for changes in ({"run_id": "../other"}, {"attempt_id": "att-invalid"}):
            with self.subTest(changes=changes), self.assertRaises(score.ScorecardError):
                score.assess_scorecard(None, **({"run_id": RUN, "attempt_id": ATTEMPT} | changes))

    def test_invalid_json_duplicate_fields_and_byte_bounds_fail_safely(self):
        contents = encode(card())
        duplicates = [contents.replace(b'"identity":3', b'"identity":1,"identity":3'),
                      contents.replace(b'"runId":"review-run"', b'"runId":"private-secret","runId":"review-run"')]
        for value in (b"", b"\xff", b'{"private-secret":', b"[]", "not bytes", bytearray(contents), *duplicates):
            with self.subTest(type=type(value).__name__), self.assertRaises(score.ScorecardError) as caught:
                score.assess_scorecard(value, run_id=RUN, attempt_id=ATTEMPT)
            self.assertNotIn("private-secret", str(caught.exception))
        with patch.object(score, "MAX_SCORECARD_BYTES", len(contents) - 1):
            with self.assertRaises(score.ScorecardError):
                score.assess_scorecard(contents, run_id=RUN, attempt_id=ATTEMPT)

    def test_unsigned_data_still_has_to_meet_the_rest_of_the_contract(self):
        data = card(signature="")
        for changes in ({"scores": {}}, {"reviewedAtUtc": "invalid"}, {"reviewerName": None},
                        {"hardFailures": [event(1, 0)]}, {"signature": False}):
            with self.subTest(changes=changes), self.assertRaises(score.ScorecardError):
                self.assess(deepcopy(data) | changes)

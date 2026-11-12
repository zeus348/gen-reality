"""Inventory real prepared media and mocked completed tasks without provider calls."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jsonschema import ValidationError

from core.budget import BudgetLedger
from core.storage import StoredArtifact, publish_bytes, read_verified
from features.inference.frozen import FrozenCreate
from features.inference.submission import SubmissionCoordinator
from features.review import export
import test_scorecard_reader as fixtures


RUN, ATTEMPT_A, ATTEMPT_B = fixtures.RUN, fixtures.ATTEMPT_A, fixtures.ATTEMPT_B


class ExportInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reader_fixture = type("ExportReaderFixture", (fixtures.ScorecardReaderTests,), {})
        reader_fixture.setUpClass()
        cls.addClassCleanup(reader_fixture.doClassCleanups)
        cls.fixture = reader_fixture
        cls.base = reader_fixture.base
        cls.manifests = reader_fixture.manifests
        binding_fixture = reader_fixture.fixture
        cls.media = binding_fixture.fixture

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        shutil.copytree(self.base, self.root, dirs_exist_ok=True)
        self.run_root = self.root / RUN
        self.ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 5000.0)
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.original_review = self.data(self.manifests[ATTEMPT_A])
        self.original_metadata = self.data(self.media.completed[ATTEMPT_A].metadata)

    def data(self, record):
        return json.loads(read_verified(self.run_root, record))

    def build(self, *, attempt=ATTEMPT_A, manifest=None, binding=None, **changes):
        arguments = {"run_id": RUN, "attempt_id": attempt, "scorecard_binding": binding} | changes
        return export.build_export_inventory(self.owner, self.root, manifest or self.manifests[attempt], **arguments)

    def assert_no_inventory(self):
        for attempt in (ATTEMPT_A, ATTEMPT_B):
            self.assertFalse((self.run_root / f"attempts/{attempt}/export").exists())

    def replace_review(self, data):
        record = self.manifests[ATTEMPT_A]
        contents = json.dumps(data, sort_keys=True).encode()
        (self.run_root / record.path).write_bytes(contents)
        return StoredArtifact(record.path, sha256(contents).hexdigest(), len(contents))

    def replace_metadata(self, data, **review_changes):
        old = self.media.completed[ATTEMPT_A].metadata
        contents = json.dumps(data, sort_keys=True).encode()
        (self.run_root / old.path).write_bytes(contents)
        record = StoredArtifact(old.path, sha256(contents).hexdigest(), len(contents))
        with self.ledger._connection() as connection:
            connection.execute("UPDATE completed_outputs SET metadata_sha256=?, metadata_size=? WHERE attempt_id=?",
                               (record.sha256, record.size, ATTEMPT_A))
        return self.replace_review(self.original_review | {"rawMetadata": asdict(record)} | review_changes)

    def replace_check(self, changes):
        data = self.data(self.media.checks[ATTEMPT_A].evidence) | changes
        if "frozen" in changes:
            data["payloadSha256"] = FrozenCreate(**data["frozen"]).payload_sha256
        contents = json.dumps(data, sort_keys=True).encode()
        record = publish_bytes(self.run_root,
            f"attempts/{ATTEMPT_A}/provider_checks/{sha256(contents).hexdigest()}.json", contents)
        return self.replace_review(self.original_review | {"requestCheck": asdict(record)})

    def ledger_dump(self):
        with self.ledger._connection() as connection:
            return list(connection.iterdump())

    def test_real_modes_include_all_originals_both_guides_and_exact_selected_samples(self):
        for attempt, mode, count in ((ATTEMPT_A, "A", 23), (ATTEMPT_B, "B", 28)):
            with self.subTest(mode=mode):
                record = self.build(attempt=attempt)
                inventory = self.data(record)
                files = {entry["path"]: entry for entry in inventory["files"]}
                preparation = self.data(self.media.manifests[attempt])
                review = self.data(self.manifests[attempt])
                expected = {self.media.manifests[attempt].path, self.manifests[attempt].path,
                            self.media.completed[attempt].raw.path, self.media.completed[attempt].metadata.path}
                expected.update(value["path"] for value in preparation["artifacts"].values())
                expected.update(sample[view]["image"]["path"] for sample in review["samples"]
                                for view in ("source", "guide", "raw"))
                self.assertEqual(set(files), expected)
                self.assertEqual(len(files), count)
                self.assertEqual(list(files), sorted(files))
                self.assertEqual((inventory["runId"], inventory["attemptId"], inventory["mode"]), (RUN, attempt, mode))
                self.assertEqual(inventory["sourceKindDeclared"], "synthetic_fixture")
                self.assertEqual(inventory["selection"], {"preparationManifest": asdict(self.media.manifests[attempt]),
                    "reviewManifest": asdict(self.manifests[attempt]), "scorecardBinding": None, "scorecard": None})
                self.assertEqual(record.path, f"attempts/{attempt}/export/inventories/{record.sha256}.json")
                for entry in files.values():
                    self.assertEqual(entry["roles"], sorted(set(entry["roles"])))
                    read_verified(self.run_root, StoredArtifact(**{key: entry[key] for key in ("path", "sha256", "size")}))
                prepared = self.media.prepared[attempt]
                for retained in (prepared.source_original, prepared.asset_original, prepared.keyframe_a, prepared.guide_b,
                                 self.media.completed[attempt].raw):
                    self.assertEqual({name: files[retained.path][name] for name in ("path", "sha256", "size")}, asdict(retained))
                if mode == "A":
                    self.assertEqual(files[prepared.keyframe_a.path]["roles"], ["guide_a", "guide_sample", "submitted_guide"])
                    self.assertEqual(files[prepared.normalized_source.path]["roles"], ["provider_video_input", "source_normalized"])
                else:
                    self.assertEqual(files[prepared.guide_b.path]["roles"], ["guide_b", "provider_video_input", "submitted_guide"])
                self.assertEqual(inventory["totalFileBytes"], sum(entry["size"] for entry in files.values()))
                self.assertEqual(inventory["visualEffectDemonstrated"], "NOT_RUN")
                self.assertEqual(inventory["automatedWorkflowComplete"], "NOT_RUN")
                self.assertEqual(inventory["availability"], {
                    "scorecard": "NOT_SELECTED", "providerProvenance": "NOT_ESTABLISHED", "humanProvenance": "NOT_ESTABLISHED",
                    "actualChargeAudit": "NOT_RUN", "runBudgetAudit": "NOT_RUN", "elapsedBudgetAudit": "NOT_RUN",
                    "recordedTestAudit": "NOT_RUN"})

    def test_private_records_are_verified_hashes_only_and_provider_summary_is_fixed(self):
        inventory = self.data(self.build())
        paths = {entry["path"] for entry in inventory["files"]}
        self.assertEqual(set(inventory["privateEvidenceHashes"]), {"requestCheck", "request", "frozenRequest", "providerSuccess"})
        for name, record in inventory["privateEvidenceHashes"].items():
            self.assertEqual(record, {field: self.original_review[name][field] for field in ("sha256", "size")})
            self.assertNotIn(self.original_review[name]["path"], paths)
        self.assertEqual(inventory["provider"], {
            "vendor": "Runway", "model": "aleph2", "route": "POST /v1/video_to_video", "sdkVersion": "5.24.0",
            "taskId": "fixture-task-a", "recordedStatus": "SUCCEEDED",
            "payloadSha256": self.media.checks[ATTEMPT_A].frozen.payload_sha256,
            "requestBodySha256": self.original_review["request"]["sha256"], "reservedCents": 140, "completedUnix": 1001.0})
        self.assertNotIn("actualCents", inventory["provider"])

    def test_explicit_binding_selection_preserves_old_corrected_and_unselected_records(self):
        for attempt, binding, count in ((ATTEMPT_A, self.fixture.bindings[ATTEMPT_A], 25),
                                        (ATTEMPT_A, self.fixture.corrected, 25),
                                        (ATTEMPT_B, self.fixture.bindings[ATTEMPT_B], 30)):
            with self.subTest(binding=binding):
                inventory = self.data(self.build(attempt=attempt, binding=binding))
                card = self.data(binding)["scorecard"]
                self.assertEqual(inventory["selection"]["scorecardBinding"], asdict(binding))
                self.assertEqual(inventory["selection"]["scorecard"], card)
                selected = [entry["path"] for entry in inventory["files"]
                            if "scorecard_binding" in entry["roles"] or "scorecard" in entry["roles"]]
                self.assertEqual(set(selected), {binding.path, card["path"]})
                self.assertEqual(len(inventory["files"]), count)
                self.assertEqual(inventory["availability"]["scorecard"], "SELECTED")
                self.assertEqual(inventory["visualEffectDemonstrated"], "NOT_RUN")
        unselected = self.data(self.build())
        self.assertFalse(any("scorecard" in entry["path"] for entry in unselected["files"]))
        for binding in (*self.fixture.bindings.values(), self.fixture.corrected):
            read_verified(self.run_root, binding)

    def test_no_directory_scan_or_private_url_secret_html_or_other_attempt_is_selected(self):
        sentinel = "private-export-environment-secret"
        for relative in (f"attempts/{ATTEMPT_A}/review/index.html", f"attempts/{ATTEMPT_A}/extra-secret.json",
                         f"attempts/{ATTEMPT_B}/other-secret.json"):
            publish_bytes(self.run_root, relative, sentinel.encode())
        with patch.dict(os.environ, {"EXPORT_TEST_SECRET": sentinel}), \
                patch.object(Path, "rglob", side_effect=AssertionError("No recursive scan")), \
                patch.object(Path, "glob", side_effect=AssertionError("No directory scan")):
            record = self.build()
        inventory = self.data(record)
        contents = [read_verified(self.run_root, record)]
        for entry in inventory["files"]:
            self.assertTrue(entry["path"].startswith(f"attempts/{ATTEMPT_A}/"))
            self.assertNotIn(".html", entry["path"])
            contents.append(read_verified(self.run_root, StoredArtifact(**{key: entry[key] for key in ("path", "sha256", "size")})))
        for data in contents:
            for secret in (sentinel.encode(), b"runway://", b"private-output-token", b"signature="):
                self.assertNotIn(secret, data)

    def test_inventory_uses_no_current_clock_recovery_or_ledger_writes(self):
        before = self.ledger_dump()
        real_connection = self.ledger._connection
        statements = []

        @contextmanager
        def guarded_connection():
            with real_connection() as connection:
                connection.set_authorizer(lambda action, *details: sqlite3.SQLITE_OK if action in
                    (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION) else sqlite3.SQLITE_DENY)
                connection.set_trace_callback(statements.append)
                yield connection

        with patch.object(self.ledger, "_connection", guarded_connection), \
                patch.object(self.ledger, "_now", side_effect=RuntimeError("Broken clock")), \
                patch.object(SubmissionCoordinator, "__init__", side_effect=AssertionError("No recovery")):
            record = self.build(binding=self.fixture.bindings[ATTEMPT_A])
        self.assertEqual(self.data(record)["provider"]["completedUnix"], 1001.0)
        self.assertTrue(statements)
        self.assertTrue(all(statement.lstrip().upper().startswith("SELECT") for statement in statements))
        self.assertEqual(self.ledger_dump(), before)

    def test_mixed_review_preparation_check_and_mode_fail_before_publication(self):
        for record in (self.manifests[ATTEMPT_B], replace(self.manifests[ATTEMPT_A], sha256="f" * 64)):
            with self.subTest(record=record), self.assertRaises(export.ExportInventoryError):
                self.build(manifest=record)
            self.assert_no_inventory()
        other_review = self.data(self.manifests[ATTEMPT_B])
        changes = [{field: other_review[field]} for field in ("preparation", "requestCheck", "sourceOriginal", "rawMetadata")]
        changes += [{"payloadSha256": "f" * 64}, {"sourceKind": "clean_camera"}, {"mode": "B"}]
        for change in changes:
            record = self.replace_review(self.original_review | change)
            with self.subTest(change=change), self.assertRaises(export.ExportInventoryError):
                self.build(manifest=record)
            self.assert_no_inventory()

    def test_frozen_preparation_prompt_and_source_kind_must_match_the_saved_check(self):
        frozen = asdict(self.media.checks[ATTEMPT_A].frozen)
        changes = [{"frozen": frozen | {"preparation_sha256": "f" * 64}},
                   {"frozen": frozen | {"prompt": "Changed fixture instruction."}},
                   {"sourceKind": "clean_camera"}, {"preparation": asdict(self.media.manifests[ATTEMPT_B])}]
        for change in changes:
            record = self.replace_check(change)
            with self.subTest(change=change), self.assertRaises(export.ExportInventoryError):
                self.build(manifest=record)
            self.assert_no_inventory()

    def test_reservation_run_payload_and_reserved_cost_must_match(self):
        self.ledger.approve_run("other-run", approved_by="fixture", limit_cents=140, max_attempts=1, cost_cents=140)
        cases = [("run_id", "other-run", RUN), ("payload_sha256", "f" * 64,
                  self.media.checks[ATTEMPT_A].frozen.payload_sha256), ("cost_cents", 141, 140)]
        for field, value, original in cases:
            with self.ledger._connection() as connection:
                connection.execute(f"UPDATE reservations SET {field}=? WHERE attempt_id=?", (value, ATTEMPT_A))
            with self.subTest(field=field), self.assertRaises(export.ExportInventoryError):
                self.build()
            self.assert_no_inventory()
            with self.ledger._connection() as connection:
                connection.execute(f"UPDATE reservations SET {field}=? WHERE attempt_id=?", (original, ATTEMPT_A))

    def test_private_evidence_corruption_is_rejected_even_though_bytes_are_excluded(self):
        for name in ("requestCheck", "request", "frozenRequest", "providerSuccess"):
            path = self.run_root / self.original_review[name]["path"]
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(record=name), self.assertRaises(export.ExportInventoryError):
                self.build()
            self.assert_no_inventory()
            path.write_bytes(original)

    def test_missing_asset_unused_guide_and_selected_sample_are_not_silently_omitted(self):
        prepared = self.media.prepared[ATTEMPT_A]
        sample = StoredArtifact(**self.original_review["samples"][3]["raw"]["image"])
        for record in (prepared.asset_original, prepared.guide_b, sample):
            path = self.run_root / record.path
            original = path.read_bytes()
            path.unlink()
            with self.subTest(path=record.path), self.assertRaises(export.ExportInventoryError):
                self.build()
            self.assert_no_inventory()
            self.assertFalse(path.exists())
            path.write_bytes(original)

    def test_invalid_native_metadata_is_rejected_without_a_selected_scorecard(self):
        for change in ({"durationSeconds": 21}, {"frameSeconds": [0, 0]}, {"frameSeconds": [0, True]}):
            record = self.replace_metadata(self.original_metadata | change)
            with self.subTest(change=change), self.assertRaises(export.ExportInventoryError):
                self.build(manifest=record)
            self.assert_no_inventory()

    def test_current_completed_task_and_valid_task_identifier_are_required(self):
        for state in ("SUBMISSION_UNKNOWN", "RUNNING", "FAILED"):
            with self.ledger._connection() as connection:
                connection.execute("UPDATE submission_jobs SET state=? WHERE attempt_id=?", (state, ATTEMPT_A))
            with self.subTest(state=state), self.assertRaises(export.ExportInventoryError):
                self.build()
            self.assert_no_inventory()
        task = "invalid?private-task"
        success_bytes = json.dumps({"id": task, "status": "SUCCEEDED"}).encode()
        success = publish_bytes(self.run_root, f"provider_responses/{sha256(success_bytes).hexdigest()}.json", success_bytes)
        with self.ledger._connection() as connection:
            connection.execute("UPDATE submission_jobs SET state='SUCCEEDED', provider_task_id=? WHERE attempt_id=?", (task, ATTEMPT_A))
            connection.execute("UPDATE completed_outputs SET provider_task_id=?, provider_evidence_sha256=? WHERE attempt_id=?",
                               (task, success.sha256, ATTEMPT_A))
        record = self.replace_metadata(self.original_metadata | {"providerTaskId": task},
                                       providerTaskId=task, providerSuccess=asdict(success))
        with self.assertRaises(export.ExportInventoryError):
            self.build(manifest=record)
        self.assert_no_inventory()

    def test_foreign_or_forged_selected_binding_is_rejected(self):
        with self.assertRaises(export.ExportInventoryError):
            self.build(binding=self.fixture.bindings[ATTEMPT_B])
        self.assert_no_inventory()
        data = self.data(self.fixture.corrected) | {"rubricVerdict": "PASS"}
        contents = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()
        record = publish_bytes(self.run_root,
            f"attempts/{ATTEMPT_A}/review/scorecard-bindings/{sha256(contents).hexdigest()}.json", contents)
        with self.assertRaises(export.ExportInventoryError):
            self.build(binding=record)
        self.assert_no_inventory()

    def test_repeats_are_stable_and_corrected_selection_preserves_prior_inventory(self):
        first = self.build(binding=self.fixture.bindings[ATTEMPT_A])
        first_bytes = read_verified(self.run_root, first)
        self.ledger.clock = lambda: -1
        self.assertEqual(self.build(binding=self.fixture.bindings[ATTEMPT_A]), first)
        corrected = self.build(binding=self.fixture.corrected)
        self.assertNotEqual(corrected, first)
        self.assertEqual(read_verified(self.run_root, first), first_bytes)
        self.assertEqual(self.data(corrected)["selection"]["scorecardBinding"], asdict(self.fixture.corrected))
        path = self.run_root / first.path
        path.write_bytes(b"changed immutable inventory")
        with self.assertRaises(export.ExportInventoryError):
            self.build(binding=self.fixture.bindings[ATTEMPT_A])
        self.assertEqual(path.read_bytes(), b"changed immutable inventory")

    def test_only_inventory_is_published_last_and_an_interruption_can_retry(self):
        with patch.object(export, "_publish_or_verify", side_effect=OSError("private interruption")) as publisher:
            with self.assertRaises(export.ExportInventoryError) as raised:
                self.build()
        self.assertEqual(publisher.call_count, 1)
        self.assert_no_inventory()
        self.assertEqual(str(raised.exception), "Export inventory could not be verified or published for this attempt.")
        real_publish = export._publish_or_verify
        calls = []

        def record_publish(root, path, contents):
            calls.append(path)
            return real_publish(root, path, contents)

        with patch.object(export, "_publish_or_verify", side_effect=record_publish):
            record = self.build()
        self.assertEqual(calls, [record.path])
        self.assertEqual(self.build(), record)

    def test_schema_rejects_extra_claims_invalid_roles_paths_hashes_and_selection(self):
        original = self.data(self.build(attempt=ATTEMPT_B, binding=self.fixture.bindings[ATTEMPT_B]))
        export._VALIDATOR.validate(original)
        cases = []
        for field, value in (("visualEffectDemonstrated", "PASS"), ("automatedWorkflowComplete", "PASS"), ("extra", True)):
            cases.append(original | {field: value})
        for field, value in (("providerProvenance", "LIVE"), ("actualChargeAudit", "PASS"), ("runBudgetAudit", "PASS")):
            cases.append(original | {"availability": original["availability"] | {field: value}})
        cases.append(original | {"provider": original["provider"] | {"outputUrl": "https://private.example/output"}})
        cases.append(original | {"selection": original["selection"] | {"scorecardBinding": None}})
        cases.append(original | {"selection": original["selection"] | {"reviewManifest": asdict(self.media.completed[ATTEMPT_B].raw)}})
        for field, value in (("path", "../secret"), ("path", f"attempts/{ATTEMPT_B}/provider/request.json"),
                             ("sha256", "A" * 64), ("size", True), ("size", 0), ("size", 536870913),
                             ("roles", ["provider_authentication"]), ("roles", ["raw_output", "raw_output"])):
            changed = deepcopy(original)
            changed["files"][0][field] = value
            cases.append(changed)
        missing_private = deepcopy(original)
        del missing_private["privateEvidenceHashes"]["frozenRequest"]
        cases.append(missing_private)
        extra_private_path = deepcopy(original)
        extra_private_path["privateEvidenceHashes"]["request"]["path"] = "provider/request.json"
        cases.append(extra_private_path)
        too_many = deepcopy(original)
        too_many["files"].append(original["files"][0] | {
            "path": f"attempts/{ATTEMPT_B}/review/raw/frame-000001.png", "roles": ["raw_sample"]})
        cases.append(too_many)
        for index, data in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValidationError):
                export._VALIDATOR.validate(data)

    def test_schema_records_aggregate_bytes_above_archive_limit_without_claiming_zip_fitness(self):
        inventory = self.data(self.build())
        inventory["files"][0]["size"] = 536870912
        inventory["totalFileBytes"] = sum(entry["size"] for entry in inventory["files"])
        self.assertGreater(inventory["totalFileBytes"], 536870912)
        export._VALIDATOR.validate(inventory)
        self.assertNotIn("archiveReady", inventory)
        self.assertNotIn("archiveSize", inventory)

    def test_bad_expected_identity_or_closed_owner_publishes_nothing(self):
        for changes in ({"run_id": "../invalid"}, {"attempt_id": ATTEMPT_B}):
            with self.subTest(changes=changes), self.assertRaises(export.ExportInventoryError):
                self.build(**changes)
            self.assert_no_inventory()
        self.owner.close()
        with self.assertRaises(export.ExportInventoryError):
            self.build()
        self.assert_no_inventory()


if __name__ == "__main__":
    unittest.main()

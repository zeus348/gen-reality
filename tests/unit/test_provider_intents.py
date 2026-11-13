"""Bind durable provider intent to real historical media without spending."""

import asyncio
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
import runwayml.resources.uploads as sdk_uploads

from app import prepare, preparation, provider_intents as intents
from core import budget
from core.storage import StoredArtifact, publish_bytes, read_verified
from features.inference import account, preflight, prepared, pricing, submission, upload
from features.inference.frozen import FrozenCreate
from features.placement.math import Placement
import test_prepare as media_fixtures


_BASE_LEDGER = budget.BudgetLedger
_BASE_OWNER = submission.SubmissionCoordinator
_BASE_SERVICE = preparation.PreparationService
KEY = "synthetic-intent-fixture-key"


class ProviderIntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media = type("IntentMediaFixture", (media_fixtures.PrepareTests,), {})
        cls.media.setUpClass()
        cls.addClassCleanup(cls.media.doClassCleanups)

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.state, self.outputs = self.root / "private", self.root / "runs"
        self.state.mkdir(mode=0o700)
        self.ledger = _BASE_LEDGER(self.state / "budget.sqlite", clock=lambda: 1001.0)
        self.owner = _BASE_OWNER(self.ledger)
        self.addCleanup(self.owner.close)
        self.service = _BASE_SERVICE(self.state, self.outputs, self.owner, clock=lambda: 1000.0)
        self.addCleanup(self.service.close)
        self.selected = self.service.create_run()
        self.run_id = self.selected.run_id
        self.run_root = self.outputs / self.run_id
        self.attempts, self.manifests = [], {}

        async def prepare_history():
            async with self.service.media_operation(self.run_id, expected_revision=0) as operation:
                selected = await operation.import_source(self.chunks(self.media.video.original_bytes),
                    source_suffix=".mp4", clean_source=True, source_kind="synthetic_fixture")
            async with self.service.media_operation(self.run_id, expected_revision=selected.revision) as operation:
                selected = await operation.import_asset(self.chunks(self.media.validated_asset.original_bytes), non_human=True)
            selected = self.service.set_placement(self.run_id, expected_revision=selected.revision,
                                                  placement=Placement(0.5, 0.5, 0.25))
            for _ in range(3):
                async with self.service.media_operation(self.run_id, expected_revision=selected.revision) as operation:
                    selected = await operation.build_guides()
                attempt = selected.attempts[-1].attempt_id
                self.attempts.append(attempt)
                self.manifests[attempt] = selected.prepared
            return selected

        # The cached objects contain real validated media. Reuse that expensive
        # normalization/encoding while exercising real registration/publication.
        with patch.object(preparation, "normalize_video", return_value=self.media.video), \
                patch.object(prepare, "compile_guidance", return_value=self.media.guidance):
            self.selected = asyncio.run(prepare_history())
        self.approved = False
        self.evidence = self.root / "provider-evidence"
        self.price = None
        self.uploads = {}

    @staticmethod
    async def chunks(contents):
        yield contents[:17]
        yield contents[17:]

    def approve(self):
        if not self.approved:
            self.ledger.approve_run(self.run_id, approved_by="fixture operator", limit_cents=1000,
                max_attempts=6, cost_cents=140, deadline_unix_cap=11800.0, only_run=True)
            self.approved = True

    def store(self, *, approved=True):
        if approved:
            self.approve()
        return intents.ProviderIntents(self.owner, self.service)

    def admit(self, store, index=0, *, key="client-key", mode="A", run_id=None):
        return store.admit(run_id or self.run_id, prepared_attempt_id=self.attempts[index],
                           mode=mode, idempotency_key=key)

    def rows(self):
        with sqlite3.connect(self.ledger.path) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute("SELECT * FROM provider_intents ORDER BY rowid")]

    def financial_rows(self):
        with sqlite3.connect(self.ledger.path) as connection:
            return {name: connection.execute("SELECT * FROM " + name + " ORDER BY 1").fetchall()
                    for name in ("approved_runs", "reservations", "actual_charges", "submission_jobs",
                                 "submission_events", "run_events")}

    def assert_code(self, code, operation):
        with self.assertRaises(intents.ProviderIntentError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)

    def make_upload(self, source):
        if source in self.uploads:
            return self.uploads[source]

        def storage_client(**arguments):
            return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(204)), **arguments)

        sdk_httpx = SimpleNamespace(**(vars(httpx) | {"Client": storage_client}))
        response = {"runwayUri": "runway://" + source.sha256,
                    "uploadUrl": "https://uploads.example.com/file", "fields": {}}
        with patch.object(sdk_uploads, "httpx", sdk_httpx):
            result = upload.upload_frozen_artifact(self.run_root, source, KEY, self.evidence,
                allowed_upload_hosts=frozenset({"uploads.example.com"}), clock=lambda: 1000.0,
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)))
        self.uploads[source] = result
        return result

    def check(self, index=0, mode="A", *, account_time=1000.0, checked_time=1001.0):
        attempt = self.attempts[index]
        selection = prepared.read_prepared_attempt(self.run_root, self.manifests[attempt],
                                                   run_id=self.run_id, attempt_id=attempt)
        if self.price is None:
            source = publish_bytes(self.evidence, "price.txt", b"Synthetic price evidence; no live provider request.\n")
            self.price = pricing.record_price_quote(self.evidence, source, checked_by="fixture operator",
                checked_unix=1000.0, credits_per_second=28, minimum_credits=56,
                usd_per_credit="0.01", maximum_additional_cents=0, clock=lambda: 1000.0)
        body = {"creditBalance": 500,
            "tier": {"models": {"aleph2": {"maxConcurrentGenerations": 5, "maxDailyGenerations": 20}}},
            "usage": {"models": {"aleph2": {"dailyGenerations": 0}}}}
        account_check = account.check_account_once(KEY, 140, self.evidence, clock=lambda: account_time,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
        video, keyframe = selection.media_for_mode(mode)
        return preflight.check_prepared_request(self.run_root, self.manifests[attempt],
            run_id=self.run_id, attempt_id=attempt, mode=mode,
            video=self.make_upload(video), keyframe=self.make_upload(keyframe) if keyframe else None,
            price=self.price, account=account_check, evidence_root=self.evidence, clock=lambda: checked_time)

    def changed_check(self, checked, changes):
        data = json.loads(read_verified(self.run_root, checked.evidence))
        data.update(changes)
        if "frozen" in changes:
            data["payloadSha256"] = FrozenCreate(**data["frozen"]).payload_sha256
        contents = (json.dumps(data, sort_keys=True) + "\n").encode()
        return publish_bytes(self.run_root,
            f"attempts/{checked.attempt_id}/provider_checks/{sha256(contents).hexdigest()}.json", contents)

    def test_modes_bind_literal_key_digest_and_reservation_identity_without_spending(self):
        store = self.store()
        before = self.financial_rows()
        clock = Mock(side_effect=budget.BudgetError("fixture clock must not run"))
        with patch.object(self.ledger, "clock", clock), \
                patch.object(self.ledger, "reserve", side_effect=budget.BudgetError("fixture reserve forbidden")) as reserve, \
                patch.object(self.ledger, "approve_run", side_effect=budget.BudgetError("fixture approval forbidden")) as approve:
            for index, mode, key in ((0, "A", "Client-Key_!~"), (1, "B", "X" * 128)):
                result = self.admit(store, index, mode=mode, key=key)
                manifest = self.manifests[self.attempts[index]]
                expected_data = {"formatVersion": "provider-intent-1", "model": "aleph2", "runId": self.run_id,
                    "attemptId": self.attempts[index], "mode": mode, "preparation": asdict(manifest)}
                expected_digest = sha256(json.dumps(expected_data, sort_keys=True,
                    separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
                key_hash = sha256(key.encode("ascii")).hexdigest()
                self.assertEqual((result.run_id, result.key_sha256, result.attempt_id, result.mode,
                                  result.preparation, result.input_sha256, result.checked_request),
                    (self.run_id, key_hash, self.attempts[index], mode, manifest, expected_digest, None))
                self.assertEqual(result.reservation_key, "provider-intent-v1:" + self.run_id + ":" + key_hash)
                self.assertEqual(store.get(self.run_id, prepared_attempt_id=self.attempts[index]), result)
                with self.assertRaises(FrozenInstanceError):
                    result.mode = "changed"
                self.assertNotIn(key, json.dumps(self.rows()) + repr(result))
            for forbidden in (clock, reserve, approve):
                forbidden.assert_not_called()
        self.assertEqual(self.financial_rows(), before)

    def test_invalid_scalar_inputs_and_keys_fail_before_database_access(self):
        store = self.store()
        arguments = {"prepared_attempt_id": self.attempts[0], "mode": "A", "idempotency_key": "valid-key"}
        cases = [(self.run_id, arguments | {"idempotency_key": key}) for key in
                 (None, True, 1, b"key", "", "x" * 129, "with space", "\t", "\n", "\x00", "\x7f", "é")]
        cases += [(self.run_id, arguments | {"mode": value}) for value in (None, True, "a", "C", " A")]
        cases += [(value, arguments) for value in (None, "run-1", "../run", "run-" + "g" * 32)]
        cases += [(self.run_id, arguments | {"prepared_attempt_id": value})
                  for value in (None, "att-short", "att-" + "g" * 32)]
        for run_id, supplied in cases:
            with self.subTest(arguments=supplied, run_id=run_id), \
                    patch.object(self.ledger, "_connection", side_effect=budget.BudgetError("fixture SQL forbidden")) as connection, \
                    patch.object(self.ledger, "_transaction", side_effect=budget.BudgetError("fixture SQL forbidden")) as transaction:
                self.assert_code("INVALID_INPUT", lambda: store.admit(run_id, **supplied))
                connection.assert_not_called()
                transaction.assert_not_called()
        self.assertEqual(self.rows(), [])

    def test_duplicate_and_reopened_store_return_held_unknown_identity_without_media_or_clock(self):
        store = self.store()
        original = self.admit(store)
        checked = self.check()
        selected = store.select_checked(self.run_id, prepared_attempt_id=original.attempt_id,
                                         checked_request=checked.evidence)
        reserved = self.ledger.reserve(self.run_id, selected.reservation_key, checked.frozen.payload_sha256,
                                      attempt_id=selected.attempt_id)
        unknown = self.owner.submit_once(reserved.attempt_id, Mock(side_effect=RuntimeError("synthetic uncertain result")))
        self.assertEqual(unknown.state, "SUBMISSION_UNKNOWN")
        before = self.financial_rows()
        (self.run_root / original.preparation.path).write_bytes(b"damaged after intent admission")
        clock = Mock(side_effect=budget.BudgetError("fixture clock must not run"))
        with patch.object(self.ledger, "clock", clock), \
                patch.object(intents, "read_prepared_attempt", side_effect=ValueError("fixture media forbidden")) as media, \
                patch.object(intents, "read_run_start", side_effect=ValueError("fixture start forbidden")) as start, \
                patch.object(self.service, "get_run", side_effect=ValueError("fixture draft forbidden")) as current, \
                patch.object(self.service._state, "get", side_effect=ValueError("fixture draft forbidden")) as snapshot, \
                patch.object(self.ledger, "reserve", side_effect=budget.BudgetError("fixture reserve forbidden")) as reserve:
            reopened = intents.ProviderIntents(self.owner, self.service)
            self.assertEqual(self.admit(reopened), selected)
            self.assertEqual(reopened.get(self.run_id, prepared_attempt_id=original.attempt_id), selected)
            for forbidden in (clock, media, start, current, snapshot, reserve):
                forbidden.assert_not_called()
        self.assertEqual(self.financial_rows(), before)
        self.assertEqual(self.owner.get(reserved.attempt_id), unknown)

    def test_historical_preparation_does_not_depend_on_corrupt_newer_draft_cache(self):
        store = self.store()

        async def replace_asset():
            async with self.service.media_operation(self.run_id, expected_revision=self.selected.revision) as operation:
                return await operation.import_asset(self.chunks(self.media.validated_asset.original_bytes), non_human=True)

        current = asyncio.run(replace_asset())
        (self.run_root / current.asset.record.path).write_bytes(b"damaged newer draft cache")
        with self.assertRaises(preparation.ServiceError):
            self.service.get_run(self.run_id)
        with patch.object(self.service, "get_run", side_effect=ValueError("fixture current draft forbidden")) as get_run, \
                patch.object(self.service, "_snapshot", side_effect=ValueError("fixture current draft forbidden")) as snapshot, \
                patch.object(self.service._state, "get", side_effect=ValueError("fixture current joins forbidden")) as state_get:
            result = self.admit(store)
            for forbidden in (get_run, snapshot, state_get):
                forbidden.assert_not_called()
        self.assertEqual(result.preparation, self.manifests[self.attempts[0]])

    def test_changed_key_mode_attempt_or_run_conflicts_without_replacing_identity(self):
        store = self.store()
        original = self.admit(store)
        before = self.rows()
        cases = [lambda: self.admit(store, key="different-key"), lambda: self.admit(store, mode="B"),
                 lambda: self.admit(store, index=1),
                 lambda: self.admit(store, run_id="run-" + "f" * 32)]
        for operation in cases:
            self.assert_code("INTENT_CONFLICT", operation)
            self.assertEqual(self.rows(), before)
        self.assert_code("INTENT_NOT_FOUND", lambda: store.get(self.run_id, prepared_attempt_id=self.attempts[1]))
        self.assertEqual(store.get(self.run_id, prepared_attempt_id=original.attempt_id), original)

    def test_new_admission_requires_approval_registered_complete_history_and_unchanged_evidence(self):
        store = self.store(approved=False)
        self.assert_code("APPROVAL_REQUIRED", lambda: self.admit(store))
        self.approve()
        self.assert_code("INPUT_REQUIRED", lambda: store.admit(self.run_id,
            prepared_attempt_id="att-" + "f" * 32, mode="A", idempotency_key="missing-history"))
        unfinished = "att-" + "f" * 32
        self.service._state.begin_attempt(self.service._snapshot(self.run_id), unfinished, self.media.guidance.prompt)
        self.assert_code("INPUT_REQUIRED", lambda: store.admit(self.run_id,
            prepared_attempt_id=unfinished, mode="A", idempotency_key="unfinished-history"))
        self.assert_code("STATE_UNAVAILABLE", lambda: self.admit(store, run_id="run-" + "f" * 32))
        selected = prepared.read_prepared_attempt(self.run_root, self.manifests[self.attempts[0]],
                                                   run_id=self.run_id, attempt_id=self.attempts[0])
        for record in (self.manifests[self.attempts[0]], selected.normalized_source, selected.keyframe_a):
            path = self.run_root / record.path
            original = path.read_bytes()
            try:
                path.write_bytes(b"X" + original[1:])
                self.assert_code("STATE_UNAVAILABLE", lambda: self.admit(store))
                self.assertEqual(self.rows(), [])
            finally:
                path.write_bytes(original)
        start = self.run_root / "run-start.json"
        original = start.read_bytes()
        value = json.loads(original)
        value["startedAtUnix"] += 1
        value["deadlineAtUnix"] += 1
        try:
            start.write_text(json.dumps(value))
            self.assert_code("STATE_UNAVAILABLE", lambda: self.admit(store))
        finally:
            start.write_bytes(original)
        self.assertEqual(self.rows(), [])

    def test_owner_mismatch_and_closed_lifecycle_fail_before_sql(self):
        store = self.store()
        with patch.object(self.ledger, "_connection", side_effect=budget.BudgetError("fixture SQL forbidden")) as connection, \
                patch.object(self.ledger, "_transaction", side_effect=budget.BudgetError("fixture SQL forbidden")) as transaction:
            for owner, service in ((None, None), (object(), self.service), (self.owner, object())):
                self.assert_code("STATE_UNAVAILABLE", lambda: intents.ProviderIntents(owner, service))
            connection.assert_not_called()
            transaction.assert_not_called()
        other = self.root / "other-output"
        with patch.object(self.service, "output_root", other), \
                patch.object(self.ledger, "_connection", side_effect=budget.BudgetError("fixture SQL forbidden")) as connection, \
                patch.object(self.ledger, "_transaction", side_effect=budget.BudgetError("fixture SQL forbidden")) as transaction:
            self.assert_code("OWNER_MISMATCH", lambda: self.admit(store))
            connection.assert_not_called()
            transaction.assert_not_called()
        with patch.object(self.owner, "_preparation_service", None), \
                patch.object(self.ledger, "_transaction", side_effect=budget.BudgetError("fixture SQL forbidden")) as transaction:
            self.assert_code("OWNER_MISMATCH", lambda: intents.ProviderIntents(self.owner, self.service))
            transaction.assert_not_called()
        self.service.close()
        with patch.object(self.ledger, "_connection", side_effect=budget.BudgetError("fixture SQL forbidden")) as connection, \
                patch.object(self.ledger, "_transaction", side_effect=budget.BudgetError("fixture SQL forbidden")) as transaction:
            self.assert_code("SERVICE_CLOSED", lambda: store.get(self.run_id, prepared_attempt_id=self.attempts[0]))
            self.assert_code("SERVICE_CLOSED", lambda: intents.ProviderIntents(self.owner, self.service))
            connection.assert_not_called()
            transaction.assert_not_called()

    def test_partial_or_changed_feature_schema_is_rejected_without_repair(self):
        self.approve()
        with sqlite3.connect(self.ledger.path) as connection:
            connection.execute("CREATE TABLE provider_intents (run_id TEXT)")
        self.assert_code("STATE_UNAVAILABLE", lambda: intents.ProviderIntents(self.owner, self.service))
        with sqlite3.connect(self.ledger.path) as connection:
            self.assertEqual(connection.execute("PRAGMA table_info(provider_intents)").fetchall(),
                             [(0, "run_id", "TEXT", 0, None, 0)])
            connection.execute("DROP TABLE provider_intents")
        store = self.store()
        self.admit(store)
        with sqlite3.connect(self.ledger.path) as connection:
            connection.execute("DROP TRIGGER provider_intents_no_delete")
        self.assert_code("STATE_UNAVAILABLE", lambda: intents.ProviderIntents(self.owner, self.service))
        self.assert_code("STATE_UNAVAILABLE", lambda: store.get(self.run_id, prepared_attempt_id=self.attempts[0]))
        with sqlite3.connect(self.ledger.path) as connection:
            self.assertIsNone(connection.execute("SELECT sql FROM sqlite_schema WHERE name='provider_intents_no_delete'").fetchone())
        self.assertEqual(len(self.rows()), 1)

    def test_row_reader_recomputes_input_identity_after_trusted_database_damage(self):
        store = self.store()
        self.admit(store)
        with sqlite3.connect(self.ledger.path) as connection:
            sql = connection.execute("SELECT sql FROM sqlite_schema WHERE name='provider_intents_fixed_identity'").fetchone()[0]
            connection.execute("DROP TRIGGER provider_intents_fixed_identity")
            connection.execute("UPDATE provider_intents SET input_sha256=?", ("0" * 64,))
            connection.execute(sql)
        self.assert_code("STATE_UNAVAILABLE", lambda: store.get(self.run_id, prepared_attempt_id=self.attempts[0]))
        self.assert_code("STATE_UNAVAILABLE", lambda: self.admit(store))
        self.assertEqual(self.rows()[0]["input_sha256"], "0" * 64, "Reading repaired a damaged intent identity.")

    def test_sql_guards_preserve_identity_against_delete_update_and_each_replace_collision(self):
        store = self.store()
        self.admit(store)
        before = self.rows()
        row = before[0]
        primary_collision = row | {"attempt_id": self.attempts[1],
                                  "preparation_path": f"attempts/{self.attempts[1]}/manifest.json"}
        attempt_collision = row | {"key_sha256": "c" * 64}
        insert = "INSERT OR REPLACE INTO provider_intents VALUES (" + ",".join("?" for _ in row) + ")"
        operations = [("DELETE FROM provider_intents", ()), ("UPDATE provider_intents SET mode='B'", ()),
                      (insert, tuple(primary_collision.values())), (insert, tuple(attempt_collision.values())),
                      ("UPDATE provider_intents SET checked_path='partial'", ())]
        for sql, parameters in operations:
            with self.subTest(sql=sql, parameters=parameters):
                connection = sqlite3.connect(self.ledger.path)
                try:
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(sql, parameters)
                finally:
                    connection.rollback()
                    connection.close()
                self.assertEqual(self.rows(), before)
        checked = self.check()
        store.select_checked(self.run_id, prepared_attempt_id=self.attempts[0], checked_request=checked.evidence)
        retained = self.rows()
        with sqlite3.connect(self.ledger.path) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE provider_intents SET checked_path=NULL,checked_sha256=NULL,checked_size=NULL")
        self.assertEqual(self.rows(), retained)

    def test_capacity_accepts_exact_boundary_and_rejects_one_more_for_each_limit(self):
        store = self.store()
        with patch.object(intents, "MAX_INTENTS_PER_RUN", 1), patch.object(intents, "MAX_INTENTS", 100):
            first = self.admit(store, key="first")
            self.assert_code("CAPACITY_EXCEEDED", lambda: self.admit(store, index=1, key="second", mode="B"))
        with patch.object(intents, "MAX_INTENTS_PER_RUN", 100), patch.object(intents, "MAX_INTENTS", 2):
            self.admit(store, index=1, key="second", mode="B")
            self.assert_code("CAPACITY_EXCEEDED", lambda: self.admit(store, index=2, key="third"))
            self.assertEqual(self.admit(store, key="first"), first)
        self.assertEqual(len(self.rows()), 2)

    def test_two_store_races_converge_or_conflict_without_replacing_rows(self):
        stores = [self.store(), self.store()]

        def race(arguments):
            barrier = Barrier(2, timeout=5)
            outcomes = []
            originals = [store._prepared for store in stores]

            def read(index, *args):
                result = originals[index](*args)
                barrier.wait(timeout=5)
                return result

            def admit(index):
                try:
                    outcomes.append(self.admit(stores[index], **arguments[index]))
                except BaseException as error:
                    outcomes.append(error)

            with patch.object(stores[0], "_prepared", side_effect=lambda *args: read(0, *args)), \
                    patch.object(stores[1], "_prepared", side_effect=lambda *args: read(1, *args)):
                threads = [Thread(target=admit, args=(index,), name=f"intent-race-{index}") for index in range(2)]
                started = []
                try:
                    for thread in threads:
                        thread.start()
                        started.append(thread)
                    for thread in started:
                        thread.join(timeout=8)
                finally:
                    barrier.abort()
                    for thread in started:
                        thread.join(timeout=8)
                self.assertTrue(all(not thread.is_alive() for thread in started), "The bounded provider-intent race did not finish.")
            return outcomes

        same = race([{"index": 0, "key": "same"}, {"index": 0, "key": "same"}])
        self.assertEqual(len(same), 2)
        self.assertTrue(all(type(result) is intents.ProviderIntent for result in same),
                        "Identical concurrent admissions did not both return an intent.")
        self.assertEqual(same[0], same[1])
        conflicting = race([{"index": 1, "key": "conflicting"}, {"index": 2, "key": "conflicting"}])
        winners = [result for result in conflicting if type(result) is intents.ProviderIntent]
        failures = [result for result in conflicting if isinstance(result, BaseException)]
        self.assertEqual((len(winners), len(failures)), (1, 1))
        self.assertIs(type(failures[0]), intents.ProviderIntentError)
        self.assertEqual(failures[0].code, "INTENT_CONFLICT")
        self.assertEqual(len(self.rows()), 2)

    def test_real_checked_selection_is_immutable_and_expiry_does_not_refresh_it(self):
        store = self.store()
        checked_a, checked_b = self.check(), self.check(index=1, mode="B")
        refreshed = self.check(account_time=1005.0, checked_time=1006.0)
        self.assertEqual(refreshed.frozen, checked_a.frozen)
        self.assertNotEqual(refreshed.evidence, checked_a.evidence)
        self.admit(store)
        self.admit(store, index=1, key="mode-b", mode="B")
        before = self.financial_rows()
        clock = Mock(side_effect=budget.BudgetError("fixture clock must not run"))
        with patch.object(self.ledger, "clock", clock):
            for checked in (checked_a, checked_b):
                result = store.select_checked(self.run_id, prepared_attempt_id=checked.attempt_id,
                                               checked_request=checked.evidence)
                self.assertEqual(result.checked_request, checked.evidence)
                self.assertEqual(store.select_checked(self.run_id, prepared_attempt_id=checked.attempt_id,
                                                       checked_request=checked.evidence), result)
            self.assert_code("INTENT_CONFLICT", lambda: store.select_checked(self.run_id,
                prepared_attempt_id=self.attempts[0], checked_request=refreshed.evidence))
            clock.assert_not_called()
        self.assertEqual(self.financial_rows(), before)
        self.assertEqual(store.get(self.run_id, prepared_attempt_id=self.attempts[0]).checked_request, checked_a.evidence)

    def test_checked_selection_rejects_forged_identity_full_reference_and_corrupt_bytes(self):
        store = self.store()
        self.admit(store)
        checked = self.check()
        frozen = asdict(checked.frozen)
        changes = [({"runId": "run-" + "f" * 32}, "STATE_UNAVAILABLE"),
            ({"attemptId": self.attempts[1]}, "STATE_UNAVAILABLE"),
            ({"sourceKind": "clean_camera"}, "INTENT_CONFLICT"),
            ({"frozen": frozen | {"prompt": "Different valid prompt."}}, "INTENT_CONFLICT"),
            ({"frozen": frozen | {"preparation_sha256": "f" * 64}}, "INTENT_CONFLICT"),
            ({"frozen": frozen | {"mode": "B", "keyframe_uri": None}}, "INTENT_CONFLICT"),
            ({"preparation": asdict(self.manifests[self.attempts[0]]) | {"size": 1}}, "STATE_UNAVAILABLE"),
            ({"preparation": asdict(self.manifests[self.attempts[0]]) | {
                "path": f"attempts/{self.attempts[1]}/manifest.json"}}, "STATE_UNAVAILABLE")]
        before = self.rows()
        for changed, code in changes:
            with self.subTest(changes=changed):
                record = self.changed_check(checked, changed)
                self.assert_code(code, lambda: store.select_checked(self.run_id,
                    prepared_attempt_id=checked.attempt_id, checked_request=record))
                self.assertEqual(self.rows(), before)
        for record in (replace(checked.evidence, size=True), replace(checked.evidence, size=65537),
                       replace(checked.evidence, path="provider_checks/foreign.json")):
            with self.subTest(record=record):
                self.assert_code("INVALID_INPUT", lambda: store.select_checked(self.run_id,
                    prepared_attempt_id=checked.attempt_id, checked_request=record))
        path = self.run_root / checked.evidence.path
        contents = path.read_bytes()
        path.write_bytes(b"X" + contents[1:])
        self.assert_code("STATE_UNAVAILABLE", lambda: store.select_checked(self.run_id,
            prepared_attempt_id=checked.attempt_id, checked_request=checked.evidence))
        self.assertEqual(self.rows(), before)


if __name__ == "__main__":
    unittest.main()

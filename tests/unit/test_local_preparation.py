"""Check private preparation state with synthetic media and no provider work."""

import asyncio
from dataclasses import FrozenInstanceError, asdict
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from app import prepare, preparation, preparation_state, server
from core import budget, run_start
from core.storage import read_verified
from features.inference import submission
from features.placement.math import Placement
import test_prepare as media_fixtures
from starlette.testclient import TestClient


class LocalPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = type("LocalPreparationMedia", (media_fixtures.PrepareTests,), {})
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.second_placement = Placement(0.7, 0.5, 0.25)
        cls.second_guidance = prepare.compile_guidance(
            cls.fixture.video, cls.fixture.validated_asset, cls.second_placement)

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.state = self.root / "private"
        self.state.mkdir(mode=0o700)
        self.outputs = self.root / "outputs"
        self.placement = Placement(0.5, 0.5, 0.25)

    def owner(self, path=None):
        ledger = budget.BudgetLedger(path or self.state / "budget.sqlite", clock=lambda: 1000.0)
        owner = submission.SubmissionCoordinator(ledger)
        self.addCleanup(owner.close)
        return owner

    def service(self, *, owner=None, state=None, outputs=None, clock=None):
        owner = owner or self.owner()
        service = preparation.PreparationService(
            state or self.state, outputs or self.outputs, owner, clock=clock or (lambda: 1000.0))
        self.addCleanup(service.close)
        return service

    @staticmethod
    async def chunks(contents):
        yield contents[:17]
        yield contents[17:1000]
        yield contents[1000:]

    def import_source(self, service, run, **changes):
        arguments = {"source_suffix": ".mp4", "clean_source": True,
                     "source_kind": "synthetic_fixture"} | changes

        async def operation():
            async with service.media_operation(run.run_id, expected_revision=run.revision) as media:
                return await media.import_source(self.chunks(self.fixture.video.original_bytes), **arguments)

        return asyncio.run(operation())

    def import_asset(self, service, run, **changes):
        arguments = {"non_human": True} | changes

        async def operation():
            async with service.media_operation(run.run_id, expected_revision=run.revision) as media:
                return await media.import_asset(self.chunks(self.fixture.validated_asset.original_bytes), **arguments)

        return asyncio.run(operation())

    def ready(self, service):
        run = service.create_run()
        with patch.object(preparation, "normalize_video", return_value=self.fixture.video):
            run = self.import_source(service, run)
        run = self.import_asset(service, run)
        return service.set_placement(run.run_id, expected_revision=run.revision, placement=self.placement)

    def build(self, service, run, **arguments):
        async def operation():
            async with service.media_operation(run.run_id, expected_revision=run.revision) as media:
                return await media.build_guides(**arguments)

        return asyncio.run(operation())

    def cached_guidance(self, video, asset, placement, prompt):
        self.assertEqual(video, self.fixture.video)
        self.assertEqual(asset, self.fixture.validated_asset)
        self.assertEqual(prompt, self.fixture.guidance.prompt)
        return self.second_guidance if placement == self.second_placement else self.fixture.guidance

    def files(self, root=None):
        root = root or self.outputs
        return {path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}

    def assert_code(self, code, operation):
        with self.assertRaises(preparation.ServiceError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_create_run_starts_original_clock_and_grants_no_authority(self):
        clock = Mock(return_value=1000.25)
        owner = self.owner()
        service = self.service(owner=owner, clock=clock)
        clock.assert_not_called()
        run = service.create_run()
        clock.assert_called_once_with()
        self.assertRegex(run.run_id, r"\Arun-[0-9a-f]{32}\Z")
        self.assertEqual(run.revision, 0)
        self.assertIsNone(run.source)
        self.assertIsNone(run.asset)
        self.assertIsNone(run.placement)
        self.assertIsNone(run.prepared)
        self.assertEqual(run.attempts, ())
        started = run_start.read_run_start(self.outputs, run_id=run.run_id)
        self.assertEqual(started.record, run.run_start)
        self.assertEqual((started.started_unix, started.deadline_unix), (1000.25, 11800.25))
        self.assertEqual(service.get_run(run.run_id), run)
        self.assertNotIn(str(self.root), json.dumps(asdict(run)))
        with self.assertRaises(FrozenInstanceError):
            run.revision = 5
        with owner.ledger._connection() as connection:
            for table in ("approved_runs", "reservations", "submission_jobs"):
                self.assertEqual(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
        other = service.create_run()
        self.assertNotEqual(other.run_id, run.run_id)

    def test_real_imports_validate_once_after_start_and_preserve_exact_bytes(self):
        service = self.service()
        run = service.create_run()
        original_normalize = preparation.normalize_video
        calls = []

        def normalize(path):
            calls.append(path.read_bytes())
            self.assertEqual(run_start.read_run_start(self.outputs, run_id=run.run_id).record, run.run_start)
            return original_normalize(path)

        with patch.object(preparation, "normalize_video", side_effect=normalize) as source_validator, \
                patch.object(preparation, "load_asset", wraps=preparation.load_asset) as asset_validator:
            source_run = self.import_source(service, run, source_suffix=".mov")
            selected = self.import_asset(service, source_run)
        source_validator.assert_called_once()
        asset_validator.assert_called_once()
        self.assertEqual(calls, [self.fixture.video.original_bytes])
        self.assertEqual((source_run.revision, selected.revision), (1, 2))
        self.assertEqual(selected.source, source_run.source)
        self.assertEqual(selected.source.source_suffix, ".mov")
        self.assertEqual(selected.source.source_kind, "synthetic_fixture")
        root = self.outputs / run.run_id
        self.assertEqual(read_verified(root, selected.source.original), self.fixture.video.original_bytes)
        self.assertEqual(read_verified(root, selected.source.normalized), self.fixture.video.video_bytes)
        self.assertEqual(read_verified(root, selected.asset.original), self.fixture.validated_asset.original_bytes)
        self.assertEqual((selected.asset.width, selected.asset.height), (16, 16))
        for selected_input in (selected.source, selected.asset):
            contents = read_verified(root, selected_input.record)
            self.assertIn(selected_input.record.sha256, selected_input.record.path)
            self.assertNotIn(str(self.root).encode(), contents)
        self.assertEqual(service.get_run(run.run_id), selected)

    def test_invalid_import_declarations_fail_without_consuming_body(self):
        service = self.service()
        run = service.create_run()
        before = self.files()
        consumed = []

        async def unread():
            consumed.append(True)
            yield b"private malformed body"

        async def source(arguments):
            async with service.media_operation(run.run_id, expected_revision=run.revision) as media:
                return await media.import_source(unread(), **arguments)

        base = {"source_suffix": ".mp4", "clean_source": True, "source_kind": "synthetic_fixture"}
        for changes in ({"source_suffix": ".avi"}, {"source_suffix": ".MP4"},
                        {"source_suffix": "../source.mp4"}, {"clean_source": False},
                        {"clean_source": 1}, {"source_kind": "demo_reference"}):
            with self.subTest(changes=changes):
                self.assert_code("INVALID_INPUT", lambda: asyncio.run(source(base | changes)))

        async def asset(non_human):
            async with service.media_operation(run.run_id, expected_revision=run.revision) as media:
                return await media.import_asset(unread(), non_human=non_human)

        for value in (False, 1, None):
            self.assert_code("INVALID_INPUT", lambda: asyncio.run(asset(value)))
        self.assertEqual(consumed, [])
        self.assertEqual(self.files(), before)
        self.assertEqual(service.get_run(run.run_id), run)

        async def receive_input(selected, kind, parts):
            async def body():
                for part in parts:
                    yield part

            async with service.media_operation(selected.run_id, expected_revision=selected.revision) as media:
                if kind == "source":
                    return await media.import_source(body(), **base)
                return await media.import_asset(body(), non_human=True)

        fixtures = (
            ("source", "MAX_INPUT_BYTES", "normalize_video", self.fixture.video,
             self.fixture.video.original_bytes),
            ("asset", "MAX_FILE_BYTES", "load_asset", self.fixture.validated_asset,
             self.fixture.validated_asset.original_bytes),
        )
        for kind, limit_name, validator_name, validated, contents in fixtures:
            with self.subTest(kind=kind), patch.object(preparation, limit_name, len(contents)), \
                    patch.object(preparation, validator_name, return_value=validated) as validator:
                # The real cached validator result isolates streaming limits from
                # decoder rejection. Exactly the declared local cap is accepted.
                run = asyncio.run(receive_input(run, kind, [contents[:17], contents[17:]]))
                validator.assert_called_once()
                validator.reset_mock()
                selected_files = self.files()
                for parts in ([contents, b" "], [b""], [memoryview(contents)], ["not bytes"]):
                    with self.subTest(kind=kind, part_types=[type(part).__name__ for part in parts]):
                        self.assert_code("INVALID_INPUT", lambda: asyncio.run(receive_input(run, kind, parts)))
                        validator.assert_not_called()
                        self.assertEqual(service.get_run(run.run_id), run)
                        self.assertEqual(self.files(), selected_files)
                        self.assertEqual(list(self.state.glob(".preparation-*")), [])

    def test_placement_requires_exact_revision_and_keeps_previous_selection(self):
        service = self.service()
        run = service.create_run()
        changed = service.set_placement(run.run_id, expected_revision=0, placement=self.placement)
        self.assertEqual(changed.revision, 1)
        self.assertEqual(changed.placement, self.placement)
        self.assert_code("REVISION_CONFLICT", lambda: service.set_placement(
            run.run_id, expected_revision=0, placement=self.second_placement))
        for revision in (True, False, None, -1, 1.0, "1"):
            with self.subTest(revision=revision):
                self.assert_code("INVALID_INPUT", lambda: service.set_placement(
                    run.run_id, expected_revision=revision, placement=self.second_placement))
        self.assertEqual(service.get_run(run.run_id), changed)
        with self.assertRaises(FrozenInstanceError):
            changed.placement.u = 0.2

    def test_build_requires_all_inputs_and_creates_no_attempt_on_rejection(self):
        service = self.service()
        run = service.create_run()
        with patch.object(prepare, "compile_guidance", side_effect=ValueError("Unexpected renderer.")) as compiler:
            self.assert_code("INPUT_REQUIRED", lambda: self.build(service, run))
            with patch.object(preparation, "normalize_video", return_value=self.fixture.video):
                run = self.import_source(service, run)
            self.assert_code("INPUT_REQUIRED", lambda: self.build(service, run))
            run = self.import_asset(service, run)
            self.assert_code("INPUT_REQUIRED", lambda: self.build(service, run))
            compiler.assert_not_called()
        self.assertEqual(service.get_run(run.run_id).attempts, ())
        self.assertFalse((self.outputs / run.run_id / "attempts").exists())

    def test_two_placements_keep_original_attempt_and_select_only_new_guides(self):
        service = self.service()
        run = self.ready(service)
        with patch.object(prepare, "compile_guidance", side_effect=self.cached_guidance) as compiler, \
                patch.object(preparation, "normalize_video", side_effect=ValueError("Repeated normalization.")) as normalize, \
                patch.object(preparation, "load_asset", side_effect=ValueError("Repeated asset validation.")) as asset:
            first = self.build(service, run)
            self.assertEqual(first.revision, run.revision + 1)
            self.assertEqual(len(first.attempts), 1)
            attempt = first.attempts[0]
            self.assertEqual((attempt.revision, attempt.status, attempt.manifest, attempt.error_code),
                             (run.revision, "PREPARED", first.prepared, None))
            root = self.outputs / run.run_id
            before = self.files(root / "attempts" / attempt.attempt_id)
            changed = service.set_placement(run.run_id, expected_revision=first.revision,
                                             placement=self.second_placement)
            self.assertIsNone(changed.prepared)
            self.assertEqual(changed.attempts, first.attempts)
            second = self.build(service, changed)
            self.assertEqual(second.revision, changed.revision + 1)
            self.assertEqual(len(second.attempts), 2)
            self.assertEqual(second.attempts[0], attempt)
            self.assertNotEqual(second.attempts[1].attempt_id, attempt.attempt_id)
            self.assertEqual(before, self.files(root / "attempts" / attempt.attempt_id))
            old_manifest = json.loads(read_verified(root, first.prepared))
            new_manifest = json.loads(read_verified(root, second.prepared))
            self.assertNotEqual(old_manifest["artifacts"]["frozen/keyframe_A.png"]["sha256"],
                                new_manifest["artifacts"]["frozen/keyframe_A.png"]["sha256"])
            self.assertEqual(old_manifest["artifacts"]["frozen/source_original.mp4"]["sha256"],
                             run.source.original.sha256)
            self.assertEqual(new_manifest["artifacts"]["frozen/asset_original.png"]["sha256"],
                             run.asset.original.sha256)
            self.assertEqual(compiler.call_count, 2)
            normalize.assert_not_called()
            asset.assert_not_called()

    def test_placement_edit_during_render_keeps_completed_attempt_out_of_current_selection(self):
        service = self.service()
        run = self.ready(service)
        changed = []

        def render(video, asset, placement, prompt):
            self.assertEqual(placement, self.placement)
            changed.append(service.set_placement(run.run_id, expected_revision=run.revision,
                                                 placement=self.second_placement))
            return self.fixture.guidance

        with patch.object(prepare, "compile_guidance", side_effect=render) as compiler:
            result = self.build(service, run)
        compiler.assert_called_once()
        self.assertEqual(len(changed), 1)
        self.assertEqual(result.revision, run.revision + 1)
        self.assertEqual(result.placement, self.second_placement)
        self.assertIsNone(result.prepared)
        self.assertEqual(len(result.attempts), 1)
        attempt = result.attempts[0]
        self.assertEqual((attempt.revision, attempt.status, attempt.error_code),
                         (run.revision, "PREPARED", None))
        self.assertIsNotNone(attempt.manifest)
        manifest = json.loads(read_verified(self.outputs / run.run_id, attempt.manifest))
        placement_ref = manifest["artifacts"]["frozen/placement.json"]
        self.assertEqual(json.loads((self.outputs / run.run_id / placement_ref["path"]).read_bytes()),
                         {"u": 0.5, "v": 0.5, "heightFrac": 0.25, "rotationDeg": 0})
        self.assertEqual(service.get_run(run.run_id), result)

    def test_partial_build_is_failed_without_replacing_files_or_adopting_a_manifest(self):
        service = self.service()
        run = self.ready(service)
        publish = prepare.publish_bytes
        calls = []

        def fail_after_three(root, path, contents):
            calls.append(path)
            if len(calls) == 4:
                raise OSError("private storage details must not be recorded")
            return publish(root, path, contents)

        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance), \
                patch.object(prepare, "publish_bytes", side_effect=fail_after_three):
            self.assert_code("PREPARATION_FAILED", lambda: self.build(service, run))
        failed = service.get_run(run.run_id)
        self.assertEqual(failed.revision, run.revision)
        self.assertIsNone(failed.prepared)
        self.assertEqual(len(failed.attempts), 1)
        attempt = failed.attempts[0]
        self.assertEqual((attempt.status, attempt.manifest, attempt.error_code),
                         ("FAILED", None, "WORK_FAILED"))
        folder = self.outputs / run.run_id / "attempts" / attempt.attempt_id
        before = self.files(folder)
        self.assertEqual(set(before), {"frozen/placement.json", "frozen/source_original.mp4",
                                       "frozen/source_normalized.mp4"})
        self.assertNotIn("private storage details", json.dumps(asdict(failed)))
        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance):
            completed = self.build(service, failed)
        self.assertEqual(len(completed.attempts), 2)
        self.assertEqual(completed.attempts[0], attempt)
        self.assertNotEqual(completed.attempts[1].attempt_id, attempt.attempt_id)
        self.assertEqual(self.files(folder), before)

    def test_restart_reuses_verified_inputs_without_repeating_media_validation(self):
        owner = self.owner()
        service = self.service(owner=owner)
        run = self.ready(service)
        before = self.files()
        service.close()
        with patch.object(preparation, "normalize_video", side_effect=ValueError("Repeated normalization.")) as normalize, \
                patch.object(preparation, "load_asset", side_effect=ValueError("Repeated asset validation.")) as asset, \
                patch.object(prepare, "compile_guidance", side_effect=self.cached_guidance) as compiler:
            restarted = self.service(owner=owner, clock=Mock(side_effect=ValueError("No new start clock.")))
            restored = restarted.get_run(run.run_id)
            self.assertEqual(restored, run)
            self.assertEqual(self.files(), before)
            completed = self.build(restarted, restored)
            self.assertIsNotNone(completed.prepared)
            self.assertEqual(completed.source, run.source)
            self.assertEqual(completed.asset, run.asset)
            self.assertEqual(completed.run_start, run.run_start)
            compiler.assert_called_once()
            normalize.assert_not_called()
            asset.assert_not_called()

    def test_changed_saved_input_bytes_cannot_reach_the_guide_compiler(self):
        service = self.service()
        run = self.ready(service)
        root = self.outputs / run.run_id
        source_record = root / run.source.record.path
        original_record = source_record.read_bytes()
        self.assertTrue(original_record.endswith(b"\n"))
        # Keep valid JSON and the same byte length, so a shape/length check alone
        # cannot stand in for the recorded content hash.
        source_record.write_bytes(original_record[:-1] + b" ")
        self.assert_code("STATE_UNAVAILABLE", lambda: service.get_run(run.run_id))
        source_record.write_bytes(original_record)
        normalized = root / run.source.normalized.path
        original_video = normalized.read_bytes()
        normalized.write_bytes(bytes([original_video[0] ^ 1]) + original_video[1:])
        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance) as compiler:
            with self.assertRaises(preparation.ServiceError):
                self.build(service, run)
            compiler.assert_not_called()
        self.assertIsNone(service.get_run(run.run_id).prepared)

    def test_restart_marks_unfinished_build_interrupted_and_never_adopts_published_files(self):
        owner = self.owner()
        service = self.service(owner=owner)
        run = self.ready(service)
        database = self.state / "preparation.sqlite"
        # Insert the exact supported BUILDING transition, then publish complete
        # files as if the process stopped before committing the result reference.
        attempt_id = "att-" + "a" * 32
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            row = connection.execute("SELECT source_id,asset_id,placement FROM preparation_runs WHERE run_id=?",
                                     (run.run_id,)).fetchone()
            connection.execute(
                "INSERT INTO preparation_attempts VALUES (?,?,?,?,?,?,?,'BUILDING',NULL,NULL)",
                (attempt_id, run.run_id, run.revision, *row, self.fixture.guidance.prompt))
        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance):
            result = prepare.prepare_validated_attempt(
                self.fixture.video, self.fixture.validated_asset, self.outputs, run.run_id,
                attempt_id, self.placement, source_suffix=".mp4", non_human=True,
                clean_source=True, source_kind="synthetic_fixture")
        before = self.files()
        service.close()
        with patch.object(prepare, "compile_guidance", side_effect=ValueError("Restart cannot render.")) as compiler:
            restarted = self.service(owner=owner)
            recovered = restarted.get_run(run.run_id)
            compiler.assert_not_called()
        self.assertEqual(recovered.revision, run.revision)
        self.assertIsNone(recovered.prepared)
        self.assertEqual(len(recovered.attempts), 1)
        attempt = recovered.attempts[0]
        self.assertEqual((attempt.attempt_id, attempt.status, attempt.manifest, attempt.error_code),
                         (attempt_id, "INTERRUPTED", None, "WORK_INTERRUPTED"))
        self.assertEqual(self.files(), before)
        self.assertTrue((self.outputs / run.run_id / result["manifest"]["path"]).exists())
        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance):
            completed = self.build(restarted, recovered)
        self.assertEqual(completed.attempts[0], attempt)
        self.assertNotEqual(completed.attempts[1].attempt_id, attempt_id)

    def test_existing_run_folders_and_orphaned_starts_are_not_adopted(self):
        service = self.service()
        legacy_id = "run-" + "b" * 32
        started = run_start.start_run(self.outputs, run_id=legacy_id, clock=lambda: 900.0)
        legacy_bytes = read_verified(self.outputs / legacy_id, started.record)
        self.assert_code("RUN_NOT_FOUND", lambda: service.get_run(legacy_id))
        with patch.object(preparation_state.State, "create", side_effect=preparation.ServiceError("STATE_UNAVAILABLE")):
            self.assert_code("STATE_UNAVAILABLE", service.create_run)
        orphan_ids = {path.name for path in self.outputs.iterdir()} - {legacy_id}
        self.assertEqual(len(orphan_ids), 1)
        orphan_id = next(iter(orphan_ids))
        before = self.files(self.outputs / orphan_id)
        self.assertEqual(set(before), {"run-start.json"})
        self.assert_code("RUN_NOT_FOUND", lambda: service.get_run(orphan_id))
        new = service.create_run()
        self.assertNotIn(new.run_id, {legacy_id, orphan_id})
        self.assertEqual(self.files(self.outputs / orphan_id), before)
        self.assertEqual(read_verified(self.outputs / legacy_id, started.record), legacy_bytes)

    def test_foreign_schema_and_owner_mismatch_are_rejected_without_rewriting_database(self):
        owner = self.owner()
        service = self.service(owner=owner)
        run = service.create_run()
        service.close()
        database = self.state / "preparation.sqlite"
        before = database.read_bytes()
        self.assert_code("STATE_UNAVAILABLE", lambda: self.service(owner=owner, outputs=self.root / "other-outputs"))
        self.assertEqual(database.read_bytes(), before)
        other_owner = self.owner(self.state / "different-budget.sqlite")
        self.assert_code("STATE_UNAVAILABLE", lambda: self.service(owner=other_owner))
        self.assertEqual(database.read_bytes(), before)
        # Retain every expected table and owner so a removed schema guard would
        # admit this database, rather than fail on a later missing-table query.
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.execute("INSERT INTO unrelated VALUES ('retain-me')")
        before = database.read_bytes()
        self.assert_code("STATE_UNAVAILABLE", lambda: self.service(owner=owner))
        self.assertEqual(database.read_bytes(), before)
        with sqlite3.connect(database) as connection:
            self.assertEqual(connection.execute("SELECT value FROM unrelated").fetchone()[0], "retain-me")
            self.assertEqual(connection.execute("SELECT run_id FROM preparation_runs").fetchone()[0], run.run_id)

    def test_unsafe_state_paths_closed_owner_and_sqlite_aliases_fail_before_connections(self):
        owner = self.owner()
        public = self.root / "public-state"
        public.mkdir(mode=0o755)
        public.chmod(0o755)
        linked = self.root / "linked-state"
        linked.symlink_to(self.state, target_is_directory=True)
        for state, output in ((public, self.outputs), (linked, self.outputs), (self.state, self.root)):
            with self.subTest(state=state, output=output):
                with patch.object(preparation_state.sqlite3, "connect",
                                  side_effect=preparation.ServiceError("STATE_UNAVAILABLE")) as connect:
                    self.assert_code("STATE_UNAVAILABLE", lambda: self.service(owner=owner, state=state, outputs=output))
                    connect.assert_not_called()
        owner.close()
        with self.assertRaises(preparation.ServiceError):
            self.service(owner=owner)
        self.assertFalse((self.state / "preparation.sqlite").exists())
        alias_state = self.root / "alias-state"
        alias_state.mkdir(mode=0o700)
        alias_owner = self.owner(alias_state / "preparation.sqlite-wal")
        alias_owner.ledger.path.chmod(0o600)
        before = alias_owner.ledger.path.read_bytes()
        with patch.object(preparation_state.sqlite3, "connect",
                          side_effect=preparation.ServiceError("STATE_UNAVAILABLE")) as connect:
            self.assert_code("STATE_UNAVAILABLE", lambda: self.service(owner=alias_owner, state=alias_state))
            connect.assert_not_called()
        self.assertFalse((alias_state / "preparation.sqlite").exists())
        self.assertEqual(alias_owner.ledger.path.read_bytes(), before)

    def test_new_sidecar_hardlink_is_rejected_before_any_sqlite_connection(self):
        owner = self.owner()
        service = self.service(owner=owner)
        run = service.create_run()
        owner.ledger.path.chmod(0o600)
        sidecar = self.state / "preparation.sqlite-journal"
        os.link(owner.ledger.path, sidecar)
        ledger_before = owner.ledger.path.read_bytes()
        database = self.state / "preparation.sqlite"
        state_before = database.read_bytes()
        with patch.object(preparation_state.sqlite3, "connect",
                          side_effect=preparation.ServiceError("STATE_UNAVAILABLE")) as connect:
            self.assert_code("STATE_UNAVAILABLE", lambda: service.get_run(run.run_id))
            self.assert_code("STATE_UNAVAILABLE", lambda: service.set_placement(
                run.run_id, expected_revision=run.revision, placement=self.placement))
            service.close()
            self.assert_code("STATE_UNAVAILABLE", lambda: self.service(owner=owner))
            connect.assert_not_called()
        self.assertEqual(owner.ledger.path.read_bytes(), ledger_before)
        self.assertEqual(sidecar.read_bytes(), ledger_before)
        self.assertEqual(database.read_bytes(), state_before)

    def test_run_input_attempt_and_database_caps_reject_further_state(self):
        service = self.service()
        run = service.create_run()
        with patch.object(preparation_state, "MAX_RUNS", 1):
            self.assert_code("CAPACITY_EXCEEDED", service.create_run)
        with patch.object(preparation, "normalize_video", return_value=self.fixture.video):
            run = self.import_source(service, run)
        with patch.object(preparation_state, "MAX_INPUTS_PER_RUN", 1):
            self.assert_code("CAPACITY_EXCEEDED", lambda: self.import_asset(service, run))
        self.assertIsNone(service.get_run(run.run_id).asset)
        run = self.import_asset(service, run)
        run = service.set_placement(run.run_id, expected_revision=run.revision, placement=self.placement)
        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance):
            run = self.build(service, run)
        with patch.object(preparation_state, "MAX_ATTEMPTS_PER_RUN", 1), \
                patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance) as compiler:
            self.assert_code("CAPACITY_EXCEEDED", lambda: self.build(service, run))
            compiler.assert_not_called()
        self.assertEqual(service.get_run(run.run_id), run)
        database = self.state / "preparation.sqlite"
        before = database.read_bytes()
        with patch.object(preparation_state, "MAX_DATABASE_BYTES", len(before) - 1):
            self.assert_code("STATE_UNAVAILABLE", lambda: service.get_run(run.run_id))
        self.assertEqual(database.read_bytes(), before)
        with sqlite3.connect(database) as connection:
            counts = [connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                      for table in ("preparation_runs", "preparation_inputs", "preparation_attempts")]
        self.assertEqual(counts, [1, 2, 1])

    def test_one_preparation_service_per_owner_and_close_allows_reopen(self):
        owner = self.owner()
        service = self.service(owner=owner)
        run = service.create_run()
        with patch.object(preparation_state.State, "__init__",
                          side_effect=preparation.ServiceError("STATE_UNAVAILABLE")) as construct:
            self.assert_code("MEDIA_BUSY", lambda: self.service(owner=owner))
            construct.assert_not_called()
        self.assertEqual(service.get_run(run.run_id), run)
        service.close()
        restarted = self.service(owner=owner)
        self.assertEqual(restarted.get_run(run.run_id), run)

    def test_current_prepared_reference_requires_matching_completed_history(self):
        service = self.service()
        run = service.create_run()
        fake = {"path": "attempts/att-" + "a" * 32 + "/manifest.json",
                "sha256": "b" * 64, "size": 100}
        with sqlite3.connect(self.state / "preparation.sqlite") as connection:
            connection.execute("UPDATE preparation_runs SET prepared=?,revision=revision+1 WHERE run_id=?",
                               (json.dumps(fake), run.run_id))
        self.assert_code("STATE_UNAVAILABLE", lambda: service.get_run(run.run_id))

    def test_server_lifespan_uses_existing_owner_and_closes_service_before_owner(self):
        config = server.LocalServerConfig(self.state, self.outputs)
        actual_owner = server.SubmissionCoordinator
        owners = []
        order = []

        def coordinator(ledger):
            owner = actual_owner(ledger)
            owners.append(owner)
            original_close = owner.close

            def close():
                order.append("owner")
                original_close()

            owner.close = close
            return owner

        class Service:
            def close(self):
                owners[0]._require_open()
                order.append("service")

        service = Service()

        def construct(state_dir, output_root, owner):
            self.assertEqual((state_dir, output_root), (self.state, self.outputs))
            self.assertEqual(len(owners), 1)
            self.assertIs(owner, owners[0])
            owner._require_open()
            return service

        with patch.object(server, "SubmissionCoordinator", side_effect=coordinator) as make_owner, \
                patch.object(server, "PreparationService", side_effect=construct) as make_service:
            app = server.create_app(config)
            make_owner.assert_not_called()
            make_service.assert_not_called()
            with TestClient(app, base_url=config.origin):
                self.assertIs(app.state.preparation, service)
                self.assertIs(app.state.coordinator, owners[0])
                self.assertEqual(order, [])
                make_owner.assert_called_once()
                make_service.assert_called_once()
        self.assertEqual(order, ["service", "owner"])

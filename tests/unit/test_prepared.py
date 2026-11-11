"""Check frozen record integrity with the real writer and stub media bytes."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app import prepare
from core.storage import StorageError, StoredArtifact, publish_bytes
from features.inference import prepared as prepared_module
from features.inference.prepared import PreparationError, read_prepared_attempt
from features.placement.math import Placement


class PreparedTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.run_root = self.root / "run-1"
        self.video = SimpleNamespace(original_bytes=b"synthetic original", video_bytes=b"synthetic normalized")
        self.asset = SimpleNamespace(original_bytes=b"synthetic asset")
        self.guidance = SimpleNamespace(keyframe_png=b"synthetic keyframe", guide_mp4=b"synthetic guide",
                                        prompt="Turn the fixture creature's head.")
        self.result = self.write("att-1")
        self.manifest = StoredArtifact(**self.result["manifest"])
        self.data = json.loads((self.run_root / self.manifest.path).read_bytes())

    def write(self, attempt, suffix=".mp4"):
        # The reader checks byte identity. Media decoding has its own real-media tests.
        with patch.object(prepare, "normalize_video", return_value=self.video), \
                patch.object(prepare, "load_asset", return_value=self.asset), \
                patch.object(prepare, "compile_guidance", return_value=self.guidance):
            return prepare.prepare_attempt(self.root / ("source" + suffix), self.root / "asset.png",
                                           self.root, "run-1", attempt, Placement(0.5, 0.6, 0.25),
                                           non_human=True, clean_source=True, source_kind="synthetic_fixture")

    def read(self, **changes):
        arguments = {"run_id": "run-1", "attempt_id": "att-1"} | changes
        return read_prepared_attempt(self.run_root, self.manifest, **arguments)

    def replace_file(self, record, body):
        (self.run_root / record.path).write_bytes(body)
        return StoredArtifact(record.path, sha256(body).hexdigest(), len(body))

    def save_manifest(self):
        self.manifest = self.replace_file(self.manifest, json.dumps(self.data).encode())

    def change_control(self, name, body):
        record = StoredArtifact(**self.data["artifacts"][name])
        self.data["artifacts"][name] = asdict(self.replace_file(record, body))
        self.save_manifest()

    def test_writer_and_reader_preserve_identity_and_choose_matching_guides(self):
        result = self.read()
        self.assertEqual((result.run_id, result.attempt_id, result.source_kind), ("run-1", "att-1", "synthetic_fixture"))
        self.assertEqual(result.manifest, self.manifest)
        self.assertEqual(result.prompt, self.guidance.prompt)
        self.assertEqual(result.placement, Placement(0.5, 0.6, 0.25))
        records = self.data["artifacts"]
        self.assertEqual(result.media_for_mode("A"), (StoredArtifact(**records["frozen/source_normalized.mp4"]),
                                                     StoredArtifact(**records["frozen/keyframe_A.png"])))
        self.assertEqual(result.media_for_mode("B"), (StoredArtifact(**records["frozen/guide_B.mp4"]), None))
        for mode in [None, "", "C", True]:
            with self.subTest(mode=mode), self.assertRaises(PreparationError):
                result.media_for_mode(mode)
        mov = self.write("att-mov", suffix=".mov")
        loaded = read_prepared_attempt(self.run_root, StoredArtifact(**mov["manifest"]),
                                       run_id="run-1", attempt_id="att-mov")
        self.assertTrue(loaded.source_original.path.endswith("source_original.mov"))

    def test_expected_ids_and_paths_cannot_switch_attempts(self):
        for changes in [{"run_id": "other"}, {"attempt_id": "other"},
                        {"run_id": "../escape"}, {"attempt_id": "x" * 65}]:
            with self.subTest(changes=changes), self.assertRaises(PreparationError):
                self.read(**changes)
        record = StoredArtifact(**self.data["artifacts"]["frozen/guide_B.mp4"])
        other = publish_bytes(self.run_root, "attempts/other/frozen/guide_B.mp4",
                              (self.run_root / record.path).read_bytes())
        self.data["artifacts"]["frozen/guide_B.mp4"] = asdict(other)
        self.save_manifest()
        with self.assertRaises(PreparationError):
            self.read()

    def test_every_recorded_file_is_verified_without_repair(self):
        for record in [self.manifest, *[StoredArtifact(**value) for value in self.data["artifacts"].values()]]:
            with self.subTest(path=record.path):
                path = self.run_root / record.path
                original = path.read_bytes()
                damaged = b"!" + original[1:]
                path.write_bytes(damaged)
                with self.assertRaises(PreparationError):
                    self.read()
                self.assertEqual(path.read_bytes(), damaged)
                path.write_bytes(original)

    def test_artifact_map_requires_exactly_the_nine_frozen_files(self):
        original = dict(self.data["artifacts"])
        extra = publish_bytes(self.run_root, "attempts/att-1/frozen/extra.bin", b"extra fixture")
        cases = [{name: value for name, value in original.items() if name != "frozen/guide_B.mp4"},
                 original | {"frozen/extra.bin": asdict(extra)},
                 original | {"frozen/source_original.mov": original["frozen/source_original.mp4"]}]
        for files in cases:
            with self.subTest(files=list(files)):
                self.data["artifacts"] = files
                self.save_manifest()
                with self.assertRaises(PreparationError):
                    self.read()

    def test_locked_configuration_is_validated_from_verified_bytes(self):
        original = json.loads((self.run_root / self.data["artifacts"]["frozen/config.json"]["path"]).read_bytes())
        for field, value in [("model", "other"), ("version", "unverified")]:
            data = json.loads(json.dumps(original))
            data["sdk"][field] = value
            self.change_control("frozen/config.json", json.dumps(data).encode())
            with self.subTest(field=field), self.assertRaises(PreparationError):
                self.read()
        original["asset"]["alphaRequired"] = 1
        self.change_control("frozen/config.json", json.dumps(original).encode())
        with self.assertRaises(PreparationError):
            self.read()

    def test_invalid_placement_or_prompt_cannot_be_used(self):
        placement = {"u": 0.5, "v": 0.6, "heightFrac": 0.25, "rotationDeg": 0}
        for field, value in [("u", 2), ("v", True), ("heightFrac", 0), ("rotationDeg", 1)]:
            with self.subTest(field=field):
                self.change_control("frozen/placement.json", json.dumps(placement | {field: value}).encode())
                with self.assertRaises(PreparationError):
                    self.read()
        self.change_control("frozen/placement.json", json.dumps(placement).encode())
        for body in [b" ", b"x" * 1001, b"\xff", b"x" * 4001]:
            with self.subTest(body_size=len(body)):
                self.change_control("frozen/prompt.txt", body)
                with self.assertRaises(PreparationError):
                    self.read()

    def test_input_declarations_and_references_must_match_the_snapshot(self):
        original = json.loads((self.run_root / self.data["artifacts"]["frozen/inputs.json"]["path"]).read_bytes())
        cases = [original | {"sourceKind": "clean_camera"}, original | {"cleanSourceDeclared": False},
                 original | {"nonHumanDeclared": 1}, original | {"source": original["asset"]},
                 original | {"normalizedSource": original["source"]}]
        for inputs in cases:
            with self.subTest(inputs=inputs):
                self.change_control("frozen/inputs.json", json.dumps(inputs).encode())
                with self.assertRaises(PreparationError):
                    self.read()

    def test_preparation_cannot_claim_submitted_or_completed_provider_work(self):
        original = json.loads(json.dumps(self.data))
        cases = [original | {"automatedWorkflowComplete": "PASS"},
                 original | {"provider": original["provider"] | {"submitted": True}},
                 original | {"provider": original["provider"] | {"taskId": "fixture-task"}},
                 original | {"sourceKind": "unknown"}]
        for data in cases:
            with self.subTest(data=data):
                self.data = data
                self.save_manifest()
                with self.assertRaises(PreparationError):
                    self.read()

    def test_json_is_bounded_and_rejects_duplicate_or_nonfinite_fields(self):
        original = json.loads(json.dumps(self.data))
        body = json.dumps(original).replace('"state": "PREPARED"', '"state":"DRAFT","state":"PREPARED"').encode()
        self.manifest = self.replace_file(self.manifest, body)
        with self.assertRaises(PreparationError):
            self.read()
        for extra in [float("nan"), "x" * (64 * 1024)]:
            self.data = original | {"extra": extra}
            self.save_manifest()
            with self.subTest(extra_type=type(extra).__name__), self.assertRaises(PreparationError):
                self.read()
        self.data = original
        self.save_manifest()
        name = "frozen/config.json"
        body = (self.run_root / self.data["artifacts"][name]["path"]).read_bytes() + b" " * (64 * 1024)
        self.change_control(name, body)
        with self.assertRaises(PreparationError):
            self.read()

    def test_missing_or_linked_files_fail_with_a_safe_message(self):
        with patch.object(prepared_module, "read_verified", side_effect=StorageError("private fixture path and secret")):
            with self.assertRaises(PreparationError) as caught:
                self.read()
        self.assertEqual(str(caught.exception), "Prepared snapshot could not be verified.")
        record = StoredArtifact(**self.data["artifacts"]["frozen/guide_B.mp4"])
        path = self.run_root / record.path
        body = path.read_bytes()
        path.unlink()
        with self.assertRaises(PreparationError):
            self.read()
        target = self.root / "linked.mp4"
        target.write_bytes(body)
        path.symlink_to(target)
        with self.assertRaises(PreparationError):
            self.read()

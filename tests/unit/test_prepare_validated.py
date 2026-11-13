"""Freeze validated media without reading or validating the originals again."""

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app import prepare
from core.storage import ArtifactExistsError, StorageError
from features.placement.math import Placement
import test_prepare as media_fixtures


NOW = datetime(2026, 11, 13, 12, 34, 56, tzinfo=timezone.utc)


def json_bytes(value):
    """State the historical JSON encoding independently of the producer."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def artifact(path, contents):
    return {"path": path, "sha256": sha256(contents).hexdigest(), "size": len(contents)}


class PrepareValidatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The local subclass caches real normalized video, PNG, and both guides.
        # It is not another discovered TestCase in this module.
        cls.fixture = type("ValidatedMediaFixture", (media_fixtures.PrepareTests,), {})
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.settings = prepare.load_config()

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.runs = Path(self.folder.name)
        self.placement = Placement(0.5, 0.5, 0.25)

    def build(self, **changes):
        arguments = {
            "video": self.fixture.video, "asset": self.fixture.validated_asset,
            "output_root": self.runs, "run_id": "run-1", "attempt_id": "att-1",
            "placement": self.placement, "source_suffix": ".mp4",
            "non_human": True, "clean_source": True, "source_kind": "synthetic_fixture",
        } | changes
        return prepare.prepare_validated_attempt(**arguments)

    def snapshot(self):
        return {path.relative_to(self.runs).as_posix(): path.read_bytes()
                for path in self.runs.rglob("*") if path.is_file()}

    def expected_files(self, *, attempt_id, suffix, kind):
        prefix = f"attempts/{attempt_id}/"
        original = "frozen/source_original" + suffix
        files = {
            "frozen/placement.json": json_bytes({"u": 0.5, "v": 0.5,
                                                 "heightFrac": 0.25, "rotationDeg": 0}),
            original: self.fixture.video.original_bytes,
            "frozen/source_normalized.mp4": self.fixture.video.video_bytes,
            "frozen/asset_original.png": self.fixture.validated_asset.original_bytes,
            "frozen/config.json": json_bytes(self.settings),
            "frozen/prompt.txt": self.fixture.guidance.prompt.encode("utf-8"),
            "frozen/keyframe_A.png": self.fixture.guidance.keyframe_png,
            "frozen/guide_B.mp4": self.fixture.guidance.guide_mp4,
        }
        files["frozen/inputs.json"] = json_bytes({
            "sourceKind": kind, "cleanSourceDeclared": True, "nonHumanDeclared": True,
            "source": artifact(prefix + original, files[original]),
            "normalizedSource": artifact(prefix + "frozen/source_normalized.mp4",
                                          files["frozen/source_normalized.mp4"]),
            "asset": artifact(prefix + "frozen/asset_original.png", files["frozen/asset_original.png"]),
        })
        records = {name: artifact(prefix + name, contents) for name, contents in files.items()}
        files["manifest.json"] = json_bytes({
            "schemaVersion": "1.4", "manifestType": "offline_preparation", "runId": "run-1",
            "attemptId": attempt_id, "state": "PREPARED", "createdAtUtc": "2026-11-13T12:34:56+00:00",
            "sourceKind": kind, "artifacts": records,
            "provider": {"vendor": "Runway", "model": "aleph2", "taskId": None, "submitted": False},
            "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN",
            "releaseClassification": "LOCAL_PIPELINE_ONLY",
            "reason": "Offline inputs and guides prepared; no generation or human review has run.",
        })
        return {prefix + name: contents for name, contents in files.items()}

    def test_both_source_suffixes_preserve_literal_manifest_and_nine_original_artifacts(self):
        for suffix, kind in ((".mp4", "synthetic_fixture"), (".mov", "clean_camera")):
            with self.subTest(suffix=suffix):
                attempt = "att-" + suffix[1:]
                with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance) as compile_guide, \
                        patch.object(prepare, "datetime") as clock:
                    clock.now.return_value = NOW
                    result = self.build(attempt_id=attempt, source_suffix=suffix, source_kind=kind)
                compile_guide.assert_called_once()
                clock.now.assert_called_once_with(timezone.utc)
                expected = self.expected_files(attempt_id=attempt, suffix=suffix, kind=kind)
                folder = self.runs / "run-1"
                actual = {path.relative_to(folder).as_posix(): path.read_bytes()
                          for path in (folder / "attempts" / attempt).rglob("*") if path.is_file()}
                self.assertEqual(actual, expected)
                manifest_path = f"attempts/{attempt}/manifest.json"
                self.assertEqual(result, {
                    "runId": "run-1", "attemptId": attempt, "state": "PREPARED",
                    "manifest": artifact(manifest_path, expected[manifest_path]), "outputRoot": str(folder),
                })
                self.assertFalse((folder / "run-start.json").exists())

    def test_validated_objects_and_prompt_reach_one_compiler_without_original_reads(self):
        prompt = "Keep the camera fixed and turn the creature's head."
        guidance = replace(self.fixture.guidance, prompt=prompt)
        original_open = Path.open
        original_paths = {self.fixture.source, self.fixture.asset, self.fixture.mov}
        reads = []

        def open_file(path, *arguments, **keywords):
            if path in original_paths:
                reads.append(path)
                raise ValueError("An original path was read again.")
            return original_open(path, *arguments, **keywords)

        with patch.object(prepare, "normalize_video", side_effect=ValueError("Repeated normalization.")) as normalize, \
                patch.object(prepare, "load_asset", side_effect=ValueError("Repeated asset validation.")) as asset, \
                patch.object(prepare, "load_config", wraps=prepare.load_config) as config, \
                patch.object(prepare, "compile_guidance", return_value=guidance) as compile_guide, \
                patch.object(Path, "open", open_file):
            self.build(prompt=prompt)
        normalize.assert_not_called()
        asset.assert_not_called()
        config.assert_called_once_with()
        compile_guide.assert_called_once()
        supplied_video, supplied_asset, supplied_placement, supplied_prompt = compile_guide.call_args.args
        self.assertIs(supplied_video, self.fixture.video)
        self.assertIs(supplied_asset, self.fixture.validated_asset)
        self.assertIs(supplied_placement, self.placement)
        self.assertEqual(supplied_prompt, prompt)
        self.assertEqual(reads, [])
        self.assertEqual((self.runs / "run-1/attempts/att-1/frozen/prompt.txt").read_bytes(), prompt.encode())

    def test_bad_ids_declarations_kind_and_suffix_fail_before_config_compiler_or_publication(self):
        class StringSubclass(str):
            pass

        cases = []
        for field in ("run_id", "attempt_id"):
            cases.extend({field: value} for value in (None, True, 1, "", "../escape", "a/b", "a" * 65, "é"))
        for field in ("non_human", "clean_source"):
            cases.extend({field: value} for value in (False, 0, 1, None, "true"))
        cases.extend({"source_kind": value} for value in (None, True, "", "demo_reference"))
        cases.extend({"source_suffix": value} for value in
                     (None, True, 1, b".mp4", "", "mp4", ".MP4", ".MOV", ".avi", ".mov/../x", StringSubclass(".mp4")))
        for changes in cases:
            with self.subTest(changes=changes):
                with patch.object(prepare, "load_config", side_effect=ValueError("Unexpected config.")) as config, \
                        patch.object(prepare, "compile_guidance", side_effect=ValueError("Unexpected compiler.")) as compiler, \
                        patch.object(prepare, "publish_bytes", side_effect=ValueError("Unexpected publisher.")) as publisher:
                    with self.assertRaises(ValueError):
                        self.build(**changes)
                    config.assert_not_called()
                    compiler.assert_not_called()
                    publisher.assert_not_called()
                self.assertEqual(list(self.runs.iterdir()), [])

    def test_configuration_and_compiler_failures_publish_nothing(self):
        with patch.object(prepare, "load_config", side_effect=ValueError("Configuration unavailable.")), \
                patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance) as compiler, \
                patch.object(prepare, "publish_bytes", side_effect=ValueError("Unexpected publication.")) as publisher:
            with self.assertRaisesRegex(ValueError, "Configuration unavailable"):
                self.build()
            compiler.assert_not_called()
            publisher.assert_not_called()
        with patch.object(prepare, "compile_guidance", side_effect=ValueError("Guidance unavailable.")) as compiler, \
                patch.object(prepare, "publish_bytes", side_effect=ValueError("Unexpected publication.")) as publisher:
            with self.assertRaisesRegex(ValueError, "Guidance unavailable"):
                self.build()
            compiler.assert_called_once()
            publisher.assert_not_called()
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_repeated_attempt_and_changed_extension_never_replace_or_add_files(self):
        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance):
            self.build()
            before = self.snapshot()
            for changes in ({}, {"source_suffix": ".mov"}, {"placement": Placement(0.7, 0.5, 0.25)}):
                with self.subTest(changes=changes):
                    with self.assertRaises(ArtifactExistsError):
                        self.build(**changes)
                    self.assertEqual(self.snapshot(), before)
        self.assertNotIn("run-1/attempts/att-1/frozen/source_original.mov", before)

    def test_partial_publication_keeps_claim_and_originals_but_never_manifest(self):
        publish = prepare.publish_bytes
        calls = []

        def fail_after_three(root, name, contents):
            calls.append(name)
            if len(calls) == 4:
                raise StorageError("Fixture storage failure.")
            return publish(root, name, contents)

        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance), \
                patch.object(prepare, "publish_bytes", side_effect=fail_after_three):
            with self.assertRaises(StorageError):
                self.build()
        expected_names = {
            "run-1/attempts/att-1/frozen/placement.json",
            "run-1/attempts/att-1/frozen/source_original.mp4",
            "run-1/attempts/att-1/frozen/source_normalized.mp4",
        }
        before = self.snapshot()
        self.assertEqual(set(before), expected_names)
        self.assertEqual(before["run-1/attempts/att-1/frozen/source_original.mp4"], self.fixture.video.original_bytes)
        self.assertEqual(before["run-1/attempts/att-1/frozen/source_normalized.mp4"], self.fixture.video.video_bytes)
        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance):
            with self.assertRaises(ArtifactExistsError):
                self.build(source_suffix=".mov")
        self.assertEqual(self.snapshot(), before)

    def test_manifest_is_published_only_after_all_nine_files_and_fixed_claim(self):
        publish = prepare.publish_bytes
        calls = []
        files_at_manifest = None

        def capture(root, name, contents):
            nonlocal files_at_manifest
            if name.endswith("/manifest.json"):
                files_at_manifest = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
            calls.append(name)
            return publish(root, name, contents)

        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guidance), \
                patch.object(prepare, "publish_bytes", side_effect=capture):
            self.build(source_suffix=".mov")
        expected = set(self.expected_files(attempt_id="att-1", suffix=".mov", kind="synthetic_fixture"))
        manifest = "attempts/att-1/manifest.json"
        self.assertEqual(calls[0], "attempts/att-1/frozen/placement.json")
        self.assertEqual(calls[-1], manifest)
        self.assertEqual(len(calls), 10)
        self.assertEqual(set(calls), expected)
        self.assertEqual(files_at_manifest, expected - {manifest})

    def test_path_wrapper_preserves_single_config_read_before_asset_and_normalization(self):
        events = []

        def config():
            events.append("config")
            return self.settings

        def asset(path, *, non_human):
            events.append("asset")
            return self.fixture.validated_asset

        def video(path):
            events.append("video")
            return self.fixture.video

        def guidance(*arguments):
            events.append("guidance")
            return self.fixture.guidance

        with patch.object(prepare, "load_config", side_effect=config) as config_call, \
                patch.object(prepare, "load_asset", side_effect=asset) as asset_call, \
                patch.object(prepare, "normalize_video", side_effect=video) as video_call, \
                patch.object(prepare, "compile_guidance", side_effect=guidance) as guidance_call:
            prepare.prepare_attempt(self.fixture.mov, self.fixture.asset, self.runs, "run-1", "att-1",
                                    self.placement, non_human=True, clean_source=True)
        self.assertEqual(events, ["config", "asset", "video", "guidance"])
        config_call.assert_called_once_with()
        asset_call.assert_called_once_with(self.fixture.asset, non_human=True)
        video_call.assert_called_once_with(self.fixture.mov)
        guidance_call.assert_called_once()
        self.assertIs(guidance_call.call_args.args[0], self.fixture.video)
        self.assertIs(guidance_call.call_args.args[1], self.fixture.validated_asset)
        self.assertEqual((self.runs / "run-1/attempts/att-1/frozen/source_original.mov").read_bytes(),
                         self.fixture.video.original_bytes)
        before = self.snapshot()
        with patch.object(prepare, "load_config", side_effect=ValueError("Configuration first.")) as config_call, \
                patch.object(prepare, "load_asset", side_effect=ValueError("Asset second.")) as asset_call, \
                patch.object(prepare, "normalize_video", side_effect=ValueError("Video third.")) as video_call:
            with self.assertRaisesRegex(ValueError, "Configuration first"):
                prepare.prepare_attempt(self.fixture.source, self.fixture.asset, self.runs, "run-1", "att-2",
                                        self.placement, non_human=False, clean_source=True)
            config_call.assert_called_once_with()
            asset_call.assert_not_called()
            video_call.assert_not_called()
        self.assertEqual(self.snapshot(), before)

"""Exercise the real offline CLI, immutable snapshots, and honest outcomes."""

from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from app import prepare
from app.prepare import main, prepare_attempt
from core.storage import ArtifactExistsError, StorageError, StoredArtifact, read_verified
from features.inference.prepared import read_prepared_attempt
from features.placement.math import Placement


class PrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.root = Path(cls.folder.name)
        cls.source = cls.root / "source.mp4"
        cls.asset = cls.root / "asset.png"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=0x204060:s=64x36:r=24:d=5", "-c:v", "libx264",
                        "-threads", "2", "-pix_fmt", "yuv420p", str(cls.source)],
                       capture_output=True, check=True, timeout=30)
        image = Image.new("RGBA", (16, 16), (240, 120, 20, 255))
        image.putpixel((0, 0), (0, 0, 0, 0))
        image.save(cls.asset)
        cls.video = prepare.normalize_video(cls.source)
        cls.validated_asset = prepare.load_asset(cls.asset, non_human=True)
        cls.guidance = prepare.compile_guidance(cls.video, cls.validated_asset,
                                               Placement(0.5, 0.5, 0.25))
        cls.mov = cls.root / "source.mov"
        cls.mov.write_bytes(cls.source.read_bytes())

    def setUp(self):
        self.output = TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.runs = Path(self.output.name)

    def arguments(self):
        return ["--source", str(self.source), "--asset", str(self.asset),
                "--output-root", str(self.runs), "--run-id", "run-1", "--attempt-id", "att-1",
                "--u", "0.5", "--v", "0.5", "--height-frac", "0.25", "--non-human",
                "--confirm-clean-source", "--source-kind", "synthetic_fixture"]

    def prepare(self, attempt_id, placement=Placement(0.5, 0.5, 0.25)):
        return prepare_attempt(self.source, self.asset, self.runs, "run-1", attempt_id,
                               placement, non_human=True, clean_source=True,
                               source_kind="synthetic_fixture")

    def test_real_cli_exports_verified_snapshot_without_secrets_or_paid_claims(self):
        environment = os.environ.copy()
        environment["RUNWAYML_API_SECRET"] = "dummy-secret-do-not-export"
        result = subprocess.run([sys.executable, "-m", "app.prepare", *self.arguments()],
                                env=environment, capture_output=True, check=True, timeout=90)
        response = json.loads(result.stdout)
        root = self.runs / "run-1"
        manifest_bytes = read_verified(root, StoredArtifact(**response["manifest"]))
        manifest = json.loads(manifest_bytes)
        self.assertEqual(response["state"], "PREPARED")
        self.assertEqual(manifest["state"], "PREPARED")
        self.assertEqual(manifest["manifestType"], "offline_preparation")
        self.assertEqual(manifest["sourceKind"], "synthetic_fixture")
        self.assertEqual(manifest["visualEffectDemonstrated"], "NOT_RUN")
        self.assertEqual(manifest["automatedWorkflowComplete"], "NOT_RUN")
        self.assertEqual(manifest["releaseClassification"], "LOCAL_PIPELINE_ONLY")
        self.assertFalse(manifest["provider"]["submitted"])
        self.assertIsNone(manifest["provider"]["taskId"])
        loaded = read_prepared_attempt(root, StoredArtifact(**response["manifest"]),
                                       run_id="run-1", attempt_id="att-1")
        self.assertEqual(loaded.media_for_mode("B"),
                         (StoredArtifact(**manifest["artifacts"]["frozen/guide_B.mp4"]), None))
        self.assertEqual(loaded.prompt, (root / "attempts/att-1/frozen/prompt.txt").read_text())
        self.assertEqual(len(manifest["artifacts"]), 9)
        for name, record in manifest["artifacts"].items():
            with self.subTest(name=name):
                self.assertTrue(record["path"].startswith("attempts/att-1/frozen/"))
                contents = read_verified(root, StoredArtifact(**record))
                self.assertNotIn(b"dummy-secret-do-not-export", contents)
        self.assertEqual((root / "attempts/att-1/frozen/source_original.mp4").read_bytes(),
                         self.source.read_bytes())
        self.assertEqual((root / "attempts/att-1/frozen/asset_original.png").read_bytes(),
                         self.asset.read_bytes())
        self.assertNotIn(b"dummy-secret-do-not-export", result.stdout + result.stderr + manifest_bytes)

    def test_second_placement_keeps_first_attempt_unchanged(self):
        first = self.prepare("att-1")
        root = self.runs / "run-1"
        folder = root / "attempts/att-1"
        before = {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
        self.prepare("att-2", Placement(0.7, 0.5, 0.25))
        after = {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(sha256((root / first["manifest"]["path"]).read_bytes()).hexdigest(),
                         first["manifest"]["sha256"])
        self.assertNotEqual((root / "attempts/att-1/frozen/keyframe_A.png").read_bytes(),
                            (root / "attempts/att-2/frozen/keyframe_A.png").read_bytes())
        with self.assertRaises(ArtifactExistsError):
            self.prepare("att-1", Placement(0.2, 0.5, 0.25))
        self.assertEqual({path.relative_to(folder): path.read_bytes()
                          for path in folder.rglob("*") if path.is_file()}, before)

    def test_invalid_ids_and_declarations_fail_before_media_work(self):
        for run_id, attempt_id, clean, kind, non_human in [
            ("../escape", "att-1", True, "clean_camera", True),
            ("run-1", "a/b", True, "clean_camera", True),
            ("run-1", "", True, "clean_camera", True),
            ("run-1", "x" * 65, True, "clean_camera", True),
            ("run-1", "att-1", False, "clean_camera", True),
            ("run-1", "att-1", True, "demo_reference", True),
            ("run-1", "att-1", True, "clean_camera", False),
        ]:
            with self.subTest(run_id=run_id, attempt_id=attempt_id):
                with patch.object(prepare, "normalize_video", side_effect=
                                  AssertionError("Invalid inputs reached media work.")) as normalize:
                    with self.assertRaises(ValueError):
                        prepare_attempt(self.source, self.asset, self.runs, run_id, attempt_id,
                                        Placement(0.5, 0.5, 0.25), non_human=non_human,
                                        clean_source=clean, source_kind=kind)
                    normalize.assert_not_called()
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_cli_validation_error_returns_nonzero_without_output(self):
        arguments = self.arguments()
        arguments[arguments.index("--u") + 1] = "nan"
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main(arguments), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("u", errors.getvalue())
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_partial_publish_does_not_create_manifest_or_replace_files(self):
        publish = prepare.publish_bytes
        count = 0

        def fail_after_three(*args):
            nonlocal count
            count += 1
            if count == 4:
                raise StorageError("Simulated storage failure.")
            return publish(*args)

        with patch.object(prepare, "publish_bytes", side_effect=fail_after_three):
            with self.assertRaises(StorageError):
                self.prepare("att-1")
        root = self.runs / "run-1/attempts/att-1"
        self.assertFalse((root / "manifest.json").exists())
        self.assertEqual((root / "frozen/source_original.mp4").read_bytes(), self.source.read_bytes())
        self.assertTrue((root / "frozen/source_normalized.mp4").is_file())

    def test_changed_source_extension_cannot_add_files_to_completed_attempt(self):
        with patch.object(prepare, "normalize_video", return_value=self.video), \
                patch.object(prepare, "load_asset", return_value=self.validated_asset), \
                patch.object(prepare, "compile_guidance", return_value=self.guidance):
            self.prepare("att-1")
            folder = self.runs / "run-1/attempts/att-1"
            before = {path.relative_to(folder): path.read_bytes()
                      for path in folder.rglob("*") if path.is_file()}
            with self.assertRaises(ArtifactExistsError):
                prepare_attempt(self.mov, self.asset, self.runs, "run-1", "att-1",
                                Placement(0.5, 0.5, 0.25), non_human=True, clean_source=True)
        self.assertEqual({path.relative_to(folder): path.read_bytes()
                          for path in folder.rglob("*") if path.is_file()}, before)

    def test_concurrent_mixed_extensions_have_one_complete_attempt(self):
        def publish(index):
            try:
                return prepare_attempt(self.source if index % 2 == 0 else self.mov,
                                       self.asset, self.runs, "run-1", "att-1",
                                       Placement(0.5, 0.5, 0.25), non_human=True, clean_source=True)
            except ArtifactExistsError:
                return None

        with patch.object(prepare, "normalize_video", return_value=self.video), \
                patch.object(prepare, "load_asset", return_value=self.validated_asset), \
                patch.object(prepare, "compile_guidance", return_value=self.guidance):
            with ThreadPoolExecutor(max_workers=20) as pool:
                results = list(pool.map(publish, range(20)))
        self.assertEqual(sum(result is not None for result in results), 1)
        root = self.runs / "run-1"
        path = root / "attempts/att-1/manifest.json"
        manifest = json.loads(path.read_text())
        recorded = {record["path"] for record in manifest["artifacts"].values()}
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        self.assertEqual(actual, recorded | {"attempts/att-1/manifest.json"})

    def test_cli_filesystem_failure_has_safe_message(self):
        output, errors = StringIO(), StringIO()
        with patch.object(prepare, "normalize_video", side_effect=OSError("dummy-secret/path")), \
                redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main(self.arguments()), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("Preparation failed", errors.getvalue())
        self.assertNotIn("dummy-secret", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())
        self.assertEqual(list(self.runs.iterdir()), [])

"""Check local page boundaries with small saved artifacts and real CLI processes."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from html.parser import HTMLParser
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app import review as command
from core.storage import StoredArtifact, publish_bytes, read_verified
from features.output import view


RUN = "page-run"
ATTEMPT_A = "att-" + "a" * 32
ATTEMPT_B = "att-" + "b" * 32


class Elements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.tags = []
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        self.tags.append((tag, dict(attributes)))


class ReviewPageTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.base = Path(folder.name)
        self.root = self.base / RUN
        self.data = self.fixture(ATTEMPT_A, "A")
        self.record = self.save(self.data)

    def fixture(self, attempt, mode):
        prefix = f"attempts/{attempt}/"

        def file(name):
            return asdict(publish_bytes(self.root, prefix + name, (attempt + name).encode()))

        original = file("frozen/source_original.mov")
        normalized = file("frozen/source_normalized.mp4")
        guide = file("frozen/keyframe_A.png" if mode == "A" else "frozen/guide_B.mp4")
        raw = file("output/raw.mp4")
        samples = []
        for second in range(5):
            source_index, raw_index = second * 24, second * 30
            guide_sample = {"kind": "static_keyframe", "referenceSeconds": 0.0, "image": guide}
            if mode == "B":
                guide_sample = {"kind": "video_frame", "frameIndex": source_index, "ptsSeconds": float(second),
                                "image": file(f"review/guide/frame-{source_index:06d}.png")}
            samples.append({"requestedSeconds": float(second),
                "source": {"frameIndex": source_index, "ptsSeconds": float(second),
                           "image": file(f"review/source/frame-{source_index:06d}.png")},
                "guide": guide_sample,
                "raw": {"frameIndex": raw_index, "ptsSeconds": second + 0.01, "offsetSeconds": 0.01,
                        "spanPosition": "within_span", "withinDeclaredDuration": True,
                        "image": file(f"review/raw/frame-{raw_index:06d}.png")}})
        return {"schemaVersion": "1.4", "manifestType": "indexed_review_sources", "runId": RUN,
                "attemptId": attempt, "mode": mode, "sourceKind": "synthetic_fixture",
                "providerTaskId": "task-A", "sourceOriginal": original, "sourceNormalized": normalized,
                "providerVideoInput": normalized if mode == "A" else guide, "rawOutput": raw,
                "guide": {"kind": "static_keyframe" if mode == "A" else "video", "artifact": guide},
                "samples": samples, "samplePixelFormat": "8-bit RGB", "sampleAutomaticRotationApplied": False,
                "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN"}

    def save(self, data):
        path = f'attempts/{data["attemptId"]}/review/manifest.json'
        body = json.dumps(data, sort_keys=True).encode()
        (self.root / path).write_bytes(body)
        return StoredArtifact(path, sha256(body).hexdigest(), len(body))

    def build(self, record=None, attempt=ATTEMPT_A):
        return view.publish_review_page(self.base, record or self.record, run_id=RUN, attempt_id=attempt)

    def html(self, record):
        return read_verified(self.root, record).decode()

    def reject(self, data):
        with self.assertRaises(view.ReviewPageError):
            self.build(self.save(data))
        self.assertFalse((self.root / f"attempts/{ATTEMPT_A}/review/index.html").exists())

    def arguments(self, record=None):
        record = record or self.record
        return ["--output-root", str(self.base), "--run-id", RUN, "--attempt-id", ATTEMPT_A,
                "--review-path", record.path, "--review-sha256", record.sha256, "--review-size", str(record.size)]

    def test_mode_a_keeps_native_originals_static_guide_and_five_timed_samples(self):
        html = self.html(self.build())
        elements = Elements(html).tags
        videos = [attrs["src"] for tag, attrs in elements if tag == "video"]
        self.assertEqual(videos, ["../frozen/source_original.mov", "../output/raw.mp4"])
        self.assertEqual(sum(tag == "img" and attrs["src"] == "../frozen/keyframe_A.png"
                             for tag, attrs in elements), 6)
        self.assertEqual([attrs["data-sample"] for _, attrs in elements if "data-sample" in attrs], list("01234"))
        self.assertEqual(sum("data-view" in attrs for _, attrs in elements), 3)
        self.assertIn("prepared five-second source", html)
        self.assertIn("Play the raw output twice at normal speed", html)
        self.assertIn("independent identity, motion, scene, and placement scores", html)
        self.assertIn("persistent hard failure", html)
        self.assertIn("PTS 4.010000 s. Offset from source: +0.010000 s.", html)
        self.assertIn("frame 120", html)
        self.assertIn("Visual effect demonstrated: NOT_RUN", html)
        self.assertIn("Automated workflow complete: NOT_RUN", html)
        self.assertIn("Span position: within raw frame range. Within declared duration: yes.", html)
        downloads = [attrs["href"] for tag, attrs in elements if tag == "a" and "download" in attrs]
        self.assertIn("../frozen/source_original.mov", downloads)
        self.assertIn("../frozen/keyframe_A.png", downloads)
        self.assertIn("../output/raw.mp4", downloads)
        self.assertNotIn("autoplay", html)

    def test_mode_b_uses_its_submitted_video_and_five_guide_frames(self):
        data = self.fixture(ATTEMPT_B, "B")
        html = self.html(self.build(self.save(data), ATTEMPT_B))
        elements = Elements(html).tags
        self.assertEqual([attrs["src"] for tag, attrs in elements if tag == "video"],
                         ["../frozen/source_original.mov", "../frozen/guide_B.mp4", "../output/raw.mp4"])
        self.assertEqual([attrs["src"] for tag, attrs in elements if tag == "img" and attrs["src"].startswith("guide/")],
                         [f"guide/frame-{second * 24:06d}.png" for second in range(5)])
        self.assertNotIn(ATTEMPT_A, html)
        self.assertNotIn("keyframe_A", html)
        for change in (lambda value: value["samples"][0]["guide"].update(kind="static_keyframe"),
                       lambda value: value["samples"][0]["guide"].update(frameIndex=1),
                       lambda value: value["samples"][0]["guide"].update(ptsSeconds=True)):
            changed = deepcopy(data)
            change(changed)
            with self.assertRaises(view.ReviewPageError):
                self.build(self.save(changed), ATTEMPT_B)

    def test_rejects_changed_identity_mode_or_outcomes_before_publication(self):
        changes = {"schemaVersion": "1.3", "manifestType": "other", "runId": "other-run", "mode": "C",
                   "sourceKind": "unknown", "samplePixelFormat": "YUV", "sampleAutomaticRotationApplied": 0,
                   "visualEffectDemonstrated": "PASS", "automatedWorkflowComplete": "PASS"}
        for name, value in changes.items():
            with self.subTest(field=name):
                self.reject(self.data | {name: value})
        with self.assertRaises(view.ReviewPageError):
            self.build(attempt=ATTEMPT_B)
        with self.assertRaises(view.ReviewPageError):
            view.publish_review_page(self.base, self.record, run_id="../escape", attempt_id=ATTEMPT_A)

    def test_rejects_foreign_remote_and_mode_mismatched_references(self):
        for path in ("https://example.com/raw.mp4", "../raw.mp4", f"attempts/{ATTEMPT_B}/output/raw.mp4"):
            with self.subTest(path=path):
                data = deepcopy(self.data)
                data["rawOutput"]["path"] = path
                self.reject(data)
        for change in (lambda data: data["guide"].update(kind="video"),
                       lambda data: data.update(providerVideoInput=data["guide"]["artifact"]),
                       lambda data: data["samples"][0]["guide"].update(image=data["sourceOriginal"])):
            data = deepcopy(self.data)
            change(data)
            self.reject(data)

    def test_checks_each_displayed_file_hash_before_publication(self):
        records = [self.data[name] for name in ("sourceOriginal", "sourceNormalized", "rawOutput")]
        records += [self.data["guide"]["artifact"]]
        records += [sample[name]["image"] for sample in self.data["samples"] for name in ("source", "raw")]
        for record in records:
            with self.subTest(path=record["path"]):
                path = self.root / record["path"]
                original = path.read_bytes()
                path.write_bytes(b"X" * len(original))
                with self.assertRaises(view.ReviewPageError):
                    self.build()
                self.assertFalse((self.root / f"attempts/{ATTEMPT_A}/review/index.html").exists())
                path.write_bytes(original)
        data = self.fixture(ATTEMPT_B, "B")
        record = self.save(data)
        path = self.root / data["samples"][4]["guide"]["image"]["path"]
        path.write_bytes(b"changed")
        with self.assertRaises(view.ReviewPageError):
            self.build(record, ATTEMPT_B)

    def test_conflicting_records_for_one_sample_path_are_rejected(self):
        data = deepcopy(self.data)
        data["samples"][1]["source"] = deepcopy(data["samples"][0]["source"])
        data["samples"][1]["source"]["image"]["sha256"] = "0" * 64
        data["samples"][1]["raw"]["offsetSeconds"] = 1.01
        self.reject(data)

    def test_rejects_invalid_sample_timing_indexes_and_labels(self):
        changes = [lambda data: data["samples"].pop(),
                   lambda data: data["samples"][0].update(requestedSeconds=True),
                   lambda data: data["samples"][0].update(requestedSeconds=1),
                   lambda data: data["samples"][0]["source"].update(frameIndex=120),
                   lambda data: data["samples"][0]["raw"].update(frameIndex=True),
                   lambda data: data["samples"][0]["raw"].update(ptsSeconds="1"),
                   lambda data: data["samples"][0]["raw"].update(offsetSeconds=1),
                   lambda data: data["samples"][0]["raw"].update(spanPosition="unknown"),
                   lambda data: data["samples"][0]["raw"].update(withinDeclaredDuration=1),
                   lambda data: data["samples"][0]["guide"].update(referenceSeconds=1)]
        for number, change in enumerate(changes):
            with self.subTest(case=number):
                data = deepcopy(self.data)
                change(data)
                self.reject(data)

    def test_manifest_hash_size_duplicate_names_and_symlink_are_rejected(self):
        for record in (replace(self.record, sha256="0" * 64), replace(self.record, size=128 * 1024 + 1)):
            with self.subTest(record=record):
                with self.assertRaises(view.ReviewPageError):
                    self.build(record)
        path = self.root / self.record.path
        original = path.read_bytes()
        duplicate = original[:-1] + b', "runId": "page-run"}'
        path.write_bytes(duplicate)
        with self.assertRaises(view.ReviewPageError):
            self.build(replace(self.record, sha256=sha256(duplicate).hexdigest(), size=len(duplicate)))
        path.unlink()
        target = self.base / "outside.json"
        target.write_bytes(original)
        path.symlink_to(target)
        with self.assertRaises(view.ReviewPageError):
            self.build()
        self.assertFalse(path.with_name("index.html").exists())

    def test_page_is_immutable_repeatable_and_independent_of_another_attempt(self):
        record = self.build()
        path = self.root / record.path
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        self.assertEqual(self.build(), record)
        data = self.fixture(ATTEMPT_B, "B")
        self.build(self.save(data), ATTEMPT_B)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        path.write_bytes(b"existing damaged page")
        with self.assertRaises(view.ReviewPageError):
            self.build()
        self.assertEqual(path.read_bytes(), b"existing damaged page")

    def test_text_is_escaped_and_page_has_no_script_or_remote_resource(self):
        data = self.data | {"providerTaskId": '<script src="https://evil.example/x">& secret</script>'}
        html = self.html(self.build(self.save(data)))
        elements = Elements(html).tags
        self.assertIn("&lt;script src=&quot;https://evil.example/x&quot;&gt;&amp; secret&lt;/script&gt;", html)
        self.assertNotIn("script", [tag for tag, _ in elements])
        for _, attributes in elements:
            self.assertFalse(any(name.startswith("on") for name in attributes))
            for name in ("src", "href"):
                if name in attributes:
                    self.assertFalse(attributes[name].startswith(("http:", "https:", "//", "data:", "javascript:")))
        policy = next(attrs["content"] for tag, attrs in elements
                      if tag == "meta" and attrs.get("http-equiv") == "Content-Security-Policy")
        self.assertIn("default-src 'none'", policy)
        self.assertIn("style-src 'sha256-", policy)
        self.assertNotIn("unsafe-inline", policy)

    def test_cli_subprocess_publishes_json_and_reuses_the_page_without_credentials(self):
        environment = {name: value for name, value in os.environ.items() if "KEY" not in name and "TOKEN" not in name}
        invocation = [sys.executable, "-m", "app.review", *self.arguments()]
        first = subprocess.run(invocation, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2],
                               env=environment, timeout=20)
        self.assertEqual(first.returncode, 0, first.stderr)
        result = json.loads(first.stdout)
        self.assertEqual((result["runId"], result["attemptId"]), (RUN, ATTEMPT_A))
        page = StoredArtifact(**result["page"])
        self.assertIn("Original raw output", self.html(page))
        second = subprocess.run(invocation, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2],
                                env=environment, timeout=20)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stderr + second.stderr, "")
        self.assertFalse((self.base / "budget.sqlite").exists())

    def test_cli_failure_is_nonzero_and_does_not_print_exception_details(self):
        failure = subprocess.run([sys.executable, "-m", "app.review",
                                  *self.arguments(replace(self.record, sha256="0" * 64))],
                                 capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2], timeout=20)
        self.assertEqual(failure.returncode, 1)
        self.assertEqual(failure.stdout, "")
        self.assertEqual(failure.stderr, "Review page could not be verified or published for this attempt.\n")
        output, errors = StringIO(), StringIO()
        with patch.object(command, "publish_review_page", side_effect=RuntimeError("private-output-token")):
            with redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(command.main(self.arguments()), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn("private-output-token", errors.getvalue())


if __name__ == "__main__":
    unittest.main()

"""Open real review evidence over loopback HTTP with a local Chromium browser."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

from core.budget import BudgetLedger
from core.storage import read_verified
from features.inference.submission import SubmissionCoordinator
from features.output.review import build_review_evidence
from features.output.view import publish_review_page

# Import the module, so unittest does not collect its TestCase a second time.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
import test_review_evidence as evidence_fixture


def encode_browser_video(path, contents, width, height, fps):
    """Use a browser-supported profile while preserving each encoded original."""
    pixels = path.with_suffix(".rgb")
    pixels.write_bytes(contents)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps),
                    "-i", str(pixels), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-preset", "ultrafast", "-threads", "2", "-video_track_timescale", "24000", str(path)],
                   capture_output=True, check=True, timeout=30)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class ReviewPageBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(evidence_fixture.ReviewEvidenceTests.doClassCleanups)
        with patch.object(evidence_fixture.ReviewEvidenceTests, "encode", staticmethod(encode_browser_video)):
            evidence_fixture.ReviewEvidenceTests.setUpClass()
        fixture = evidence_fixture.ReviewEvidenceTests
        cls.base = fixture.base
        cls.pages = {}
        ledger = BudgetLedger(cls.base / "budget.sqlite", clock=lambda: 5000.0)
        with SubmissionCoordinator(ledger) as owner:
            for attempt in (evidence_fixture.ATTEMPT_A, evidence_fixture.ATTEMPT_B):
                manifest = build_review_evidence(owner, cls.base, fixture.manifests[attempt],
                    fixture.checks[attempt].evidence, run_id=evidence_fixture.RUN, attempt_id=attempt)
                cls.pages[attempt] = publish_review_page(cls.base, manifest,
                    run_id=evidence_fixture.RUN, attempt_id=attempt)

    def test_native_views_samples_and_playback_stay_within_one_local_attempt(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=os.environ.get("GEN_REALITY_BROWSER_BIN", "/usr/bin/chromium"), headless=True)
            try:
                for attempt, mode in ((evidence_fixture.ATTEMPT_A, "A"), (evidence_fixture.ATTEMPT_B, "B")):
                    with self.subTest(mode=mode):
                        attempt_root = self.base / evidence_fixture.RUN / "attempts" / attempt
                        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(attempt_root)))
                        thread = Thread(target=server.serve_forever, daemon=True)
                        thread.start()
                        page = browser.new_page(viewport={"width": 1440, "height": 1000})
                        requests, errors = [], []
                        page.on("request", lambda request: requests.append(request.url))
                        page.on("pageerror", lambda error: errors.append(str(error)))
                        page.on("console", lambda message: errors.append(message.text)
                                if message.type == "error" and not message.location.get("url", "").endswith("/favicon.ico") else None)
                        origin = f"http://127.0.0.1:{server.server_port}"
                        try:
                            response = page.goto(origin + "/review/index.html", wait_until="networkidle")
                            self.assertEqual(response.status, 200)
                            self.assertEqual(page.locator("[data-view]").count(), 3)
                            self.assertEqual(page.locator("[data-sample]").count(), 5)
                            self.assertIn("Visual effect demonstrated: NOT_RUN", page.locator("body").inner_text())
                            self.assertIn("Automated workflow complete: NOT_RUN", page.locator("body").inner_text())
                            media = [("source", 128, 72, 6, "/frozen/source_original.mp4"),
                                     ("raw", 64, 36, 5, "/output/raw.mp4")]
                            if mode == "B":
                                media.append(("guide", 1280, 720, 5, "/frozen/guide_B.mp4"))
                            else:
                                guide = page.locator('[data-view="guide"] img')
                                guide.wait_for()
                                page.wait_for_function("image => image.complete && image.naturalWidth === 1280",
                                                       arg=guide.element_handle())
                                self.assertEqual(guide.get_attribute("src"), "../frozen/keyframe_A.png")
                            for name, width, height, duration, path in media:
                                video = page.locator(f'[data-view="{name}"] video')
                                self.assertTrue(video.evaluate("video => video.controls"))
                                page.wait_for_function("video => video.readyState >= 2", arg=video.element_handle())
                                state = video.evaluate("video => [video.videoWidth, video.videoHeight, video.duration, video.currentSrc]")
                                self.assertEqual(state[:2], [width, height])
                                self.assertAlmostEqual(state[2], duration, places=2)
                                self.assertEqual(state[3], origin + path)
                                self.assertEqual(page.locator(f'[data-view="{name}"] a[download]').get_attribute("href"), ".." + path)
                            raw = page.locator('[data-view="raw"] video')
                            raw.evaluate("async video => { video.muted = true; video.playbackRate = 1; await video.play(); }")
                            page.wait_for_function("video => video.currentTime > 0.1 && !video.paused", arg=raw.element_handle())
                            raw.evaluate("video => video.pause()")
                            for second in range(5):
                                sample = page.locator(f'[data-sample="{second}"]')
                                for index, expected_width in enumerate((1280, 1280, 64)):
                                    image = sample.locator("img").nth(index)
                                    image.scroll_into_view_if_needed()
                                    page.wait_for_function("image => image.complete && image.naturalWidth > 0", arg=image.element_handle())
                                    self.assertEqual(image.evaluate("image => image.naturalWidth"), expected_width)
                                caption = sample.inner_text()
                                self.assertIn(f"Prepared source: frame {second * 24}; PTS {second:.6f} s.", caption)
                                self.assertIn(f"Raw output: frame {second * 30}; PTS {second:.6f} s.", caption)
                                self.assertIn("Offset from source: +0.000000 s.", caption)
                            self.assertFalse(errors, errors)
                            self.assertTrue(requests)
                            self.assertTrue(all(urlsplit(url).scheme == "http" and urlsplit(url).netloc == urlsplit(origin).netloc
                                                for url in requests), requests)
                            # Browser playback and page creation must not replace the original bytes.
                            fixture = evidence_fixture.ReviewEvidenceTests
                            read_verified(self.base / evidence_fixture.RUN, fixture.prepared[attempt].source_original)
                            read_verified(self.base / evidence_fixture.RUN, fixture.completed[attempt].raw)
                            read_verified(self.base / evidence_fixture.RUN, self.pages[attempt])
                        finally:
                            page.close()
                            server.shutdown()
                            server.server_close()
                            thread.join(timeout=5)
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()

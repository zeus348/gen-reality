"""Use the real local browser flow to prepare media and retain guide history."""

from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, PngImagePlugin
from playwright.sync_api import Error as PlaywrightError, expect, sync_playwright

import test_local_session_browser as session_browser


def _media_files(root):
    source, asset = root / "distinct-first-frame.mp4", root / "original-alpha.png"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "color=c=0x204060:s=320x180:r=24:d=5,"
        "drawbox=x=0:y=0:w=iw:h=ih:color=0xa03020:t=fill:enable='eq(n,0)'",
        "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", str(source)],
        capture_output=True, check=True, timeout=30)
    image = Image.new("RGBA", (160, 120), (0, 0, 0, 0))
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((16, 16, 63, 103), fill=(0, 255, 0, 128))
    drawing.rectangle((80, 16, 143, 103), fill=(220, 150, 30, 255))
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("FixturePurpose", "Keep the exact original PNG and its alpha.")
    image.save(asset, pnginfo=metadata)
    return source, asset


def _decode_frame(video, index, destination):
    """Decode expected pixels with FFmpeg, independently of the preview service."""
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-i", str(video),
                    "-map", "0:v:0", "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
                    "-fps_mode", "vfr", "-c:v", "png", "-threads", "2", str(destination)],
                   capture_output=True, check=True, timeout=30)
    with Image.open(destination) as image:
        image.load()
        return image.convert("RGB")


def _wait_image(page, test_id, expected_path):
    image = page.get_by_test_id(test_id)
    expect(image).to_have_attribute("src", expected_path, timeout=30_000)
    expect(image).to_be_visible(timeout=30_000)
    page.wait_for_function("""id => {
        const image = document.querySelector(`[data-testid="${id}"]`);
        return image && image.complete && image.naturalWidth === 1280 && image.naturalHeight === 720;
    }""", arg=test_id, timeout=30_000)
    return image.get_attribute("src")


def _wait_video(page, expected_path):
    video = page.get_by_test_id("video-guide")
    expect(video).to_have_attribute("src", expected_path, timeout=30_000)
    expect(video).to_be_visible(timeout=30_000)
    page.wait_for_function("""() => {
        const video = document.querySelector('[data-testid="video-guide"]');
        return video && video.readyState >= 1 && video.videoWidth === 1280 && video.videoHeight === 720;
    }""", timeout=30_000)
    return video.get_attribute("src")


def _screenshot(page):
    target = os.environ.get("GEN_REALITY_PREPARATION_SCREENSHOT")
    if target is None:
        return
    path = Path(target)
    if not path.is_absolute() or path.resolve().is_relative_to(session_browser.PROJECT_ROOT.resolve()):
        raise AssertionError("The optional browser screenshot must stay outside the repository.")
    page.screenshot(path=str(path), full_page=True)


class LocalPreparationBrowserTests(unittest.TestCase):
    def _response(self, page, method, path, action, status=200):
        with page.expect_response(lambda response: response.request.method == method
                                  and urlsplit(response.url).path == path, timeout=90_000) as pending:
            action()
        response = pending.value
        self.assertEqual(response.status, status, "The visible preparation action returned the wrong status.")
        if path == "/runs":
            return response.json()
        # Chromium can discard the inspector copy of a File-upload response.
        # Read the committed selection through the ordinary authenticated GET.
        run_path = path.rsplit("/", 1)[0]
        address = urlsplit(response.url)
        selected = page.request.get(f"{address.scheme}://{address.netloc}{run_path}", timeout=30_000)
        self.assertEqual(selected.status, 200, "The completed action's current run could not be read.")
        return selected.json()

    def _download(self, context, origin, path, content_type):
        self.assertTrue(path.startswith("/runs/run-"), "The displayed media path is not a local run path.")
        response = context.request.get(origin + path, timeout=30_000)
        self.assertEqual(response.status, 200, "The displayed media could not be read.")
        self.assertEqual(response.headers.get("content-type"), content_type)
        contents = response.body()
        self.assertEqual(response.headers.get("x-content-sha256"), sha256(contents).hexdigest(),
                         "The displayed media hash does not describe its exact bytes.")
        return contents

    def _video_contract(self, contents, destination):
        destination.write_bytes(contents)
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams",
                                "-show_format", "-of", "json", str(destination)],
                               capture_output=True, check=True, timeout=30)
        data = json.loads(probe.stdout)
        self.assertEqual(len(data["streams"]), 1, "Mode B contains an unexpected media stream.")
        stream = data["streams"][0]
        self.assertEqual((stream["codec_type"], stream["codec_name"], stream["width"], stream["height"],
                          stream["pix_fmt"], stream["avg_frame_rate"], stream["nb_read_frames"]),
                         ("video", "h264", 1280, 720, "yuv420p", "24/1", "120"))
        self.assertAlmostEqual(float(stream["duration"]), 5.0, places=5)
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-threads", "2",
                        "-i", str(destination), "-f", "null", "-"],
                       capture_output=True, check=True, timeout=30)

    def _saved_pixels(self, contents, source_frame, placement):
        with Image.open(BytesIO(contents)) as image:
            image.load()
            self.assertEqual((image.mode, image.size), ("RGB", (1280, 720)))
            # These positions are deep inside uniform alpha regions. No browser
            # or production renderer is used to calculate expected colors.
            height = max(1, round(placement["heightFrac"] * 720))
            width = max(1, round(height * 160 / 120))
            left = round(placement["u"] * 1280 - width / 2)
            top = round(placement["v"] * 720 - height / 2)
            transparent = (left + round(width * 0.05), top + height // 2)
            translucent = (left + round(width * 0.25), top + height // 2)
            opaque = (left + round(width * 0.7), top + height // 2)
            self.assertEqual(image.getpixel((10, 10)), source_frame.getpixel((10, 10)),
                             "The saved background is not decoded source frame zero.")
            self.assertEqual(image.getpixel(transparent), source_frame.getpixel(transparent),
                             "The original transparent region changed the source pixels.")
            self.assertEqual(image.getpixel(opaque), (220, 150, 30),
                             "The original opaque asset region changed color.")
            background = source_frame.getpixel(translucent)
            blended = tuple(round((128 * foreground + 127 * behind) / 255)
                            for foreground, behind in zip((0, 255, 0), background))
            self.assertEqual(image.getpixel(translucent), blended,
                             "The saved translucent interior does not match independent alpha math.")

    def test_operator_can_upload_place_build_and_reopen_local_guides(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, asset = _media_files(root)
            original_asset = asset.read_bytes()
            state_dir, output_root = root / "operator", root / "runs"
            state_dir.mkdir(mode=0o700)
            output_root.mkdir(mode=0o700)
            with session_browser._running_server(state_dir, output_root) as running:
                origin = running["origin"]
                token = running["token_file"].read_text().strip()
                writes, external, errors, csp_errors, urls = [], [], [], [], []
                held_previews = []
                hold_preview = False
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(
                        executable_path=os.environ.get("GEN_REALITY_BROWSER_BIN", "/usr/bin/chromium"),
                        headless=True)
                    try:
                        context = browser.new_context(viewport={"width": 1010, "height": 900},
                                                      service_workers="block")
                        context.set_default_timeout(15_000)

                        def intercept(route):
                            request = route.request
                            parsed = urlsplit(request.url)
                            urls.append(request.url)
                            if parsed.scheme != "http" or parsed.netloc != urlsplit(origin).netloc:
                                external.append(request.url)
                                route.abort()
                                return
                            if parsed.path.startswith("/runs") and request.method != "GET":
                                writes.append({"method": request.method, "path": parsed.path,
                                    "body": request.post_data_json if request.method == "PUT" else None})
                            if hold_preview and request.method == "GET" and parsed.path.endswith("/placement"):
                                held_previews.append(route)
                                return
                            route.continue_()

                        context.route("**/*", intercept)
                        page = context.new_page()
                        page.on("pageerror", lambda error: errors.append(str(error)))

                        def console(message):
                            if "Content Security Policy" in message.text or "Refused to" in message.text:
                                csp_errors.append(message.text)

                        page.on("console", console)
                        page.goto(origin, wait_until="domcontentloaded", timeout=15_000)
                        try:
                            page.get_by_test_id("login-token").fill(token)
                        except PlaywrightError:
                            raise AssertionError("The operator login field was unavailable.") from None
                        page.get_by_test_id("login-button").click()
                        expect(page.get_by_test_id("prepare-link")).to_be_visible()
                        page.get_by_test_id("prepare-link").click()
                        expect(page.get_by_role("heading", name="Prepare local guides", exact=True)).to_be_visible()
                        expect(page.get_by_test_id("start-run")).to_be_enabled()
                        self.assertEqual(writes, [], "Opening preparation silently created or changed a run.")

                        selected = self._response(page, "POST", "/runs",
                                                  page.get_by_test_id("start-run").click, status=201)
                        run_id = selected["runId"]
                        run_path = "/runs/" + run_id
                        self.assertRegex(run_id, r"\Arun-[0-9a-f]{32}\Z")
                        expect(page.get_by_test_id("source-file")).to_be_enabled()
                        self.assertFalse(page.get_by_test_id("clean-source").is_checked())
                        self.assertFalse(page.get_by_test_id("non-human").is_checked())
                        self.assertEqual(page.get_by_test_id("source-kind").input_value(), "")
                        page.get_by_test_id("source-file").set_input_files(source)
                        page.get_by_test_id("source-kind").select_option("synthetic_fixture")
                        page.get_by_test_id("clean-source").check()
                        self._response(page, "POST", run_path + "/source",
                                       page.get_by_test_id("upload-source").click)
                        expect(page.get_by_test_id("asset-file")).to_be_enabled(timeout=30_000)
                        page.get_by_test_id("asset-file").set_input_files(asset)
                        page.get_by_test_id("non-human").check()
                        selected = self._response(page, "POST", run_path + "/asset",
                                                  page.get_by_test_id("upload-asset").click)
                        expect(page.get_by_test_id("height-number")).to_be_enabled(timeout=30_000)
                        expect(page.get_by_test_id("create-guides")).to_be_disabled()
                        self.assertIsNone(selected["placement"], "Uploads silently saved the default placement.")

                        normalized = list((output_root / run_id / "inputs").glob("src-*/source_normalized.mp4"))
                        self.assertEqual(len(normalized), 1)
                        frame_zero = _decode_frame(normalized[0], 0, root / "expected-frame-zero.png")
                        later_frame = _decode_frame(normalized[0], 1, root / "expected-later-frame.png")
                        self.assertNotEqual(frame_zero.getpixel((10, 10)), later_frame.getpixel((10, 10)),
                                            "The fixture cannot distinguish frame zero from a later frame.")
                        source_png = self._download(context, origin, selected["source"]["previewUrl"], "image/png")
                        with Image.open(BytesIO(source_png)) as image:
                            self.assertEqual(image.convert("RGB").tobytes(), frame_zero.tobytes(),
                                             "The source preview is not the independently decoded first frame.")
                        asset_png = self._download(context, origin, selected["asset"]["previewUrl"], "image/png")
                        self.assertEqual(asset_png, original_asset, "The asset preview changed original PNG bytes or metadata.")

                        canvas = page.get_by_test_id("placement-canvas")
                        self.assertEqual(canvas.evaluate("node => [node.width, node.height]"), [1280, 720])
                        background = canvas.evaluate("node => Array.from(node.getContext('2d').getImageData(10,10,1,1).data)")
                        self.assertEqual(background, list(frame_zero.getpixel((10, 10))) + [255],
                                         "The placement canvas is not drawing source frame zero.")
                        before_draft = list(writes)
                        height_range = page.get_by_test_id("height-range")
                        height_range.focus()
                        height_range.press("ArrowRight")
                        self.assertAlmostEqual(float(page.get_by_test_id("height-number").input_value()),
                                               float(height_range.input_value()), places=9)
                        page.get_by_test_id("position-u").fill("0.42")
                        page.get_by_test_id("position-v").fill("0.48")
                        page.get_by_test_id("height-number").fill("0.23456")
                        page.get_by_test_id("height-number").press("Tab")
                        self.assertEqual(float(page.get_by_test_id("height-number").input_value()), 0.23456,
                                         "The slider rounded a valid numeric height draft.")
                        canvas.scroll_into_view_if_needed()
                        box = canvas.bounding_box()
                        self.assertIsNotNone(box)
                        self.assertNotEqual(round(box["width"]), 1280, "The drag did not exercise CSS-scaled coordinates.")
                        self.assertAlmostEqual(box["width"] / box["height"], 16 / 9, places=2)
                        grab = (box["x"] + box["width"] * 0.45, box["y"] + box["height"] * 0.46)
                        delta = (box["width"] * 0.11, -box["height"] * 0.07)
                        page.mouse.move(*grab)
                        page.mouse.down()
                        page.mouse.move(grab[0] + delta[0], grab[1] + delta[1], steps=5)
                        page.mouse.up()
                        page.wait_for_timeout(100)
                        self.assertEqual(writes, before_draft, "A draft edit or pointer release sent an automatic mutation.")
                        expect(page.get_by_test_id("create-guides")).to_be_disabled()
                        expected = {"u": 0.42 + delta[0] / box["width"],
                                    "v": 0.48 + delta[1] / box["height"],
                                    "heightFrac": 0.23456, "rotationDeg": 0}

                        hold_preview = True
                        saved = self._response(page, "PUT", run_path + "/placement",
                                              page.get_by_test_id("save-placement").click)
                        put = writes[-1]
                        self.assertEqual((put["method"], put["path"]), ("PUT", run_path + "/placement"))
                        self.assertEqual(set(put["body"]), {"expectedRevision", *expected})
                        self.assertEqual(put["body"]["expectedRevision"], selected["revision"])
                        for key, value in expected.items():
                            self.assertAlmostEqual(put["body"][key], value, places=6,
                                msg="The saved placement lost its normalized drag, grab offset or numeric height: " + key)
                        deadline = time.monotonic() + 30
                        while not held_previews and time.monotonic() < deadline:
                            page.wait_for_timeout(25)
                        self.assertEqual(len(held_previews), 1, "The exact saved preview was not requested.")
                        expect(page.get_by_test_id("create-guides")).to_be_disabled()
                        self.assertFalse(any(row["path"].endswith("/guides") for row in writes),
                                         "Guides were requested before the exact saved preview loaded.")
                        hold_preview = False
                        held_previews.pop().abort()
                        expect(page.get_by_test_id("run-status")).to_have_text(
                            "The local request or media load failed. Use Refresh run before trying again.")
                        expect(page.get_by_test_id("refresh-run")).to_be_enabled()
                        expect(page.get_by_test_id("create-guides")).to_be_disabled()
                        page.get_by_test_id("refresh-run").click()
                        preview_url = _wait_image(page, "saved-preview", saved["placementPreviewUrl"])
                        self.assertEqual(preview_url, saved["placementPreviewUrl"])
                        expect(page.get_by_test_id("create-guides")).to_be_enabled(timeout=30_000)
                        preview_bytes = self._download(context, origin, preview_url, "image/png")
                        self._saved_pixels(preview_bytes, frame_zero, expected)

                        first = self._response(page, "POST", run_path + "/guides",
                                               page.get_by_test_id("create-guides").click)
                        first_a_url = _wait_image(page, "keyframe-guide", first["prepared"]["keyframeAUrl"])
                        first_b_url = _wait_video(page, first["prepared"]["guideBUrl"])
                        self.assertEqual((first_a_url, first_b_url),
                                         (first["prepared"]["keyframeAUrl"], first["prepared"]["guideBUrl"]))
                        first_a = self._download(context, origin, first_a_url, "image/png")
                        first_b = self._download(context, origin, first_b_url, "video/mp4")
                        self._saved_pixels(first_a, frame_zero, expected)
                        self._video_contract(first_b, root / "first-guide.mp4")
                        video = page.get_by_test_id("video-guide")
                        self.assertTrue(video.evaluate("node => node.controls && node.paused && !node.autoplay"),
                                        "Mode B did not remain a paused video with native controls.")

                        expect(page.get_by_test_id("position-u")).to_be_enabled(timeout=30_000)
                        before_second = len(writes)
                        page.get_by_test_id("position-u").fill("0.7")
                        page.get_by_test_id("position-v").fill("0.55")
                        page.get_by_test_id("height-number").fill("0.2")
                        page.get_by_test_id("height-number").press("Tab")
                        expect(page.get_by_test_id("create-guides")).to_be_disabled()
                        self.assertEqual(len(writes), before_second, "Editing a saved placement sent an automatic mutation.")
                        self._response(page, "PUT", run_path + "/placement", page.get_by_test_id("save-placement").click)
                        expect(page.get_by_test_id("create-guides")).to_be_enabled(timeout=30_000)
                        second = self._response(page, "POST", run_path + "/guides",
                                                page.get_by_test_id("create-guides").click)
                        second_a_url = _wait_image(page, "keyframe-guide", second["prepared"]["keyframeAUrl"])
                        second_b_url = _wait_video(page, second["prepared"]["guideBUrl"])
                        self.assertNotEqual(first["prepared"]["attemptId"], second["prepared"]["attemptId"])
                        second_a = self._download(context, origin, second_a_url, "image/png")
                        second_b = self._download(context, origin, second_b_url, "video/mp4")
                        self.assertNotEqual(first_a, second_a, "The second placement reused the first Mode A bytes.")
                        self.assertNotEqual(first_b, second_b, "The second placement reused the first Mode B bytes.")
                        self._video_contract(second_b, root / "second-guide.mp4")
                        first_hashes = (sha256(first_a).hexdigest(), sha256(first_b).hexdigest())
                        history = page.get_by_test_id("guide-history")
                        expect(history).to_be_enabled()
                        self.assertEqual(history.locator("option").evaluate_all("nodes => nodes.filter(n => n.value).length"), 2)
                        before_history = list(writes)
                        history.select_option(first["prepared"]["attemptId"])
                        self.assertEqual(_wait_image(page, "keyframe-guide", first_a_url), first_a_url)
                        self.assertEqual(_wait_video(page, first_b_url), first_b_url)
                        self.assertEqual(writes, before_history, "Choosing an old guide changed server selection or placement.")
                        self.assertEqual(self._download(context, origin, first_a_url, "image/png"), first_a)
                        self.assertEqual(self._download(context, origin, first_b_url, "video/mp4"), first_b)
                        storage = page.evaluate("() => ({...localStorage})")
                        self.assertEqual(list(storage.values()), [run_id], "The page persisted more than the registered run ID.")
                        self.assertEqual(page.evaluate("() => ({...sessionStorage})"), {})
                        before_reload = list(writes)
                        page.reload(wait_until="domcontentloaded")
                        expect(page.get_by_test_id("create-guides")).to_be_enabled(timeout=30_000)
                        self.assertEqual(_wait_image(page, "keyframe-guide", second_a_url), second_a_url)
                        self.assertEqual(_wait_video(page, second_b_url), second_b_url)
                        self.assertEqual(float(page.get_by_test_id("position-u").input_value()), 0.7)
                        self.assertEqual(float(page.get_by_test_id("position-v").input_value()), 0.55)
                        self.assertEqual(float(page.get_by_test_id("height-number").input_value()), 0.2)
                        history.select_option(first["prepared"]["attemptId"])
                        self.assertEqual(_wait_image(page, "keyframe-guide", first_a_url), first_a_url)
                        self.assertEqual(_wait_video(page, first_b_url), first_b_url)
                        after_reload_a = self._download(context, origin, first_a_url, "image/png")
                        after_reload_b = self._download(context, origin, first_b_url, "video/mp4")
                        self.assertEqual((after_reload_a, after_reload_b), (first_a, first_b),
                                         "The first guide pair changed after a second build or reload.")
                        self.assertEqual((sha256(after_reload_a).hexdigest(), sha256(after_reload_b).hexdigest()), first_hashes)
                        self.assertEqual(writes, before_reload, "Reload or restored history sent a mutation request.")
                        self.assertEqual(page.evaluate("() => ({...localStorage})"), storage)
                        self.assertEqual(external, [], "The preparation page attempted an external request.")
                        self.assertEqual(errors, [], "The preparation page raised a JavaScript error.")
                        self.assertEqual(csp_errors, [], "The preparation page violated its content security policy.")
                        self.assertFalse(any(token in url for url in urls), "A request URL contained the operator token.")
                        _screenshot(page)
                    finally:
                        for route in held_previews:
                            route.abort()
                        browser.close()
                with sqlite3.connect((state_dir / "budget.sqlite").as_uri() + "?mode=ro", uri=True) as connection:
                    for table in ("approved_runs", "reservations", "submission_jobs"):
                        self.assertEqual(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0,
                                         "Local preparation created provider or budget authority.")
                databases = sorted(path.name for path in state_dir.glob("*.sqlite"))
                self.assertEqual(databases, ["budget.sqlite", "preparation.sqlite"])
                for name in databases:
                    with sqlite3.connect((state_dir / name).as_uri() + "?mode=ro", uri=True) as connection:
                        self.assertEqual(connection.execute(
                            "SELECT count(*) FROM sqlite_schema WHERE name='runtime_pins'").fetchone()[0], 0)
                self.assertEqual(list(output_root.rglob("runtime")), [], "Local preparation wrote runtime provider receipts.")
            self.assertFalse(token.encode("ascii") in running["logs"], "The local server logged its operator token.")


if __name__ == "__main__":
    unittest.main()

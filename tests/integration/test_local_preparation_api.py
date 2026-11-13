"""Check authenticated preparation routes with cached real synthetic media."""

from contextlib import ExitStack, contextmanager
import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

from app import local_session, prepare, preparation, preparation_routes as routes, server


TOP_FIELDS = {"runId", "revision", "runStart", "source", "asset", "placement",
              "placementPreviewUrl", "prepared", "history"}
PRIVATE = "private-route-exception-and-path"


def _media_fixtures():
    # Import the retained fixture module, not its TestCase into test globals.
    # This works when only tests/integration is on the discovery path.
    path = Path(__file__).resolve().parents[1] / "unit" / "test_preparation_preview.py"
    spec = importlib.util.spec_from_file_location("local_api_media_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LocalPreparationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = _media_fixtures()
        cls.fixture = type("ApiPreviewFixture", (module.PreparationPreviewTests,), {})
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.state = self.root / "private"
        self.state.mkdir(mode=0o700)
        self.outputs = self.root / "outputs"

    @contextmanager
    def client(self):
        config = server.LocalServerConfig(self.state, self.outputs)
        app = server.create_app(config)
        with TestClient(app, base_url=config.origin, raise_server_exceptions=False) as client:
            yield app, client, config

    def login(self, client, config):
        response = client.post("/session", json={"token": config.token_path.read_text().strip()},
                               headers={"Origin": config.origin})
        self.assertEqual(response.status_code, 200, response.text)
        return {"Origin": config.origin, "X-CSRF-Token": client.get("/session").json()["csrfToken"]}

    def error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.headers.get("content-type"), "application/json")
        self.assertEqual(response.json(), {"error": {"code": code}})
        self.assertNotIn("set-cookie", response.headers)
        for private in (PRIVATE, str(self.root), "Traceback"):
            self.assertNotIn(private, response.text)

    def selection(self, response, status=200):
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.headers.get("content-type"), "application/json")
        document = response.json()
        self.assertEqual(set(document), TOP_FIELDS)
        self.assertNotIn(str(self.root), response.text)
        self.assertNotIn("inputs/", response.text)
        self.assertNotIn("source_original", response.text)
        return document

    def media(self, response, contents, mime):
        self.assertEqual(response.status_code, 200, response.text[:100])
        self.assertEqual(response.content, contents)
        self.assertEqual(response.headers["content-type"], mime)
        self.assertEqual(response.headers["x-content-sha256"], sha256(contents).hexdigest())
        self.assertIn("no-store", response.headers["cache-control"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertNotIn("content-disposition", response.headers)

    def create(self, client, headers):
        return self.selection(client.post("/runs", json={}, headers=headers), 201)

    def upload_headers(self, headers, revision, kind, *, mime=None):
        values = dict(headers) | {"X-Expected-Revision": str(revision)}
        if kind == "source":
            return values | {"Content-Type": mime or "video/mp4", "X-Clean-Source": "true",
                             "X-Source-Kind": "synthetic_fixture"}
        return values | {"Content-Type": mime or "image/png", "X-Non-Human": "true"}

    def ready(self, client, headers):
        selected = self.create(client, headers)
        base = "/runs/" + selected["runId"]
        with patch.object(preparation, "normalize_video", return_value=self.fixture.video), \
                patch.object(preparation, "load_asset", return_value=self.fixture.asset):
            selected = self.selection(client.post(base + "/source", content=self.fixture.video.original_bytes,
                headers=self.upload_headers(headers, selected["revision"], "source")))
            selected = self.selection(client.post(base + "/asset", content=self.fixture.asset.original_bytes,
                headers=self.upload_headers(headers, selected["revision"], "asset")))
        return self.place(client, headers, selected, self.fixture.first_placement)

    def place(self, client, headers, selected, placement):
        return self.selection(client.put("/runs/" + selected["runId"] + "/placement", headers=headers, json={
            "expectedRevision": selected["revision"], "u": placement.u, "v": placement.v,
            "heightFrac": placement.height_frac, "rotationDeg": placement.rotation_deg,
        }))

    def build(self, client, headers, selected, placement):
        with patch.object(prepare, "compile_guidance", return_value=self.fixture.guides[placement]) as compiler:
            result = self.selection(client.post("/runs/" + selected["runId"] + "/guides",
                                               headers=headers, json={"expectedRevision": selected["revision"]}))
        compiler.assert_called_once()
        self.assertEqual(compiler.call_args.args[3], self.fixture.guides[placement].prompt)
        return result

    def expected_inputs(self, run_id, revision, placement=None):
        preview = f"/runs/{run_id}/preview/{revision}"
        return {
            "source": {"originalSha256": sha256(self.fixture.video.original_bytes).hexdigest(),
                       "normalizedSha256": sha256(self.fixture.video.video_bytes).hexdigest(),
                       "width": 1280, "height": 720, "frames": 120, "fps": 24, "durationSeconds": 5,
                       "sourceKind": "synthetic_fixture", "previewUrl": preview + "/source"},
            "asset": {"sha256": sha256(self.fixture.asset.original_bytes).hexdigest(),
                      "width": 18, "height": 14, "previewUrl": preview + "/asset"},
            "placement": None if placement is None else {
                "u": placement.u, "v": placement.v, "heightFrac": placement.height_frac,
                "rotationDeg": placement.rotation_deg},
            "placementPreviewUrl": None if placement is None else preview + "/placement",
        }

    def expected_attempt(self, run_id, attempt_id, input_revision):
        contents = (self.outputs / run_id / "attempts" / attempt_id / "manifest.json").read_bytes()
        base = f"/runs/{run_id}/guides/{attempt_id}"
        return {"attemptId": attempt_id, "inputRevision": input_revision,
                "manifestSha256": sha256(contents).hexdigest(),
                "keyframeAUrl": base + "/keyframe-a", "guideBUrl": base + "/guide-b"}

    def test_real_media_flow_has_literal_public_projection_and_preserves_old_guides(self):
        with self.client() as (app, client, config):
            headers = self.login(client, config)
            created = self.create(client, headers)
            run_id = created["runId"]
            start = (self.outputs / run_id / "run-start.json").read_bytes()
            expected = {"runId": run_id, "revision": 0,
                        "runStart": {"path": "run-start.json", "sha256": sha256(start).hexdigest(), "size": len(start)},
                        "source": None, "asset": None, "placement": None,
                        "placementPreviewUrl": None, "prepared": None, "history": []}
            self.assertEqual(created, expected)
            base = "/runs/" + run_id
            with patch.object(preparation, "normalize_video", wraps=preparation.normalize_video) as normalize, \
                    patch.object(preparation, "load_asset", wraps=preparation.load_asset) as validate:
                source = self.selection(client.post(base + "/source", content=self.fixture.video.original_bytes,
                    headers=self.upload_headers(headers, 0, "source")))
                expected |= {"revision": 1, "source": self.expected_inputs(run_id, 1)["source"]}
                self.assertEqual(source, expected)
                asset = self.selection(client.post(base + "/asset", content=self.fixture.asset.original_bytes,
                    headers=self.upload_headers(headers, 1, "asset")))
                normalize.assert_called_once()
                validate.assert_called_once()
            expected |= {"revision": 2} | self.expected_inputs(run_id, 2)
            self.assertEqual(asset, expected)
            placed = self.place(client, headers, asset, self.fixture.first_placement)
            expected |= {"revision": 3} | self.expected_inputs(run_id, 3, self.fixture.first_placement)
            self.assertEqual(placed, expected)
            self.media(client.get(placed["source"]["previewUrl"]), self.fixture.frame_zero, "image/png")
            self.media(client.get(placed["asset"]["previewUrl"]), self.fixture.asset.original_bytes, "image/png")
            self.media(client.get(placed["placementPreviewUrl"]),
                       self.fixture.expected_placement[self.fixture.first_placement][1], "image/png")
            first = self.build(client, headers, placed, self.fixture.first_placement)
            saved = app.state.preparation.get_run(run_id)
            first_attempt = self.expected_attempt(run_id, saved.attempts[0].attempt_id, 3)
            expected |= {"revision": 4, "prepared": first_attempt,
                         "history": [first_attempt | {"status": "PREPARED"}]} | self.expected_inputs(
                             run_id, 4, self.fixture.first_placement)
            self.assertEqual(first, expected)
            changed = self.place(client, headers, first, self.fixture.second_placement)
            self.assertIsNone(changed["prepared"])
            self.assertEqual(changed["history"], expected["history"])
            second = self.build(client, headers, changed, self.fixture.second_placement)
            saved = app.state.preparation.get_run(run_id)
            second_attempt = self.expected_attempt(run_id, saved.attempts[1].attempt_id, 5)
            expected |= {"revision": 6, "prepared": second_attempt,
                         "history": [first_attempt | {"status": "PREPARED"},
                                     second_attempt | {"status": "PREPARED"}]} | self.expected_inputs(
                                         run_id, 6, self.fixture.second_placement)
            self.assertEqual(second, expected)
            self.assertEqual(self.selection(client.get(base)), expected)
            for attempt, placement in ((first_attempt, self.fixture.first_placement),
                                       (second_attempt, self.fixture.second_placement)):
                self.media(client.get(attempt["keyframeAUrl"]), self.fixture.guides[placement].keyframe_png, "image/png")
                self.media(client.get(attempt["guideBUrl"]), self.fixture.guides[placement].guide_mp4, "video/mp4")
            self.error(client.get(placed["source"]["previewUrl"]), 409, "REVISION_CONFLICT")
            with app.state.coordinator.ledger._connection() as connection:
                for table in ("approved_runs", "reservations", "submission_jobs"):
                    self.assertEqual(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)

    def test_every_route_requires_the_current_session_before_service_calls(self):
        with self.client() as (app, client, config):
            selected = app.state.preparation.create_run()
            base = "/runs/" + selected.run_id
            headers = {"Origin": config.origin, "X-CSRF-Token": "a" * 64}
            requests = [
                ("POST", "/runs", {"json": {}}), ("GET", base, {}),
                ("POST", base + "/source", {"content": self.fixture.video.original_bytes,
                 "headers": self.upload_headers(headers, 0, "source")}),
                ("POST", base + "/asset", {"content": self.fixture.asset.original_bytes,
                 "headers": self.upload_headers(headers, 0, "asset")}),
                ("PUT", base + "/placement", {"json": {"expectedRevision": 0, "u": 0.5,
                                                        "v": 0.5, "heightFrac": 0.2, "rotationDeg": 0}}),
                ("POST", base + "/guides", {"json": {"expectedRevision": 0}}),
                ("GET", base + "/preview/0/source", {}),
                ("GET", base + "/guides/att-" + "a" * 32 + "/guide-b", {}),
            ]
            with ExitStack() as stack:
                calls = [stack.enter_context(patch.object(app.state.preparation, name,
                    side_effect=preparation.ServiceError("STATE_UNAVAILABLE"))) for name in
                    ("create_run", "get_run", "set_placement", "media_operation", "preview", "prepared_media")]
                for method, path, values in requests:
                    with self.subTest(path=path):
                        self.error(client.request(method, path, **({"headers": headers} | values)), 401, "SESSION_REQUIRED")
                for call in calls:
                    call.assert_not_called()

    def test_every_preparation_write_requires_csrf_with_a_valid_session(self):
        with self.client() as (app, client, config):
            headers = self.login(client, config)
            run = app.state.preparation.create_run()
            base = "/runs/" + run.run_id
            wrong = headers | {"X-CSRF-Token": "0" * 64}
            requests = [
                ("POST", "/runs", {"json": {}}),
                ("POST", base + "/source", {"content": self.fixture.video.original_bytes,
                 "headers": self.upload_headers(wrong, 0, "source")}),
                ("POST", base + "/asset", {"content": self.fixture.asset.original_bytes,
                 "headers": self.upload_headers(wrong, 0, "asset")}),
                ("PUT", base + "/placement", {"json": {"expectedRevision": 0, "u": 0.5,
                                                        "v": 0.5, "heightFrac": 0.2, "rotationDeg": 0}}),
                ("POST", base + "/guides", {"json": {"expectedRevision": 0}}),
            ]
            with ExitStack() as stack:
                calls = [stack.enter_context(patch.object(app.state.preparation, name,
                    side_effect=preparation.ServiceError("STATE_UNAVAILABLE"))) for name in
                    ("create_run", "set_placement", "media_operation")]
                for method, path, values in requests:
                    with self.subTest(path=path):
                        self.error(client.request(method, path, **({"headers": wrong} | values)), 403, "CSRF_INVALID")
                for call in calls:
                    call.assert_not_called()

    def test_create_requires_exact_utf8_json_and_one_supported_content_type(self):
        with self.client() as (app, client, config):
            headers = self.login(client, config)
            bodies = [b"", b"{", b"\xff", b"[]", b"null", b"true", b"NaN", b"Infinity",
                      b'{"runId":"chosen-by-browser"}', b'{"path":"/private"}',
                      b'{"approved":true}', b'{"token":1,"token":1}', "{}".encode("utf-16")]
            with patch.object(app.state.preparation, "create_run",
                              side_effect=preparation.ServiceError("STATE_UNAVAILABLE")) as create:
                for body in bodies:
                    with self.subTest(body=body[:60]):
                        self.error(client.post("/runs", content=body,
                            headers=headers | {"Content-Type": "application/json"}), 400, "INVALID_REQUEST")
                self.error(client.post("/runs", content=b"{}", headers=headers), 400, "INVALID_REQUEST")
                self.error(client.post("/runs", content=b"{}", headers=headers | {"Content-Type": "text/plain"}),
                           415, "UNSUPPORTED_MEDIA_TYPE")
                self.error(client.post("/runs", content=b"{}", headers=list(headers.items()) +
                    [("Content-Type", "application/json"), ("Content-Type", "application/json")]), 400, "INVALID_REQUEST")
                create.assert_not_called()
            self.selection(client.post("/runs", content=b"{}", headers=headers | {
                "Content-Type": "application/json; charset=utf-8"}), 201)
            self.error(client.post("/runs?approved=true", json={}, headers=headers), 400, "INVALID_REQUEST")

    def test_placement_requires_exact_fields_native_revision_and_valid_fractions(self):
        with self.client() as (app, client, config):
            headers = self.login(client, config)
            selected = self.create(client, headers)
            path = "/runs/" + selected["runId"] + "/placement"
            body = {"expectedRevision": 0, "u": 0.37, "v": 0.63, "heightFrac": 0.21, "rotationDeg": 0}
            malformed = [body | {"prompt": "browser prompt"}, body | {"mask": "path"},
                         {key: value for key, value in body.items() if key != "rotationDeg"}]
            malformed.extend(body | {"expectedRevision": value} for value in (True, False, None, "0", 0.0, -1, 2147483648))
            with patch.object(app.state.preparation, "set_placement",
                              side_effect=preparation.ServiceError("STATE_UNAVAILABLE")) as place:
                for value in malformed:
                    with self.subTest(value=value):
                        self.error(client.put(path, json=value, headers=headers), 400, "INVALID_REQUEST")
                for changes in ({"u": True}, {"u": -0.1}, {"v": 1.1}, {"heightFrac": 0},
                                {"heightFrac": "0.2"}, {"rotationDeg": 1}, {"rotationDeg": True}):
                    with self.subTest(changes=changes):
                        self.error(client.put(path, json=body | changes, headers=headers), 422, "INVALID_INPUT")
                duplicate = b'{"expectedRevision":0,"u":0.37,"u":0.37,"v":0.63,"heightFrac":0.21,"rotationDeg":0}'
                self.error(client.put(path, content=duplicate, headers=headers | {"Content-Type": "application/json"}),
                           400, "INVALID_REQUEST")
                nonfinite = json.dumps(body | {"u": float("nan")}).encode()
                self.error(client.put(path, content=nonfinite, headers=headers | {"Content-Type": "application/json"}),
                           400, "INVALID_REQUEST")
                place.assert_not_called()
            changed = self.selection(client.put(path, json=body, headers=headers))
            self.assertEqual(changed["placement"], {"u": 0.37, "v": 0.63, "heightFrac": 0.21, "rotationDeg": 0})
            self.error(client.put(path, json=body, headers=headers), 409, "REVISION_CONFLICT")

    def test_guides_accept_only_the_expected_revision_and_fixed_default_prompt(self):
        with self.client() as (app, client, config):
            headers = self.login(client, config)
            selected = self.ready(client, headers)
            path = "/runs/" + selected["runId"] + "/guides"
            valid = {"expectedRevision": selected["revision"]}
            with patch.object(app.state.preparation, "media_operation",
                              side_effect=preparation.ServiceError("STATE_UNAVAILABLE")) as admission:
                for body in ({}, valid | {"prompt": "custom"}, valid | {"model": "other"},
                             valid | {"approved": True}, {"expectedRevision": True},
                             {"expectedRevision": float(selected["revision"])}):
                    with self.subTest(body=body):
                        self.error(client.post(path, json=body, headers=headers), 400, "INVALID_REQUEST")
                revision = str(selected["revision"])
                duplicate = ('{"expectedRevision":' + revision + ',"expectedRevision":' + revision + '}').encode()
                self.error(client.post(path, content=duplicate, headers=headers | {"Content-Type": "application/json"}),
                           400, "INVALID_REQUEST")
                admission.assert_not_called()
            prepared = self.build(client, headers, selected, self.fixture.first_placement)
            self.assertEqual(prepared["prepared"]["inputRevision"], selected["revision"])
            self.assertEqual(prepared["revision"], selected["revision"] + 1)

    def test_upload_headers_mime_and_declarations_are_strict_and_quicktime_keeps_mov_suffix(self):
        with self.client() as (app, client, config):
            headers = self.login(client, config)
            selected = self.create(client, headers)
            base = "/runs/" + selected["runId"]
            with patch.object(app.state.preparation, "media_operation",
                              side_effect=preparation.ServiceError("STATE_UNAVAILABLE")) as admission:
                for kind, contents in (("source", self.fixture.video.original_bytes),
                                       ("asset", self.fixture.asset.original_bytes)):
                    good = self.upload_headers(headers, 0, kind)
                    required = ["Content-Type", "X-Expected-Revision"] + (
                        ["X-Clean-Source", "X-Source-Kind"] if kind == "source" else ["X-Non-Human"])
                    for name in required:
                        with self.subTest(kind=kind, missing=name):
                            missing = {key: value for key, value in good.items() if key != name}
                            self.error(client.post(base + "/" + kind, content=contents, headers=missing),
                                       400, "INVALID_REQUEST")
                        with self.subTest(kind=kind, duplicate=name):
                            duplicate = list(good.items()) + [(name, good[name])]
                            self.error(client.post(base + "/" + kind, content=contents, headers=duplicate),
                                       400, "INVALID_REQUEST")
                    for revision in ("", "-1", "+0", "0.0", "true", "2147483648", "0" * 11):
                        with self.subTest(kind=kind, revision=revision):
                            self.error(client.post(base + "/" + kind, content=contents,
                                headers=good | {"X-Expected-Revision": revision}), 400, "INVALID_REQUEST")
                    for mime in ("application/octet-stream", "text/plain", "image/jpeg"):
                        self.error(client.post(base + "/" + kind, content=contents,
                            headers=good | {"Content-Type": mime}), 415, "UNSUPPORTED_MEDIA_TYPE")
                    declaration = "X-Clean-Source" if kind == "source" else "X-Non-Human"
                    for value in ("false", "True", "1", ""):
                        self.error(client.post(base + "/" + kind, content=contents,
                            headers=good | {declaration: value}), 422, "INVALID_INPUT")
                    if kind == "source":
                        for value in ("clean", "demo_reference", "Synthetic_fixture", ""):
                            self.error(client.post(base + "/source", content=contents,
                                headers=good | {"X-Source-Kind": value}), 422, "INVALID_INPUT")
                admission.assert_not_called()
            with patch.object(preparation, "normalize_video", wraps=preparation.normalize_video) as normalize:
                source = self.selection(client.post(base + "/source", content=self.fixture.video.original_bytes,
                    headers=self.upload_headers(headers, 0, "source", mime="video/quicktime")))
            normalize.assert_called_once()
            self.assertEqual(normalize.call_args.args[0].suffix, ".mov")
            saved = app.state.preparation.get_run(selected["runId"])
            self.assertEqual(saved.source.source_suffix, ".mov")
            self.assertTrue(saved.source.original.path.endswith("/source_original.mov"))
            self.assertEqual(source["source"]["originalSha256"], sha256(self.fixture.video.original_bytes).hexdigest())

    def test_read_routes_report_missing_stale_foreign_and_corrupt_inputs_safely(self):
        with self.client() as (app, client, config):
            headers = self.login(client, config)
            empty = self.create(client, headers)
            base = "/runs/" + empty["runId"]
            self.error(client.get(base + "/preview/0/source"), 409, "INPUT_REQUIRED")
            self.error(client.get("/runs/run-" + "f" * 32), 404, "RUN_NOT_FOUND")
            for revision in ("-1", "true", "0.0", "2147483648", "0" * 11):
                self.error(client.get(base + "/preview/" + revision + "/source"), 400, "INVALID_REQUEST")
            self.error(client.get(base + "/preview/0/not-a-role"), 422, "INVALID_INPUT")
            self.error(client.get(base + "/guides/att-" + "f" * 32 + "/guide-b"), 404, "ATTEMPT_NOT_FOUND")
            self.error(client.get(base + "/guides/not-an-attempt/guide-b"), 422, "INVALID_INPUT")
            ready = self.ready(client, headers)
            built = self.build(client, headers, ready, self.fixture.first_placement)
            attempt = built["prepared"]["attemptId"]
            self.error(client.get(base + "/guides/" + attempt + "/guide-b"), 404, "ATTEMPT_NOT_FOUND")
            self.error(client.get("/runs/" + built["runId"] + "/guides/" + attempt + "/source"), 422, "INVALID_INPUT")
            self.error(client.get(ready["source"]["previewUrl"]), 409, "REVISION_CONFLICT")
            saved = app.state.preparation.get_run(built["runId"])
            path = self.outputs / built["runId"] / saved.source.normalized.path
            contents = path.read_bytes()
            path.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
            self.error(client.get(built["source"]["previewUrl"]), 500, "STATE_UNAVAILABLE")

    def test_unfinished_history_projects_only_fixed_nullable_public_fields(self):
        with self.client() as (app, client, config):
            headers = self.login(client, config)
            selected = self.ready(client, headers)
            run_id, revision = selected["runId"], selected["revision"]
            with sqlite3.connect(self.state / "preparation.sqlite") as connection:
                row = connection.execute("SELECT source_id,asset_id,placement FROM preparation_runs WHERE run_id=?",
                                         (run_id,)).fetchone()
                for index, status in enumerate(("BUILDING", "FAILED", "INTERRUPTED"), 1):
                    error = {"BUILDING": None, "FAILED": "WORK_FAILED", "INTERRUPTED": "WORK_INTERRUPTED"}[status]
                    connection.execute("INSERT INTO preparation_attempts VALUES (?,?,?,?,?,?,?,?,NULL,?)",
                        ("att-" + str(index) * 32, run_id, revision, *row, self.fixture.guides[
                            self.fixture.first_placement].prompt, status, error))
            response = client.get("/runs/" + run_id)
            current = self.selection(response)
            expected = [{"attemptId": "att-" + str(index) * 32, "inputRevision": revision,
                         "status": status, "manifestSha256": None, "keyframeAUrl": None, "guideBUrl": None}
                        for index, status in enumerate(("BUILDING", "FAILED", "INTERRUPTED"), 1)]
            self.assertEqual(current, selected | {"history": expected})
            for item in expected:
                self.error(client.get(f"/runs/{run_id}/guides/{item['attemptId']}/guide-b"), 409, "INPUT_REQUIRED")
            self.assertNotIn("error_code", response.text)
            self.assertNotIn("WORK_FAILED", response.text)
            self.assertNotIn("WORK_INTERRUPTED", response.text)

    def test_service_error_codes_and_unexpected_errors_have_fixed_status_and_json(self):
        with self.client() as (app, client, config):
            headers = self.login(client, config)
            selected = self.create(client, headers)
            path = "/runs/" + selected["runId"]
            cases = [("INVALID_INPUT", 422), ("RUN_NOT_FOUND", 404), ("ATTEMPT_NOT_FOUND", 404),
                     ("MEDIA_BUSY", 409), ("REVISION_CONFLICT", 409), ("INPUT_REQUIRED", 409),
                     ("CAPACITY_EXCEEDED", 409), ("SERVICE_CLOSED", 503),
                     ("STATE_UNAVAILABLE", 500), ("PREPARATION_FAILED", 500)]
            for code, status in cases:
                with self.subTest(code=code), patch.object(app.state.preparation, "get_run",
                        side_effect=preparation.ServiceError(code)):
                    self.error(client.get(path), status, code)
            for error in (preparation.ServiceError("OPERATION_USED"), preparation.ServiceError(PRIVATE),
                          RuntimeError(PRIVATE)):
                with self.subTest(error=type(error).__name__), patch.object(app.state.preparation, "get_run", side_effect=error):
                    self.error(client.get(path), 500, "INTERNAL_ERROR")
            original = app.state.preparation
            try:
                app.state.preparation = None
                self.error(client.get(path), 503, "SERVICE_CLOSED")
            finally:
                app.state.preparation = original

    def test_restart_requires_new_session_and_keeps_registered_media_urls(self):
        with self.client() as (app, client, config):
            headers = self.login(client, config)
            built = self.build(client, headers, self.ready(client, headers), self.fixture.first_placement)
            old_cookie = "gen_reality_session=" + client.cookies.get("gen_reality_session")
            path = "/runs/" + built["runId"]
        with self.client() as (app, client, config):
            self.error(client.get(path, headers={"Cookie": old_cookie}), 401, "SESSION_REQUIRED")
            self.login(client, config)
            self.assertEqual(self.selection(client.get(path)), built)
            self.media(client.get(built["prepared"]["keyframeAUrl"]),
                       self.fixture.guides[self.fixture.first_placement].keyframe_png, "image/png")
            self.media(client.get(built["prepared"]["guideBUrl"]),
                       self.fixture.guides[self.fixture.first_placement].guide_mp4, "video/mp4")

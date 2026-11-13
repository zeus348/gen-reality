"""Expose fixed local preparation operations through the current operator session."""

import asyncio
from contextlib import aclosing
import json
import re

from fastapi import APIRouter, Request
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response

from app import local_session
from app.preparation_state import MAX_REVISION, ServiceError
from core.json_data import unique_fields
from features.placement.math import Placement


UPLOAD_RECEIVE_SECONDS = 300
router = APIRouter()


def service_error_response(error):
    statuses = {
        "INVALID_INPUT": 422, "RUN_NOT_FOUND": 404, "ATTEMPT_NOT_FOUND": 404,
        "MEDIA_BUSY": 409, "REVISION_CONFLICT": 409, "INPUT_REQUIRED": 409,
        "CAPACITY_EXCEEDED": 409, "SERVICE_CLOSED": 503,
        "STATE_UNAVAILABLE": 500, "PREPARATION_FAILED": 500,
    }
    status = statuses.get(error.code)
    if status is None:
        return local_session.error_response(500, "INTERNAL_ERROR")
    return local_session.error_response(status, error.code)


def _service(request):
    service = request.app.state.preparation
    if service is None:
        raise ServiceError("SERVICE_CLOSED")
    return service


def _header(request, name):
    values = request.headers.getlist(name)
    if len(values) != 1:
        raise HTTPException(400)
    return values[0]


def _content_type(request, allowed):
    value = _header(request, "content-type").lower().strip()
    if value not in allowed:
        raise HTTPException(415)
    return value


def _revision(value):
    if type(value) is not int or not 0 <= value <= MAX_REVISION:
        raise HTTPException(400)
    return value


def _revision_text(value):
    if re.fullmatch(r"[0-9]{1,10}", value) is None:
        raise HTTPException(400)
    return _revision(int(value))


def _reject_constant(value):
    raise ValueError("JSON constants are not allowed.")


async def _json(request, fields):
    _content_type(request, ("application/json", "application/json;charset=utf-8",
                            "application/json; charset=utf-8"))
    contents = await request.body()
    try:
        value = json.loads(contents.decode("utf-8"), object_pairs_hook=unique_fields,
                           parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise HTTPException(400) from None
    if type(value) is not dict or set(value) != fields:
        raise HTTPException(400)
    return value


def _selection(selected):
    """Return only fixed browser fields, never private cache or state paths."""
    base = f"/runs/{selected.run_id}"
    preview = f"{base}/preview/{selected.revision}"
    source = asset = placement = prepared = None
    if selected.source is not None:
        source = {
            "originalSha256": selected.source.original.sha256,
            "normalizedSha256": selected.source.normalized.sha256,
            "width": 1280, "height": 720, "frames": 120, "fps": 24,
            "durationSeconds": 5, "sourceKind": selected.source.source_kind,
            "previewUrl": preview + "/source",
        }
    if selected.asset is not None:
        asset = {"sha256": selected.asset.original.sha256,
                 "width": selected.asset.width, "height": selected.asset.height,
                 "previewUrl": preview + "/asset"}
    if selected.placement is not None:
        placement = {"u": selected.placement.u, "v": selected.placement.v,
                     "heightFrac": selected.placement.height_frac,
                     "rotationDeg": selected.placement.rotation_deg}
    history = []
    for attempt in selected.attempts:
        manifest_hash = keyframe_url = guide_url = None
        if attempt.status == "PREPARED" and attempt.manifest is not None:
            manifest_hash = attempt.manifest.sha256
            keyframe_url = f"{base}/guides/{attempt.attempt_id}/keyframe-a"
            guide_url = f"{base}/guides/{attempt.attempt_id}/guide-b"
        item = {"attemptId": attempt.attempt_id, "inputRevision": attempt.revision,
                "status": attempt.status, "manifestSha256": manifest_hash,
                "keyframeAUrl": keyframe_url, "guideBUrl": guide_url}
        history.append(item)
        if selected.prepared is not None and attempt.manifest == selected.prepared:
            prepared = {"attemptId": attempt.attempt_id, "inputRevision": attempt.revision,
                        "manifestSha256": manifest_hash, "keyframeAUrl": keyframe_url,
                        "guideBUrl": guide_url}
    start = selected.run_start
    return {
        "runId": selected.run_id, "revision": selected.revision,
        "runStart": {"path": start.path, "sha256": start.sha256, "size": start.size},
        "source": source, "asset": asset, "placement": placement,
        "placementPreviewUrl": preview + "/placement" if source and asset and placement else None,
        "prepared": prepared, "history": history,
    }


async def _timed_stream(request, timeout):
    # Exhausting this iterator ends the receive deadline before the worker starts.
    async with timeout:
        async for chunk in request.stream():
            yield chunk


async def _upload(request, run_id, kind):
    error = local_session.require_session(request, write=True)
    if error is not None:
        return error
    revision = _revision_text(_header(request, "x-expected-revision"))
    if kind == "source":
        content_type = _content_type(request, ("video/mp4", "video/quicktime"))
        if _header(request, "x-clean-source") != "true":
            raise ServiceError("INVALID_INPUT")
        source_kind = _header(request, "x-source-kind")
        if source_kind not in ("clean_camera", "synthetic_fixture"):
            raise ServiceError("INVALID_INPUT")
        suffix = ".mp4" if content_type == "video/mp4" else ".mov"
    else:
        _content_type(request, ("image/png",))
        if _header(request, "x-non-human") != "true":
            raise ServiceError("INVALID_INPUT")
    async with _service(request).media_operation(run_id, expected_revision=revision) as operation:
        timeout = asyncio.timeout(UPLOAD_RECEIVE_SECONDS)
        try:
            async with aclosing(_timed_stream(request, timeout)) as chunks:
                if kind == "source":
                    selected = await operation.import_source(chunks, source_suffix=suffix,
                        clean_source=True, source_kind=source_kind)
                else:
                    selected = await operation.import_asset(chunks, non_human=True)
        except HTTPException as error:
            if error.status_code == 413:
                return local_session.error_response(413, "BODY_TOO_LARGE")
            raise
        except TimeoutError:
            if timeout.expired():
                return local_session.error_response(408, "UPLOAD_RECEIVE_TIMEOUT")
            raise
    return JSONResponse(_selection(selected))


@router.post("/runs")
async def create_run(request: Request):
    error = local_session.require_session(request, write=True)
    if error is not None:
        return error
    await _json(request, set())
    return JSONResponse(_selection(_service(request).create_run()), status_code=201)


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: str):
    error = local_session.require_session(request)
    if error is not None:
        return error
    return JSONResponse(_selection(_service(request).get_run(run_id)))


@router.post("/runs/{run_id}/source")
async def upload_source(request: Request, run_id: str):
    return await _upload(request, run_id, "source")


@router.post("/runs/{run_id}/asset")
async def upload_asset(request: Request, run_id: str):
    return await _upload(request, run_id, "asset")


@router.put("/runs/{run_id}/placement")
async def set_placement(request: Request, run_id: str):
    error = local_session.require_session(request, write=True)
    if error is not None:
        return error
    value = await _json(request, {"expectedRevision", "u", "v", "heightFrac", "rotationDeg"})
    revision = _revision(value["expectedRevision"])
    try:
        placement = Placement(value["u"], value["v"], value["heightFrac"], value["rotationDeg"])
    except ValueError:
        raise ServiceError("INVALID_INPUT") from None
    selected = _service(request).set_placement(run_id, expected_revision=revision, placement=placement)
    return JSONResponse(_selection(selected))


@router.post("/runs/{run_id}/guides")
async def build_guides(request: Request, run_id: str):
    error = local_session.require_session(request, write=True)
    if error is not None:
        return error
    value = await _json(request, {"expectedRevision"})
    revision = _revision(value["expectedRevision"])
    async with _service(request).media_operation(run_id, expected_revision=revision) as operation:
        selected = await operation.build_guides()
    return JSONResponse(_selection(selected))


def _media_response(media):
    return Response(media.contents, media_type=media.media_type,
                    headers={"X-Content-SHA256": media.sha256})


@router.get("/runs/{run_id}/preview/{revision}/{role}")
async def preview(request: Request, run_id: str, revision: str, role: str):
    error = local_session.require_session(request)
    if error is not None:
        return error
    media = await _service(request).preview(run_id, expected_revision=_revision_text(revision), role=role)
    return _media_response(media)


@router.get("/runs/{run_id}/guides/{attempt_id}/{role}")
async def prepared_media(request: Request, run_id: str, attempt_id: str, role: str):
    error = local_session.require_session(request)
    if error is not None:
        return error
    media = await _service(request).prepared_media(run_id, attempt_id, role=role)
    return _media_response(media)

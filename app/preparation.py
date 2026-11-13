"""Own one local editor and one media worker without provider or budget actions."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore, Event, RLock
import time
from uuid import uuid4

from app import preparation_media, preparation_state
from app.preparation_media import AssetSelection, SourceSelection
from app.preparation_state import ServiceError
from app.prepare import prepare_validated_attempt
from core import run_start
from core.storage import StoredArtifact
from features.asset.ingest import AssetError, MAX_FILE_BYTES, load_asset
from features.guidance.video import DEFAULT_PROMPT
from features.inference.prepared import read_prepared_attempt
from features.media.ingest import MAX_INPUT_BYTES, MediaError, normalize_video
from features.placement.math import Placement


@dataclass(frozen=True)
class GuideAttempt:
    attempt_id: str
    revision: int
    status: str
    manifest: StoredArtifact | None
    error_code: str | None


@dataclass(frozen=True)
class RunSelection:
    run_id: str
    run_start: StoredArtifact
    revision: int
    source: SourceSelection | None
    asset: AssetSelection | None
    placement: Placement | None
    prepared: StoredArtifact | None
    attempts: tuple[GuideAttempt, ...]


def _placement(value):
    if type(value) is not dict or set(value) != {"u", "v", "height_frac", "rotation_deg"}:
        raise ServiceError("STATE_UNAVAILABLE")
    return Placement(**value)


class PreparationService:
    """Use an existing coordinator; the private editor does not approve runs."""

    def __init__(self, state_dir, output_root, coordinator, *, clock=time.time):
        self.coordinator = coordinator
        self.clock = clock
        self._mutex = RLock()
        self._slot = BoundedSemaphore(1)
        self._drained = Event()
        self._drained.set()
        self._closed = False
        self._cleanup_failed = False
        self._executor = None
        claimed = False
        try:
            with coordinator._mutex:
                coordinator._require_open()
                if getattr(coordinator, "_preparation_service", None) is not None:
                    raise ServiceError("MEDIA_BUSY")
                coordinator._preparation_service = self
                claimed = True
            self._state = preparation_state.State(state_dir, output_root, coordinator.ledger.path)
            self.state_dir = self._state.state_dir
            self.output_root = self._state.output_root
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-preparation")
        except Exception as error:
            if claimed:
                with coordinator._mutex:
                    coordinator._preparation_service = None
            self._closed = True
            if isinstance(error, ServiceError):
                raise
            raise ServiceError("STATE_UNAVAILABLE") from None

    def _require_open(self):
        if self._closed:
            raise ServiceError("SERVICE_CLOSED")
        try:
            self.coordinator._require_open()
            if Path(os.path.abspath(self.coordinator.ledger.path)) != self._state.ledger_path:
                raise ServiceError("STATE_UNAVAILABLE")
        except Exception:
            raise ServiceError("SERVICE_CLOSED") from None

    def close(self):
        """Stop admission and drain workers; a receiving request owns its cleanup.

        An unused or receiving handle cannot submit after close. Its context
        releases staging when the request cancels or next yields a body chunk.
        The server must close this service before its coordinator.
        """
        with self._mutex:
            self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._drained.wait()
        with self.coordinator._mutex:
            if getattr(self.coordinator, "_preparation_service", None) is self:
                self.coordinator._preparation_service = None
        if self._cleanup_failed:
            raise ServiceError("STATE_UNAVAILABLE")

    def _snapshot(self, run_id):
        self._require_open()
        preparation_state.check_run_id(run_id)
        snapshot = self._state.get(run_id)
        saved_start = preparation_state.reference(preparation_state.decode(snapshot["start_record"]))
        if run_start.read_run_start(self.output_root, run_id=run_id).record != saved_start:
            raise ServiceError("STATE_UNAVAILABLE")
        return snapshot

    def _selection(self, snapshot):
        run_id = snapshot["run_id"]
        root = self.output_root / run_id
        source = asset = None
        if "source" in snapshot["inputs"]:
            row = snapshot["inputs"]["source"]
            source, _ = preparation_media.load_source(root, run_id, row["input_id"],
                preparation_state.reference(preparation_state.decode(row["record"])))
        if "asset" in snapshot["inputs"]:
            row = snapshot["inputs"]["asset"]
            asset, _ = preparation_media.load_asset(root, run_id, row["input_id"],
                preparation_state.reference(preparation_state.decode(row["record"])))
        placement = _placement(preparation_state.decode(snapshot["placement"])) if snapshot["placement"] else None
        prepared = preparation_state.reference(preparation_state.decode(snapshot["prepared"])) if snapshot["prepared"] else None
        attempts = []
        for row in snapshot["attempts"]:
            manifest = preparation_state.reference(preparation_state.decode(row["manifest"])) if row["manifest"] else None
            if (type(row["attempt_id"]) is not str
                    or re.fullmatch(r"att-[0-9a-f]{32}", row["attempt_id"]) is None
                    or type(row["revision"]) is not int or not 0 <= row["revision"] <= snapshot["revision"]
                    or (manifest is not None and (
                        manifest.path != f"attempts/{row['attempt_id']}/manifest.json"
                        or manifest.size > preparation_state.MAX_RECORD_BYTES))):
                raise ServiceError("STATE_UNAVAILABLE")
            attempts.append(GuideAttempt(row["attempt_id"], row["revision"], row["status"], manifest, row["error_code"]))
        if prepared is not None:
            matches = [attempt for attempt in attempts if attempt.status == "PREPARED"
                       and attempt.manifest == prepared and attempt.revision == snapshot["revision"] - 1]
            if len(matches) != 1:
                raise ServiceError("STATE_UNAVAILABLE")
        return RunSelection(run_id, preparation_state.reference(preparation_state.decode(snapshot["start_record"])),
                            snapshot["revision"], source, asset, placement, prepared, tuple(attempts))

    def create_run(self) -> RunSelection:
        try:
            with self._mutex:
                self._require_open()
                self._state.require_capacity()
                run_id = "run-" + uuid4().hex
                if (self.output_root / run_id).exists():
                    raise ServiceError("STATE_UNAVAILABLE")
                started = run_start.start_run(self.output_root, run_id=run_id, clock=self.clock)
                self._state.create(run_id, started.record)
                return self.get_run(run_id)
        except ServiceError:
            raise
        except Exception:
            raise ServiceError("STATE_UNAVAILABLE") from None

    def get_run(self, run_id) -> RunSelection:
        try:
            return self._selection(self._snapshot(run_id))
        except ServiceError:
            raise
        except Exception:
            raise ServiceError("STATE_UNAVAILABLE") from None

    def set_placement(self, run_id, *, expected_revision, placement: Placement) -> RunSelection:
        try:
            with self._mutex:
                self._snapshot(run_id)
                preparation_state.check_revision(expected_revision)
                if type(placement) is not Placement:
                    raise ServiceError("INVALID_INPUT")
                # Reuse the existing constructor before publishing values.
                try:
                    checked = Placement(**asdict(placement))
                except ValueError:
                    raise ServiceError("INVALID_INPUT") from None
                self._state.set_placement(run_id, expected_revision, asdict(checked))
                return self.get_run(run_id)
        except ServiceError:
            raise
        except Exception:
            raise ServiceError("STATE_UNAVAILABLE") from None

    @asynccontextmanager
    async def media_operation(self, run_id, *, expected_revision):
        with self._mutex:
            self._require_open()
            if not self._slot.acquire(blocking=False):
                raise ServiceError("MEDIA_BUSY")
        operation = None
        try:
            preparation_state.check_revision(expected_revision)
            snapshot = self._snapshot(run_id)
            if snapshot["revision"] != expected_revision:
                raise ServiceError("REVISION_CONFLICT")
            self._selection(snapshot)  # Check the selected producer records before receiving bytes.
            operation = _MediaOperation(self, snapshot)
            yield operation
        except ServiceError:
            raise
        except Exception:
            raise ServiceError("STATE_UNAVAILABLE") from None
        finally:
            if operation is None:
                self._slot.release()
            else:
                operation._exit()

    def _import_source(self, snapshot, path, source_suffix, clean_source, source_kind):
        try:
            video = normalize_video(path)
        except MediaError:
            raise ServiceError("INVALID_INPUT") from None
        try:
            run_id, input_id = snapshot["run_id"], "src-" + uuid4().hex
            record = preparation_media.save_source(self.output_root / run_id, run_id, input_id,
                                                   video, source_suffix, source_kind)
            preparation_media.load_source(self.output_root / run_id, run_id, input_id, record)
            self._state.add_input(run_id, snapshot["revision"], input_id, "source", record)
            return self._selection(self._state.get(run_id))
        except ServiceError:
            raise
        except Exception:
            raise ServiceError("STATE_UNAVAILABLE") from None

    def _import_asset(self, snapshot, path, non_human):
        try:
            asset = load_asset(path, non_human=non_human)
        except AssetError:
            raise ServiceError("INVALID_INPUT") from None
        try:
            run_id, input_id = snapshot["run_id"], "asset-" + uuid4().hex
            record = preparation_media.save_asset(self.output_root / run_id, run_id, input_id, asset)
            preparation_media.load_asset(self.output_root / run_id, run_id, input_id, record)
            self._state.add_input(run_id, snapshot["revision"], input_id, "asset", record)
            return self._selection(self._state.get(run_id))
        except ServiceError:
            raise
        except Exception:
            raise ServiceError("STATE_UNAVAILABLE") from None

    def _build_guides(self, snapshot, prompt):
        attempt_id = None
        try:
            selected = self._selection(snapshot)
            root = self.output_root / selected.run_id
            _, video = preparation_media.load_source(root, selected.run_id, selected.source.input_id,
                                                     selected.source.record, with_bytes=True)
            _, asset = preparation_media.load_asset(root, selected.run_id, selected.asset.input_id,
                                                    selected.asset.record, with_bytes=True)
            attempt_id = "att-" + uuid4().hex
            self._state.begin_attempt(snapshot, attempt_id, prompt)
            result = prepare_validated_attempt(video, asset, self.output_root, selected.run_id,
                attempt_id, selected.placement, source_suffix=selected.source.source_suffix,
                non_human=True, clean_source=True, source_kind=selected.source.source_kind, prompt=prompt)
            manifest = preparation_state.reference(result["manifest"])
            prepared = read_prepared_attempt(root, manifest, run_id=selected.run_id, attempt_id=attempt_id)
            pairs = ((prepared.source_original, selected.source.original),
                     (prepared.normalized_source, selected.source.normalized),
                     (prepared.asset_original, selected.asset.original))
            if (any((actual.sha256, actual.size) != (expected.sha256, expected.size) for actual, expected in pairs)
                    or prepared.source_kind != selected.source.source_kind
                    or prepared.placement != selected.placement or prepared.prompt != prompt):
                raise ServiceError("STATE_UNAVAILABLE")
            self._state.finish_attempt(attempt_id, manifest)
            return self._selection(self._state.get(selected.run_id))
        except Exception as error:
            if attempt_id is not None:
                try:
                    self._state.fail_attempt(attempt_id)
                except Exception:
                    pass  # Restart reports any still-BUILDING row as INTERRUPTED.
            if isinstance(error, ServiceError):
                raise
            raise ServiceError("PREPARATION_FAILED") from None


class _MediaOperation:
    """One admitted operation. Only its actual worker can release a submitted slot."""

    def __init__(self, service, snapshot):
        self._service = service
        self._snapshot = snapshot
        self._used = False
        self._exited = False
        self._submitted = False
        self._temporary = None

    def _take(self):
        with self._service._mutex:
            self._service._require_open()
            if self._used or self._exited:
                raise ServiceError("OPERATION_USED")
            self._used = True

    async def _stage(self, chunks, suffix, limit):
        self._service._state.require_capacity(self._snapshot["run_id"])
        self._temporary = TemporaryDirectory(prefix=".preparation-", dir=self._service.state_dir)
        path = Path(self._temporary.name) / ("upload" + suffix)
        size = 0
        with path.open("xb") as file:
            os.chmod(path, 0o600)
            async for chunk in chunks:
                self._service._require_open()
                if type(chunk) is not bytes:
                    raise ServiceError("INVALID_INPUT")
                size += len(chunk)
                if size > limit:
                    raise ServiceError("INVALID_INPUT")
                file.write(chunk)
        if size == 0:
            raise ServiceError("INVALID_INPUT")
        return path

    def _cleanup(self):
        try:
            if self._temporary is not None:
                self._temporary.cleanup()
        except Exception:
            with self._service._mutex:
                self._service._cleanup_failed = True
                self._service._closed = True
        finally:
            self._service._slot.release()

    def _complete(self, future):
        try:
            self._cleanup()
        finally:
            self._service._drained.set()

    def _exit(self):
        self._exited = True
        if not self._submitted:
            self._cleanup()

    async def _submit(self, worker, *arguments):
        with self._service._mutex:
            self._service._require_open()
            try:
                future = self._service._executor.submit(worker, self._snapshot, *arguments)
            except Exception:
                raise ServiceError("STATE_UNAVAILABLE") from None
            self._service._drained.clear()
            self._submitted = True
            future.add_done_callback(self._complete)
        wrapped = asyncio.wrap_future(future)
        # Retrieve late exceptions even if the HTTP waiter has been cancelled.
        wrapped.add_done_callback(lambda result: result.exception() if not result.cancelled() else None)
        return await asyncio.shield(wrapped)

    async def import_source(self, chunks, *, source_suffix, clean_source, source_kind="clean_camera"):
        self._take()
        if (type(source_suffix) is not str or source_suffix not in (".mp4", ".mov")
                or clean_source is not True or source_kind not in ("clean_camera", "synthetic_fixture")):
            raise ServiceError("INVALID_INPUT")
        path = await self._stage(chunks, source_suffix, MAX_INPUT_BYTES)
        return await self._submit(self._service._import_source, path, source_suffix, clean_source, source_kind)

    async def import_asset(self, chunks, *, non_human):
        self._take()
        if non_human is not True:
            raise ServiceError("INVALID_INPUT")
        path = await self._stage(chunks, ".png", MAX_FILE_BYTES)
        return await self._submit(self._service._import_asset, path, non_human)

    async def build_guides(self, *, prompt=DEFAULT_PROMPT):
        self._take()
        if type(prompt) is not str or not prompt.strip() or len(prompt) > 1000:
            raise ServiceError("INVALID_INPUT")
        if any(self._snapshot[name] is None for name in ("source_id", "asset_id", "placement")):
            raise ServiceError("INPUT_REQUIRED")
        self._service._state.require_capacity(self._snapshot["run_id"], attempts=True)
        return await self._submit(self._service._build_guides, prompt)

"""Prepare one offline attempt without provider credentials or paid work."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from uuid import uuid4

from core import run_start
from core.config import load_config
from core.storage import publish_bytes
from features.asset.ingest import load_asset
from features.guidance.video import DEFAULT_PROMPT, compile_guidance
from features.media.ingest import normalize_video
from features.placement.math import Placement


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def prepare_attempt(
    source_path: Path,
    asset_path: Path,
    output_root: Path,
    run_id: str,
    attempt_id: str,
    placement: Placement,
    *,
    non_human: bool,
    clean_source: bool,
    source_kind: str = "clean_camera",
    prompt: str = DEFAULT_PROMPT,
) -> dict:
    """Publish a complete offline snapshot; publish its manifest last.

    Each attempt owns copies of its inputs. A failed publication can leave an
    incomplete attempt, which must use a new ID rather than replacing any file.
    Human declarations record input provenance; they do not establish quality.
    This offline primitive does not start an experiment clock. Application
    callers use prepare_run_attempt before any media work.
    """
    for name in (run_id, attempt_id):
        if type(name) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None:
            raise ValueError("Run and attempt IDs must contain 1 to 64 letters, digits, underscores, or hyphens.")
    if clean_source is not True:
        raise ValueError("Confirm that the source is a clean fixed-camera clip.")
    if source_kind not in ("clean_camera", "synthetic_fixture"):
        raise ValueError("Source kind must be clean_camera or synthetic_fixture.")
    config = load_config()
    asset = load_asset(asset_path, non_human=non_human)
    video = normalize_video(source_path)
    guidance = compile_guidance(video, asset, placement, prompt)
    root = Path(output_root) / run_id
    prefix = f"attempts/{attempt_id}/"
    placement_data = {"u": placement.u, "v": placement.v,
                      "heightFrac": placement.height_frac, "rotationDeg": placement.rotation_deg}
    records = {}

    def publish(name, contents):
        record = publish_bytes(root, prefix + name, contents)
        records[name] = asdict(record)
        return record

    # This fixed path claims the attempt before any source-dependent path exists.
    publish("frozen/placement.json", _json_bytes(placement_data))
    original_name = "frozen/source_original" + Path(source_path).suffix.lower()
    publish(original_name, video.original_bytes)
    publish("frozen/source_normalized.mp4", video.video_bytes)
    publish("frozen/asset_original.png", asset.original_bytes)
    publish("frozen/config.json", _json_bytes(config))
    publish("frozen/prompt.txt", guidance.prompt.encode("utf-8"))
    publish("frozen/keyframe_A.png", guidance.keyframe_png)
    publish("frozen/guide_B.mp4", guidance.guide_mp4)
    publish("frozen/inputs.json", _json_bytes({
        "sourceKind": source_kind, "cleanSourceDeclared": True, "nonHumanDeclared": True,
        "source": records[original_name], "normalizedSource": records["frozen/source_normalized.mp4"],
        "asset": records["frozen/asset_original.png"],
    }))
    manifest = {
        "schemaVersion": "1.4", "manifestType": "offline_preparation",
        "runId": run_id, "attemptId": attempt_id, "state": "PREPARED",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceKind": source_kind, "artifacts": records,
        "provider": {"vendor": "Runway", "model": "aleph2", "taskId": None, "submitted": False},
        "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN",
        "releaseClassification": "LOCAL_PIPELINE_ONLY",
        "reason": "Offline inputs and guides prepared; no generation or human review has run.",
    }
    # Do not include the manifest in its own content hash.
    record = publish_bytes(root, prefix + "manifest.json", _json_bytes(manifest))
    return {"runId": run_id, "attemptId": attempt_id, "state": "PREPARED",
            "manifest": asdict(record), "outputRoot": str(root)}


def prepare_run_attempt(source_path: Path, asset_path: Path, output_root: Path,
                        run_id: str, attempt_id: str, placement: Placement, *,
                        non_human: bool, clean_source: bool, source_kind: str = "clean_camera",
                        prompt: str = DEFAULT_PROMPT, clock=time.time) -> dict:
    """Start the experiment clock before preparing its immutable offline inputs."""
    started = run_start.start_run(output_root, run_id=run_id, clock=clock)
    result = prepare_attempt(source_path, asset_path, output_root, run_id, attempt_id, placement,
                             non_human=non_human, clean_source=clean_source,
                             source_kind=source_kind, prompt=prompt)
    return result | {"runStart": asdict(started.record)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", default="run-" + uuid4().hex)
    parser.add_argument("--attempt-id", default="att-" + uuid4().hex)
    parser.add_argument("--u", type=float, required=True)
    parser.add_argument("--v", type=float, required=True)
    parser.add_argument("--height-frac", type=float, required=True)
    parser.add_argument("--non-human", action="store_true", required=True)
    parser.add_argument("--confirm-clean-source", action="store_true", required=True)
    parser.add_argument("--source-kind", choices=["clean_camera", "synthetic_fixture"], default="clean_camera")
    arguments = parser.parse_args(argv)
    try:
        placement = Placement(arguments.u, arguments.v, arguments.height_frac)
        result = prepare_run_attempt(arguments.source, arguments.asset, arguments.output_root,
                                 arguments.run_id, arguments.attempt_id, placement,
                                 non_human=arguments.non_human, clean_source=arguments.confirm_clean_source,
                                 source_kind=arguments.source_kind)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    except OSError:
        print("Preparation failed because storage or media files are unavailable.", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

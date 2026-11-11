"""Publish a local review page from a trusted review-manifest reference."""

import base64
from hashlib import sha256
from html import escape
import math
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from core.budget import BudgetLedger
from core.storage import StoredArtifact, read_verified
from features.inference.preflight import _match_fields, _read_record
from features.output.complete import _publish_or_verify


class ReviewPageError(ValueError):
    """The saved review cannot support a local page."""


STYLE = """
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #182a33; background: #f2f5f6; }
body { max-width: 1500px; margin: auto; padding: 24px; line-height: 1.5; }
h1, h2, h3 { line-height: 1.2; }
h1 { margin-bottom: 8px; }
a { color: #064f83; overflow-wrap: anywhere; }
.meta { overflow-wrap: anywhere; }
.notice, .sample { background: white; padding: 20px; border: 1px solid #b8c8cf; border-radius: 8px; margin: 20px 0; }
.views { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; }
figure { margin: 0; min-width: 0; }
video, img { display: block; width: 100%; height: auto; max-height: 400px; object-fit: contain; background: #15232c; }
figcaption { margin-top: 8px; overflow-wrap: anywhere; }
.status { font-weight: 700; }
@media (max-width: 850px) { .views { grid-template-columns: 1fr; } body { padding: 12px; } }
"""


def _number(value, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ReviewPageError("Review timing is invalid.")
    return value


def _index(value, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ReviewPageError("Review frame index is invalid.")
    return value


def _verify_views(root: Path, data: dict, prefix: str) -> None:
    """Check every linked media file before the page is published."""
    verified = {}

    def check(record_data, paths, limit=256 * 1024 * 1024):
        record = StoredArtifact(**record_data)
        if record.path not in paths or type(record.size) is not int or not 0 < record.size <= limit:
            raise ReviewPageError("Review media does not match its expected path or size.")
        if record.path in verified:
            if verified[record.path] != record:
                raise ReviewPageError("Review media has conflicting file records.")
        else:
            read_verified(root, record)
            verified[record.path] = record

    mode = data["mode"]
    check(data["sourceOriginal"], {prefix + "frozen/source_original.mp4", prefix + "frozen/source_original.mov"})
    check(data["sourceNormalized"], {prefix + "frozen/source_normalized.mp4"})
    check(data["rawOutput"], {prefix + "output/raw.mp4"})
    guide = data["guide"]
    guide_kind = "static_keyframe" if mode == "A" else "video"
    if guide["kind"] != guide_kind:
        raise ReviewPageError("Review guide does not match its mode.")
    guide_name = "keyframe_A.png" if mode == "A" else "guide_B.mp4"
    check(guide["artifact"], {prefix + "frozen/" + guide_name},
          32 * 1024 * 1024 if mode == "A" else 256 * 1024 * 1024)
    _match_fields(data, {"providerVideoInput": data["sourceNormalized"] if mode == "A" else guide["artifact"]})
    samples = data["samples"]
    if type(samples) is not list or len(samples) != 5:
        raise ReviewPageError("Review needs five ordered samples.")
    for second, sample in enumerate(samples):
        if _number(sample["requestedSeconds"], 0, 4) != second:
            raise ReviewPageError("Review samples must use source times zero through four.")
        source, raw, guide_sample = sample["source"], sample["raw"], sample["guide"]
        source_index = _index(source["frameIndex"], 119)
        raw_index = _index(raw["frameIndex"], 2399)
        source_time = _number(source["ptsSeconds"], -0.00001, 5)
        raw_time = _number(raw["ptsSeconds"], 0, 20)
        offset = _number(raw["offsetSeconds"], -21, 21)
        if not math.isclose(offset, raw_time - source_time, rel_tol=0, abs_tol=1e-9):
            raise ReviewPageError("Raw offset does not match the saved timestamps.")
        if (raw["spanPosition"] not in ("before_first", "within_span", "after_last")
                or type(raw["withinDeclaredDuration"]) is not bool):
            raise ReviewPageError("Raw timing labels are invalid.")
        check(source["image"], {prefix + f"review/source/frame-{source_index:06d}.png"}, 32 * 1024 * 1024)
        check(raw["image"], {prefix + f"review/raw/frame-{raw_index:06d}.png"}, 32 * 1024 * 1024)
        if mode == "A":
            if (guide_sample["kind"] != "static_keyframe"
                    or _number(guide_sample["referenceSeconds"], 0, 0) != 0):
                raise ReviewPageError("Mode A samples must use the submitted static keyframe.")
            _match_fields(guide_sample, {"image": guide["artifact"]})
        else:
            if (guide_sample["kind"] != "video_frame"
                    or _index(guide_sample["frameIndex"], 119) != source_index):
                raise ReviewPageError("Mode B samples must use the matched guide frames.")
            _number(guide_sample["ptsSeconds"], -0.00001, 5)
            check(guide_sample["image"], {prefix + f"review/guide/frame-{source_index:06d}.png"}, 32 * 1024 * 1024)


def _url(record: dict, prefix: str) -> str:
    # All callers use paths checked by _verify_views. Keep every link in this attempt.
    within_attempt = PurePosixPath(record["path"]).relative_to(PurePosixPath(prefix))
    path = str(within_attempt)
    relative = path.removeprefix("review/") if path.startswith("review/") else "../" + path
    return escape(quote(relative, safe="/"), quote=True)


def _frame_caption(label: str, frame: dict) -> str:
    return f'{label}: frame {frame["frameIndex"]}; PTS {frame["ptsSeconds"]:.6f} s.'


def _render(data: dict, prefix: str) -> bytes:
    style_hash = base64.b64encode(sha256(STYLE.encode()).digest()).decode()
    policy = ("default-src 'none'; img-src 'self'; media-src 'self'; "
              f"style-src 'sha256-{style_hash}'; base-uri 'none'; form-action 'none'; object-src 'none'")

    def media(record, label, name, *, still=False):
        url = _url(record, prefix)
        element = (f'<img src="{url}" alt="{escape(label)}">' if still else
                   f'<video controls preload="metadata" playsinline src="{url}">'
                   'Use the download link if this browser cannot play the video.</video>')
        return (f'<figure data-view="{name}"><h3>{escape(label)}</h3>{element}'
                f'<figcaption><a href="{url}" download>Download {escape(label.lower())}</a></figcaption></figure>')

    guide_label = "Submitted guide — static keyframe" if data["mode"] == "A" else "Submitted guide — video"
    main_views = (media(data["sourceOriginal"], "Original source", "source")
                  + media(data["guide"]["artifact"], guide_label, "guide", still=data["mode"] == "A")
                  + media(data["rawOutput"], "Original raw output", "raw"))
    span_labels = {"before_first": "before first raw frame", "within_span": "within raw frame range",
                   "after_last": "after last raw frame"}
    sample_views = []
    for second, sample in enumerate(data["samples"]):
        captions = [_frame_caption("Prepared source", sample["source"]),
                    "Submitted static keyframe; reference 0.000000 s." if data["mode"] == "A" else
                    _frame_caption("Submitted guide", sample["guide"]),
                    _frame_caption("Raw output", sample["raw"]) +
                    f' Offset from source: {sample["raw"]["offsetSeconds"]:+.6f} s.' +
                    f' Span position: {span_labels[sample["raw"]["spanPosition"]]}.' +
                    f' Within declared duration: {"yes" if sample["raw"]["withinDeclaredDuration"] else "no"}.']
        figures = []
        for name, caption in zip(("source", "guide", "raw"), captions):
            url = _url(sample[name]["image"], prefix)
            figures.append(f'<figure><img loading="lazy" src="{url}" alt="{escape(caption)}">'
                           f'<figcaption>{escape(caption)}</figcaption></figure>')
        sample_views.append(f'<section class="sample" data-sample="{second}"><h3>Source time {second} s</h3>'
                            f'<div class="views">{"".join(figures)}</div></section>')
    source_kind = "Synthetic input fixture" if data["sourceKind"] == "synthetic_fixture" else "Clean camera input (declared)"
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{escape(policy, quote=True)}">
<title>Generative Reality — local review</title><style>{STYLE}</style></head>
<body><header><h1>Generative Reality — local review</h1>
<p class="meta">Run: {escape(data["runId"])} · Attempt: {escape(data["attemptId"])} · Mode {data["mode"]}<br>
Provider task: {escape(data["providerTaskId"])} · {source_kind}</p></header>
<section class="notice"><h2>Review is not complete</h2>
<p class="status">Visual effect demonstrated: NOT_RUN<br>Automated workflow complete: NOT_RUN</p>
<p>This page presents saved evidence. It does not record scores or change an acceptance decision.</p>
<ol><li>Play the raw output twice at normal speed. Compare it with the original source and submitted guide.</li>
<li>Compare the matched samples at source times 0, 1, 2, 3, and 4 seconds. Inspect nearby video frames when needed.</li>
<li>Record independent identity, motion, scene, and placement scores. Record each persistent hard failure in the review record.</li></ol>
<p>Use a named human reviewer. Inspect unoccluded frames for identity changes and frame-step around suspected artifacts.
Each score must be at least 3 out of 5 for acceptance. Check for a visibly doubled subject, a missing major limb, subject replacement,
or unexplained teleporting or sliding. Record a hard failure when any one of these artifacts persists continuously for at least 0.25 seconds
in an unoccluded segment. Use native presentation times to record its start and end; annotate any other catastrophic failure.
Keep the reviewer name, UTC time, plays, inspected times, scores, and signature in a separate scorecard. This page leaves both outcomes at NOT_RUN.</p></section>
<section><h2>Original media</h2><p>These players and downloads use the saved original bytes. Browser playback depends on the native codec.
Use a local media player if a video cannot play. No video is converted for this page.</p>
<div class="views">{main_views}</div></section>
<section><h2>Five matched samples</h2>
<p>Source samples use the <a href="{_url(data["sourceNormalized"], prefix)}" download>prepared five-second source</a>.
The source player above uses the original source. PTS is the saved frame presentation time.
Raw offset is raw PTS minus source PTS. Sample PNGs keep decoded pixel geometry without automatic display rotation.
A duration label does not prove that all requested times have video coverage.</p>
{"".join(sample_views)}</section>
<p><a href="manifest.json" download>Download the review manifest</a>. Keep this page with its attempt files.</p>
</body></html>
'''
    return page.encode("utf-8")


def publish_review_page(output_root: Path, review_manifest: StoredArtifact, *,
                        run_id: str, attempt_id: str) -> StoredArtifact:
    """Verify visible media and publish deterministic HTML without provider calls.

    Supply the reference returned by build_review_evidence. This reader checks
    page inputs, not provider provenance or human visual quality. Keep the run
    directory private. Later file changes are detected when this function runs
    again; a browser does not recheck hashes while it plays linked files.
    """
    try:
        BudgetLedger._run_id(run_id)
        BudgetLedger._attempt_id(attempt_id)
        root = Path(output_root) / run_id
        prefix = f"attempts/{attempt_id}/"
        if review_manifest.path != prefix + "review/manifest.json":
            raise ReviewPageError("Review manifest must belong to the requested attempt.")
        data = _read_record(root, review_manifest, 128 * 1024)
        _match_fields(data, {"schemaVersion": "1.4", "manifestType": "indexed_review_sources",
                            "runId": run_id, "attemptId": attempt_id, "samplePixelFormat": "8-bit RGB",
                            "sampleAutomaticRotationApplied": False,
                            "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN"})
        if (data["mode"] not in ("A", "B") or data["sourceKind"] not in ("clean_camera", "synthetic_fixture")
                or type(data["providerTaskId"]) is not str or not 1 <= len(data["providerTaskId"]) <= 128):
            raise ReviewPageError("Review identity is invalid.")
        _verify_views(root, data, prefix)
        return _publish_or_verify(root, prefix + "review/index.html", _render(data, prefix))
    except Exception:
        raise ReviewPageError("Review page could not be verified or published for this attempt.") from None

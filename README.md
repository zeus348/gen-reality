# gen-reality

Prepare an offline attempt from a clean fixed-camera clip and a prepared PNG:

```sh
python -m app.prepare --source scene.mp4 --asset creature.png \
  --u 0.5 --v 0.6 --height-frac 0.25 \
  --non-human --confirm-clean-source --output-root runs
```

Install the requirements below and FFmpeg first. The command emits JSON with
the manifest path, size, and hash. It keeps originals, normalized video, placement,
prompt, and both guidance artifacts under one immutable attempt. Give a new
`--attempt-id` for each placement edit. Mark test inputs with
`--source-kind synthetic_fixture`; use fresh clean camera footage for a real
visual experiment. The clean-source and non-human flags record your declarations.

The command saves `run-start.json` before media work and returns its reference
as `runStart`. Its deadline is 180 minutes after that original start. Reusing a
run ID keeps the same start and deadline, including after preparation fails.
An existing nonempty run without this record cannot be adopted. If another
starter is still publishing, retry after it finishes. A start published before
an error is reused. A stranded pending file without a published start blocks
that run; use a new run ID or explicit trusted repair. No media work starts
until the record is durable. Library applications use `prepare_run_attempt`;
the lower-level `prepare_attempt` remains an offline primitive without lifecycle
evidence. Old preparation manifests and files are not rewritten.

The manifest is an **offline preparation** record. It reports `PREPARED` and
`LOCAL_PIPELINE_ONLY`; visual and automated generation outcomes remain `NOT_RUN`.
This CLI does not authorize spending or call a provider. Server primitives for
budgets, submission, and polling are described below. The live workflow, human
review, UI, and full release audit remain pending. No provider key is needed for
this command.

The manifest is published last. If preparation fails while writing files, retain
the incomplete attempt and use a new attempt ID. Existing files are never replaced.

The loader reads the locked version 1.4 configuration. It does not submit
provider tasks or authorize spending.

Use Python 3.11 or later. Install the Python dependencies:

```sh
python -m pip install -r requirements.txt
```

Load the default configuration, or pass a path to a JSON file:

```python
from pathlib import Path
from core.config import load_config

settings = load_config()
settings = load_config(Path("config/build_config.json"))
```

The loader raises `ConfigError` for unreadable files, invalid UTF-8 or JSON,
duplicate fields, files larger than 65,536 bytes, and settings that differ from
the locked schema. It returns a new dictionary on each call. The SDK version is
pinned to the tested release. Provider URIs in the config are shape examples,
not uploaded inputs. The adapter checks the installed SDK version before use.

Run the offline tests from the repository root:

```sh
python -m unittest discover -s tests/unit -v
```

Prepared assets must be single-frame, 8-bit RGBA PNGs. Confirm that the asset is
non-human; the program does not detect its subject. It preserves the original
bytes and all alpha values. At least one pixel must have zero alpha and one must
have nonzero alpha. Files are limited to 32 MiB, each dimension to 16,384 pixels,
and total size to 16,777,216 pixels.

```python
from features.asset.ingest import load_asset

asset = load_asset(Path("creature.png"), non_human=True)
```

Artifact storage currently requires Unix directory handles and hard links. Use a
private application-owned root on one filesystem. `publish_bytes` publishes each
relative path once; `read_verified` checks its recorded size and SHA-256. New
placement attempts must use new paths. These functions do not prevent the owner
of the storage directory from changing files outside the application.

Video normalization requires `ffmpeg` and `ffprobe` with the H.264 encoder.
The current checks use FFmpeg 7.1.5. Inputs must be MP4 or MOV files of 5–20
seconds, up to 256 MiB, 4096 pixels on either axis, and 8,847,360 pixels per
frame. Normalization keeps the original bytes and returns a five-second silent
H.264 clip with 120 frames at 1280×720 and 24 FPS. It preserves the display aspect
ratio with black bars and decodes the whole source before accepting the trim.

```python
from features.media.ingest import normalize_video

video = normalize_video(Path("clean-scene.mp4"))
```

Compile both guidance modes from that normalized video, asset, and placement:

```python
from features.guidance.video import compile_guidance
from features.placement.math import Placement

guidance = compile_guidance(video, asset, Placement(0.5, 0.6, 0.25))
```

The result contains a first-frame PNG, a five-second MP4, their hashes, and one
shared prompt. Compilation needs about 633 MiB of temporary disk space for raw
RGB frames, plus input and encoded files. Rendering and encoding run offline.
Repeated artifact hashes are checked with the installed Pillow and FFmpeg
versions; byte identity across different tool versions is not established.

The budget ledger is a separate, local server primitive. Only trusted operator
code may record an approval and per-attempt quote. Its database needs an existing
private, durable parent directory. It reserves integer USD cents under a SQLite
write transaction and retains idempotency across restarts. The P0 limits are
US$10, six attempts, and 180 minutes; an approval may set lower limits.
Verified final charges replace reserved amounts in the total. Unknown charges
keep their reservation. The receipt hash and final amount are recorded once;
any charge above its quote is retained and blocks further spending. A confirmed zero
charge does not refund an attempt or reopen the elapsed-time limit.
Reservations alone do not authorize provider calls. The offline CLI does not
create approvals, record final charges, or call a provider.
For a new experiment, trusted operator code reads `read_run_start(output_root,
run_id=...)` and passes its `deadline_unix` to `approve_run` as
`deadline_unix_cap`. Approval stores the earlier of that cap and its requested
duration, so delayed approval cannot restart the experiment clock. A cap can
leave less than one minute. Reservation and final-send checks enforce the same
stored deadline. Legacy approvals retain their original approval-time basis;
no missing experiment start is inferred.
Trusted server code can pass the preparation's default `att-<32 lowercase hex
digits>` ID to `reserve(attempt_id=...)`. That ID stays the same through submission
and billing. A different idempotency key or run cannot claim it. Reusing the key
with a different attempt ID fails. Custom offline IDs must use this standard
format before they can be reserved. Omitting the ID retains generated-ID behavior.

`SubmissionCoordinator` commits `SUBMITTING` before calling a trusted adapter.
It rechecks the run limits and permits one create call per reservation. The
adapter must disable its own retries. A timeout, invalid task ID, or process
exit leaves `SUBMISSION_UNKNOWN` with the cost held. A restart never repeats
that create call. Only verified task evidence can reconcile an unknown result.
One coordinator owns each private Unix ledger until it closes; concurrent
callers share that coordinator. Known task IDs survive restarts.

The coordinator appends local `submission_events` in the same transaction as
each claim, task record, unknown result, owner recovery, or reconciliation.
`SEND_GATE_PASSED` records a successful final authority and freshness check;
it does not prove that a POST reached the provider. Lower-level adapter calls
can have no such event. SQL triggers reject event updates, deletions, and
replacement of an existing event ID.
Existing ledgers gain an empty journal, with no reconstructed history. Recovery
records only the recovery performed now. Claim and send checks require valid
timestamps. If an ordinary clock error occurs while recording a result, recovery, or
reconciliation, its timestamp is NULL and the state is still durable. These
records establish neither live execution nor actual charges. Approval time is
recorded separately from the experiment-start artifact and its deadline cap.

`run_events` puts new approvals, reservations, final charges, and submission-event
links in one committed order. Each journal entry shares its source transaction.
Use its sequence to reconstruct actual-or-reserved amounts when timestamps tie.
Financial entries hold fixed snapshots of approved limits, reserved cents and
payload/key hashes, or actual cents and the receipt hash. Snapshots use at most
4096 ASCII JSON bytes. The frozen payload still binds the separate pricing
evidence; the journal does not verify pricing or billing receipts. A final charge
survives an ordinary clock error with a NULL observation time. Identical retries
add no entry. SQL triggers block updates, deletions, and replacement through
either journal sequence or submission-event link. Existing financial records
and old submission events remain unlinked; their missing history is not filled
in. This journal records local changes, not remote receipt or live provenance.

`core.run_snapshot.read_run_snapshot(path, run_id=...)` captures the eight fixed
ledger tables in one SQLite read transaction, including committed WAL records.
It opens the existing database with `mode=ro`, `query_only=ON`, and
`trusted_schema=OFF`; it performs no application/database-state writes, recovery,
table creation, or checkpoint. SQLite can still create or manage WAL/SHM
coordination files. Keep the existing parent directory private and its ancestors
trusted throughout the read.

The frozen result contains named tuple rows for the exact requested run, missing
table names, and global presence flags for orphaned or cross-run links. Flags
contain no unrelated row contents. Missing approval or journal history is never
filled in. Table columns and scalar storage types are checked, but financial
replay, record semantics, and timing verdicts are separate work. The reader does
not open payloads, billing receipts, provider evidence, or output files.

Reads are limited to 64 MiB each for the database, WAL, and rollback journal,
4 MiB for SHM, 10,000 rows per table and 20,000 captured rows overall. SQL checks
cell lengths before returning their values: 4096 bytes for journal JSON and
1024 for other cells, with 8 MiB of captured scalar data overall. Query work is
limited to five million SQLite VM instructions and lock waits to 0.25 seconds.
Unsupported schemas, non-UTF-8 databases, or exceeded limits fail the whole read.
The SHA-256 covers versioned, ordered, typed logical records plus missing-table
and relationship flags, streamed with a 64 MiB encoding limit. It does not hash
the bare database file or establish authentic billing, provider origin, or an
acceptance outcome. Changes to unrelated valid rows do not enter this hash.

`core.run_audit.audit_run(snapshot)` checks that immutable snapshot without any
I/O, current-clock reading, recovery, or writer call. Its frozen report separates
record consistency, current recorded limits, spending-boundary replay, and
recorded elapsed checks. Each assessment has a verdict and separate
`COMPLETE`/`INCOMPLETE` coverage. Missing evidence produces `BLOCKED`; a known
failure stays `FAIL` even when other evidence is missing. A stale snapshot hash
is an explicit failure, and the report binds the recomputed hash of the rows
actually assessed. Malformed or oversized snapshot structures are rejected.

The audit sums integer recorded actual charges plus reserves without recorded
final charges. A zero charge releases held money but never an attempt. Missing
final billing does not itself fail conservative accounting. Overquote attempt
IDs are separate facts: a final charge above its quote can still be within the
approved dollar cap. Replay fails a later spending boundary after that charge;
it does not invalidate an earlier valid boundary. Shared journal
sequence reconstructs those amounts at reservations, claims, and passed send
gates, including charges between boundaries with equal timestamps. Fixed event
snapshots and submission links must match their current records. The report
also retains unknown/in-flight and unresolved-billing attempt IDs, approved
limits, and boundary facts; it does not duplicate the full captured row graph.

Timing uses `APPROVAL_START`, not the start of the entire experiment. Repeated
passed gates are allowed; a direct callback with no gate leaves coverage
incomplete. Nullable or backward charge/result observation times do not reorder
money. Later polling and completion do not retroactively invalidate a timely
spending boundary. Retained polling evidence is distinguished from a fresh
observation. These are consistency checks of declared local records, not proof
of an actual POST, authentic billing, provider origin, human review, or release
acceptance. Malformed attempt identifiers are reported without echoing their
text into issue or summary fields.

`RunwayCreateAdapter` uses the official `runwayml` 5.24.0 SDK. Mode A sends one
keyframe at zero seconds; Mode B omits `keyframes`. It disables SDK retries and
HTTP redirects. Its required recorder receives the serialized request body
before the POST, without authentication headers. Call it only through the
coordinator after checking the account, pricing, uploaded media, and frozen
inputs. The offline tests use HTTPX's fake transport and make no paid calls.
Full submission preflight is still pending.

The checked PyPI wheel is `runwayml-5.24.0-py3-none-any.whl`, SHA-256
`28e67a64ee34b23cb27873b58a912526a48b449d8285a726518a98b2516f0e03`.

`FrozenCreate` binds the request bytes, mode, SDK version, preparation manifest
hash, pricing evidence hash, and quoted cents. Trusted server code reserves its
`payload_sha256`, then calls `submit_frozen`. That function checks the database
reservation and quote, compares the SDK's actual body with the frozen bytes,
and publishes both records before sending. Changed inputs cannot use the old
reservation. These hashes identify evidence; the caller must still verify the
media, uploaded URIs, account access, and current price. The offline CLI does
not call this path.

`TaskPoller` appends polling outcomes for a known provider task ID. A local
deadline or GET timeout preserves the job state and held cost; a later call
resumes the same ID. Failed and canceled tasks stay terminal. Provider-reported
success stays separate from local success until raw video validation finishes.
The retrieval callback must use GET only and save the provider response before
returning its evidence hash.

`retrieve_task_once` uses the pinned SDK's GET endpoint. It limits successful
JSON response bodies to 1 MiB, checks the task identity and status, and stores
each response by its SHA-256. Repeated responses verify the existing file.
Pass this function as the poller's retrieval callback. It does not download
output video or mark local success.

`validate_raw_video` fully decodes a raw MP4 with FFmpeg and records its actual
frame timestamps. It preserves the original bytes and accepts timing that
differs from the normalized input. Resource limits are 256 MiB, twenty seconds,
4096 pixels per axis, 8,847,360 pixels per frame, 120 FPS, and 2400 decoded frames.
Supported codecs are H.264, HEVC, AV1, and MPEG-4. It accepts one video stream
and at most one audio stream, with up to eight audio channels at 192 kHz.
Every decoded frame must keep the declared dimensions; resolution changes fail.
The validator requires the MP4's declared container `start_time` and records it
as `timelineOriginSeconds`. Frame PTS remain native; elapsed frame time is PTS
minus that origin and must fit the declared container duration. Thus a one-second
clip starting at PTS 1.25 can retain frames at 1.25 through 2.216667. The origin
is not inferred from the first video frame: earlier audio can leave a real gap
before video starts. Missing or invalid origin in a fresh probe is rejected.
Content validation alone does not prove a real provider generation or visual
quality.

`plan_review_frames` plans the five review samples at 0, 1, 2, 3, and 4 seconds.
Pass the validated normalized source timestamps, native raw timestamps, declared
duration, and `raw_timeline_origin_seconds` (default zero). It records actual
source and nearest raw frame indexes and timestamps, choosing the earlier frame
on an exact tie. It preserves timestamp
values, compares elapsed raw times to source PTS, and records offset as elapsed
raw time minus source PTS, including the source validator's small tolerance
around zero. It reports position within the raw timestamp span separately from the
declared duration flag. A positive first timestamp is valid; the last timestamp
is not playback end. Container duration can include longer audio, so the flag
does not prove video coverage. This helper creates no images or acceptance
verdicts. The review builder below binds its plan to one verified attempt.

`extract_frame_png` extracts one decoded frame index from already validated MP4
bytes and matching dimensions and frame count. FFmpeg keeps decoded geometry
without automatic rotation, scaling, or frame-rate conversion. The helper uses
private temporary files, then checks that the one output is a fully decodable
8-bit RGB PNG with the expected dimensions and at most 32 MiB. It returns derived PNG
bytes; the caller retains the original video and timestamps. The review builder
handles attempt binding and immutable publication; visual acceptance is separate.

`build_review_evidence` verifies one attempt's preparation, saved request check,
reservation, actual submitted request bytes, completed raw output, metadata, and
saved provider success response before publishing review samples. Its manifest
references the original source, normalized five-second sample basis, actual
provider input and guide, unchanged raw video, and five source/guide/raw PNG
references. Mode A reuses its static keyframe PNG; Mode B samples its guide clip.
It records native frame indexes, timestamps, offsets, and duration annotations.
New raw metadata carries the origin into the review's `rawTimelineOriginSeconds`.
Saved metadata and reviews without these fields retain their original zero
basis; rebuilding those legacy reviews preserves their bytes. New review pages
label native PTS and elapsed time separately. Browser control time is not claimed
to measure native PTS, and no seek synchronization is introduced. Scorecard
endpoints still require exact saved native PTS; container duration never supplies
an inferred final frame boundary.
If an older completion saved raw bytes and metadata before its final database
commit failed, retry can reuse the exact old metadata only after verifying every
field and its zero-based timeline. Recovery keeps its bytes and hash; fresh
completions always publish the explicit origin. Changed metadata or a
contradictory explicit origin is rejected rather than rewritten.
PNGs keep decoded geometry without applying rotation metadata; these records do
not verify display orientation or video coverage. Matching repeats verify and
reuse existing bytes, interrupted publication can resume, and the manifest is
published last. Expired checks remain readable. Visual and automated workflow
outcomes stay `NOT_RUN`; the offline fixtures do not establish a live generation.

`assess_scorecard` validates up to 64 KiB of scorecard JSON for the expected run
and attempt. It requires exactly four integer scores from 1 to 5, at least two
declared full normal-speed plays, inspected source times `[0,1,2,3,4]`, and a
written observation. Absent scorecards and otherwise valid drafts without a
nonblank reviewer name, UTC review time, or signature return `NOT_RUN`. Malformed
or mismatched data raises `ScorecardError`. Complete declarations get a
`rubric_verdict`: all four scores must reach 3 and no qualifying failure may remain.
This result does not authenticate a human, generated clip, or actual viewing;
it never sets either global outcome. Input verdict and provenance flags are rejected.

Each `hardFailures` entry declares `kind`, numeric `startPTS` and `endPTS`,
`reason`, `continuous`, and `unoccluded`. Ordinary kinds are `duplicate`,
`missing_limb`, `replaced_subject`, and `teleport_or_slip`. Report each continuous
run separately. An unoccluded ordinary run fails at 0.250 seconds; a justified
`catastrophic` entry fails at any positive duration. Intervals are half-open:
the end is the next unaffected frame boundary, so six 24 FPS frames span 0.250
seconds. Same-kind unoccluded intervals must not overlap or touch; positive-gap
runs are never summed. Decimal arithmetic preserves up to twelve whole and nine
fractional digits, bounds numeric tokens and exponents, and ignores caller decimal
settings. The validator does not verify endpoints against native media or infer
the final video display end. Storage and evidence binding are separate operations.

`bind_scorecard` accepts an existing open coordinator, output root, trusted review
manifest reference, original scorecard bytes, and expected run and attempt IDs.
It verifies the displayed files and current completed task, raw output, metadata,
and saved provider success response before it publishes anything. A complete
reviewer name, UTC time, and signature are required. Review time must be at or
after completion and at or before the ledger's trusted current time. Each failure
endpoint must exactly equal a saved native frame timestamp. The final display
end is unavailable unless separately established; container duration and FPS do
not supply it.

The original card is saved at `review/scorecards/{SHA256}.json` within the attempt.
Its receipt is published last at `review/scorecard-bindings/{SHA256}.json` and binds
the card, review manifest, completed output, metadata, task, and declared rubric
result. Retain the returned receipt reference to select that review. Exact retries
reuse the records; corrected cards preserve earlier records. There is no automatic
latest-review selection. Review pages and manifests stay unchanged. A binding
does not authenticate a human or prove live generation; both global outcomes
remain `NOT_RUN`. This library operation makes no provider calls.

`read_scorecard_binding` verifies an explicitly selected binding reference with
an existing open coordinator and expected run and attempt IDs. It returns frozen
binding, card, and review-manifest references plus a recomputed rubric assessment.
It checks current completed-task records, source hashes, native endpoints, and
the complete canonical receipt. Extra fields or changed verdicts are rejected.
The review must follow completion, but historical reads do not consult today's
clock or the spending deadline. They create no schema, change no ledger rows, and
publish no artifacts. The reader uses the existing ledger connection; it is not
a standalone SQLite snapshot auditor. It neither selects a latest review nor
establishes human participation or live provider provenance. The writer still
checks that a newly bound review time is no later than its trusted current time.

`build_export_inventory` takes an existing open coordinator, output root, trusted
review-manifest reference, expected run and attempt IDs, and an optional explicit
scorecard-binding reference. It requires completed output and verifies the saved
preparation, request, reservation, task, native timing, and selected review before
publishing one `export/inventories/{SHA256}.json` within the attempt. It needs no
current clock, fresh provider check, recovery, or provider call. Exact repeats
reuse the inventory; a corrected selection preserves earlier inventories.

The strict inventory schema permits at most 30 unique payload files. The list
contains the preparation manifest, all nine frozen files (including both local
guides), raw output and metadata, the review manifest and selected PNGs, and only
the explicitly selected binding/card. Paths and roles are sorted, and a reused
file appears once. Original bytes stay unchanged. There is no directory scan,
implicit HTML inclusion, or automatic selection of another attempt or scorecard.

The embedded provider summary contains fixed fields and a reserved cost, not an
actual-charge claim. Exactly four private records have separate hash/size entries:
the request check, request, frozen request, and provider success response. Their
bytes, uploaded URIs, and signed output URLs are excluded. References to these
private records inside bundled manifests do not mean their bytes are included.
Human/provider provenance remains `NOT_ESTABLISHED`; billing, full-run budget,
elapsed-time, and recorded-test audits remain `NOT_RUN`, as do both global outcomes.

This operation creates an inventory, not a ZIP. Its `files` list defines future
ZIP payload members; a future ZIP must also contain the exact inventory bytes as
`inventory.json`. `totalFileBytes` counts payload bytes without asserting that an
archive fits the 512 MiB artifact limit. ZIP creation uses the separate API below;
independent release auditing remains future work.

`build_export_archive` takes the output root, a trusted inventory reference, and
expected run and attempt IDs. It checks inventory identity, schema, same-attempt
paths and roles, selected references, and byte totals. It packages the exact
inventory bytes as `inventory.json` first, then only the sorted listed payload
members. Every payload is hash-verified before its bytes are written. Original
media bytes stay unchanged. The operation needs no ledger, private provider
records, current clock, directory scan, or provider call.

ZIP members use `ZIP_STORED`, fixed 1980-01-01 timestamps, Unix regular-file mode
`0600`, and no comments or extra fields. ZIP64 is disabled. An empty ZIP with the
same names and metadata supplies the exact overhead calculation. The complete
size must fit 512 MiB before any payload is read. Construction uses a private
bounded temporary file and checks its final size before create-only publication
at `export/archives/{SHA256}.zip` within the attempt. It holds one payload at a
time during construction; final publication uses the existing byte-based storage
API. Repeats return the same artifact, and changed existing archives are rejected.

This is an evidence archive, not an installable software package or a release
verdict. It preserves the inventory's outcomes and provenance declarations; it
does not audit the complete evidence graph or establish human/provider authenticity.

`verify_export_archive` takes the output root, trusted archive and inventory
references, and expected run and attempt IDs. It reads only the ZIP. The original
inventory, payload files, private evidence, and ledger need not remain available.
It verifies the archive hash and size, bounds the actual central-directory entries
before parsing, checks fixed ZIP metadata and contiguous extents, and rejects
duplicate, extra, missing, or unsafe members. The exact `inventory.json` bytes must
match the retained inventory reference. Its schema, identity, paths, roles,
selection, and byte totals are checked using the shared inventory contract. Each
payload is streamed through CRC, length, and SHA-256 checks against that inventory.
The full ZIP is held in memory within the 512 MiB storage limit; payload reads use
bounded chunks. No files are extracted or written, and no clock or network is used.

The frozen `ExportArchiveIntegrity` result contains the verified references,
payload file count, and total payload bytes. It makes no authenticity or release
claim. Retain both trusted references independently: a consistently replaced ZIP
and references cannot be authenticated by hashes alone. Linked-record graph,
media decoding, human/provider provenance, full-run budget, and release audits
remain separate. The future QA owner must implement `qa/verify_run.py` separately.

`verify_export_preparation` takes the same trusted archive and inventory
references and expected IDs. It verifies the entire ZIP, then checks preparation
records retained from that same byte snapshot: the manifest, configuration,
placement, input declarations, and prompt. Four JSON records are limited to
64 KiB each; the prompt is limited to 4000 bytes and 1000 characters. All nine
frozen references must match the inventory, and the declared source kind must
agree. The existing filesystem reader shares the preparation rules, including
the locked configuration schema and placement validation. No original files,
extraction, writes, clock, or network are needed.

The frozen `ExportPreparationIntegrity` result contains the archive integrity
result, preparation reference, and declared source kind. These checks establish
record consistency, not media correctness or authenticity. Review, raw metadata,
selected scorecard, private evidence, full-run audits, and global outcome
evaluation remain separate.

`verify_export_review` adds review-manifest and raw-metadata consistency checks
to archive and preparation verification. It retains those two fixed JSON records
at up to 128 KiB each from the same fully verified snapshot. Review identity,
mode, declared source kind, task, payload, exact media/sample references, and
private evidence hash/size references must agree with the inventory. Private
record bytes stay excluded. The raw metadata must reference the same output and
task, with bounded geometry, duration, and strictly increasing recorded PTS.

Each raw sample must use its recorded native PTS and the nearest elapsed raw frame
to the declared source PTS, choosing the earlier frame on a tie. Metadata and
review origins must agree, with absent saved fields meaning zero. Offset, span, and
duration labels are checked. Shared pure rules keep filesystem review readers
consistent. These checks do not decode media or prove that saved timestamps
describe actual frames. Full source/guide timelines are unavailable and are
not inferred from frame rate; the final display end is not inferred either.
The frozen `ExportReviewIntegrity` result contains preparation integrity and the
review/metadata references. Selected scorecards, private evidence authenticity,
full-run audits, and global outcome evaluation remain separate.

`verify_export_scorecard` completes the selected scorecard and binding checks
within that same fully verified archive snapshot. Only the explicitly selected
card and binding are retained, each at up to 64 KiB. The frozen result reports
`NOT_SELECTED` with no card, binding, or assessment when the inventory selects
none. It does not choose another review automatically.

For a selected card, the original bytes and references must match. Shared pure
binding rules require complete declared sign-off, recompute the rubric, require
failure endpoints to equal exact saved native PTS, and require review time at or
after recorded completion. The entire binding must equal the reconstructed
canonical receipt bytes. A complete rubric `FAIL` is valid consistent evidence.
Historical verification uses no current-clock upper bound. These checks do not
authenticate a human, actual viewing, provider execution, or decoded media, and
they do not establish a global outcome or release verdict.

Save an entered scorecard with the trusted review manifest's path, hash, and size:

```sh
python -m app.scorecard --output-root runs --ledger PRIVATE_LEDGER.sqlite \
  --run-id RUN_ID --attempt-id ATTEMPT_ID \
  --review-path attempts/ATTEMPT_ID/review/manifest.json \
  --review-sha256 REVIEW_SHA256 --review-size REVIEW_SIZE \
  --scorecard-file entered-scorecard.json
```

The command requires an existing trusted ledger. It reads a regular scorecard
file of at most 64 KiB without following symlinks, then validates the complete
card and expected IDs before opening the coordinator. Enter the reviewer name,
UTC time, and signature yourself; the command does not fill them in. The original
bytes are preserved. Opening the coordinator can recover abandoned `SUBMITTING`
jobs as `SUBMISSION_UNKNOWN`, with their reserves held. It creates no approval,
reservation, or provider request and needs no API key.

Successful JSON contains `runId`, `attemptId`, and the explicit `binding` artifact
reference. Exit code 0 means the binding was saved, including a rubric `FAIL`;
it does not mean visual acceptance. Exit code 1 reports a safe command error.
Retain each returned binding reference. Repeating the same input returns the same
record, and a corrected card creates a separate record without replacing the old
one. `bind_scorecard_file` exposes the same operation to trusted local callers,
with an injectable clock for tests.

`OutputCompleter` accepts a trusted GET-only download callback after a recorded
provider success. It validates the entire MP4, publishes raw bytes and metadata,
then commits both artifact records and local `SUCCEEDED` in one transaction.
An interrupted commit can resume using the same verified bytes. Repeated calls
verify completed artifacts and return the same record without downloading again.
Money stays held until final billing is verified. Output metadata records content
validation as `PASS`; visual and automated workflow outcomes remain `NOT_RUN`.

`download_task_output` reads the saved success response by its hash and selects
its first output URL. The server must supply an exact host allowlist from trusted
provider documentation or verified operator configuration. Never derive it from
the response or a client. The downloader requires HTTPS on port 443 and rejects
credentials in URLs, IP addresses, and redirects. A separate HTTPX client sends
one GET without provider credentials. Identity-encoded bytes are limited to
256 MiB. It checks a 60-second deadline between chunks; a stalled read can take
another 30 seconds. Use this function as the completion callback. Failed downloads
can resume the same task without a new generation or a released reservation.

`upload_frozen_artifact` verifies a frozen PNG or MP4, then gives those exact
bytes to the official SDK's `uploads.create_ephemeral` method. The server must
supply an approved upload host set. The SDK starts the upload at Runway, then
sends the multipart body with a separate HTTPX client. Neither request retries
or follows redirects. The source file is not reopened after verification.
Successful upload metadata must use identity encoding and fit within 1 MiB.
Oversized or encoded metadata closes the response before any multipart upload.
A successful upload publishes an immutable source-hash-to-URI record with SDK
version and times. Its conservative expiry is 24 hours after call start.
Current [Runway upload limits](https://docs.dev.runwayml.com/assets/uploads/) are
512 bytes to 200 MB, and the account must have purchased credits. This path
creates no generation. Tests intercept both HTTP clients; they make no uploads.

`check_account_once` reads the account through the SDK and stores its exact
response and a check summary. It checks credits, Aleph 2 model availability,
daily usage, and configured limits against the required credit quote. Missing
model limits or usage fail closed; explicit null limits mean unlimited.
Successful bodies must use identity encoding and fit within 1 MiB. A ten-second
deadline is checked between chunks; a stalled read can take another ten seconds.
`require_current` rejects blocked checks, checks older than sixty seconds, and
future-dated checks. Repeat this check before each paid submission, together
with current pricing and budget approval. This account snapshot authorizes no
spending and cannot guarantee that a later provider request will succeed.

`record_price_quote` records facts checked by a trusted operator: current credits
per second, minimum credits, USD per credit, and the maximum additional cents
that can apply. None of these values has a default. The operator must retain
supporting evidence and supply its verified file record. The evidence hash
identifies those bytes; it does not verify their origin or the operator's reading.
The quote uses the greater of the five-second cost and minimum credits, rounds
fractional USD cents up, then adds all additional cents. Totals above US$10 fail.
Decimal arithmetic uses its own context. Quotes expire five minutes after the
recorded price check and never grant spending authority. Current pricing must
still be checked by the operator before each paid submission.

`read_prepared_attempt` verifies the expected run, attempt, manifest, and all nine
frozen files. It checks attempt paths, hashes, locked configuration, placement,
prompt, and input declarations. JSON is limited to 64 KiB per file. Mode A
selects the normalized source and first keyframe; Mode B selects the composited
guide and omits the keyframe. It retains the declared source kind. This reader
checks frozen record identity and consistency; media validation and guidance
compilation remain the preparation writer's responsibility.

`check_prepared_request` joins the verified snapshot, matching upload records,
current price quote, and account evidence. It checks their hashes, recorded
values, credits, and expiry times, then saves complete `FrozenCreate` fields
before reservation. This permits request reconstruction after a restart.
An account refresh creates a new check record while preserving the frozen
payload identity. The result grants no spending authority. Its freshness must
be checked again at the final send boundary.

`submit_checked` is the application submission entry point. It matches the saved
check to the run and reservation, then uses the existing one-shot SDK path.
After both request records are durable, it verifies the check again and rechecks
run limits, unsettled cost, clock order, and evidence freshness before the POST.
An expired initial check leaves the reservation untouched. A failure after the
durable submission claim remains `SUBMISSION_UNKNOWN`, with no automatic replay.
Known or uncertain submissions return their saved state even after evidence
expires. The application must use the same provider account key for upload,
account checks, and submission.

`python -m app.submit` submits an existing saved request check. Supply
`--output-root`, `--ledger`, `--run-id`, `--attempt-id`, and the check record's
`--check-path`, `--check-sha256`, and `--check-size`. The path is relative to the
run directory. The ledger must already contain operator approval with the same
quote. `--api-key-env` names a server environment variable and defaults to
`RUNWAY_API_KEY`; never put the key value on the command line. The key must match
the account used to prepare the check. The command does not grant approval,
refresh evidence, or upload media. It uses a stable key for a new reservation
and preserves an existing reservation. New work, including an existing RESERVED
attempt, needs verified run-start evidence and approval within its fixed deadline.
Known or uncertain submissions return their stored state after a restart, even
without credentials, fresh evidence, or the run-start file.
JSON output contains submission state and task ID. Exit codes are 0 for a known
submission, 2 for an uncertain submission, and 1 for a command failure. Upload
and preflight command orchestration remain pending.

`python -m app.resume` inspects or resumes an existing attempt. Supply
`--output-root`, `--ledger`, `--run-id`, and `--attempt-id`. Active tasks use one
SDK GET with the key named by `--api-key-env` (default `RUNWAY_API_KEY`). After
provider success, the command downloads, validates, and saves the raw output.
Supply each trusted output host with `--output-host`; the operator must obtain
these hosts from trusted configuration, never from a task response or browser
client. A later invocation gets fresh task evidence, so an expired output URL
can be replaced without another generation. Each invocation polls at most once;
its polling deadline is not an overall command timeout.

`--status-only` makes no provider calls. Opening the ledger still recovers an
abandoned submission claim as `SUBMISSION_UNKNOWN`. Reserved, uncertain, and
terminal tasks require no key; uncertain submissions need verified manual
reconciliation. Completed output is hash-verified before it is reported.
Resume never reserves funds, records a charge, or submits a generation. JSON
contains the saved state and output records, without signed URLs. Exit code 0
means status inspection or local success, 2 means another state, and 1 means
the command failed. Local success does not establish human visual acceptance.
The downloader installs a persistent HTTPX logging filter that retains request
methods, origins, and response status while hiding URL paths and queries. It
also protects concurrent downloads when SDK INFO logging is enabled.

Both commands add a `runtime` summary with immutable receipt references and
explicit evidence gaps. Trusted service callers can pass `RuntimeEvidence`;
service return types stay unchanged. The recorder saves checked creation inputs
before the final send gate and a download intent before the GET. Failure to save
either prerequisite blocks that operation. Ordinary failures to save later
observations preserve the known task or returned bytes. Failed observation
clocks produce null times; interruptions still propagate.

Receipts link the saved request, local submission events, a separate SDK task
GET, and the exact returned download bytes. Response-hook time means local
response availability; body-finish time means the bounded read loop completed.
Neither is a provider receipt time. Restart reads fixed creation records and
reports missing legacy links. Each GET or download retry has distinct receipts.
`DEFAULT_NETWORK` describes the actual selected default HTTPX transport;
`INJECTED` marks supplied or replaced transports. Neither proves provider
authentication or a real generation. Retain selected receipt references outside
the run/export tree in trusted operator configuration for later QA. Do not
reconstruct that trust record from exported receipt files. Receipts omit keys,
headers, signed output URLs, and acceptance claims. Their 16 KiB limit does not
bound the SDK's pre-existing eager read of non-success HTTP response bodies;
that remains an owner follow-up.

Use `--trusted-runtime-index /private/operator/run-pins.sqlite` with submit and
resume to retain fresh receipt pins outside `--output-root`. Its existing parent
must be private to the current operator. One index belongs to one run, output
root, and ledger; it cannot be a ledger or an exported file. The fixed SQLite
files and their sidecars must not share names or file identities with ledger files.
The index uses DELETE journals, 4096-byte pages, a 16 MiB file limit, 10,000 pins,
bounded reads, and short lock waits. Foreign databases and WAL sidecars are
rejected. Interrupted initial creation can leave an unusable index; use a new
private path or trusted repair rather than adopting its contents automatically.

`RuntimePinStore` is optional for trusted service callers. A fresh creation
context must be pinned before POST when the store is requested. Ordinary pin
failures after a response preserve the saved task or returned bytes. Restart
only compares old receipts: missing pins stay missing, and changed receipts
report `PIN_MISMATCH`. A new GET can pin its own receipt without establishing
missing creation history. Repeated reads and identical existing artifacts never
backfill pins. `read_runtime_pins` returns a bounded immutable snapshot without
schema or row writes. Paths remain private; pins identify retained local bytes,
not provider authentication or acceptance. Later QA must receive this index from
trusted operator configuration rather than reconstructing it from an export.

`publish_review_page` creates a static local page from the artifact reference
returned by `build_review_evidence`. The page shows the original source, the
actual submitted guide (a static PNG for Mode A or a video for Mode B), and the
original raw output. Five saved sample groups use the prepared source at times
0 through 4 seconds, with frame indexes, presentation times, and raw offsets.
Downloads retain the original bytes. Browser playback depends on the codec;
use a local media player if a native file cannot play. No video is converted.

Create the page with the trusted review manifest's path, hash, and size:

```sh
python -m app.review --output-root runs --run-id RUN_ID --attempt-id ATTEMPT_ID \
  --review-path attempts/ATTEMPT_ID/review/manifest.json \
  --review-sha256 REVIEW_SHA256 --review-size REVIEW_SIZE
```

The command checks the manifest identity and every linked media file before
publishing `attempts/ATTEMPT_ID/review/index.html`. It needs no credentials or
ledger and makes no provider calls. It returns the page record as JSON with
exit code 0, or a safe error with exit code 1. Repeating the same command checks
and reuses the same page; it never replaces an existing page. Keep the run
folder private and retain the trusted manifest reference. The page reader does
not independently establish provider provenance. Browsers do not recheck file
hashes during playback, so rerun the command if local files might have changed.

To view one attempt, start this local server and open
`http://127.0.0.1:8765/review/index.html` in a browser. Stop the server with Ctrl-C.
The directory limit keeps other attempts and the run ledger outside this server.

```sh
python -m http.server 8765 --bind 127.0.0.1 \
  --directory runs/RUN_ID/attempts/ATTEMPT_ID
```

The page has no scripts or remote assets. Human instructions require two full
normal-speed plays, matched-frame inspection, independent scores, and annotated
hard failures. Both outcomes remain `NOT_RUN`; the page does not save or sign a
scorecard. Direct `file://` viewing is not tested because the managed browser
blocks that scheme. The supported viewing path is the loopback server above.

Optional browser verification uses a local Chromium executable, FFmpeg, and
separate test dependencies. It builds real Mode A and B manifests with mocked
provider calls and H.264/yuv420p fixtures, then checks the three native views,
playback, five sample groups, and requests limited to the local server. This
checks page behavior; it does not perform the human acceptance review.

```sh
python -m pip install -r requirements-ui.txt
GEN_REALITY_BROWSER_BIN=/usr/bin/chromium python -m unittest discover -s tests/ui -v
```

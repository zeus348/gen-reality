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
Content validation alone does not prove a real provider generation or visual
quality.

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
be checked again at the final send boundary; that integration is still pending.

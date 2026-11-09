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
Provider submission, budget authorization, human review, UI, and full release
auditing are not implemented yet. No provider key is needed for this command.

The manifest is published last. If preparation fails while writing files, retain
the incomplete attempt and use a new attempt ID. Existing files are never replaced.

The first slice loads the locked version 1.4 configuration. It does not submit
provider tasks, authorize spending, or verify the SDK version.

Use Python 3.11 or later. Install the configuration validator:

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
the locked schema. It returns a new dictionary on each call. The SDK version and
provider URI remain unverified template values. Changing them requires a separate
schema update and review before provider integration.

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
Reservations do not authorize provider calls. Submission states are still
pending, and the offline CLI does not create approvals or record final charges.

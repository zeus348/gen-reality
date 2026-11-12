"""Plan review samples from existing frame timestamps without changing media."""

from bisect import bisect_left
from dataclasses import dataclass
import math

from features.output.validate import MAX_RAW_FRAMES, _check_origin


@dataclass(frozen=True)
class FrameMatch:
    """One source frame and its nearest raw frame, with original timestamps.

    raw_span_position is before_first, within_span, or after_last. It compares
    source elapsed time with raw elapsed frame times, not frame display intervals.
    within_declared_duration uses the media interval [0, duration). Container
    duration can include longer audio, so this flag does not prove video coverage.
    """

    requested_seconds: float
    source_index: int
    source_seconds: float
    raw_index: int
    raw_seconds: float
    offset_seconds: float
    raw_span_position: str
    within_declared_duration: bool


def _validate_timeline(times: tuple[float, ...], *, minimum_count: int,
                       maximum_count: int, duration_seconds: float,
                       minimum_seconds: float = 0.0) -> None:
    if type(times) is not tuple or not minimum_count <= len(times) <= maximum_count:
        raise ValueError("Frame timestamps must be a tuple with the allowed frame count.")
    if any(type(value) not in (int, float) or not minimum_seconds <= value <= duration_seconds
           or not math.isfinite(value) for value in times):
        raise ValueError("Frame timestamps must be finite numbers within the allowed time bounds.")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("Frame timestamps must increase strictly.")


def _nearest_index(times: tuple[float, ...], target_seconds: float) -> int:
    right = bisect_left(times, target_seconds)
    if right == 0:
        return 0
    if right == len(times):
        return right - 1
    left = right - 1
    if target_seconds - times[left] <= times[right] - target_seconds:
        return left
    return right


def _raw_elapsed_times(native_times: tuple[float, ...], duration_seconds: float,
                       origin_seconds: float = 0.0) -> tuple[float, ...]:
    """Validate native PTS and return elapsed times for comparison only."""
    _check_origin(origin_seconds)
    if (type(duration_seconds) not in (int, float)
            or not 0 < duration_seconds <= 20 or not math.isfinite(duration_seconds)):
        raise ValueError("Raw duration must be positive, finite, and at most twenty seconds.")
    if (type(native_times) is not tuple or not 1 <= len(native_times) <= MAX_RAW_FRAMES
            or any(type(value) not in (int, float) for value in native_times)):
        raise ValueError("Raw timestamps must be a bounded tuple of numbers.")
    try:
        elapsed = tuple(value - origin_seconds for value in native_times)
    except OverflowError:
        raise ValueError("Raw timestamps exceed the supported numeric range.") from None
    _validate_timeline(elapsed, minimum_count=1, maximum_count=MAX_RAW_FRAMES,
                       duration_seconds=duration_seconds)
    return elapsed


def plan_review_frames(source_frame_seconds: tuple[float, ...],
                       raw_frame_seconds: tuple[float, ...], *,
                       raw_duration_seconds: float,
                       raw_timeline_origin_seconds: float = 0.0) -> tuple[FrameMatch, ...]:
    """Match source inspection times 0 through 4 to nearest decoded raw frames.

    Pass the validated normalized source timeline and RawVideo's timeline and
    duration and declared container origin. Check numeric bounds here; media
    validation belongs to the caller. Saved native timestamps remain unchanged.
    Use actual source timestamps for raw selection and choose the earlier frame
    on an exact tie. A positive first raw timestamp is valid. A final timestamp
    does not mark playback end. Targets outside duration still get a nearest
    existing frame, with an explicit duration flag. This plan assigns no verdict.
    """
    elapsed = _raw_elapsed_times(raw_frame_seconds, raw_duration_seconds, raw_timeline_origin_seconds)
    # Normalized source validation permits a 0.00001-second error at time zero.
    _validate_timeline(source_frame_seconds, minimum_count=120, maximum_count=120,
                       duration_seconds=5.0, minimum_seconds=-0.00001)
    matches = []
    for requested_seconds in (0.0, 1.0, 2.0, 3.0, 4.0):
        source_index = _nearest_index(source_frame_seconds, requested_seconds)
        source_seconds = source_frame_seconds[source_index]
        raw_index = _nearest_index(elapsed, source_seconds)
        raw_seconds = raw_frame_seconds[raw_index]
        if source_seconds < elapsed[0]:
            position = "before_first"
        elif source_seconds > elapsed[-1]:
            position = "after_last"
        else:
            position = "within_span"
        matches.append(FrameMatch(requested_seconds, source_index, source_seconds,
                                  raw_index, raw_seconds, elapsed[raw_index] - source_seconds,
                                  position, 0 <= source_seconds < raw_duration_seconds))
    return tuple(matches)

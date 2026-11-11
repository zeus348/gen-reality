"""Check frame selection with synthetic timelines and no provider access."""

import unittest

from features.output import align


SOURCE_TIMES = tuple(index / 24 for index in range(120))
RAW_TIMES = tuple(index / 30 for index in range(150))


class AlignmentTests(unittest.TestCase):
    def plan(self, source=SOURCE_TIMES, raw=RAW_TIMES, duration=5.0):
        return align.plan_review_frames(source, raw, raw_duration_seconds=duration)

    def test_24_fps_source_matches_30_fps_raw_without_changing_timestamps(self):
        matches = self.plan()
        self.assertEqual(tuple(item.requested_seconds for item in matches), (0, 1, 2, 3, 4))
        self.assertEqual(tuple(item.source_index for item in matches), (0, 24, 48, 72, 96))
        self.assertEqual(tuple(item.raw_index for item in matches), (0, 30, 60, 90, 120))
        self.assertEqual(tuple(item.source_seconds for item in matches), (0, 1, 2, 3, 4))
        self.assertEqual(tuple(item.raw_seconds for item in matches), (0, 1, 2, 3, 4))
        self.assertTrue(all(item.offset_seconds == 0 for item in matches))
        self.assertTrue(all(item.raw_span_position == "within_span" for item in matches))
        self.assertTrue(all(item.within_declared_duration for item in matches))

    def test_irregular_timestamps_use_nearest_frame_and_choose_earlier_exact_tie(self):
        raw = (0.0, 0.25, 0.75, 1.25, 1.8, 2.4, 2.9, 3.875, 4.2)
        matches = self.plan(raw=raw)
        self.assertEqual(tuple(item.raw_index for item in matches), (0, 2, 4, 6, 7))
        self.assertEqual(tuple(item.raw_seconds for item in matches), (0.0, 0.75, 1.8, 2.9, 3.875))
        for item, expected in zip(matches, (0.0, -0.25, -0.2, -0.1, -0.125)):
            self.assertAlmostEqual(item.offset_seconds, expected)

    def test_actual_source_pts_controls_selection_offset_and_duration_flag(self):
        source = list(SOURCE_TIMES)
        source[24] = 1.000001
        raw = (0.0, 0.9999997, 1.0000006)
        sample = self.plan(source=tuple(source), raw=raw, duration=1.0000008)[1]
        self.assertEqual((sample.requested_seconds, sample.source_index, sample.source_seconds),
                         (1.0, 24, 1.000001))
        self.assertEqual((sample.raw_index, sample.raw_seconds), (2, 1.0000006))
        self.assertAlmostEqual(sample.offset_seconds, -0.0000004, places=12)
        self.assertEqual(sample.raw_span_position, "after_last")
        self.assertFalse(sample.within_declared_duration)

    def test_positive_first_raw_pts_is_preserved(self):
        raw = (0.033, 1.033, 2.033, 3.033, 4.033)
        sample = self.plan(raw=raw)[0]
        self.assertEqual((sample.raw_index, sample.raw_seconds, sample.offset_seconds), (0, 0.033, 0.033))
        self.assertEqual(sample.raw_span_position, "before_first")
        self.assertTrue(sample.within_declared_duration)

    def test_valid_small_negative_source_pts_is_preserved(self):
        source = (-0.000001,) + SOURCE_TIMES[1:]
        sample = self.plan(source=source)[0]
        self.assertEqual((sample.requested_seconds, sample.source_index, sample.source_seconds),
                         (0.0, 0, -0.000001))
        self.assertEqual((sample.raw_index, sample.raw_seconds, sample.offset_seconds), (0, 0.0, 0.000001))
        self.assertEqual(sample.raw_span_position, "before_first")
        self.assertFalse(sample.within_declared_duration)

    def test_target_after_last_pts_can_be_within_declared_duration(self):
        sample = self.plan(raw=(0.0, 1.0, 2.0, 3.0, 3.967))[4]
        self.assertEqual((sample.raw_index, sample.raw_seconds), (4, 3.967))
        self.assertAlmostEqual(sample.offset_seconds, -0.033)
        self.assertEqual(sample.raw_span_position, "after_last")
        self.assertTrue(sample.within_declared_duration)

    def test_targets_at_or_beyond_duration_keep_nearest_frame_with_false_flag(self):
        matches = self.plan(raw=(0.0, 1.0, 2.5, 3.0), duration=3.0)
        at_end, beyond_end = matches[3:]
        self.assertEqual((at_end.raw_index, at_end.raw_seconds, at_end.offset_seconds), (3, 3.0, 0.0))
        self.assertEqual(at_end.raw_span_position, "within_span")
        self.assertFalse(at_end.within_declared_duration)
        self.assertEqual((beyond_end.raw_index, beyond_end.raw_seconds, beyond_end.offset_seconds), (3, 3.0, -1.0))
        self.assertEqual(beyond_end.raw_span_position, "after_last")
        self.assertFalse(beyond_end.within_declared_duration)

    def test_raw_frame_count_boundaries_remain_valid(self):
        single = self.plan(raw=(0.5,))
        self.assertEqual(tuple(item.raw_index for item in single), (0, 0, 0, 0, 0))
        self.assertEqual(tuple(item.raw_seconds for item in single), (0.5, 0.5, 0.5, 0.5, 0.5))
        maximum = self.plan(raw=tuple(index / 120 for index in range(2400)), duration=20.0)
        self.assertEqual(tuple(item.raw_index for item in maximum), (0, 120, 240, 360, 480))

    def test_invalid_timeline_types_and_frame_counts_are_rejected(self):
        for field in ("source", "raw"):
            for value in (None, True, "0,1,2", [], (), [0.0]):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.plan(**{field: value})
        for source in (SOURCE_TIMES[:-1], SOURCE_TIMES + (5.0,)):
            with self.subTest(source_count=len(source)), self.assertRaises(ValueError):
                self.plan(source=source)
        with self.assertRaises(ValueError):
            self.plan(raw=tuple(index / 480 for index in range(2401)))

    def test_invalid_timestamp_values_and_order_are_rejected(self):
        for field, original in (("source", SOURCE_TIMES), ("raw", RAW_TIMES)):
            invalid = [(0, False), (original.index(1.0), True), (0, -0.1), (0, -float("inf"))]
            invalid += [(len(original) - 1, value) for value in
                        (None, "1", float("nan"), float("inf"), 5.1, 10 ** 1000)]
            for index, value in invalid:
                changed = list(original)
                changed[index] = value
                with self.subTest(field=field, value=str(value)[:20]), self.assertRaises(ValueError):
                    self.plan(**{field: tuple(changed)})
            for value in (original[0], original[3]):
                changed = list(original)
                changed[1] = value
                with self.subTest(field=field, order=value), self.assertRaises(ValueError):
                    self.plan(**{field: tuple(changed)})
        with self.assertRaises(ValueError):
            self.plan(raw=(-0.000001,) + RAW_TIMES[1:])

    def test_invalid_durations_are_rejected(self):
        for value in (True, False, None, "5", 0, -1, 20.1, float("nan"),
                      float("inf"), -float("inf"), 10 ** 1000):
            with self.subTest(duration=str(value)[:20]), self.assertRaises(ValueError):
                self.plan(raw=(0.0,), duration=value)

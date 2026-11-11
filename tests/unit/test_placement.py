"""Check the placement contract without media or provider access."""

from dataclasses import FrozenInstanceError
import unittest

from features.placement.math import Placement, SpriteBounds, sprite_bounds


class PlacementTests(unittest.TestCase):
    def test_center_and_aspect_ratio(self):
        bounds = sprite_bounds(Placement(0.5, 0.5, 0.25), 200, 100)
        self.assertEqual(bounds, SpriteBounds(460, 270, 360, 180))

    def test_edge_placements_preserve_unclipped_rectangle(self):
        self.assertEqual(sprite_bounds(Placement(0, 0, 1), 100, 100),
                         SpriteBounds(-360, -360, 720, 720))
        self.assertEqual(sprite_bounds(Placement(1, 1, 1), 100, 100),
                         SpriteBounds(920, 360, 720, 720))

    def test_rounding_and_small_sizes_are_deterministic(self):
        placement = Placement(0.5, 0.5, 1e-300)
        expected = SpriteBounds(640, 360, 1, 1)
        self.assertEqual(sprite_bounds(placement, 1, 100), expected)
        self.assertEqual(sprite_bounds(placement, 1, 100), expected)
        self.assertEqual(sprite_bounds(Placement(0.5, 0.5, 0.25), 1, 1, 10, 10),
                         SpriteBounds(4, 4, 2, 2))

    def test_invalid_fractions_are_rejected(self):
        for field in ["u", "v", "height_frac"]:
            for value in [float("nan"), float("inf"), -float("inf"), -0.1,
                          1.1, True, None, "0.5", 10 ** 1000]:
                with self.subTest(field=field, value=str(value)[:20]):
                    values = {"u": 0.5, "v": 0.5, "height_frac": 0.25}
                    values[field] = value
                    with self.assertRaises(ValueError):
                        Placement(**values)

    def test_zero_height_and_rotation_are_rejected(self):
        with self.assertRaises(ValueError):
            Placement(0.5, 0.5, 0)
        for rotation in [1, True, "0", float("nan"), float("inf")]:
            with self.subTest(rotation=rotation):
                with self.assertRaises(ValueError):
                    Placement(0.5, 0.5, 0.25, rotation)

    def test_invalid_dimensions_are_rejected(self):
        for index in range(4):
            for value in [0, -1, 16_385, True, 1.5, "100", None]:
                with self.subTest(index=index, value=value):
                    dimensions = [100, 100, 1280, 720]
                    dimensions[index] = value
                    with self.assertRaises(ValueError):
                        sprite_bounds(Placement(0.5, 0.5, 0.25), *dimensions)

    def test_placement_is_immutable(self):
        placement = Placement(0.5, 0.5, 0.25)
        with self.assertRaises(FrozenInstanceError):
            placement.u = 0.9

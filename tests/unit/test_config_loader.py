"""Check offline configuration loading and rejection of unsafe input."""

import copy
from contextlib import chdir
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jsonschema import Draft202012Validator

from core.config import CONFIG_PATH, SCHEMA_PATH, ConfigError, load_config


class ConfigLoaderTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "settings.json"
        self.settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def write_settings(self, settings):
        self.path.write_text(json.dumps(settings), encoding="utf-8")

    def test_schema_is_valid(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

    def test_default_configuration_loads(self):
        self.assertEqual(load_config(), self.settings)

    def test_custom_path_loads_without_using_working_directory(self):
        self.write_settings(self.settings)
        with chdir(self.folder.name):
            self.assertEqual(load_config(self.path), self.settings)
        self.path.unlink()
        with self.assertRaises(ConfigError):
            load_config(self.path)

    def test_calls_return_independent_settings(self):
        first = load_config()
        first["sdk"]["createMaxRetries"] = 9
        self.assertEqual(load_config()["sdk"]["createMaxRetries"], 0)

    def test_missing_file_has_a_clear_error(self):
        with self.assertRaisesRegex(ConfigError, "cannot be read"):
            load_config(self.path)

    def test_invalid_text_is_rejected_without_echoing_values(self):
        for contents in [b'\xff', b'{"secret":"dummy-secret",', b'[' * 2000]:
            with self.subTest(contents=contents[:20]):
                self.path.write_bytes(contents)
                with self.assertRaises(ConfigError) as caught:
                    load_config(self.path)
                self.assertNotIn("dummy-secret", str(caught.exception))

    def test_duplicate_fields_are_rejected(self):
        contents = json.dumps(self.settings)
        contents = contents.replace('"createMaxRetries": 0',
                                    '"createMaxRetries": 5, "createMaxRetries": 0')
        self.path.write_text(contents, encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "duplicate fields"):
            load_config(self.path)

    def test_oversized_integer_has_a_clear_error(self):
        self.path.write_text('{"fixture":' + '9' * 5000 + '}', encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "valid UTF-8 JSON"):
            load_config(self.path)

    def test_nonstandard_numbers_are_rejected(self):
        for value in ["NaN", "Infinity", "-Infinity"]:
            with self.subTest(value=value):
                self.path.write_text('{"unexpected":' + value + '}', encoding="utf-8")
                with self.assertRaisesRegex(ConfigError, "standard JSON"):
                    load_config(self.path)

    def test_file_size_limit_includes_boundary(self):
        contents = json.dumps(self.settings).encode("utf-8")
        self.path.write_bytes(contents.ljust(65_536, b" "))
        self.assertEqual(load_config(self.path), self.settings)
        self.path.write_bytes(contents.ljust(65_537, b" "))
        with self.assertRaisesRegex(ConfigError, "exceeds 65536 bytes"):
            load_config(self.path)

    def test_invalid_root_types_are_rejected(self):
        for value in [None, [], "settings", 1, True]:
            with self.subTest(value=value):
                self.write_settings(value)
                with self.assertRaisesRegex(ConfigError, "at root"):
                    load_config(self.path)

    def test_missing_and_extra_fields_are_rejected(self):
        for section in [None, "sdk", "paidBudget", "review"]:
            for change in ["missing", "extra"]:
                with self.subTest(section=section, change=change):
                    settings = copy.deepcopy(self.settings)
                    target = settings if section is None else settings[section]
                    if change == "missing":
                        target.pop(next(iter(target)))
                    else:
                        target["dummy-secret"] = "dummy-secret"
                    self.write_settings(settings)
                    with self.assertRaises(ConfigError) as caught:
                        load_config(self.path)
                    self.assertNotIn("dummy-secret", str(caught.exception))

    def test_changed_settings_are_rejected_with_field_name(self):
        for section, field, value in [
            ("sdk", "createMaxRetries", 1),
            ("sdk", "createMaxRetries", False),
            ("sdk", "version", "unverified-release"),
            ("paidBudget", "maxUsd", "100.00"),
            ("paidBudget", "priceMustBeReverified", False),
            ("review", "humanReviewerRequired", False),
            ("fixture", "frames", 121),
        ]:
            with self.subTest(section=section, field=field, value=value):
                settings = copy.deepcopy(self.settings)
                settings[section][field] = value
                self.write_settings(settings)
                with self.assertRaises(ConfigError) as caught:
                    load_config(self.path)
                self.assertIn(f"{section}.{field}", str(caught.exception))
                self.assertNotIn("unverified-release", str(caught.exception))

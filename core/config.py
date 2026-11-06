"""Load the locked offline configuration. This does not authorize paid work."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "build_config.json"
SCHEMA_PATH = PROJECT_ROOT / "config" / "build_config.schema.json"
MAX_CONFIG_BYTES = 65_536


class ConfigError(ValueError):
    """The configuration cannot be read or does not match the locked settings."""


def _reject_constant(value):
    raise ConfigError("Configuration must use standard JSON values.")


def _unique_fields(pairs):
    fields = {}
    for name, value in pairs:
        if name in fields:
            raise ConfigError("Configuration must not contain duplicate fields.")
        fields[name] = value
    return fields


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read and validate settings. Return a fresh dictionary for each call.

    Version 1.4 is locked. Changes, including an SDK version pin, require a
    separate schema update and review. Never pass this template to a provider.
    """
    try:
        with Path(path).open("rb") as config_file:
            contents = config_file.read(MAX_CONFIG_BYTES + 1)
    except OSError:
        raise ConfigError("Configuration file cannot be read.") from None

    if len(contents) > MAX_CONFIG_BYTES:
        raise ConfigError("Configuration file exceeds 65536 bytes.")

    try:
        settings = json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=_unique_fields,
            parse_constant=_reject_constant,
        )
    except ConfigError:
        raise
    except (ValueError, RecursionError):
        raise ConfigError("Configuration must be valid UTF-8 JSON.") from None

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    error = next(validator.iter_errors(settings), None)
    if error is not None:
        # Read field names from the trusted schema, never from input values.
        schema_path = list(error.absolute_schema_path)
        fields = [
            schema_path[index + 1]
            for index, part in enumerate(schema_path[:-1])
            if part == "properties"
        ]
        location = ".".join(fields) or "root"
        raise ConfigError(f"Configuration at {location} must match locked version 1.4.")
    return settings

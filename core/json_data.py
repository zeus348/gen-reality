"""Shared checks for JSON read from configuration or a provider."""


def unique_fields(pairs):
    """Reject repeated decoded names when used as json.loads' object_pairs_hook."""
    fields = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("JSON must not contain duplicate fields.")
        fields[name] = value
    return fields

"""Fixed offline test commands; run evidence cannot supply commands."""

import json


PYTEST_VERSION = "9.2.0"
REPORTLOG_VERSION = "1.0.0"
DEFAULT_SUITES = ("unit", "integration_offline", "ui")
SUITES = {
    "unit": ("tests/unit", "not live_provider"),
    "integration_offline": ("tests/integration", "not live_provider"),
    "ui": ("tests/ui", "not live_provider"),
}
PYTEST_OPTIONS = (
    "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
    "-p", "pytest_reportlog.plugin", "-o", "addopts=",
    "-o", "junit_logging=all", "-o", "junit_log_passing_tests=true",
    "--strict-markers",
)
VIEWER_ONLY_TESTS = (
    "tests/ui/test_review_page_browser.py::ReviewPageBrowserTests::"
    "test_native_views_samples_and_playback_stay_within_one_local_attempt",
)


def registry_bytes() -> bytes:
    """Describe the trusted selection without assigning acceptance coverage."""
    data = {
        "schemaVersion": "1.4", "registryVersion": "offline-tests-1",
        "pytestVersion": PYTEST_VERSION, "reportlogVersion": REPORTLOG_VERSION,
        "options": list(PYTEST_OPTIONS),
        "suites": {name: {"path": path, "marker": marker}
                   for name, (path, marker) in SUITES.items()},
        "viewerOnlyTests": list(VIEWER_ONLY_TESTS),
        "scope": "Record available offline tests; acceptance coverage is a separate audit.",
    }
    return (json.dumps(data, sort_keys=True, indent=2) + "\n").encode("utf-8")

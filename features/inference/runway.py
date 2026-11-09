"""Use the pinned Runway SDK for one paid task-creation request."""

from importlib.metadata import version
import re
from typing import Callable

import httpx
from runwayml import RunwayML


SDK_VERSION = "5.24.0"
API_URL = "https://api.dev.runwayml.com"


class ProviderError(ValueError):
    """The adapter cannot create a task with the supplied inputs."""


def _require_upload_uri(uri):
    if type(uri) is not str or re.fullmatch(r"runway://[A-Za-z0-9._~/-]{1,2000}", uri) is None:
        raise ProviderError("Input must be a Runway upload URI.")


class RunwayCreateAdapter:
    """Trusted create adapter; call only through SubmissionCoordinator.

    The caller must verify account access, current prices, uploaded media, and
    frozen inputs first. This class does not grant spending authority. Save the
    actual request body before sending it; authentication headers are excluded.
    A custom transport is for controlled tests and must never retry requests.
    """

    def __init__(self, api_key: str, *, record_request: Callable[[bytes], None],
                 transport: httpx.BaseTransport | None = None):
        if version("runwayml") != SDK_VERSION:
            raise ProviderError("Install the pinned Runway SDK before creating tasks.")
        if type(api_key) is not str or not api_key.strip():
            raise ProviderError("A server-side Runway API key is required.")

        def record_body(request):
            record_request(request.read())

        http_client = httpx.Client(
            transport=transport, timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=False, event_hooks={"request": [record_body]},
        )
        try:
            self._client = RunwayML(api_key=api_key, base_url=API_URL, max_retries=0,
                                    timeout=httpx.Timeout(60.0, connect=10.0), http_client=http_client)
        except BaseException:
            http_client.close()
            raise

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def create_task(self, video_uri: str, prompt: str, *, keyframe_uri: str | None = None) -> str:
        """Mode A supplies a first-frame URI. Mode B leaves that argument out."""
        _require_upload_uri(video_uri)
        if type(prompt) is not str or not prompt.strip() or len(prompt) > 1000:
            raise ProviderError("Prompt must contain 1 to 1000 characters.")
        arguments = {"model": "aleph2", "video_uri": video_uri, "prompt_text": prompt}
        if keyframe_uri is not None:
            _require_upload_uri(keyframe_uri)
            arguments["keyframes"] = [{"uri": keyframe_uri, "seconds": 0}]
        try:
            response = self._client.video_to_video.create(**arguments)
            if type(response.id) is not str or not response.id:
                raise ProviderError("Provider task ID is missing.")
            return response.id
        except Exception:
            # Do not expose SDK errors that can contain response bodies or secrets.
            raise ProviderError("Task creation is uncertain; do not repeat the request.") from None

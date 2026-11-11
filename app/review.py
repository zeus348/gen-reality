"""Create a local review page without provider credentials or paid work."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from core.storage import StoredArtifact
from features.output.view import publish_review_page


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--review-path", required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--review-size", type=int, required=True)
    arguments = parser.parse_args(argv)
    try:
        record = StoredArtifact(arguments.review_path, arguments.review_sha256, arguments.review_size)
        page = publish_review_page(arguments.output_root, record,
                                   run_id=arguments.run_id, attempt_id=arguments.attempt_id)
        result = {"runId": arguments.run_id, "attemptId": arguments.attempt_id, "page": asdict(page)}
    except Exception:
        print("Review page could not be verified or published for this attempt.", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

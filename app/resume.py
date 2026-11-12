"""Inspect or resume one saved attempt without creating another generation."""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
import time

import httpx

from core.budget import BudgetError, BudgetLedger
from features.inference.retrieve import retrieve_task_once
from features.inference.runtime import RuntimeEvidence
from features.inference.submission import SubmissionCoordinator, SubmissionError
from features.output.complete import CompletedOutput, OutputCompleter, OutputError
from features.output.fetch import download_task_output


@dataclass(frozen=True)
class ResumeResult:
    run_id: str
    attempt_id: str
    state: str
    provider_task_id: str | None
    provider_status: str | None
    poll_status: str
    output: CompletedOutput | None


def resume_attempt(output_root: Path, ledger_path: Path, *, run_id: str, attempt_id: str,
                   output_hosts: frozenset[str], api_key_env: str = "RUNWAY_API_KEY",
                   status_only: bool = False, clock=time.time,
                   task_transport: httpx.BaseTransport | None = None,
                   output_transport: httpx.BaseTransport | None = None,
                   runtime: RuntimeEvidence | None = None) -> ResumeResult:
    """Poll a known task once, then validate and save its output after success.

    The trusted operator supplies exact output hosts. A fresh task GET can
    replace an expired output URL with new evidence. No create-time evidence or
    budget deadline blocks recovery. Status inspection makes no provider calls;
    opening the coordinator still recovers abandoned claims as UNKNOWN.
    """
    BudgetLedger._run_id(run_id)
    BudgetLedger._attempt_id(attempt_id)
    if runtime is not None:
        runtime.require_identity(output_root, run_id, attempt_id)
    if not Path(ledger_path).is_file():
        raise BudgetError("Resume requires an existing budget database.")
    ledger = BudgetLedger(ledger_path, clock=clock)
    with SubmissionCoordinator(ledger) as coordinator:
        with ledger._connection() as connection:
            row = connection.execute("SELECT run_id FROM reservations WHERE attempt_id=?", (attempt_id,)).fetchone()
        if row is None or row["run_id"] != run_id:
            raise SubmissionError("Attempt does not belong to this run.")
        completer = OutputCompleter(coordinator, output_root)
        poller = completer.poller

        def snapshot():
            job = coordinator.get(attempt_id)
            observed = poller.get(attempt_id)
            output = completer.get(attempt_id)
            if job.state == "SUCCEEDED" and output is None:
                raise OutputError("Completed attempt has no verified output record.")
            return ResumeResult(run_id, attempt_id, job.state, job.provider_task_id,
                                observed.provider_status, observed.poll_status, output)

        current = snapshot()
        if runtime is not None:
            runtime.bind_existing(coordinator)
        if status_only or current.state not in ("QUEUED", "RUNNING"):
            return current
        if type(api_key_env) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", api_key_env) is None:
            raise ValueError("Supply an environment variable name for the API key.")
        api_key = os.environ.get(api_key_env, "")
        if not api_key.strip():
            raise ValueError("The named environment variable must contain a Runway API key.")
        evidence_root = Path(output_root) / run_id
        observed = poller.poll_once(attempt_id, lambda task_id: retrieve_task_once(
            task_id, api_key, evidence_root, transport=task_transport, runtime=runtime),
            deadline_unix=ledger._now() + 10)
        if observed.provider_status == "SUCCEEDED" and observed.poll_status == "OBSERVED":
            completer.complete(attempt_id, lambda task_id, evidence_hash: download_task_output(
                task_id, evidence_hash, evidence_root, allowed_hosts=output_hosts,
                transport=output_transport, runtime=runtime))
        return snapshot()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--api-key-env", default="RUNWAY_API_KEY", help="Environment variable name; never the key value.")
    parser.add_argument("--output-host", action="append", default=[],
                        help="Exact trusted output host; repeat for each approved host.")
    parser.add_argument("--status-only", action="store_true", help="Inspect local state without provider calls.")
    arguments = parser.parse_args(argv)
    try:
        runtime = RuntimeEvidence(arguments.output_root, arguments.run_id, arguments.attempt_id)
        result = resume_attempt(arguments.output_root, arguments.ledger, run_id=arguments.run_id,
            attempt_id=arguments.attempt_id, api_key_env=arguments.api_key_env,
            output_hosts=frozenset(arguments.output_host), status_only=arguments.status_only, runtime=runtime)
    except Exception:
        print("Resume could not proceed. Check the saved attempt, credentials, and trusted output hosts.",
              file=sys.stderr)
        return 1
    print(json.dumps(asdict(result) | {"runtime": runtime.summary()}, sort_keys=True))
    return 0 if arguments.status_only or result.state == "SUCCEEDED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

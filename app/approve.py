"""Allocate existing one-run authority to a registered local experiment."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import time

from app.operator_config import read_operator_config
from app.preparation_state import check_run_id, read_registered_run_start
from core.budget import BudgetError, BudgetLedger
from core.run_snapshot import ApprovedRunRow, read_run_snapshot
from core.run_start import MAX_UNIX_SECONDS, read_run_start
from core.storage import StoredArtifact
from features.inference.pricing import PricingError, read_price_quote


_REQUIRED_TABLES = frozenset({"approved_runs", "reservations", "actual_charges",
                              "submission_jobs", "submission_events", "run_events"})


class ApprovalError(ValueError):
    """Return a fixed code without private paths or database exception details."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ApprovalResult:
    run_id: str
    status: str
    approval: ApprovedRunRow
    run_start: StoredArtifact
    price_quote: StoredArtifact

    def record(self):
        """The selected quote is evidence, not a claim about the original quote."""
        return {"runId": self.run_id, "status": self.status,
                "limitCents": self.approval.limit_cents,
                "maxAttempts": self.approval.max_attempts,
                "costCents": self.approval.cost_cents,
                "startedUnix": self.approval.started_unix,
                "deadlineUnix": self.approval.deadline_unix,
                "runStart": asdict(self.run_start), "priceQuote": asdict(self.price_quote)}


def _read_approval(ledger_path, run_id):
    # The server checks this ownership too. This command can run on its own.
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            info = Path(str(ledger_path) + suffix).lstat()
        except FileNotFoundError:
            if suffix:
                continue
            raise
        if info.st_uid != os.getuid():
            raise ApprovalError("STATE_UNAVAILABLE")
    snapshot = read_run_snapshot(ledger_path, run_id=run_id)
    if (_REQUIRED_TABLES.intersection(snapshot.missing_tables)
            or snapshot.relationship_issues or len(snapshot.approved_runs) > 1):
        raise ApprovalError("STATE_UNAVAILABLE")
    return snapshot.approved_runs[0] if snapshot.approved_runs else None


def _require_matching(approval, config, start, quote):
    times = (approval.started_unix, approval.deadline_unix, approval.last_checked_unix)
    if (approval.run_id != start.run_id or approval.approved_by != config.approved_by
            or approval.limit_cents != 1000 or approval.max_attempts != 6
            or approval.cost_cents != quote.reserved_cents
            or any(type(value) not in (int, float) or not 0 <= value <= MAX_UNIX_SECONDS
                   or not math.isfinite(value) for value in times)
            or not start.started_unix <= approval.started_unix < start.deadline_unix
            or approval.deadline_unix != start.deadline_unix
            or approval.last_checked_unix < approval.started_unix
            or approval.time_stopped not in (0, 1)):
        raise ApprovalError("APPROVAL_MISMATCH")


def approve_registered_run(state_dir: Path, run_id: str, *, clock=time.time) -> ApprovalResult:
    """Use the same dedicated private ledger as the running local server.

    The read-only precheck validates fixed table columns, not all ledger DDL.
    The operator owns the directory and database. A retained result describes
    historical terms; it does not renew permission or certify a global registry.
    """
    try:
        check_run_id(run_id)
        config = read_operator_config(state_dir)
        state_dir = Path(os.path.abspath(state_dir))
    except Exception:
        raise ApprovalError("CONFIG_INVALID") from None
    ledger_path = state_dir / "budget.sqlite"
    try:
        registered = read_registered_run_start(
            state_dir, config.output_root, ledger_path, run_id=run_id)
        start = read_run_start(config.output_root, run_id=run_id)
        if start.record != registered:
            raise ApprovalError("STATE_UNAVAILABLE")
        approval = _read_approval(ledger_path, run_id)
    except Exception:
        raise ApprovalError("STATE_UNAVAILABLE") from None
    try:
        quote = read_price_quote(config.output_root / run_id, config.price_quote)
    except Exception:
        raise ApprovalError("PRICE_INVALID") from None
    if approval is not None:
        _require_matching(approval, config, start, quote)
        return ApprovalResult(run_id, "ALREADY_APPROVED", approval, registered, quote.record)

    def approval_clock():
        now = clock()
        if (type(now) not in (int, float) or not 0 <= now <= MAX_UNIX_SECONDS
                or not math.isfinite(now) or not start.started_unix <= now < start.deadline_unix):
            raise ApprovalError("APPROVAL_TIME_INVALID")
        try:
            quote.require_current(now)
        except PricingError:
            raise ApprovalError("PRICE_STALE") from None
        return now

    try:
        ledger = BudgetLedger(ledger_path, clock=approval_clock)
        ledger.approve_run(run_id, approved_by=config.approved_by, limit_cents=1000,
                           max_attempts=6, cost_cents=quote.reserved_cents,
                           max_elapsed_minutes=180, deadline_unix_cap=start.deadline_unix,
                           only_run=True)
        approval = _read_approval(ledger_path, run_id)
        if approval is None:
            raise ApprovalError("STATE_UNAVAILABLE")
        _require_matching(approval, config, start, quote)
        return ApprovalResult(run_id, "APPROVED", approval, registered, quote.record)
    except ApprovalError:
        raise
    except BudgetError:
        raise ApprovalError("APPROVAL_DENIED") from None
    except Exception:
        raise ApprovalError("APPROVAL_UNAVAILABLE") from None


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ApprovalError("INVALID_ARGUMENTS")


def main(argv=None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    try:
        arguments = parser.parse_args(argv)
        result = approve_registered_run(arguments.state_dir, arguments.run_id)
        print(json.dumps(result.record(), sort_keys=True, allow_nan=False))
        return 0 if result.status == "APPROVED" else 2
    except ApprovalError as error:
        print(json.dumps({"error": error.code}, sort_keys=True), file=sys.stderr)
        return 1
    except Exception:
        print('{"error": "APPROVAL_UNAVAILABLE"}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

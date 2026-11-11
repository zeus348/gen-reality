"""Record a trusted price check and calculate a conservative five-second quote."""

from dataclasses import asdict, dataclass
from decimal import Context, Decimal, ROUND_CEILING, localcontext
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time
from typing import Callable

from core.storage import ArtifactExistsError, StoredArtifact, publish_bytes, read_verified


PRICE_SOURCE_URL = "https://docs.dev.runwayml.com/guides/pricing/"
PRICE_MAX_AGE_SECONDS = 300.0
MAX_PRICE_EVIDENCE_BYTES = 2 * 1024 * 1024
CLIP_SECONDS = 5


class PricingError(ValueError):
    """A price check cannot support the requested reservation."""


def _require_current(checked_unix: float, now_unix: float) -> None:
    if (type(now_unix) not in (int, float) or not 0 <= now_unix <= 253_402_300_799
            or not math.isfinite(now_unix) or not 0 <= now_unix - checked_unix <= PRICE_MAX_AGE_SECONDS):
        raise PricingError("Price check is stale or future-dated; check pricing again.")


@dataclass(frozen=True)
class PriceQuote:
    checked_unix: float
    required_credits: int
    reserved_cents: int
    source: StoredArtifact
    record: StoredArtifact

    def require_current(self, now_unix: float) -> None:
        _require_current(self.checked_unix, now_unix)


def record_price_quote(evidence_root: Path, source: StoredArtifact, *, checked_by: str,
                        checked_unix: float, credits_per_second: int, minimum_credits: int,
                        usd_per_credit: str, maximum_additional_cents: int,
                        clock: Callable[[], float] = time.time) -> PriceQuote:
    """Use operator-verified price facts; never accept them from a client.

    The operator must check current official pricing, the minimum charge, and
    all applicable additional charges, and retain the supporting evidence.
    The source hash identifies those bytes; it does not verify their origin or
    interpretation. No rates or fee allowances have defaults. This function
    fetches no prices, grants no approval, and records no actual charge.
    """
    if type(checked_by) is not str or not checked_by.strip() or len(checked_by) > 200:
        raise PricingError("Price evidence needs an operator identity.")
    if (type(checked_unix) not in (int, float) or not 0 <= checked_unix <= 253_402_300_799
            or not math.isfinite(checked_unix)):
        raise PricingError("Price check time must be a valid UTC timestamp.")
    for value, lower, upper in [(credits_per_second, 1, 1_000_000),
                                (minimum_credits, 0, 1_000_000),
                                (maximum_additional_cents, 0, 1000)]:
        if type(value) is not int or not lower <= value <= upper:
            raise PricingError("Price counts and additional cents must be integers within the supported limits.")
    if (type(usd_per_credit) is not str
            or re.fullmatch(r"(?:0|[1-9][0-9]{0,3})(?:\.[0-9]{1,8})?", usd_per_credit) is None):
        raise PricingError("USD per credit must be a decimal string with at most eight fractional digits.")
    required = max(CLIP_SECONDS * credits_per_second, minimum_credits)
    # Bound input digits above; isolate arithmetic from the caller's decimal settings.
    with localcontext(Context(prec=40, rounding=ROUND_CEILING, Emin=-999, Emax=999)):
        unit_price = Decimal(usd_per_credit)
        if unit_price <= 0:
            raise PricingError("USD per credit must be positive.")
        base_cents = int((required * unit_price * 100).to_integral_value(rounding=ROUND_CEILING))
    reserved = base_cents + maximum_additional_cents
    if not 1 <= reserved <= 1000:
        raise PricingError("The conservative quote exceeds the P0 limit of 1000 cents.")
    if (type(source) is not StoredArtifact or type(source.size) is not int
            or not 0 < source.size <= MAX_PRICE_EVIDENCE_BYTES):
        raise PricingError("Price evidence must contain 1 byte to 2 MiB.")
    try:
        read_verified(evidence_root, source)
        # Reject stale facts before publishing a new record; do not refresh their check time.
        _require_current(checked_unix, clock())
        details = {"schemaVersion": "1.4", "manifestType": "operator_price_quote", "model": "aleph2",
                   "clipSeconds": CLIP_SECONDS, "priceSourceUrl": PRICE_SOURCE_URL,
                   "checkedBy": checked_by, "checkedAtUnix": checked_unix,
                   "creditsPerSecond": credits_per_second, "minimumCredits": minimum_credits,
                   "usdPerCredit": usd_per_credit, "requiredCredits": required,
                   "baseCents": base_cents, "maximumAdditionalCents": maximum_additional_cents,
                   "reservedCents": reserved, "source": asdict(source), "spendingAuthorized": False}
        body = (json.dumps(details, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        digest = sha256(body).hexdigest()
        path = f"provider_pricing/{digest}.json"
        try:
            record = publish_bytes(evidence_root, path, body)
        except ArtifactExistsError:
            record = StoredArtifact(path, digest, len(body))
            read_verified(evidence_root, record)
        return PriceQuote(checked_unix, required, reserved, source, record)
    except PricingError:
        raise
    except Exception:
        raise PricingError("Price evidence could not be verified or recorded.") from None

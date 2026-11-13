"""Read historical price evidence without refreshing or authorizing it."""

from dataclasses import FrozenInstanceError, replace
from decimal import Context, Decimal, localcontext
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, call, patch

from core.storage import StorageError, StoredArtifact, publish_bytes, read_verified
from features.inference import pricing


class PriceQuoteReaderTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = publish_bytes(self.root, "evidence/price.txt", b"Offline price facts.\n")

    def details(self, **changes):
        return {
            "schemaVersion": "1.4", "manifestType": "operator_price_quote",
            "model": "aleph2", "clipSeconds": 5,
            "priceSourceUrl": "https://docs.dev.runwayml.com/guides/pricing/",
            "checkedBy": "fixture operator", "checkedAtUnix": 1000,
            "creditsPerSecond": 28, "minimumCredits": 56, "usdPerCredit": "0.01",
            "requiredCredits": 140, "baseCents": 140, "maximumAdditionalCents": 0,
            "reservedCents": 140,
            "source": {"path": self.source.path, "sha256": self.source.sha256,
                       "size": self.source.size},
            "spendingAuthorized": False,
        } | changes

    def save_body(self, body):
        digest = sha256(body).hexdigest()
        return publish_bytes(self.root, f"provider_pricing/{digest}.json", body)

    def save_details(self, details):
        body = (json.dumps(details, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        return self.save_body(body)

    def historical_bytes(self, timestamp):
        # The literal layout is independent of the writer and reader serializers.
        return f'''{{
  "baseCents": 140,
  "checkedAtUnix": {timestamp},
  "checkedBy": "fixture operator",
  "clipSeconds": 5,
  "creditsPerSecond": 28,
  "manifestType": "operator_price_quote",
  "maximumAdditionalCents": 0,
  "minimumCredits": 56,
  "model": "aleph2",
  "priceSourceUrl": "https://docs.dev.runwayml.com/guides/pricing/",
  "requiredCredits": 140,
  "reservedCents": 140,
  "schemaVersion": "1.4",
  "source": {{
    "path": "evidence/price.txt",
    "sha256": "{self.source.sha256}",
    "size": {self.source.size}
  }},
  "spendingAuthorized": false,
  "usdPerCredit": "0.01"
}}
'''.encode()

    def write_quote(self, **changes):
        facts = {"checked_by": "fixture operator", "checked_unix": 1000,
                 "credits_per_second": 28, "minimum_credits": 56,
                 "usd_per_credit": "0.01", "maximum_additional_cents": 0,
                 "clock": lambda: 1001}
        return pricing.record_price_quote(self.root, self.source, **(facts | changes))

    def test_literal_integer_and_float_records_read_without_clock_or_publication(self):
        records = [(self.save_body(self.historical_bytes(token)), expected)
                   for token, expected in [("1000", 1000), ("1000.25", 1000.25)]]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.root.rglob("*") if p.is_file()}
        with patch.object(pricing, "publish_bytes", side_effect=StorageError("unexpected write")) as publish, \
                patch.object(pricing.time, "time", side_effect=ValueError("no current clock")) as clock, \
                patch.object(pricing, "_require_current", side_effect=pricing.PricingError("stale")) as freshness:
            for record, expected in records:
                with self.subTest(timestamp=expected):
                    quote = pricing.read_price_quote(self.root, record)
                    self.assertEqual((quote.checked_unix, quote.required_credits, quote.reserved_cents),
                                     (expected, 140, 140))
                    self.assertIs(type(quote.checked_unix), type(expected))
                    self.assertEqual((quote.source, quote.record), (self.source, record))
                    with self.assertRaises(FrozenInstanceError):
                        quote.reserved_cents = 1
            publish.assert_not_called()
            clock.assert_not_called()
            freshness.assert_not_called()
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(after, before)
        with self.assertRaises(pricing.PricingError):
            quote.require_current(1300.251)

    def test_writer_keeps_literal_historical_bytes_and_timestamp_types(self):
        for token, value in [("1000", 1000), ("1000.25", 1000.25)]:
            with self.subTest(timestamp=token):
                expected = self.historical_bytes(token)
                quote = self.write_quote(checked_unix=value)
                self.assertEqual(read_verified(self.root, quote.record), expected)
                self.assertEqual(quote.record, StoredArtifact(
                    f"provider_pricing/{sha256(expected).hexdigest()}.json",
                    sha256(expected).hexdigest(), len(expected)))

    def test_writer_checks_inputs_and_source_before_freshness(self):
        clock = Mock(return_value=1301)
        with patch.object(pricing, "read_verified", side_effect=StorageError("private path")) as read:
            with self.assertRaises(pricing.PricingError):
                self.write_quote(checked_by="", clock=clock)
            read.assert_not_called()
            clock.assert_not_called()
            with self.assertRaisesRegex(pricing.PricingError, "could not be verified or recorded"):
                self.write_quote(clock=clock)
            read.assert_called_once_with(self.root, self.source)
            clock.assert_not_called()
        with patch.object(pricing, "read_verified", wraps=read_verified) as read:
            with self.assertRaisesRegex(pricing.PricingError, "stale"):
                self.write_quote(clock=clock)
            read.assert_called_once_with(self.root, self.source)
            clock.assert_called_once_with()
        self.assertFalse((self.root / "provider_pricing").exists())

    def test_reader_recalculates_minimum_rounding_and_budget_in_hostile_decimal_context(self):
        cases = [
            ({"minimumCredits": 200, "requiredCredits": 200, "baseCents": 200,
              "reservedCents": 200}, 200, 200),
            ({"usdPerCredit": "0.01000001", "baseCents": 141,
              "maximumAdditionalCents": 31, "reservedCents": 172}, 140, 172),
            ({"creditsPerSecond": 1, "minimumCredits": 0, "requiredCredits": 5,
              "usdPerCredit": "0.00000001", "baseCents": 1, "reservedCents": 1}, 5, 1),
            ({"maximumAdditionalCents": 860, "reservedCents": 1000}, 140, 1000),
        ]
        for changes, credits, cents in cases:
            with self.subTest(changes=changes):
                record = self.save_details(self.details(**changes))
                context = Context(prec=1, Emin=0, Emax=1, clamp=1)
                for signal in context.traps:
                    context.traps[signal] = True
                with localcontext(context):
                    quote = pricing.read_price_quote(self.root, record)
                self.assertEqual((quote.required_credits, quote.reserved_cents), (credits, cents))
        self.assertEqual(Decimal("140") * Decimal("0.01000001") * 100, Decimal("140.00014000"))
        over = self.save_details(self.details(maximumAdditionalCents=861, reservedCents=1001))
        with self.assertRaises(pricing.PricingError):
            pricing.read_price_quote(self.root, over)

    def test_fixed_identity_fields_and_derived_values_are_not_trusted(self):
        fields = {
            "schemaVersion": ["1.3", 1.4], "manifestType": ["price_quote", None],
            "model": ["other", None], "clipSeconds": [4, 5.0, True],
            "priceSourceUrl": ["https://example.invalid/prices", ""],
            "spendingAuthorized": [True, 0, None],
            "requiredCredits": [139, 140.0, True], "baseCents": [139, 140.0, True],
            "reservedCents": [139, 140.0, True],
        }
        for field, values in fields.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    record = self.save_details(self.details(**{field: value}))
                    with self.assertRaises(pricing.PricingError):
                        pricing.read_price_quote(self.root, record)
        for field in self.details():
            with self.subTest(missing=field):
                details = self.details()
                del details[field]
                with self.assertRaises(pricing.PricingError):
                    pricing.read_price_quote(self.root, self.save_details(details))
        extra = self.save_details(self.details(actualChargeCents=0))
        with self.assertRaises(pricing.PricingError):
            pricing.read_price_quote(self.root, extra)

    def test_invalid_fact_types_and_limits_are_rejected(self):
        fields = {
            "checkedBy": [None, "", " ", "x" * 201],
            "checkedAtUnix": [True, -1, 253402300800, "1000"],
            "creditsPerSecond": [28.0, "28"], "minimumCredits": [56.0, True, -1, 1000001],
            "maximumAdditionalCents": [0.0, False, -1, 1001],
            "usdPerCredit": [0.01, "0", "-0.01", "+0.01", "01", "1e-2",
                             "NaN", "Infinity", "0.010000000", "10000"],
        }
        for field, values in fields.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    record = self.save_details(self.details(**{field: value}))
                    with self.assertRaises(pricing.PricingError):
                        pricing.read_price_quote(self.root, record)
        for credits in [True, 0, -1, 1000001]:
            # Keep otherwise-derived fields consistent, so a later mismatch cannot mask a bad count.
            required = max(5 * credits, 0)
            base = max(0, (required + 999999) // 1000000)
            details = self.details(creditsPerSecond=credits, minimumCredits=0,
                                   usdPerCredit="0.00000001", requiredCredits=required,
                                   baseCents=base, maximumAdditionalCents=1, reservedCents=base + 1)
            with self.subTest(credits=credits), self.assertRaises(pricing.PricingError):
                pricing.read_price_quote(self.root, self.save_details(details))

    def test_duplicate_nonfinite_encoding_and_noncanonical_json_are_rejected(self):
        original = self.historical_bytes("1000")
        bodies = [
            original.replace(b'"checkedBy": "fixture operator",',
                             b'"checkedBy": "fixture operator", "checkedBy": "fixture operator",'),
            original.replace(b'"path": "evidence/price.txt",',
                             b'"path": "evidence/price.txt", "path": "evidence/price.txt",'),
            *[original.replace(b'"checkedAtUnix": 1000', b'"checkedAtUnix": ' + token)
              for token in [b"NaN", b"Infinity", b"-Infinity", b"1e999"]],
            original.decode().encode("utf-16"), b"\xff" + original,
            original.rstrip(b"\n"), original + b" ",
            json.dumps(self.details(), separators=(",", ":")).encode(), b"[]", b"null",
        ]
        for body in bodies:
            with self.subTest(body=body[:80]):
                record = self.save_body(body)
                with self.assertRaises(pricing.PricingError):
                    pricing.read_price_quote(self.root, record)

    def test_quote_reference_is_canonical_bounded_and_hash_verified(self):
        body = self.historical_bytes("1000")
        record = self.save_body(body)
        invalid = [None, {"path": record.path, "sha256": record.sha256, "size": record.size},
                   replace(record, path="other/" + Path(record.path).name),
                   replace(record, path="provider_pricing/" + "0" * 64 + ".json"),
                   replace(record, path="../" + record.path),
                   replace(record, sha256=record.sha256.upper()),
                   replace(record, size=0), replace(record, size=True),
                   replace(record, size=65537)]
        for reference in invalid:
            with self.subTest(reference=reference):
                with patch.object(pricing, "read_verified", side_effect=StorageError("unexpected read")) as read:
                    with self.assertRaises(pricing.PricingError):
                        pricing.read_price_quote(self.root, reference)
                    read.assert_not_called()
        with patch.object(pricing, "read_verified", side_effect=StorageError("read at exact cap")) as read:
            with self.assertRaises(pricing.PricingError):
                pricing.read_price_quote(self.root, replace(record, size=65536))
            read.assert_called_once_with(self.root, replace(record, size=65536))
        with patch.object(pricing, "MAX_PRICE_RECORD_BYTES", len(body)):
            self.assertEqual(pricing.read_price_quote(self.root, record).record, record)
        with patch.object(pricing, "MAX_PRICE_RECORD_BYTES", len(body) - 1), \
                patch.object(pricing, "read_verified", side_effect=StorageError("unexpected read")) as read:
            with self.assertRaises(pricing.PricingError):
                pricing.read_price_quote(self.root, record)
            read.assert_not_called()
        (self.root / record.path).write_bytes(body.replace(b"fixture operator", b"altered operator"))
        self.assertEqual((self.root / record.path).stat().st_size, record.size)
        with self.assertRaises(pricing.PricingError):
            pricing.read_price_quote(self.root, record)
        (self.root / record.path).unlink()
        with self.assertRaises(pricing.PricingError):
            pricing.read_price_quote(self.root, record)

    def test_source_reference_fields_bytes_and_two_mebibyte_limit_are_checked(self):
        source = self.details()["source"]
        invalid = [None, [], source | {"extra": False}, {"path": source["path"]},
                   source | {"size": 0}, source | {"size": True},
                   source | {"size": 2 * 1024 * 1024 + 1},
                   source | {"sha256": "0" * 64}, source | {"path": "missing.txt"},
                   source | {"path": "../price.txt"}]
        for value in invalid:
            with self.subTest(source=value):
                record = self.save_details(self.details(source=value))
                with self.assertRaises(pricing.PricingError):
                    pricing.read_price_quote(self.root, record)
        large = publish_bytes(self.root, "evidence/exact-limit.txt", b"x" * (2 * 1024 * 1024))
        large_ref = {"path": large.path, "sha256": large.sha256, "size": large.size}
        record = self.save_details(self.details(source=large_ref))
        self.assertEqual(pricing.read_price_quote(self.root, record).source, large)
        too_large = self.save_details(self.details(source=large_ref | {"size": large.size + 1}))
        with patch.object(pricing, "read_verified", wraps=read_verified) as read:
            with self.assertRaises(pricing.PricingError):
                pricing.read_price_quote(self.root, too_large)
            self.assertEqual(read.call_args_list, [call(self.root, too_large)])
        original = (self.root / self.source.path).read_bytes()
        (self.root / self.source.path).write_bytes(b"X" + original[1:])
        with self.assertRaises(pricing.PricingError):
            pricing.read_price_quote(self.root, self.save_details(self.details()))

    def test_read_failures_are_safe_and_do_not_publish(self):
        record = self.save_body(self.historical_bytes("1000"))
        for error in [StorageError("private evidence path"), OSError("private evidence path")]:
            with self.subTest(error=type(error).__name__):
                with patch.object(pricing, "read_verified", side_effect=error), \
                        patch.object(pricing, "publish_bytes") as publish:
                    with self.assertRaises(pricing.PricingError) as caught:
                        pricing.read_price_quote(self.root, record)
                    self.assertNotIn("private evidence path", str(caught.exception))
                    publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()

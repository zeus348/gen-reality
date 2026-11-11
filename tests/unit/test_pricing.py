"""Check conservative quote arithmetic and immutable operator evidence."""

from dataclasses import replace
from decimal import Context, localcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.storage import StorageError, publish_bytes, read_verified
from features.inference import pricing as pricing_module
from features.inference.pricing import PricingError, record_price_quote


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = publish_bytes(self.root, "evidence/price.txt", b"Synthetic pricing evidence for offline tests only.\n")
        self.facts = {"checked_by": "fixture operator", "checked_unix": 1000.0,
                      "credits_per_second": 28, "minimum_credits": 56,
                      "usd_per_credit": "0.01", "maximum_additional_cents": 0}

    def quote(self, *, source=None, now=1001.0, **changes):
        return record_price_quote(self.root, self.source if source is None else source,
                                  **(self.facts | changes), clock=lambda: now)

    def test_quote_binds_price_facts_source_and_check_time_without_approval(self):
        result = self.quote()
        self.assertEqual((result.required_credits, result.reserved_cents, result.checked_unix), (140, 140, 1000.0))
        self.assertEqual(result.source, self.source)
        result.require_current(1001.0)
        data = json.loads(read_verified(self.root, result.record))
        self.assertEqual((data["model"], data["clipSeconds"], data["checkedBy"]), ("aleph2", 5, "fixture operator"))
        self.assertEqual(data["priceSourceUrl"], "https://docs.dev.runwayml.com/guides/pricing/")
        self.assertEqual(data["source"]["sha256"], self.source.sha256)
        self.assertEqual((data["creditsPerSecond"], data["minimumCredits"], data["usdPerCredit"]), (28, 56, "0.01"))
        self.assertEqual((data["baseCents"], data["maximumAdditionalCents"], data["reservedCents"]), (140, 0, 140))
        self.assertIs(data["spendingAuthorized"], False)

    def test_minimum_credits_apply_when_larger_than_five_second_cost(self):
        result = self.quote(minimum_credits=200)
        self.assertEqual((result.required_credits, result.reserved_cents), (200, 200))

    def test_fractional_cents_round_up_before_adding_all_extra_cents(self):
        result = self.quote(usd_per_credit="0.01000001", maximum_additional_cents=31)
        self.assertEqual((result.required_credits, result.reserved_cents), (140, 172))
        data = json.loads(read_verified(self.root, result.record))
        self.assertEqual(data["baseCents"], 141)
        small = self.quote(credits_per_second=1, minimum_credits=0, usd_per_credit="0.00000001")
        self.assertEqual((small.required_credits, small.reserved_cents), (5, 1))

    def test_decimal_settings_in_the_caller_cannot_reduce_the_quote(self):
        with localcontext(Context(prec=1, Emin=0, Emax=9)) as context:
            for signal in context.traps:
                context.traps[signal] = False
            result = self.quote(usd_per_credit="0.01000001", maximum_additional_cents=31)
        self.assertEqual(result.reserved_cents, 172)

    def test_invalid_operator_or_price_facts_cannot_publish_a_quote(self):
        fields = {"checked_by": [None, "", " ", "x" * 201],
                  "credits_per_second": [True, 0, -1, 1.5, 1_000_001],
                  "minimum_credits": [True, -1, 1.5, 1_000_001],
                  "maximum_additional_cents": [True, -1, 1.5, 1001],
                  "usd_per_credit": [0.01, "0", "0.000", "-0.01", "+0.01", " 0.01", "01", "1e-2",
                                     "NaN", "Infinity", "0.000000001", "10000"]}
        for field, values in fields.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(PricingError):
                    self.quote(**{field: value})
        self.assertFalse((self.root / "provider_pricing").exists())

    def test_total_quote_including_extras_cannot_exceed_ten_dollars(self):
        exact = self.quote(maximum_additional_cents=860)
        self.assertEqual(exact.reserved_cents, 1000)
        for changes in [{"maximum_additional_cents": 861}, {"credits_per_second": 201}]:
            with self.subTest(changes=changes), self.assertRaises(PricingError):
                self.quote(**changes)
        self.assertEqual(len(list((self.root / "provider_pricing").iterdir())), 1)

    def test_damaged_missing_or_oversized_source_cannot_support_a_quote(self):
        cases = [replace(self.source, size=0), replace(self.source, size=2 * 1024 * 1024 + 1),
                 replace(self.source, size=True), replace(self.source, path="missing.txt"),
                 replace(self.source, sha256="0" * 64)]
        for source in cases:
            with self.subTest(source=source), self.assertRaises(PricingError):
                self.quote(source=source)
        (self.root / self.source.path).write_bytes(b"Damaged synthetic evidence.\n")
        with self.assertRaises(PricingError):
            self.quote()
        self.assertFalse((self.root / "provider_pricing").exists())

    def test_invalid_stale_or_future_times_cannot_create_or_refresh_a_quote(self):
        for checked in [True, -1, float("nan"), float("inf"), 10**500, 1002.0, 700.0]:
            with self.subTest(checked=checked), self.assertRaises(PricingError):
                self.quote(checked_unix=checked)
        self.assertFalse((self.root / "provider_pricing").exists())
        result = self.quote(now=1300.0)
        result.require_current(1300.0)
        for now in [True, -1, float("nan"), float("inf"), 10**500, 999.0, 1300.001]:
            with self.subTest(now=now):
                with self.assertRaises(PricingError):
                    result.require_current(now)
                with self.assertRaises(PricingError):
                    self.quote(now=now)
        self.assertEqual(len(list((self.root / "provider_pricing").iterdir())), 1)
        self.assertEqual(json.loads(read_verified(self.root, result.record))["checkedAtUnix"], 1000.0)

    def test_existing_quote_is_verified_and_changed_facts_get_a_new_record(self):
        first = self.quote()
        path = self.root / first.record.path
        original = path.read_bytes()
        modified = path.stat().st_mtime_ns
        self.assertEqual(self.quote(), first)
        second = self.quote(usd_per_credit="0.02")
        self.assertNotEqual(first.record.sha256, second.record.sha256)
        self.assertEqual(second.reserved_cents, 280)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(path.stat().st_mtime_ns, modified)
        path.write_bytes(b"damaged quote fixture")
        with self.assertRaises(PricingError):
            self.quote()
        self.assertEqual(path.read_bytes(), b"damaged quote fixture")

    def test_failed_publication_cannot_return_a_quote(self):
        with patch.object(pricing_module, "publish_bytes", side_effect=StorageError("fixture write failure")):
            with self.assertRaisesRegex(PricingError, "could not be verified or recorded"):
                self.quote()
        self.assertFalse((self.root / "provider_pricing").exists())

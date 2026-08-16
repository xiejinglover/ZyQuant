from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch


from zyquant.core.exceptions import DataContractError, FutureDataError
from zyquant.data import AdjustmentProcessor, ParquetDataProvider, SnapshotPublisher

from tests.support import CODE_A, canonical_tables


class DataTests(unittest.TestCase):
    def test_publish_materializes_post_adjusted_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("sample-v1", tables)
            post = snapshot.post_adjusted_bars(days[0], days[-1], [CODE_A], cutoff=days[-1])
            raw = snapshot.raw_bars(days[0], days[-1], [CODE_A], cutoff=days[-1])
            self.assertEqual(len(post), len(raw))
            ex = post[post["trade_date"] == days[3]].iloc[0]
            self.assertGreater(ex["adjustment_factor"], 1.0)
            self.assertAlmostEqual(ex["close_post"], 10.5, places=10)
            with patch.object(AdjustmentProcessor, "build", side_effect=AssertionError("must not recalculate")):
                reopened = ParquetDataProvider(temporary).open_snapshot("sample-v1")
                direct = reopened.post_adjusted_bars(days[3], days[4], [CODE_A], cutoff=days[4])
                self.assertEqual(len(direct), 2)

    def test_snapshot_is_immutable_and_pit_guarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            publisher = SnapshotPublisher(temporary)
            publisher.publish("sample-v1", tables)
            with self.assertRaises(DataContractError):
                publisher.publish("sample-v1", tables)
            snapshot = ParquetDataProvider(temporary).open_snapshot("sample-v1")
            with self.assertRaises(FutureDataError):
                snapshot.raw_bars(days[0], days[-1], cutoff=days[-2])

    def test_table_supports_sparse_trade_date_pushdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish(
                "sample-v1", tables
            )
            requested = [days[1], days[4]]
            frame = snapshot.table(
                "daily_raw", days[0], days[-1], cutoff=days[-1],
                fields=["paused"], dates=requested,
            )
            self.assertEqual(
                sorted(frame["trade_date"].unique()), requested
            )

    def test_vendor_factors_are_materialized_and_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            raw = tables["daily_raw"]
            factors = raw[["trade_date", "instrument_id"]].copy()
            expected = AdjustmentProcessor().build(
                raw, tables["corporate_actions"]
            ).daily_post_adjusted
            factors = factors.merge(
                expected[["trade_date", "instrument_id", "adjustment_factor"]],
                on=["trade_date", "instrument_id"], how="left",
            )
            snapshot = SnapshotPublisher(temporary).publish("vendor-v1", tables, factors)
            post = snapshot.post_adjusted_bars(days[0], days[-1], [CODE_A], cutoff=days[-1])
            self.assertEqual(set(post["factor_source"]), {"vendor"})
            self.assertAlmostEqual(post.iloc[0]["adjustment_factor"], 1.0)
            self.assertGreater(post.iloc[3]["adjustment_factor"], 1.0)

    def test_conflicting_vendor_factors_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            factors = tables["daily_raw"][["trade_date", "instrument_id"]].copy()
            factors["adjustment_factor"] = 1.0
            with self.assertRaises(DataContractError):
                SnapshotPublisher(temporary).publish("bad-v1", tables, factors)

    def test_validate_mode_publishes_event_factors_and_records_deviation(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            factors = tables["daily_raw"][[
                "trade_date", "instrument_id"
            ]].copy()
            factors["adjustment_factor"] = 1.0
            snapshot = SnapshotPublisher(temporary).publish(
                "validate-v1",
                tables,
                factors,
                vendor_factor_mode="validate",
                vendor_factor_rtol=1e-3,
            )
            self.assertEqual(
                snapshot.manifest["adjustment"]["factor_source"],
                "corporate_action",
            )
            quality = snapshot.manifest["quality"]["vendor_factors"]
            self.assertEqual(quality["mode"], "validate")
            self.assertEqual(quality["status"], "deviation_observed")
            self.assertGreater(quality["mismatches"], 0)
            self.assertGreater(quality["relative_deviation"]["max"], 0)

    def test_off_mode_ignores_supplied_vendor_factors(self):
        tables, _ = canonical_tables()
        factors = tables["daily_raw"][["trade_date", "instrument_id"]].copy()
        factors["adjustment_factor"] = -1.0
        result = AdjustmentProcessor().build(
            tables["daily_raw"],
            tables["corporate_actions"],
            factors,
            vendor_factor_mode="off",
        )
        self.assertEqual(result.diagnostics.factor_source, "corporate_action")
        self.assertEqual(result.diagnostics.vendor_factors["status"], "not_checked")


if __name__ == "__main__":
    unittest.main()

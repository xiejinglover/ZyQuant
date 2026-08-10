from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from zyquant.core.exceptions import DataContractError
from zyquant.cli.main import main as cli_main
from zyquant.connectors.jqdata import (
    JQDataAdapter,
    JQDataCredentials,
    JQFinancialRequest,
    JQDataRequest,
    JQDataSDKClient,
)
from zyquant.data import (
    AdjustmentProcessor,
    ARROW_SCHEMAS,
    ParquetDataProvider,
    SnapshotPublisher,
)
from zyquant.data.normalization import normalize_table
from tests.support import canonical_tables


class FakeJQDataClient:
    sdk_version = "1.9.8-test"

    def __init__(self):
        self.authenticated = False
        self.price_calls = []
        self.index_calls = []
        self.catalog_calls = []
        self.query_count = 1000

    def authenticate(self):
        self.authenticated = True

    def get_privilege(self):
        return ["沪深A股", "上市公司基本面", "基金", "指数"]

    def get_query_count(self):
        self.query_count -= 1
        return self.query_count

    def get_all_securities(self, types, as_of):
        self.catalog_calls.append((tuple(types), as_of))
        if types == ["stock"]:
            return pd.DataFrame(
                {
                    "display_name": ["浦发银行", "平安银行", "测试成分"],
                    "start_date": [
                        date(1999, 11, 10),
                        date(1991, 4, 3),
                        date(2020, 1, 1),
                    ],
                    "end_date": [
                        date(2200, 1, 1),
                        date(2200, 1, 1),
                        date(2200, 1, 1),
                    ],
                },
                index=["600000.XSHG", "000001.XSHE", "300001.XSHE"],
            )
        return pd.DataFrame(
            {
                "display_name": ["华泰柏瑞沪深300ETF"],
                "start_date": [date(2012, 5, 28)],
                "end_date": [date(2200, 1, 1)],
            },
            index=["510300.XSHG"],
        )

    def get_trade_days(self, start_date, end_date):
        return [
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 6),
        ]

    def get_price(self, instruments, start_date, end_date, fields):
        self.price_calls.append((tuple(instruments), tuple(fields)))
        rows = []
        prices = {
            "600000.XSHG": [10.0, 9.0, 9.2],
            "000001.XSHE": [12.0, 12.1, 12.2],
            "300001.XSHE": [20.0, 20.1, 20.2],
            "510300.XSHG": [4.0, 4.1, 4.2],
        }
        factors = {
            "600000.XSHG": [1.0, 10.0 / 9.0, 10.0 / 9.0],
            "000001.XSHE": [1.0, 1.0, 1.0],
            "300001.XSHE": [1.0, 1.0, 1.0],
            "510300.XSHG": [1.0, 1.0, 1.0],
        }
        days = self.get_trade_days(start_date, end_date)
        for code in instruments:
            for index, (day, close) in enumerate(zip(days, prices[code], strict=True)):
                previous = prices[code][index - 1] if index else close
                if code == "600000.XSHG" and index == 1:
                    previous = 9.0
                rows.append({
                    "time": pd.Timestamp(day),
                    "code": code,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "pre_close": previous,
                    "volume": 100_000.0,
                    "money": close * 100_000,
                    "paused": False,
                    "high_limit": close * 1.1,
                    "low_limit": close * 0.9,
                    "factor": factors[code][index],
                })
        return pd.DataFrame(rows)

    def get_corporate_actions(self, instruments, start_date, end_date):
        return pd.DataFrame([{
            "id": 7,
            "code": "600000.XSHG",
            "a_registration_date": date(2025, 1, 2),
            "a_xr_date": date(2025, 1, 3),
            "a_bonus_amount_rmb_date": date(2025, 1, 6),
            "implementation_pub_date": date(2024, 12, 20),
            "plan_progress": "实施方案",
            "bonus_ratio_rmb": 10.0,
            "bonus_ratio": 0.0,
            "transfer_ratio": 0.0,
        }])

    def get_index_stocks(self, universe_id, day):
        self.index_calls.append((universe_id, day))
        if day == date(2025, 1, 6):
            return ["000001.XSHE", "300001.XSHE"]
        return ["600000.XSHG", "000001.XSHE"]

    def get_history_industry(self, classification, instruments):
        return pd.DataFrame([
            {
                "id": 1,
                "code": "600000.XSHG",
                "industry_code": "801780",
                "start_date": date(2020, 1, 1),
                "end_date": None,
            },
            {
                "id": 2,
                "code": "000001.XSHE",
                "industry_code": "801780",
                "start_date": date(2020, 1, 1),
                "end_date": None,
            },
            {
                "id": 3,
                "code": "300001.XSHE",
                "industry_code": "801750",
                "start_date": date(2020, 1, 1),
                "end_date": None,
            },
        ])

    def get_industry(self, instruments, day):
        return {
            code: {
                "sw_l1": {
                    "industry_code": "801780",
                    "industry_name": "银行",
                }
            }
            for code in instruments
        }


class JQDataAdapterTests(unittest.TestCase):
    def request(self, **overrides):
        payload = {
            "start_date": date(2025, 1, 2),
            "end_date": date(2025, 1, 6),
            "batch_size": 2,
        }
        payload.update(overrides)
        return JQDataRequest(**payload)

    def test_credentials_only_load_from_environment_and_hide_repr(self):
        with patch.dict(
            os.environ,
            {"JQDATA_USERNAME": "account", "JQDATA_PASSWORD": "secret"},
            clear=False,
        ):
            credentials = JQDataCredentials.from_env()
        self.assertNotIn("account", repr(credentials))
        self.assertNotIn("secret", repr(credentials))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(DataContractError):
                JQDataCredentials.from_env()

    def test_sdk_wrapper_fetches_post_factor_separately_from_raw_prices(self):
        class SDK:
            def __init__(self):
                self.fq_values = []

            def get_price(self, instruments, **kwargs):
                self.fq_values.append(kwargs["fq"])
                base = {
                    "time": [pd.Timestamp("2025-01-02")],
                    "code": [instruments[0]],
                }
                if kwargs["fq"] == "post":
                    return pd.DataFrame({**base, "factor": [2.0]})
                return pd.DataFrame({
                    **base,
                    "open": [10.0], "high": [10.1], "low": [9.9],
                    "close": [10.0], "pre_close": [10.0],
                    "volume": [100], "money": [1000.0],
                    "paused": [False], "high_limit": [11.0],
                    "low_limit": [9.0],
                })

        client = object.__new__(JQDataSDKClient)
        client._sdk = SDK()
        result = client.get_price(
            ["600000.XSHG"],
            date(2025, 1, 2),
            date(2025, 1, 2),
            JQDataAdapter.PRICE_FIELDS,
        )
        self.assertEqual(client._sdk.fq_values, [None, "post"])
        self.assertEqual(result.iloc[0]["factor"], 2.0)

    def test_adjustment_uses_reconciled_exchange_reference_price(self):
        raw = pd.DataFrame([
            {
                "trade_date": date(2025, 6, 11),
                "instrument_id": "000001.XSHE",
                "open": 11.81, "high": 11.90, "low": 11.80,
                "close": 11.85, "pre_close": 11.81,
            },
            {
                "trade_date": date(2025, 6, 12),
                "instrument_id": "000001.XSHE",
                "open": 11.49, "high": 11.75, "low": 11.45,
                "close": 11.68, "pre_close": 11.49,
            },
        ])
        actions = pd.DataFrame([{
            "event_id": "cash",
            "instrument_id": "000001.XSHE",
            "event_type": "cash_dividend",
            "record_date": date(2025, 6, 11),
            "ex_date": date(2025, 6, 12),
            "pay_date": date(2025, 6, 12),
            "cash_per_share": 0.362,
            "share_ratio": 0.0,
            "status": "active",
            "announced_at": date(2025, 6, 5),
        }])
        factors = raw[["trade_date", "instrument_id"]].copy()
        factors["adjustment_factor"] = [1.0, 11.85 / 11.49]
        result = AdjustmentProcessor().build(raw, actions, factors)
        self.assertAlmostEqual(
            result.daily_post_adjusted.iloc[1]["adjustment_factor"],
            11.85 / 11.49,
        )

    def test_adjustment_accepts_six_decimal_vendor_factor_rounding(self):
        raw = pd.DataFrame([
            {
                "trade_date": date(2025, 6, 17),
                "instrument_id": "510300.XSHG",
                "open": 3.99, "high": 4.0, "low": 3.98,
                "close": 3.989, "pre_close": 3.990,
            },
            {
                "trade_date": date(2025, 6, 18),
                "instrument_id": "510300.XSHG",
                "open": 3.901, "high": 3.91, "low": 3.89,
                "close": 3.904, "pre_close": 3.901,
            },
        ])
        actions = pd.DataFrame([{
            "event_id": "fund-cash",
            "instrument_id": "510300.XSHG",
            "event_type": "cash_dividend",
            "record_date": date(2025, 6, 17),
            "ex_date": date(2025, 6, 18),
            "pay_date": date(2025, 6, 27),
            "cash_per_share": 0.088,
            "share_ratio": 0.0,
            "status": "active",
            "announced_at": date(2025, 6, 11),
        }])
        factors = raw[["trade_date", "instrument_id"]].copy()
        factors["adjustment_factor"] = [1.207707, 1.234951]
        result = AdjustmentProcessor().build(raw, actions, factors)
        self.assertAlmostEqual(
            result.daily_post_adjusted.iloc[1]["adjustment_factor"],
            1.234951 / 1.207707,
        )

    def test_fund_dividend_is_normalized_as_cash_per_share(self):
        fake = FakeJQDataClient()
        fake.get_corporate_actions = lambda *args: pd.DataFrame([{
            "source_table": "FUND_DIVIDEND",
            "id": 116630,
            "code": "510300",
            "record_date": date(2025, 6, 17),
            "ex_date": date(2025, 6, 18),
            "fund_paid_date": date(2025, 6, 27),
            "pay_date": None,
            "dividend_implement_date": date(2025, 6, 11),
            "pub_date": date(2025, 6, 11),
            "process": "实施方案",
            "proportion": 0.088,
            "split_ratio": None,
        }])
        actions = JQDataAdapter(client=fake)._corporate_actions(
            fake, JQDataRequest.sample_2025(), "test-batch"
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions.iloc[0]["instrument_id"], "510300.XSHG")
        self.assertAlmostEqual(actions.iloc[0]["cash_per_share"], 0.088)

    def test_adapter_maps_and_publishes_complete_sample(self):
        fake = FakeJQDataClient()
        adapter = JQDataAdapter(client=fake)
        batch = adapter.ingest(self.request())
        self.assertTrue(fake.authenticated)
        self.assertEqual(set(batch.tables), {
            "instruments", "trade_calendar", "daily_raw",
            "corporate_actions", "universe_membership",
            "industry_membership", "market_rules",
        })
        self.assertEqual(len(batch.tables["daily_raw"]), 9)
        self.assertEqual(len(batch.tables["corporate_actions"]), 1)
        self.assertEqual(
            set(batch.tables["instruments"]["instrument_id"]),
            {
                "600000.XSHG", "000001.XSHE",
                "300001.XSHE", "510300.XSHG",
            },
        )
        membership = batch.tables["universe_membership"]
        first_day = membership[
            (membership["effective_from"] <= date(2025, 1, 2))
            & (
                membership["effective_to"].isna()
                | (membership["effective_to"] >= date(2025, 1, 2))
            )
        ]
        last_day = membership[
            (membership["effective_from"] <= date(2025, 1, 6))
            & (
                membership["effective_to"].isna()
                | (membership["effective_to"] >= date(2025, 1, 6))
            )
        ]
        self.assertEqual(
            set(first_day["instrument_id"]),
            {"600000.XSHG", "000001.XSHE"},
        )
        self.assertEqual(
            set(last_day["instrument_id"]),
            {"000001.XSHE", "300001.XSHE"},
        )
        self.assertEqual(
            batch.tables["instruments"].set_index("instrument_id").loc[
                "510300.XSHG", "sell_delay_days"
            ],
            1,
        )
        self.assertNotIn("password", str(batch.source_metadata).lower())
        self.assertNotIn("username", str(batch.source_metadata).lower())
        self.assertEqual(len(fake.price_calls), 2)
        self.assertEqual(len(fake.index_calls), 3)
        self.assertEqual(
            batch.source_metadata["coverage"]["prices"]["instrument_count"],
            3,
        )
        self.assertEqual(
            batch.source_metadata["coverage"]["universe_membership"][
                "instrument_count"
            ],
            3,
        )

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = SnapshotPublisher(temporary).publish_adapter(
                "jqdata-test", adapter, self.request()
            )
            raw = snapshot.raw_bars(
                date(2025, 1, 2),
                date(2025, 1, 6),
                ["600000.XSHG"],
                cutoff=date(2025, 1, 6),
            )
            post = snapshot.post_adjusted_bars(
                date(2025, 1, 2),
                date(2025, 1, 6),
                ["600000.XSHG"],
                cutoff=date(2025, 1, 6),
            )
            self.assertEqual(len(raw), len(post))
            self.assertAlmostEqual(post.iloc[1]["close_post"], 10.0)
            reopened = ParquetDataProvider(temporary).open_snapshot("jqdata-test")
            self.assertEqual(
                reopened.metadata.fingerprint,
                snapshot.metadata.fingerprint,
            )
            with self.assertRaisesRegex(DataContractError, "price coverage"):
                reopened.raw_bars(
                    date(2025, 1, 2),
                    date(2025, 1, 6),
                    ["300001.XSHE"],
                    cutoff=date(2025, 1, 6),
                )

    def test_strict_sample_rejects_missing_actions(self):
        fake = FakeJQDataClient()
        fake.get_corporate_actions = lambda *args: pd.DataFrame()
        with self.assertRaisesRegex(DataContractError, "no implemented"):
            JQDataAdapter(client=fake).ingest(self.request())

    def test_universe_price_scope_downloads_historical_member_union(self):
        fake = FakeJQDataClient()
        batch = JQDataAdapter(client=fake).ingest(
            self.request(price_scope="universe")
        )
        self.assertEqual(
            set(batch.tables["daily_raw"]["instrument_id"]),
            {"600000.XSHG", "000001.XSHE", "300001.XSHE"},
        )
        self.assertNotIn(
            "510300.XSHG",
            set(batch.tables["daily_raw"]["instrument_id"]),
        )
        self.assertTrue(
            batch.source_metadata["coverage"]["full_universe_backtest_ready"]
        )

    def test_historical_member_metadata_is_looked_up_when_absent_at_end(self):
        fake = FakeJQDataClient()
        original = fake.get_all_securities

        def historical_catalog(types, as_of):
            frame = original(types, as_of)
            if types == ["stock"] and as_of == date(2025, 1, 6):
                return frame.drop(index="600000.XSHG")
            return frame

        fake.get_all_securities = historical_catalog
        batch = JQDataAdapter(client=fake).ingest(self.request())
        self.assertIn(
            "600000.XSHG",
            set(batch.tables["instruments"]["instrument_id"]),
        )
        self.assertIn(
            (("stock",), date(2025, 1, 2)),
            fake.catalog_calls,
        )

    def test_industry_history_permission_falls_back_to_daily_pit_queries(self):
        fake = FakeJQDataClient()
        fake.get_history_industry = lambda *args: (_ for _ in ()).throw(
            Exception("paid module")
        )
        batch = JQDataAdapter(client=fake).ingest(self.request())
        industry = batch.tables["industry_membership"]
        self.assertEqual(set(industry["instrument_id"]), {
            "600000.XSHG", "000001.XSHE", "300001.XSHE",
        })
        self.assertTrue(any(
            "reconstructed" in warning
            for warning in batch.source_metadata["warnings"]
        ))

    def test_financial_module_maps_units_and_excludes_etf(self):
        fake = FakeJQDataClient()

        def statements(statement_type, instruments, start_date, end_date):
            values = {
                "balance": {
                    "total_assets": 100.0,
                    "total_liability": 60.0,
                    "total_owner_equities": 40.0,
                    "equities_parent_company_owners": 40.0,
                    "loan_and_advance": 50.0,
                },
                "income": {
                    "total_operating_revenue": 20.0,
                    "net_profit": 4.0,
                    "np_parent_company_owners": 4.0,
                    "interest_income": 10.0,
                },
                "cash_flow": {"net_operate_cash_flow": 3.0},
            }[statement_type]
            base_id = {"balance": 101, "income": 201, "cash_flow": 301}[
                statement_type
            ]
            return pd.DataFrame([
                {
                    "id": base_id + index,
                    "code": code,
                    "pub_date": date(2025, 1, 2),
                    "start_date": date(2024, 1, 1),
                    "end_date": date(2024, 12, 31),
                    "report_date": date(2024, 12, 31),
                    "report_type": 0,
                    **values,
                }
                for index, code in enumerate(instruments)
            ])

        fake.get_financial_statements = statements
        fake.get_valuation = lambda instruments, *args: pd.DataFrame([
            {
                "code": code,
                "day": date(2025, 1, 2),
                "pe_ratio": 8.0,
                "turnover_ratio": 0.25,
                "pb_ratio": 0.6,
                "ps_ratio": 2.0,
                "pcf_ratio": -2.0,
                "capitalization": 100.0,
                "market_cap": 12.0,
                "circulating_cap": 90.0,
                "circulating_market_cap": 10.8,
                "pe_ratio_lyr": 9.0,
                "pcf_ratio2": 20.0,
                "dividend_ratio": 3.0,
                "free_cap": 80.0,
                "free_market_cap": 9.6,
                "a_cap": 100.0,
                "a_market_cap": 12.0,
            }
            for code in instruments
        ])
        fake.get_share_capital = lambda *args: pd.DataFrame([{
            "id": 501,
            "code": "600000.XSHG",
            "change_date": date(2024, 12, 31),
            "pub_date": date(2025, 1, 2),
            "change_reason_id": 1,
            "change_reason": "test",
            "share_total": 1_000_000.0,
            "share_non_trade": 0.0,
            "share_limited": 100_000.0,
            "share_trade_total": 900_000.0,
            "share_rmb": 1_000_000.0,
            "share_b": 0.0,
            "share_h": 0.0,
        }])
        batch = JQDataAdapter(client=fake).ingest(self.request(
            financial=JQFinancialRequest(
                enabled=True,
                report_start_date=date(2020, 1, 1),
                valuation_start_date=date(2025, 1, 2),
            )
        ))
        self.assertEqual(
            set(batch.tables) & {
                "financial_reports", "financial_facts",
                "fundamental_metrics", "daily_valuation", "share_capital",
            },
            {
                "financial_reports", "financial_facts",
                "fundamental_metrics", "daily_valuation", "share_capital",
            },
        )
        valuation = batch.tables["daily_valuation"].iloc[0]
        self.assertEqual(valuation["total_shares"], 1_000_000.0)
        self.assertEqual(valuation["market_cap"], 1_200_000_000.0)
        self.assertAlmostEqual(valuation["dividend_yield"], 0.03)
        coverage = batch.source_metadata["capabilities"]["financials"][
            "coverage"
        ]
        self.assertEqual(coverage["instruments"], [
            "000001.XSHE", "600000.XSHG",
        ])
        self.assertEqual(
            coverage["excluded"]["510300.XSHG"],
            "asset_type_not_stock",
        )

    def test_schema_is_machine_readable_and_delist_date_is_required(self):
        self.assertEqual(
            str(ARROW_SCHEMAS["daily_raw"].field("volume").type),
            "int64",
        )
        frame = FakeJQDataClient().get_all_securities(
            ["stock"], date(2025, 1, 1)
        ).reset_index(names="instrument_id")
        frame = pd.DataFrame([{
            "instrument_id": "600000.XSHG",
            "symbol": "600000",
            "exchange": "XSHG",
            "asset_type": "stock",
            "list_date": date(1999, 11, 10),
            "lot_size": 100,
            "sell_delay_days": 1,
        }])
        with self.assertRaisesRegex(DataContractError, "delist_date"):
            normalize_table("instruments", frame)

    def test_directory_cli_publish_remains_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "canonical"
            source.mkdir()
            tables, _ = canonical_tables()
            for name, frame in tables.items():
                frame.to_parquet(source / f"{name}.parquet", index=False)
            request = root / "directory.yaml"
            request.write_text(f"path: {source}\n", encoding="utf-8")
            status = cli_main([
                "data", "publish",
                "--source", "canonical-directory",
                "--root", str(root / "data"),
                "--request", str(request),
                "--dataset-id", "directory-v1",
            ])
            self.assertEqual(status, 0)
            self.assertTrue(
                (root / "data" / "datasets" / "directory-v1" / "manifest.json").exists()
            )

    def test_real_jqdata_sample_when_credentials_are_configured(self):
        if not (
            os.environ.get("JQDATA_USERNAME")
            and os.environ.get("JQDATA_PASSWORD")
        ):
            self.skipTest("JQData credentials are not configured")
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = SnapshotPublisher(temporary).publish_adapter(
                "jqdata-sample-2025",
                JQDataAdapter(),
                JQDataRequest.sample_2025(),
            )
            self.assertEqual(snapshot.metadata.as_of_date, date(2025, 12, 31))
            self.assertGreater(
                snapshot.manifest["quality"]["tables"]["daily_raw"]["rows"],
                500,
            )
            self.assertGreater(
                snapshot.manifest["quality"]["tables"]["corporate_actions"]["rows"],
                0,
            )


if __name__ == "__main__":
    unittest.main()

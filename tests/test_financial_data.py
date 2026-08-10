from __future__ import annotations

import tempfile
import unittest
from datetime import date

import pandas as pd

from zyquant.core.exceptions import DataContractError, FutureDataError
from zyquant.data import ParquetDataProvider, SnapshotPublisher
from zyquant.data.financial import FinancialProcessor

from tests.support import CODE_B, canonical_tables


def statement_row(
    record_id: int,
    period_end: date,
    published: date,
    report_type: int = 0,
    **facts,
):
    return {
        "id": record_id,
        "code": CODE_B,
        "pub_date": published,
        "start_date": date(period_end.year, 1, 1),
        "end_date": period_end,
        "report_date": (
            period_end if report_type == 0 else date(period_end.year + 1, 12, 31)
        ),
        "report_type": report_type,
        **facts,
    }


def financial_sources():
    income = pd.DataFrame([
        statement_row(
            10, date(2024, 3, 31), date(2024, 4, 20),
            total_operating_revenue=25.0,
            operating_cost=15.0,
            operating_profit=5.0,
            net_profit=4.0,
            np_parent_company_owners=4.0,
        ),
        statement_row(
            11, date(2024, 6, 30), date(2024, 8, 20),
            total_operating_revenue=55.0,
            operating_cost=32.0,
            operating_profit=12.0,
            net_profit=9.0,
            np_parent_company_owners=9.0,
        ),
        statement_row(
            12, date(2024, 9, 30), date(2024, 10, 20),
            total_operating_revenue=90.0,
            operating_cost=51.0,
            operating_profit=20.0,
            net_profit=15.0,
            np_parent_company_owners=15.0,
        ),
        statement_row(
            13, date(2024, 12, 31), date(2025, 1, 2),
            total_operating_revenue=130.0,
            operating_cost=72.0,
            operating_profit=30.0,
            net_profit=22.0,
            np_parent_company_owners=22.0,
        ),
    ])
    cash_flow = pd.DataFrame([
        statement_row(
            20 + index, period, published,
            net_operate_cash_flow=value,
            fix_intan_other_asset_acqui_cash=value / 5,
        )
        for index, (period, published, value) in enumerate([
            (date(2024, 3, 31), date(2024, 4, 20), 5.0),
            (date(2024, 6, 30), date(2024, 8, 20), 11.0),
            (date(2024, 9, 30), date(2024, 10, 20), 18.0),
            (date(2024, 12, 31), date(2025, 1, 2), 26.0),
        ])
    ])
    balance = pd.DataFrame([
        statement_row(
            30, date(2023, 12, 31), date(2024, 3, 15),
            total_assets=90.0,
            total_liability=55.0,
            total_owner_equities=35.0,
            equities_parent_company_owners=35.0,
        ),
        statement_row(
            31, date(2024, 12, 31), date(2025, 1, 2),
            total_assets=110.0,
            total_liability=65.0,
            total_owner_equities=45.0,
            equities_parent_company_owners=45.0,
        ),
        statement_row(
            32, date(2023, 12, 31), date(2025, 1, 2), report_type=1,
            total_assets=92.0,
            total_liability=56.0,
            total_owner_equities=36.0,
            equities_parent_company_owners=36.0,
        ),
    ])
    return {"balance": balance, "income": income, "cash_flow": cash_flow}


class FinancialDataTests(unittest.TestCase):
    def build_financial_tables(self):
        trade_days = [
            day.date()
            for day in pd.bdate_range("2024-01-01", "2025-01-10")
        ]
        result = FinancialProcessor().build(
            financial_sources(), trade_days, "financial-test"
        )
        valuation = pd.DataFrame([{
            "trade_date": date(2025, 1, 2),
            "instrument_id": CODE_B,
            "pe_ttm": 5.0,
            "pe_lyr": 5.2,
            "pb": 0.6,
            "ps_ttm": 1.5,
            "pcf_ttm": -2.0,
            "pcf_operating_ttm": 10.0,
            "dividend_yield": 0.04,
            "turnover_rate": 0.003,
            "total_shares": 1_000_000.0,
            "market_cap": 10_000_000.0,
            "circulating_shares": 900_000.0,
            "circulating_market_cap": 9_000_000.0,
            "free_float_shares": 800_000.0,
            "free_float_market_cap": 8_000_000.0,
            "a_shares": 1_000_000.0,
            "a_market_cap": 10_000_000.0,
            "available_at": date(2025, 1, 2),
        }])
        capital = pd.DataFrame([{
            "capital_event_id": "capital-1",
            "instrument_id": CODE_B,
            "effective_from": date(2024, 12, 31),
            "announced_at": date(2025, 1, 1),
            "available_at": date(2025, 1, 2),
            "change_reason_code": "1",
            "change_reason": "test",
            "total_shares": 1_000_000.0,
            "nontradable_shares": 0.0,
            "restricted_shares": 100_000.0,
            "tradable_shares": 900_000.0,
            "a_shares": 1_000_000.0,
            "b_shares": 0.0,
            "h_shares": 0.0,
        }])
        return {
            "financial_reports": result.reports,
            "financial_facts": result.facts,
            "fundamental_metrics": result.metrics,
            "daily_valuation": valuation,
            "share_capital": capital,
        }

    def test_processor_preserves_comparatives_and_builds_ttm(self):
        tables = self.build_financial_tables()
        reports = tables["financial_reports"]
        comparative = reports[
            (reports["record_kind"] == "comparative")
            & (reports["fiscal_period_end"] == date(2023, 12, 31))
        ].iloc[0]
        self.assertEqual(comparative["available_at"], date(2025, 1, 3))
        self.assertEqual(comparative["revision_sequence"], 2)
        metrics = tables["fundamental_metrics"]
        revenue = metrics[
            (metrics["metric_code"] == "revenue")
            & (metrics["basis"] == "ttm")
            & (metrics["fiscal_period_end"] == date(2024, 12, 31))
        ]
        self.assertTrue(len(revenue))
        self.assertAlmostEqual(revenue.iloc[-1]["value"], 130.0)

    def test_optional_financial_snapshot_is_pit_guarded(self):
        base, days = canonical_tables()
        base.update(self.build_financial_tables())
        lineage = {
            "capabilities": {
                "financials": {
                    "schema_version": "1.0",
                    "pit_validated": True,
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = SnapshotPublisher(temporary).publish(
                "financial-v1", base, lineage=lineage
            )
            self.assertEqual(snapshot.metadata.schema_version, "1.1")
            financial = snapshot.financial(days[-1])
            before = snapshot.financial(date(2025, 1, 2)).facts(
                date(2023, 12, 31),
                date(2023, 12, 31),
                [CODE_B],
                ["total_assets"],
            )
            self.assertEqual(set(before["value"]), {90.0})
            after = snapshot.financial(date(2025, 1, 3)).facts(
                date(2023, 12, 31),
                date(2023, 12, 31),
                [CODE_B],
                ["total_assets"],
            )
            self.assertEqual(set(after["value"]), {90.0, 92.0})
            capital = financial.share_capital(days[0], [CODE_B])
            self.assertEqual(capital.iloc[0]["total_shares"], 1_000_000.0)
            with self.assertRaises(FutureDataError):
                financial.latest_metrics(date(2025, 2, 1))
            reopened = ParquetDataProvider(temporary).open_snapshot(
                "financial-v1"
            )
            self.assertIn("financials", reopened.manifest["capabilities"])

    def test_old_snapshot_has_no_financial_capability(self):
        base, _ = canonical_tables()
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = SnapshotPublisher(temporary).publish("market-v1", base)
            with self.assertRaisesRegex(DataContractError, "financial capability"):
                snapshot.financial(date(2025, 1, 10))


if __name__ == "__main__":
    unittest.main()

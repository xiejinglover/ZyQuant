from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from zyquant.core.exceptions import DataContractError
from zyquant.data import AdjustmentProcessor
from zyquant.connectors.hermes.normalize import (
    HermesCanonicalizer, _cumulative_flow_rows,
)
from zyquant.connectors.hermes.acquisition import (
    HERMES_SOURCE_TABLES,
    HermesAcquisitionRequest,
    HermesCredentials,
    HermesDataAdapter,
    HermesExtractionPlanner,
    HermesResourceLimits,
)


def test_financial_normalization_keeps_ytd_not_standalone_quarter():
    source = pd.DataFrame([
        {
            "ID": 1, "END_DATE": "2015-09-30",
            "FISCAL_PERIOD": 3, "N_INCOME": 2.88e9,
        },
        {
            "ID": 2, "END_DATE": "2015-09-30",
            "FISCAL_PERIOD": 9, "N_INCOME": 11.83e9,
        },
        {
            "ID": 3, "END_DATE": "2015-12-31",
            "FISCAL_PERIOD": 12, "N_INCOME": 15.0e9,
        },
    ])

    income = _cumulative_flow_rows(source, "income")
    assert list(income["ID"]) == [2, 3]
    # Instantaneous statements do not use the flow-period filter.
    pd.testing.assert_frame_equal(
        _cumulative_flow_rows(source, "balance"), source
    )


class FakeHermesClient:
    def schema_inventory(self):
        return [
            {
                "TABLE_NAME": table,
                "COLUMN_NAME": "ID",
                "ORDINAL_POSITION": 1,
                "DATA_TYPE": "bigint",
                "IS_NULLABLE": "NO",
                "COLUMN_KEY": "PRI",
                "COLUMN_COMMENT": "",
            }
            for table in HERMES_SOURCE_TABLES
        ]

    def scalar(self, sql, parameters=()):
        assert sql == "SELECT UTC_TIMESTAMP(6)"
        return "2026-07-25 00:00:00.000000"

    def rows(self, sql, parameters=(), fetch_rows=50_000):
        assert not sql.lstrip().upper().startswith(
            ("INSERT", "UPDATE", "DELETE", "REPLACE", "ALTER", "DROP")
        )
        if "SELECT DISTINCT PARTY_ID" in sql:
            yield [{"PARTY_ID": 101}]
        return


def test_credentials_are_environment_only_and_redacted():
    values = {
        "HERMES_MYSQL_HOST": "db.example",
        "HERMES_MYSQL_USER": "readonly",
        "HERMES_MYSQL_PASSWORD": "do-not-persist",
        "HERMES_MYSQL_DATABASE": "hermes",
    }
    with patch.dict(os.environ, values, clear=True):
        credentials = HermesCredentials.from_env()
        assert credentials.password == "do-not-persist"
        assert "password" not in credentials.safe_metadata()
        assert "readonly" not in str(credentials.safe_metadata())
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(DataContractError, match="HERMES_MYSQL_HOST"):
            HermesCredentials.from_env()


def test_planner_is_deterministic_and_read_only():
    request = HermesAcquisitionRequest(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 24),
        limits=HermesResourceLimits(max_connections=2),
    )
    planner = HermesExtractionPlanner(
        request, "2026-07-25 00:00:00.000000"
    )
    first = planner.plan([[1, 2]])
    second = planner.plan([[1, 2]])
    assert first == second
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert all(chunk.sql.lstrip().upper().startswith("SELECT") for chunk in first)
    assert all(
        "UPDATE_TIME" in chunk.sql
        for chunk in first
    )
    assert {
        chunk.table_name for chunk in first
    } == set(HERMES_SOURCE_TABLES)


def test_empty_acquisition_resumes_without_rewriting_completed_chunks():
    with tempfile.TemporaryDirectory() as temporary:
        request = HermesAcquisitionRequest(
            job_id="contract-probe",
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 24),
            financial_warmup_start=date(2025, 1, 1),
            root=Path(temporary),
            limits=HermesResourceLimits(max_connections=2),
        )
        adapter = HermesDataAdapter(
            HermesCredentials("host", "user", "secret"),
            client=FakeHermesClient(),
        )
        first = adapter.run(request)
        assert first["job"]["status"] == "acquired"
        files = sorted((request.job_root / "raw").rglob("*.parquet"))
        mtimes = {path: path.stat().st_mtime_ns for path in files}
        second = adapter.run(request, resume=True)
        assert second["job"]["status"] == "acquired"
        assert mtimes == {path: path.stat().st_mtime_ns for path in files}
        persisted = (
            request.job_root / "state.sqlite"
        ).read_bytes() + (
            request.job_root / "source_schema.json"
        ).read_bytes()
        assert b"secret" not in persisted


def test_rights_issue_adjustment_formula():
    raw = pd.DataFrame([
        {
            "trade_date": date(2025, 1, 2),
            "instrument_id": "600000.XSHG",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "pre_close": 10.0,
        },
        {
            "trade_date": date(2025, 1, 3),
            "instrument_id": "600000.XSHG",
            "open": 9.5,
            "high": 9.5,
            "low": 9.5,
            "close": 9.5,
            "pre_close": 9.5,
        },
    ])
    actions = pd.DataFrame([{
        "instrument_id": "600000.XSHG",
        "ex_date": date(2025, 1, 3),
        "event_type": "rights_issue",
        "share_ratio": 0.2,
        "subscription_price": 7.0,
        "cash_per_share": 0.0,
        "status": "active",
    }])
    result = AdjustmentProcessor().build(raw, actions)
    theoretical = (10.0 + 0.2 * 7.0) / 1.2
    assert result.daily_post_adjusted.iloc[-1]["adjustment_factor"] == pytest.approx(
        10.0 / theoretical
    )


def _alias_canonicalizer(root: Path, traded_security_ids):
    request = HermesAcquisitionRequest(
        job_id="alias-job",
        root=root,
        start_date=date(2010, 1, 1),
        end_date=date(2026, 7, 24),
        financial_warmup_start=date(2009, 1, 1),
    )
    market = request.job_root / "raw" / "mkt_equd"
    market.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SECURITY_ID": list(traded_security_ids)}).to_parquet(
        market / "part-000.parquet", index=False
    )
    return HermesCanonicalizer(request)


def _security_records():
    return pd.DataFrame([
        {
            "SECURITY_ID": 69, "TICKER_SYMBOL": "001914", "PARTY_ID": 43,
            "SEC_SHORT_NAME": "招商积余", "LIST_STATUS_CD": "L",
            "DELIST_DATE": None, "instrument_id": "001914.XSHE",
        },
        {
            "SECURITY_ID": 77481, "TICKER_SYMBOL": "000043", "PARTY_ID": 43,
            "SEC_SHORT_NAME": "中航善达", "LIST_STATUS_CD": "DE",
            "DELIST_DATE": "2019-12-16", "instrument_id": "000043.XSHE",
        },
        {
            "SECURITY_ID": 662, "TICKER_SYMBOL": "600087", "PARTY_ID": 670,
            "SEC_SHORT_NAME": "退市长油", "LIST_STATUS_CD": "DE",
            "DELIST_DATE": "2014-06-05", "instrument_id": "600087.XSHG",
        },
        {
            "SECURITY_ID": 77002, "TICKER_SYMBOL": "601975", "PARTY_ID": 670,
            "SEC_SHORT_NAME": "招商南油", "LIST_STATUS_CD": "L",
            "DELIST_DATE": None, "instrument_id": "601975.XSHG",
        },
    ])


def test_recoded_ticker_without_market_data_is_excluded_and_recorded():
    with tempfile.TemporaryDirectory() as directory:
        # 000043 was recoded to 001914; its whole history stays under
        # SECURITY_ID 69, so the retired record owns no bar at all.
        canonicalizer = _alias_canonicalizer(
            Path(directory), [69, 662, 77002]
        )
        kept = canonicalizer._drop_superseded_aliases(_security_records())

        assert set(kept["instrument_id"]) == {
            "001914.XSHE", "600087.XSHG", "601975.XSHG",
        }
        assert [item["instrument_id"] for item in
                canonicalizer.superseded_aliases] == ["000043.XSHE"]
        excluded = canonicalizer.superseded_aliases[0]
        assert excluded["security_id"] == 77481
        assert excluded["reason"] == "no_market_data_in_window"


def test_relisting_sharing_a_party_id_is_never_excluded():
    with tempfile.TemporaryDirectory() as directory:
        # 600087 and 601975 share PARTY_ID 670 but are two real listing
        # periods, so party identity must not drive exclusion.
        canonicalizer = _alias_canonicalizer(
            Path(directory), [69, 662, 77002, 77481]
        )
        kept = canonicalizer._drop_superseded_aliases(_security_records())

        assert len(kept) == 4
        assert canonicalizer.superseded_aliases == []


def test_missing_market_data_fails_instead_of_emptying_the_universe():
    with tempfile.TemporaryDirectory() as directory:
        canonicalizer = _alias_canonicalizer(Path(directory), [])
        with pytest.raises(DataContractError):
            canonicalizer._drop_superseded_aliases(_security_records())


def _dividend_canonicalizer(root: Path, dividend: pd.DataFrame):
    request = HermesAcquisitionRequest(
        job_id="dividend-job",
        root=root,
        start_date=date(2010, 1, 1),
        end_date=date(2026, 7, 24),
        financial_warmup_start=date(2009, 1, 1),
    )
    raw = request.job_root / "raw"
    for table in ("equ_div_pit", "equ_splits", "equ_allot"):
        (raw / table).mkdir(parents=True, exist_ok=True)
    dividend.to_parquet(raw / "equ_div_pit" / "part-all.parquet", index=False)
    for table in ("equ_splits", "equ_allot"):
        pd.DataFrame().to_parquet(raw / table / "part-all.parquet")
    canonicalizer = HermesCanonicalizer(request)
    canonicalizer.instrument_by_security = {69: "001914.XSHE"}
    return canonicalizer


def _dividend_rows():
    common = {
        "SECURITY_ID": 69,
        "PER_CASH_DIV": 0.5,
        "PER_SHARE_DIV_RATIO": 0.0,
        "PER_SHARE_TRANS_RATIO": 0.0,
        "UPDATE_TIME": pd.Timestamp("2026-01-01", tz="UTC"),
    }
    return pd.DataFrame([
        {
            # Pre-2019 shape: both announcement fields present.
            **common, "ID": 1,
            "EX_DIV_DATE": date(2018, 6, 25),
            "RECORD_DATE": date(2018, 6, 22),
            "PAY_CASH_DATE": date(2018, 6, 29),
            "PUBLISH_DATE": date(2018, 4, 18),
            "IM_PUBLISH_DATE": date(2018, 6, 19),
        },
        {
            # Post-2019 shape: the source stopped populating PUBLISH_DATE.
            **common, "ID": 2,
            "EX_DIV_DATE": date(2021, 7, 15),
            "RECORD_DATE": date(2021, 7, 14),
            "PAY_CASH_DATE": date(2021, 7, 15),
            "PUBLISH_DATE": None,
            "IM_PUBLISH_DATE": date(2021, 7, 9),
        },
    ])


def test_dividends_survive_an_empty_board_proposal_date():
    with tempfile.TemporaryDirectory() as directory:
        canonicalizer = _dividend_canonicalizer(
            Path(directory), _dividend_rows()
        )
        canonicalizer._build_actions()

        actions = pd.read_parquet(
            canonicalizer.canonical / "corporate_actions"
        )
        cash = actions[actions["event_type"] == "cash_dividend"]
        # Keying on PUBLISH_DATE silently dropped the 2021 event entirely.
        assert sorted(cash["ex_date"]) == [
            date(2018, 6, 25), date(2021, 7, 15),
        ]
        assert cash["announced_at"].notna().all()


def test_dividend_visibility_uses_the_implementation_announcement():
    with tempfile.TemporaryDirectory() as directory:
        canonicalizer = _dividend_canonicalizer(
            Path(directory), _dividend_rows()
        )
        canonicalizer._build_actions()

        actions = pd.read_parquet(
            canonicalizer.canonical / "corporate_actions"
        ).set_index("ex_date")
        # One definition for the whole history, not PUBLISH_DATE pre-2019 and
        # a different field after, which would shift visibility mid-sample.
        assert actions.loc[date(2018, 6, 25), "announced_at"] == date(
            2018, 6, 19
        )
        assert actions.loc[date(2021, 7, 15), "announced_at"] == date(
            2021, 7, 9
        )
        # An announcement may never postdate the ex-date it announces.
        for ex_date, row in actions.iterrows():
            assert row["announced_at"] <= ex_date


def _state_canonicalizer(root: Path, changes: pd.DataFrame | None):
    request = HermesAcquisitionRequest(
        job_id="state-job",
        root=root,
        start_date=date(2010, 1, 1),
        end_date=date(2026, 7, 24),
        financial_warmup_start=date(2009, 1, 1),
    )
    if changes is not None:
        directory = request.job_root / "raw" / "equ_inst_sstate"
        directory.mkdir(parents=True, exist_ok=True)
        changes.to_parquet(directory / "part-all.parquet", index=False)
    canonicalizer = HermesCanonicalizer(request)
    canonicalizer.instrument_by_security = {
        98: "000100.XSHE", 2: "000001.XSHE",
    }
    return canonicalizer


def _state_changes():
    """Two transitions for one issuer, in the shape the source uses.

    `equ_inst_sstate` is a special-treatment log, not a rename history: state 2
    enters special treatment and carries a reason, state 1 leaves it.
    """
    return pd.DataFrame([
        {
            "ID": 10, "SECURITY_ID": 98, "PARTY_ID": 1,
            "SEC_SHORT_NAME": "*STTCL", "PARTY_STATE": 2, "REASON": 22.0,
            "EFF_DATE": date(2007, 5, 8), "PUBLISH_DATE": date(2007, 4, 30),
            "UPDATE_TIME": pd.Timestamp("2007-05-08"),
        },
        {
            "ID": 11, "SECURITY_ID": 98, "PARTY_ID": 1,
            "SEC_SHORT_NAME": "TCL集团", "PARTY_STATE": 1, "REASON": None,
            "EFF_DATE": date(2008, 3, 28), "PUBLISH_DATE": None,
            "UPDATE_TIME": pd.Timestamp("2008-03-28"),
        },
    ])


def _flagged_on(frame: pd.DataFrame, code: str, day: date) -> bool:
    rows = frame[
        (frame["instrument_id"] == code)
        & (frame["known_at"] <= day)
        & (frame["effective_from"] <= day)
        & (frame["effective_to"].isna() | (frame["effective_to"] > day))
    ]
    if rows.empty:
        return False
    name = str(rows.iloc[-1]["name"])
    return any(token in name for token in ("ST", "*", "退"))


def test_state_transitions_become_abutting_windows():
    with tempfile.TemporaryDirectory() as directory:
        canonicalizer = _state_canonicalizer(Path(directory), _state_changes())
        canonicalizer._build_special_treatment()
        frame = pd.read_parquet(
            canonicalizer.canonical / "special_treatment"
        )

        windows = frame[frame["instrument_id"] == "000100.XSHE"]
        assert len(windows) == 2
        # The entry window ends exactly where the exit window opens.
        assert windows.iloc[0]["effective_to"] == date(2008, 3, 28)
        assert pd.isna(windows.iloc[1]["effective_to"])
        assert list(windows["state_code"]) == [2, 1]
        assert windows.iloc[0]["reason_code"] == 22
        assert pd.isna(windows.iloc[1]["reason_code"])

        # Flagged only between entry and exit.
        assert _flagged_on(frame, "000100.XSHE", date(2007, 6, 1)) is True
        assert _flagged_on(frame, "000100.XSHE", date(2009, 6, 1)) is False


def test_an_issuer_never_flagged_gets_no_window():
    """Absence of a record means never flagged, not an unknown state.

    The log records entries and exits, so a company that was never under
    special treatment simply does not appear, and must not be synthesised
    into a window carrying today's name.
    """
    with tempfile.TemporaryDirectory() as directory:
        canonicalizer = _state_canonicalizer(Path(directory), _state_changes())
        canonicalizer._build_special_treatment()
        frame = pd.read_parquet(
            canonicalizer.canonical / "special_treatment"
        )

        assert (frame["instrument_id"] == "000001.XSHE").sum() == 0
        assert _flagged_on(frame, "000001.XSHE", date(2016, 6, 1)) is False


def test_a_transition_is_invisible_until_it_was_announced():
    with tempfile.TemporaryDirectory() as directory:
        canonicalizer = _state_canonicalizer(Path(directory), _state_changes())
        canonicalizer._build_special_treatment()
        frame = pd.read_parquet(
            canonicalizer.canonical / "special_treatment"
        ).set_index(["instrument_id", "effective_from"])

        # Announced eight days before it took effect.
        assert frame.loc[
            ("000100.XSHE", date(2007, 5, 8)), "known_at"
        ] == date(2007, 4, 30)
        # Undisclosed transition falls back to its effective date, never earlier.
        assert frame.loc[
            ("000100.XSHE", date(2008, 3, 28)), "known_at"
        ] == date(2008, 3, 28)


def test_missing_state_source_leaves_the_table_unwritten():
    with tempfile.TemporaryDirectory() as directory:
        # An acquisition captured before equ_inst_sstate was fetched must keep
        # normalizing; the table is optional, not required.
        canonicalizer = _state_canonicalizer(Path(directory), None)
        canonicalizer._build_special_treatment()

        assert not (canonicalizer.canonical / "special_treatment").exists()
        assert canonicalizer.special_treatment_windows == 0



def _valuation_canonicalizer(root: Path, dividend: pd.DataFrame):
    request = HermesAcquisitionRequest(
        job_id="valuation-job",
        root=root,
        start_date=date(2016, 1, 1),
        end_date=date(2016, 1, 31),
        financial_warmup_start=date(2015, 1, 1),
    )
    raw = request.job_root / "raw"
    partition = Path("year=2016/month=01/part-2016-01.parquet")
    day = date(2016, 1, 4)
    common = {
        "SECURITY_ID": 2, "TRADE_DATE": day, "ID": 1,
        "UPDATE_TIME": pd.Timestamp("2016-01-05"),
    }
    tables = {
        "mkt_equd_eval": pd.DataFrame([{
            **common, "PE_T": 12.0, "PB": 1.1, "PS_T": 2.0,
            "PCF_T": 8.0, "PCF_OT": 7.0,
            "MARKET_VALUE": 1.0e10, "NEG_MARKET_VALUE": 8.0e9,
        }]),
        "mkt_equd_eval_new": pd.DataFrame([{
            **common, "PE_LYR": 13.0, "FREE_MARKET_VALUE": 7.0e9,
        }]),
        "mkt_div_yield": dividend,
        "mkt_equd": pd.DataFrame([{
            **common, "TURNOVER_RATE": 0.01, "CLOSE_PRICE": 10.0,
        }]),
    }
    for name, frame in tables.items():
        target = raw / name / partition
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
    free = raw / "equ_free_shares"
    free.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "PARTY_ID": 1, "CHANGE_DATE": date(2010, 1, 1),
        "FREE_SHARES": 7.0e8,
    }]).to_parquet(free / "part-all.parquet", index=False)

    canonicalizer = HermesCanonicalizer(request)
    capital = canonicalizer.canonical / "share_capital"
    capital.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "instrument_id": "000001.XSHE",
        "effective_from": date(2010, 1, 1),
        "total_shares": 1.0e9,
        "tradable_shares": 8.0e8,
        "a_shares": 1.0e9,
    }]).to_parquet(capital / "part-000.parquet", index=False)
    canonicalizer.instrument_by_security = {2: "000001.XSHE"}
    canonicalizer.instrument_by_party = {1: "000001.XSHE"}
    return canonicalizer, day


def test_dividend_yield_uses_the_trailing_twelve_month_series():
    """DIV_RATE_TTM is empty from January to April; L12M is not.

    The source publishes several yield definitions. TTM rolls off the last
    completed report and is blank for roughly a third of every calendar year,
    which silently emptied any trailing-yield screen over that stretch.
    """
    with tempfile.TemporaryDirectory() as directory:
        dividend = pd.DataFrame([{
            "SECURITY_ID": 2, "TRADE_DATE": date(2016, 1, 4), "ID": 1,
            "UPDATE_TIME": pd.Timestamp("2016-01-05"),
            "DIV_RATE_L12M": 4.5,   # percent
            "DIV_RATE_TTM": None,   # the January hole
        }])
        canonicalizer, day = _valuation_canonicalizer(
            Path(directory), dividend
        )
        canonicalizer._build_valuation()

        frame = pd.read_parquet(
            canonicalizer.canonical / "daily_valuation"
        )
        row = frame[frame["trade_date"] == day].iloc[0]
        # Percent in the source, ratio in the contract.
        assert row["dividend_yield"] == pytest.approx(0.045)


def test_special_treatment_is_readable_but_never_required():
    from zyquant.data.contracts import (
        BASE_TABLES, FINANCIAL_TABLES, OPTIONAL_TABLES, TABLES,
    )

    assert "special_treatment" in TABLES
    assert "special_treatment" in OPTIONAL_TABLES
    # Absent from the required groups, so snapshots published before the table
    # existed still validate and still publish.
    assert "special_treatment" not in BASE_TABLES
    assert "special_treatment" not in FINANCIAL_TABLES


def test_halted_bar_may_carry_no_price_band():
    from zyquant.data.contracts import FIELD_SPECS

    # The source publishes no limit for a halted day and the execution layer
    # rejects such a bar before consulting the band, so it must be nullable.
    assert FIELD_SPECS["daily_raw"]["limit_up"].nullable
    assert FIELD_SPECS["daily_raw"]["limit_down"].nullable

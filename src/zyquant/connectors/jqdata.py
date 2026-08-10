from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import yaml

from zyquant.core.exceptions import DataContractError
from zyquant.core.hashing import hash_payload
from zyquant.core.plugins import PluginMetadata

from zyquant.data.adapters import CanonicalBatch
from zyquant.data.contracts import FIELD_SPECS, REQUIRED_COLUMNS
from zyquant.data.financial import (
    FUNDAMENTAL_CALCULATION_VERSION,
    ITEM_CATALOG_VERSION,
    FinancialProcessor,
)


DEFAULT_SAMPLE_INSTRUMENTS = (
    "600000.XSHG",
    "000001.XSHE",
    "510300.XSHG",
)


@dataclass(frozen=True)
class JQFinancialRequest:
    enabled: bool = False
    report_start_date: date = date(2020, 1, 1)
    valuation_start_date: date | None = None
    scope: str = "explicit"
    batch_size: int = 20
    strict_permissions: bool = True

    def __post_init__(self) -> None:
        if self.scope not in {"explicit", "universe"}:
            raise ValueError(
                "JQFinancialRequest.scope must be 'explicit' or 'universe'"
            )
        if self.batch_size < 1:
            raise ValueError("JQFinancialRequest.batch_size must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "JQFinancialRequest":
        payload = dict(value)
        for key in ("report_start_date", "valuation_start_date"):
            if isinstance(payload.get(key), str):
                payload[key] = date.fromisoformat(payload[key])
        return cls(**payload)

    def public_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "report_start_date": self.report_start_date.isoformat(),
            "valuation_start_date": (
                self.valuation_start_date.isoformat()
                if self.valuation_start_date is not None else None
            ),
            "scope": self.scope,
            "batch_size": self.batch_size,
            "strict_permissions": self.strict_permissions,
        }


@dataclass(frozen=True)
class JQDataCredentials:
    username: str = field(repr=False)
    password: str = field(repr=False)

    @classmethod
    def from_env(cls) -> "JQDataCredentials":
        username = os.environ.get("JQDATA_USERNAME", "").strip()
        password = os.environ.get("JQDATA_PASSWORD", "")
        if not username or not password:
            raise DataContractError(
                "JQData credentials are not configured; set JQDATA_USERNAME and "
                "JQDATA_PASSWORD in the process environment"
            )
        return cls(username, password)


@dataclass(frozen=True)
class JQDataRequest:
    start_date: date
    end_date: date
    instruments: tuple[str, ...] = DEFAULT_SAMPLE_INSTRUMENTS
    universe_id: str = "000300.XSHG"
    industry_classification: str = "sw_l1"
    batch_size: int = 20
    price_scope: str = "explicit"
    strict_permissions: bool = True
    max_retries: int = 2
    etf_sell_delay_overrides: tuple[tuple[str, int], ...] = ()
    financial: JQFinancialRequest | None = None

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("JQDataRequest.start_date must not exceed end_date")
        if not self.instruments:
            raise ValueError("JQDataRequest.instruments must not be empty")
        if len(set(self.instruments)) != len(self.instruments):
            raise ValueError("JQDataRequest.instruments must be unique")
        if self.batch_size < 1:
            raise ValueError("JQDataRequest.batch_size must be positive")
        if self.price_scope not in {"explicit", "universe"}:
            raise ValueError(
                "JQDataRequest.price_scope must be 'explicit' or 'universe'"
            )
        if self.max_retries < 0:
            raise ValueError("JQDataRequest.max_retries must be non-negative")
        for instrument, delay in self.etf_sell_delay_overrides:
            if instrument not in self.instruments or delay < 0:
                raise ValueError("invalid ETF sell-delay override")

    @classmethod
    def sample_2025(cls) -> "JQDataRequest":
        return cls(date(2025, 1, 1), date(2025, 12, 31))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "JQDataRequest":
        payload = dict(value)
        for key in ("start_date", "end_date"):
            if isinstance(payload.get(key), str):
                payload[key] = date.fromisoformat(payload[key])
        if "instruments" in payload:
            payload["instruments"] = tuple(str(item) for item in payload["instruments"])
        if "etf_sell_delay_overrides" in payload:
            overrides = payload["etf_sell_delay_overrides"]
            if isinstance(overrides, Mapping):
                payload["etf_sell_delay_overrides"] = tuple(
                    sorted((str(code), int(delay)) for code, delay in overrides.items())
                )
            else:
                payload["etf_sell_delay_overrides"] = tuple(
                    (str(code), int(delay)) for code, delay in overrides
                )
        if isinstance(payload.get("financial"), Mapping):
            payload["financial"] = JQFinancialRequest.from_mapping(
                payload["financial"]
            )
        return cls(**payload)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "JQDataRequest":
        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, Mapping):
            raise DataContractError("JQData request file must contain a mapping")
        return cls.from_mapping(payload)

    def public_payload(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "instruments": list(self.instruments),
            "universe_id": self.universe_id,
            "industry_classification": self.industry_classification,
            "batch_size": self.batch_size,
            "price_scope": self.price_scope,
            "strict_permissions": self.strict_permissions,
            "max_retries": self.max_retries,
            "etf_sell_delay_overrides": dict(self.etf_sell_delay_overrides),
            "financial": (
                self.financial.public_payload()
                if self.financial is not None else None
            ),
        }


class JQDataClientProtocol(Protocol):
    sdk_version: str

    def authenticate(self) -> None: ...

    def get_privilege(self) -> Any: ...

    def get_query_count(self) -> Any: ...

    def get_all_securities(
        self, types: Sequence[str], as_of: date,
    ) -> pd.DataFrame: ...

    def get_trade_days(self, start_date: date, end_date: date) -> Sequence[Any]: ...

    def get_price(
        self,
        instruments: Sequence[str],
        start_date: date,
        end_date: date,
        fields: Sequence[str],
    ) -> pd.DataFrame: ...

    def get_corporate_actions(
        self,
        instruments: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame: ...

    def get_index_stocks(self, universe_id: str, day: date) -> Sequence[str]: ...

    def get_history_industry(
        self, classification: str, instruments: Sequence[str],
    ) -> pd.DataFrame: ...

    def get_industry(
        self, instruments: Sequence[str], day: date,
    ) -> Mapping[str, Any]: ...

    def get_financial_statements(
        self,
        statement_type: str,
        instruments: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame: ...

    def get_valuation(
        self,
        instruments: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame: ...

    def get_share_capital(
        self,
        instruments: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame: ...


class JQDataSDKClient:
    """Thin, credential-safe wrapper around the optional jqdatasdk package."""

    def __init__(self, credentials: JQDataCredentials):
        try:
            import jqdatasdk as sdk
        except ImportError as exc:
            raise DataContractError(
                "JQData connector requires: pip install 'zyquant[jqdata]'"
            ) from exc
        self._sdk = sdk
        self._credentials = credentials
        try:
            from importlib.metadata import version

            self.sdk_version = version("jqdatasdk")
        except Exception:
            self.sdk_version = "unknown"

    def authenticate(self) -> None:
        try:
            self._sdk.auth(
                self._credentials.username,
                self._credentials.password,
            )
        except Exception as exc:
            self._raise_sanitized("authentication", exc)

    def get_privilege(self) -> Any:
        return self._sdk.get_privilege()

    def get_query_count(self) -> Any:
        return self._sdk.get_query_count()

    def get_all_securities(
        self, types: Sequence[str], as_of: date,
    ) -> pd.DataFrame:
        return self._sdk.get_all_securities(list(types), date=as_of)

    def get_trade_days(self, start_date: date, end_date: date) -> Sequence[Any]:
        return self._sdk.get_trade_days(
            start_date=start_date,
            end_date=end_date,
        )

    def get_price(
        self,
        instruments: Sequence[str],
        start_date: date,
        end_date: date,
        fields: Sequence[str],
    ) -> pd.DataFrame:
        raw_fields = [field for field in fields if field != "factor"]
        raw = self._sdk.get_price(
            list(instruments),
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=raw_fields,
            skip_paused=False,
            fq=None,
            panel=False,
            fill_paused=True,
            round=False,
        )
        factors = self._sdk.get_price(
            list(instruments),
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=["factor"],
            skip_paused=False,
            fq="post",
            panel=False,
            fill_paused=True,
            round=False,
        )
        raw = self._flat_price_response(raw, instruments)
        factors = self._flat_price_response(factors, instruments)
        keys = ["time", "code"]
        if factors.duplicated(keys).any():
            raise DataContractError("JQData factor response contains duplicate keys")
        return raw.merge(
            factors[keys + ["factor"]],
            on=keys,
            how="left",
            validate="one_to_one",
        )

    @staticmethod
    def _flat_price_response(
        frame: pd.DataFrame,
        instruments: Sequence[str],
    ) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("JQData price response must be a DataFrame")
        result = frame.copy()
        if "time" not in result:
            result.index.name = result.index.name or "time"
            result = result.reset_index()
        if "code" not in result:
            if len(instruments) != 1:
                raise DataContractError(
                    "JQData multi-security response is missing code"
                )
            result["code"] = instruments[0]
        return result

    def get_corporate_actions(
        self,
        instruments: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        try:
            stock_table = self._sdk.finance.STK_XR_XD
            stock_query = self._sdk.query(stock_table).filter(
                stock_table.code.in_(list(instruments)),
                stock_table.a_xr_date >= start_date,
                stock_table.a_xr_date <= end_date,
            ).limit(5000)
            stock = self._sdk.finance.run_query(stock_query)
            stock["source_table"] = "STK_XR_XD"

            fund_table = self._sdk.finance.FUND_DIVIDEND
            local_codes = [code.split(".", 1)[0] for code in instruments]
            fund_query = self._sdk.query(fund_table).filter(
                fund_table.code.in_(local_codes),
                fund_table.ex_date >= start_date,
                fund_table.ex_date <= end_date,
            ).limit(5000)
            fund = self._sdk.finance.run_query(fund_query)
            fund["source_table"] = "FUND_DIVIDEND"
            return pd.concat([stock, fund], ignore_index=True, sort=False)
        except Exception as exc:
            self._raise_sanitized("corporate-actions query", exc)

    def get_index_stocks(self, universe_id: str, day: date) -> Sequence[str]:
        return self._sdk.get_index_stocks(universe_id, date=day)

    def get_history_industry(
        self, classification: str, instruments: Sequence[str],
    ) -> pd.DataFrame:
        return self._sdk.get_history_industry(
            classification,
            securities=list(instruments),
        )

    def get_industry(
        self, instruments: Sequence[str], day: date,
    ) -> Mapping[str, Any]:
        return self._sdk.get_industry(
            list(instruments),
            date=day,
            df=False,
        )

    def get_financial_statements(
        self,
        statement_type: str,
        instruments: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        names = {
            "balance": "STK_BALANCE_SHEET",
            "income": "STK_INCOME_STATEMENT",
            "cash_flow": "STK_CASHFLOW_STATEMENT",
        }
        if statement_type not in names:
            raise DataContractError(
                f"unsupported financial statement type: {statement_type}"
            )
        table = getattr(self._sdk.finance, names[statement_type])
        return self._finance_paged(
            table,
            instruments,
            table.end_date >= start_date,
            table.pub_date <= end_date,
        )

    def get_valuation(
        self,
        instruments: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        try:
            return self._sdk.get_valuation(
                list(instruments),
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            self._raise_sanitized("valuation query", exc)

    def get_share_capital(
        self,
        instruments: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        table = self._sdk.finance.STK_CAPITAL_CHANGE
        return self._finance_paged(
            table,
            instruments,
            table.change_date >= start_date,
            table.pub_date <= end_date,
        )

    def _finance_paged(
        self,
        table,
        instruments: Sequence[str],
        *conditions,
    ) -> pd.DataFrame:
        outputs = []
        last_id = -1
        try:
            while True:
                query = self._sdk.query(table).filter(
                    table.code.in_(list(instruments)),
                    table.id > last_id,
                    *conditions,
                ).order_by(table.id.asc()).limit(5000)
                frame = self._sdk.finance.run_query(query)
                if not isinstance(frame, pd.DataFrame):
                    raise DataContractError(
                        "JQData finance query must return a DataFrame"
                    )
                if frame.empty:
                    break
                outputs.append(frame)
                next_id = int(pd.to_numeric(frame["id"]).max())
                if next_id <= last_id:
                    raise DataContractError(
                        "JQData finance pagination did not advance"
                    )
                last_id = next_id
                if len(frame) < 5000:
                    break
        except DataContractError:
            raise
        except Exception as exc:
            self._raise_sanitized("finance query", exc)
        return (
            pd.concat(outputs, ignore_index=True, sort=False)
            if outputs else pd.DataFrame()
        )

    def _raise_sanitized(self, operation: str, exc: Exception) -> None:
        message = str(exc)
        for secret in (
            self._credentials.username,
            self._credentials.password,
        ):
            if secret:
                message = message.replace(secret, "***REDACTED***")
        raise DataContractError(
            f"JQData {operation} failed ({type(exc).__name__}): {message}"
        ) from exc


class JQDataAdapter:
    """Convert a bounded JQData request into ZyQuant canonical tables."""

    PRICE_FIELDS = (
        "open", "high", "low", "close", "pre_close",
        "volume", "money", "paused", "high_limit", "low_limit", "factor",
    )
    CASH_FIELDS = (
        ("bonus_ratio_rmb", 10.0),
        ("cash_dividend_ratio", 10.0),
        ("cash_per_share", 1.0),
    )
    BONUS_FIELDS = (
        ("bonus_ratio", 10.0),
        ("bonus_share_ratio", 10.0),
    )
    TRANSFER_FIELDS = (
        ("transfer_ratio", 10.0),
        ("transfer_share_ratio", 10.0),
    )

    def __init__(
        self,
        credentials: JQDataCredentials | None = None,
        client: JQDataClientProtocol | None = None,
    ):
        self._credentials = credentials
        self._client = client
        self._warnings: list[str] = []

    def ingest(
        self,
        request: JQDataRequest | Mapping[str, Any] | None = None,
    ) -> CanonicalBatch:
        resolved = self._resolve_request(request)
        self._warnings = []
        client = self._client or JQDataSDKClient(
            self._credentials or JQDataCredentials.from_env()
        )
        self._client = client
        self._call(resolved, "authenticate", client.authenticate)
        privilege = self._call(resolved, "get_privilege", client.get_privilege)
        before = self._call(resolved, "get_query_count", client.get_query_count)

        request_hash = hash_payload(resolved.public_payload())
        batch_id = f"jqdata-{request_hash[:16]}"
        calendar = self._calendar(client, resolved, batch_id)
        universe = self._universe(client, resolved, calendar, batch_id)
        universe_codes = tuple(sorted(
            set(universe["instrument_id"].astype(str))
        ))
        price_codes = (
            universe_codes
            if resolved.price_scope == "universe"
            else resolved.instruments
        )
        catalog_codes = tuple(sorted(
            set(resolved.instruments) | set(universe_codes)
        ))
        first_seen = {
            str(code): min(group["effective_from"])
            for code, group in universe.groupby("instrument_id")
        }
        instruments = self._instruments(
            client, resolved, batch_id, catalog_codes, first_seen
        )
        daily_raw, factors = self._prices(
            client, resolved, batch_id, price_codes
        )
        actions = self._corporate_actions(
            client, resolved, batch_id, price_codes
        )
        industry = self._industry(
            client, resolved, instruments, calendar, batch_id
        )
        rules = self._demo_market_rules(resolved, batch_id)
        financial_tables: dict[str, pd.DataFrame] = {}
        financial_capability: dict[str, Any] | None = None
        if resolved.financial is not None and resolved.financial.enabled:
            requested_financial_codes = (
                set(universe_codes)
                if resolved.financial.scope == "universe"
                else set(resolved.instruments)
            )
            financial_codes = tuple(sorted(
                set(
                    instruments.loc[
                        instruments["asset_type"] == "stock", "instrument_id"
                    ].astype(str)
                )
                & requested_financial_codes
            ))
            financial_tables, financial_capability = self._financials(
                client,
                resolved,
                resolved.financial,
                financial_codes,
                calendar,
                batch_id,
            )

        if resolved.strict_permissions and actions.empty:
            raise DataContractError(
                "JQData returned no implemented corporate actions for the strict "
                "sample; verify STK_XR_XD entitlement and sample coverage"
            )
        if resolved.strict_permissions:
            covered_stocks = set(
                instruments.loc[
                    instruments["asset_type"] == "stock", "instrument_id"
                ].astype(str)
            )
            industry_stocks = set(industry["instrument_id"].astype(str))
            missing_industry = covered_stocks - industry_stocks
            if missing_industry:
                raise DataContractError(
                    "JQData industry history did not cover requested stocks: "
                    f"{sorted(missing_industry)}"
                )

        tables = {
            "instruments": instruments,
            "trade_calendar": calendar,
            "daily_raw": daily_raw,
            "corporate_actions": actions,
            "universe_membership": universe,
            "industry_membership": industry,
            "market_rules": rules,
            **financial_tables,
        }
        after = self._call(resolved, "get_query_count", client.get_query_count)
        table_hashes = {
            name: self._frame_hash(frame) for name, frame in sorted(tables.items())
        }
        metadata = {
            "adapter": type(self).__name__,
            "source": "JQData",
            "sdk_version": str(getattr(client, "sdk_version", "unknown")),
            "request_hash": request_hash,
            "request": resolved.public_payload(),
            "pulled_at": datetime.now(timezone.utc).isoformat(),
            "privilege": self._safe_metadata(privilege),
            "query_count_before": self._safe_metadata(before),
            "query_count_after": self._safe_metadata(after),
            "source_table_hashes": table_hashes,
            "vendor_factor_hash": self._frame_hash(factors),
            "visibility_assumptions": {
                "industry_known_at": "effective_from",
                "universe_known_at": "effective_from",
            },
            "coverage": {
                "universe_membership": {
                    "mode": "full",
                    "universe_id": resolved.universe_id,
                    "instrument_count": len(universe_codes),
                },
                "industry_membership": {
                    "mode": "full-universe-and-explicit",
                    "instrument_count": int(
                        (instruments["asset_type"] == "stock").sum()
                    ),
                },
                "prices": {
                    "mode": resolved.price_scope,
                    "instruments": list(price_codes),
                    "instrument_count": len(price_codes),
                },
                "corporate_actions": {
                    "mode": resolved.price_scope,
                    "instruments": list(price_codes),
                },
                "full_universe_backtest_ready": (
                    resolved.price_scope == "universe"
                ),
                **({
                    "financials": financial_capability["coverage"]
                } if financial_capability is not None else {}),
            },
            "capabilities": (
                {"financials": financial_capability}
                if financial_capability is not None else {}
            ),
            "warnings": list(self._warnings),
            "market_rules": "demo-scenario; replace before production use",
        }
        return CanonicalBatch(tables, factors, metadata)

    @staticmethod
    def _resolve_request(
        request: JQDataRequest | Mapping[str, Any] | None,
    ) -> JQDataRequest:
        if request is None:
            return JQDataRequest.sample_2025()
        if isinstance(request, JQDataRequest):
            return request
        return JQDataRequest.from_mapping(request)

    def _call(self, request: JQDataRequest, name: str, func, *args, **kwargs):
        attempts = 0
        while True:
            try:
                return func(*args, **kwargs)
            except DataContractError:
                raise
            except Exception as exc:
                if attempts >= request.max_retries or not self._retryable(exc):
                    raise DataContractError(
                        f"JQData {name} failed ({type(exc).__name__})"
                    ) from exc
                time.sleep(0.25 * (2 ** attempts))
                attempts += 1

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        retryable_names = {
            "Timeout", "TimeoutError", "ConnectionError",
            "ReadTimeout", "ConnectTimeout",
        }
        return type(exc).__name__ in retryable_names

    def _instruments(
        self,
        client: JQDataClientProtocol,
        request: JQDataRequest,
        batch_id: str,
        required_instruments: Sequence[str],
        historical_lookup_dates: Mapping[str, date] | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        def load_catalog(as_of: date, wanted: set[str] | None = None) -> None:
            for asset_type, source_type in (("stock", "stock"), ("etf", "etf")):
                frame = self._call(
                    request,
                    f"get_all_securities[{source_type}:{as_of}]",
                    client.get_all_securities,
                    [source_type],
                    as_of,
                )
                if not isinstance(frame, pd.DataFrame):
                    raise DataContractError(
                        "JQData get_all_securities must return DataFrame"
                    )
                current = frame.copy()
                if "instrument_id" not in current:
                    current.index = current.index.astype(str)
                    current.index.name = "instrument_id"
                    current = current.reset_index()
                current["instrument_id"] = current["instrument_id"].astype(str)
                if wanted is not None:
                    current = current[
                        current["instrument_id"].isin(wanted)
                    ].copy()
                current["asset_type"] = asset_type
                frames.append(current)

        required = set(map(str, required_instruments))
        load_catalog(request.end_date, required)
        available = {
            str(code)
            for frame in frames
            for code in frame["instrument_id"]
        }
        missing = required - available
        lookup_dates = historical_lookup_dates or {}
        grouped_missing: dict[date, set[str]] = {}
        for code in missing:
            lookup_day = lookup_dates.get(code, request.start_date)
            grouped_missing.setdefault(lookup_day, set()).add(code)
        for lookup_day, codes in sorted(grouped_missing.items()):
            load_catalog(lookup_day, codes)

        securities = pd.concat(frames, ignore_index=True, sort=False)
        securities["instrument_id"] = securities["instrument_id"].astype(str)
        securities = securities[
            securities["instrument_id"].isin(required_instruments)
        ].drop_duplicates("instrument_id", keep="first").copy()
        missing = required - set(securities["instrument_id"])
        if missing:
            raise DataContractError(
                f"JQData did not return requested instruments: {sorted(missing)}"
            )

        start_column = self._column(
            securities, ("start_date", "list_date"), "security list date"
        )
        end_column = self._column(
            securities, ("end_date", "delist_date"), "security delist date"
        )
        name_column = self._column(
            securities, ("display_name", "name"), "security name", required=False
        )
        overrides = dict(request.etf_sell_delay_overrides)
        rows = []
        for row in securities.to_dict("records"):
            code = str(row["instrument_id"])
            asset_type = str(row["asset_type"])
            list_date = self._as_date(row[start_column])
            if list_date is None:
                raise DataContractError(
                    f"JQData security metadata lacks list date for {code}"
                )
            security_name = (
                str(row.get(name_column) or "") if name_column is not None else ""
            )
            sell_delay = 1
            if asset_type == "etf":
                sell_delay = overrides.get(
                    code,
                    self._infer_etf_sell_delay(code, security_name, request),
                )
            rows.append({
                "instrument_id": code,
                "symbol": code.split(".", 1)[0],
                "exchange": code.rsplit(".", 1)[-1],
                "asset_type": asset_type,
                "list_date": list_date,
                "delist_date": self._delist_date(row[end_column]),
                "lot_size": 100,
                "sell_delay_days": sell_delay,
                "name": security_name or None,
                "currency": "CNY",
                "source_record_id": code,
                "source_batch_id": batch_id,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _infer_etf_sell_delay(
        code: str,
        security_name: str,
        request: JQDataRequest,
    ) -> int:
        t0_markers = (
            "跨境", "海外", "恒生", "标普", "纳指", "日经", "德国", "法国",
            "黄金", "商品", "原油", "债券", "货币",
        )
        t1_markers = (
            "沪深", "中证", "上证", "深证", "创业", "科创", "红利",
            "央企", "国企", "A股", "股票",
        )
        if any(marker in security_name for marker in t0_markers):
            return 0
        if any(marker in security_name for marker in t1_markers):
            return 1
        if code == "510300.XSHG":
            return 1
        if request.strict_permissions:
            raise DataContractError(
                f"cannot determine ETF settlement rule for {code}; provide "
                "etf_sell_delay_overrides"
            )
        return 1

    def _calendar(
        self,
        client: JQDataClientProtocol,
        request: JQDataRequest,
        batch_id: str,
    ) -> pd.DataFrame:
        values = self._call(
            request,
            "get_trade_days",
            client.get_trade_days,
            request.start_date,
            request.end_date,
        )
        days = sorted({
            pd.Timestamp(item).date()
            for item in values
            if pd.notna(item)
        })
        if not days:
            raise DataContractError("JQData returned an empty trade calendar")
        return pd.DataFrame([
            {
                "trade_date": day,
                "exchange": exchange,
                "source_batch_id": batch_id,
            }
            for exchange in ("XSHG", "XSHE")
            for day in days
        ])

    def _prices(
        self,
        client: JQDataClientProtocol,
        request: JQDataRequest,
        batch_id: str,
        instruments: Sequence[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        outputs = []
        for start in range(0, len(instruments), request.batch_size):
            batch = instruments[start:start + request.batch_size]
            frame = self._call(
                request,
                f"get_price[{start // request.batch_size}]",
                client.get_price,
                batch,
                request.start_date,
                request.end_date,
                self.PRICE_FIELDS,
            )
            outputs.append(self._price_frame(frame, batch))
        source = pd.concat(outputs, ignore_index=True, sort=False)
        source = source[source["instrument_id"].isin(instruments)].copy()
        price_columns = ["open", "high", "low", "close", "pre_close"]
        source = source[~source[price_columns].isna().all(axis=1)].copy()
        if source.empty:
            raise DataContractError("JQData returned no usable daily prices")
        if source[price_columns + ["factor"]].isna().any().any():
            bad = source.loc[
                source[price_columns + ["factor"]].isna().any(axis=1),
                ["trade_date", "instrument_id"],
            ].head(5)
            raise DataContractError(
                "JQData daily prices or factors contain nulls after paused-price "
                f"filling: {bad.to_dict('records')}"
            )

        factors = source[
            ["trade_date", "instrument_id", "factor"]
        ].rename(columns={"factor": "adjustment_factor"})
        raw = source.rename(columns={
            "money": "amount",
            "high_limit": "limit_up",
            "low_limit": "limit_down",
        })
        volume = pd.to_numeric(raw["volume"], errors="coerce")
        if (
            volume.isna().any()
            or not np.isfinite(volume.to_numpy(dtype=float)).all()
            or not np.allclose(
                volume.to_numpy(dtype=float),
                volume.round().to_numpy(dtype=float),
                rtol=0,
                atol=1e-9,
            )
        ):
            raise DataContractError("JQData volume cannot be converted safely to shares")
        raw["volume"] = volume.round()
        raw["paused"] = raw["paused"].fillna(False).astype(bool)
        raw["source_batch_id"] = batch_id
        raw = raw[[
            "trade_date", "instrument_id", "open", "high", "low", "close",
            "pre_close", "volume", "amount", "paused", "limit_up", "limit_down",
            "source_batch_id",
        ]]
        return (
            raw.sort_values(["trade_date", "instrument_id"], ignore_index=True),
            factors.sort_values(["trade_date", "instrument_id"], ignore_index=True),
        )

    @staticmethod
    def _price_frame(
        frame: pd.DataFrame,
        instruments: Sequence[str],
    ) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("JQData get_price must return DataFrame")
        result = frame.copy()
        if "time" not in result and "date" not in result and "day" not in result:
            index_name = result.index.name or "time"
            result.index.name = index_name
            result = result.reset_index()
        date_column = next(
            (name for name in ("time", "date", "day", "datetime") if name in result),
            None,
        )
        if date_column is None:
            raise DataContractError("JQData price response has no date column")
        code_column = next(
            (name for name in ("code", "security", "instrument_id") if name in result),
            None,
        )
        if code_column is None:
            if len(instruments) != 1:
                raise DataContractError("JQData multi-security price response has no code")
            result["code"] = instruments[0]
            code_column = "code"
        result.rename(columns={
            date_column: "trade_date",
            code_column: "instrument_id",
        }, inplace=True)
        required = set(JQDataAdapter.PRICE_FIELDS) | {
            "trade_date", "instrument_id",
        }
        missing = required - set(result)
        if missing:
            raise DataContractError(
                f"JQData price response missing fields: {sorted(missing)}"
            )
        result["trade_date"] = pd.to_datetime(
            result["trade_date"], errors="coerce"
        ).dt.date
        result["instrument_id"] = result["instrument_id"].astype(str)
        return result

    def _financials(
        self,
        client: JQDataClientProtocol,
        request: JQDataRequest,
        financial: JQFinancialRequest,
        instruments: Sequence[str],
        calendar: pd.DataFrame,
        batch_id: str,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
        if not instruments:
            raise DataContractError(
                "financial capability requires at least one stock instrument"
            )
        warmup = date(
            max(1900, financial.report_start_date.year - 1), 1, 1
        )
        statement_outputs: dict[str, list[pd.DataFrame]] = {
            "balance": [], "income": [], "cash_flow": [],
        }
        for start in range(0, len(instruments), financial.batch_size):
            batch = instruments[start:start + financial.batch_size]
            for statement_type in statement_outputs:
                frame = self._call(
                    request,
                    f"get_financial_statements[{statement_type}:{start}]",
                    client.get_financial_statements,
                    statement_type,
                    batch,
                    warmup,
                    request.end_date,
                )
                if not isinstance(frame, pd.DataFrame):
                    raise DataContractError(
                        "JQData financial statement query must return DataFrame"
                    )
                statement_outputs[statement_type].append(frame)
        statements = {
            name: pd.concat(outputs, ignore_index=True, sort=False)
            if outputs else pd.DataFrame()
            for name, outputs in statement_outputs.items()
        }
        if financial.strict_permissions:
            missing = [
                name for name, frame in statements.items() if frame.empty
            ]
            if missing:
                raise DataContractError(
                    "JQData financial statements are unavailable: "
                    f"{missing}; verify FINANCE entitlement"
                )
            for name, frame in statements.items():
                uncovered = set(instruments) - set(
                    frame["code"].astype(str)
                    if "code" in frame else ()
                )
                if uncovered:
                    raise DataContractError(
                        f"JQData {name} statements do not cover requested stocks: "
                        f"{sorted(uncovered)}"
                    )
        days = sorted(set(calendar["trade_date"]))
        valuation = self._financial_valuation(
            client, request, financial, instruments, batch_id
        )
        if financial.strict_permissions:
            missing_valuation = set(instruments) - set(
                valuation["instrument_id"].astype(str)
            )
            if missing_valuation:
                raise DataContractError(
                    "JQData valuation does not cover requested stocks: "
                    f"{sorted(missing_valuation)}"
                )
        capital = self._share_capital(
            client, request, financial, instruments, days, batch_id
        )
        built = FinancialProcessor().build(
            statements, days, batch_id, capital
        )
        capability = {
            "schema_version": "1.0",
            "item_catalog_version": ITEM_CATALOG_VERSION,
            "calculation_version": FUNDAMENTAL_CALCULATION_VERSION,
            "pit_validated": True,
            "coverage": {
                "mode": financial.scope,
                "instruments": list(instruments),
                "excluded": {
                    code: "asset_type_not_stock"
                    for code in request.instruments
                    if code not in instruments
                },
                "report_period_start": financial.report_start_date.isoformat(),
                "source_warmup_start": warmup.isoformat(),
                "valuation_start": (
                    financial.valuation_start_date or request.start_date
                ).isoformat(),
                "end_date": request.end_date.isoformat(),
            },
        }
        return {
            "financial_reports": built.reports,
            "financial_facts": built.facts,
            "fundamental_metrics": built.metrics,
            "daily_valuation": valuation,
            "share_capital": capital,
        }, capability

    def _financial_valuation(
        self,
        client: JQDataClientProtocol,
        request: JQDataRequest,
        financial: JQFinancialRequest,
        instruments: Sequence[str],
        batch_id: str,
    ) -> pd.DataFrame:
        outputs = []
        start_date = financial.valuation_start_date or request.start_date
        for start in range(0, len(instruments), financial.batch_size):
            batch = instruments[start:start + financial.batch_size]
            frame = self._call(
                request,
                f"get_valuation[{start}]",
                client.get_valuation,
                batch,
                start_date,
                request.end_date,
            )
            if not isinstance(frame, pd.DataFrame):
                raise DataContractError(
                    "JQData valuation query must return DataFrame"
                )
            outputs.append(frame)
        source = pd.concat(outputs, ignore_index=True, sort=False)
        required = {"code", "day"}
        missing = required - set(source)
        if missing:
            raise DataContractError(
                f"JQData valuation response missing fields: {sorted(missing)}"
            )

        def scaled(name: str, multiplier: float = 1.0) -> pd.Series:
            values = (
                source[name] if name in source
                else pd.Series(np.nan, index=source.index)
            )
            return pd.to_numeric(values, errors="coerce") * multiplier

        trade_dates = pd.to_datetime(source["day"], errors="coerce").dt.date
        result = pd.DataFrame({
            "trade_date": trade_dates,
            "instrument_id": source["code"].astype(str),
            "pe_ttm": scaled("pe_ratio"),
            "pe_lyr": scaled("pe_ratio_lyr"),
            "pb": scaled("pb_ratio"),
            "ps_ttm": scaled("ps_ratio"),
            "pcf_ttm": scaled("pcf_ratio"),
            "pcf_operating_ttm": scaled("pcf_ratio2"),
            "dividend_yield": scaled("dividend_ratio", 0.01),
            "turnover_rate": scaled("turnover_ratio", 0.01),
            "total_shares": scaled("capitalization", 10_000.0),
            "market_cap": scaled("market_cap", 100_000_000.0),
            "circulating_shares": scaled("circulating_cap", 10_000.0),
            "circulating_market_cap": scaled(
                "circulating_market_cap", 100_000_000.0
            ),
            "free_float_shares": scaled("free_cap", 10_000.0),
            "free_float_market_cap": scaled(
                "free_market_cap", 100_000_000.0
            ),
            "a_shares": scaled("a_cap", 10_000.0),
            "a_market_cap": scaled("a_market_cap", 100_000_000.0),
            "available_at": trade_dates,
            "source_batch_id": batch_id,
        })
        result["source_record_id"] = [
            hash_payload({
                "source": "JQData",
                "table": "valuation",
                "code": code,
                "day": day.isoformat(),
            })
            for code, day in zip(
                result["instrument_id"], result["trade_date"], strict=True
            )
        ]
        return result

    def _share_capital(
        self,
        client: JQDataClientProtocol,
        request: JQDataRequest,
        financial: JQFinancialRequest,
        instruments: Sequence[str],
        trade_days: Sequence[date],
        batch_id: str,
    ) -> pd.DataFrame:
        outputs = []
        warmup = date(
            max(1900, financial.report_start_date.year - 1), 1, 1
        )
        for start in range(0, len(instruments), financial.batch_size):
            batch = instruments[start:start + financial.batch_size]
            frame = self._call(
                request,
                f"get_share_capital[{start}]",
                client.get_share_capital,
                batch,
                warmup,
                request.end_date,
            )
            if not isinstance(frame, pd.DataFrame):
                raise DataContractError(
                    "JQData share-capital query must return DataFrame"
                )
            outputs.append(frame)
        source = pd.concat(outputs, ignore_index=True, sort=False)
        columns = [
            "capital_event_id", "instrument_id", "effective_from",
            "announced_at", "available_at", "change_reason_code",
            "change_reason", "total_shares", "nontradable_shares",
            "restricted_shares", "tradable_shares", "a_shares",
            "b_shares", "h_shares", "source_record_id", "source_batch_id",
        ]
        if source.empty:
            return pd.DataFrame(columns=columns)
        required = {"id", "code", "change_date", "pub_date", "share_total"}
        missing = required - set(source)
        if missing:
            raise DataContractError(
                f"JQData share capital missing fields: {sorted(missing)}"
            )

        def number(name: str) -> pd.Series:
            values = (
                source[name] if name in source
                else pd.Series(np.nan, index=source.index)
            )
            return pd.to_numeric(values, errors="coerce")

        announced = pd.to_datetime(
            source["pub_date"], errors="coerce"
        ).dt.date
        available = announced.map(
            lambda day: FinancialProcessor._next_trade_day(day, trade_days)
            if day is not None and not pd.isna(day) else None
        )
        source_ids = source["id"].astype(str)
        result = pd.DataFrame({
            "capital_event_id": [
                hash_payload({
                    "source": "JQData",
                    "table": "STK_CAPITAL_CHANGE",
                    "id": source_id,
                })
                for source_id in source_ids
            ],
            "instrument_id": source["code"].astype(str),
            "effective_from": pd.to_datetime(
                source["change_date"], errors="coerce"
            ).dt.date,
            "announced_at": announced,
            "available_at": available,
            "change_reason_code": source.get("change_reason_id"),
            "change_reason": source.get("change_reason"),
            "total_shares": number("share_total"),
            "nontradable_shares": number("share_non_trade"),
            "restricted_shares": number("share_limited"),
            "tradable_shares": number("share_trade_total"),
            "a_shares": number("share_rmb"),
            "b_shares": number("share_b"),
            "h_shares": number("share_h"),
            "source_record_id": source_ids,
            "source_batch_id": batch_id,
        })
        return result[columns]

    def _corporate_actions(
        self,
        client: JQDataClientProtocol,
        request: JQDataRequest,
        batch_id: str,
        instruments: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        target_instruments = tuple(instruments or request.instruments)
        source = self._call(
            request,
            "get_corporate_actions",
            client.get_corporate_actions,
            target_instruments,
            request.start_date,
            request.end_date,
        )
        if not isinstance(source, pd.DataFrame):
            raise DataContractError(
                "JQData corporate-actions query must return DataFrame"
            )
        source = self._normalize_action_sources(source, target_instruments)
        columns = list(REQUIRED_COLUMNS["corporate_actions"]) + [
            "source_record_id", "source_batch_id",
        ]
        if source.empty:
            return pd.DataFrame(columns=columns)

        code_column = self._column(
            source, ("code", "security", "instrument_id"), "action security"
        )
        ex_column = self._column(
            source, ("a_xr_date", "ex_date"), "action ex-date"
        )
        id_column = self._column(
            source, ("id", "record_id", "source_record_id"),
            "action source id", required=False,
        )
        record_column = self._column(
            source,
            ("a_registration_date", "registration_date", "record_date"),
            "action record date",
            required=False,
        )
        pay_column = self._column(
            source,
            (
                "a_bonus_amount_rmb_date", "bonus_amount_rmb_date",
                "dividend_payment_date", "pay_date",
            ),
            "action pay date",
            required=False,
        )
        announced_column = self._column(
            source,
            (
                "implementation_pub_date", "implementation_announcement_date",
                "pub_date", "announced_at",
            ),
            "implementation announcement date",
            required=request.strict_permissions,
        )
        progress_column = self._column(
            source,
            ("plan_progress", "plan_progress_code", "status"),
            "action progress",
            required=False,
        )
        rows = []
        for index, record in source.reset_index(drop=True).iterrows():
            code = str(record[code_column])
            if code not in target_instruments:
                continue
            ex_date = self._as_date(record[ex_column])
            if ex_date is None or not request.start_date <= ex_date <= request.end_date:
                continue
            if progress_column is not None:
                progress = str(record.get(progress_column, "")).lower()
                if any(marker in progress for marker in ("取消", "终止", "cancel")):
                    continue
                if (
                    request.strict_permissions
                    and progress
                    and not any(
                        marker in progress
                        for marker in ("实施", "完成", "implemented", "active")
                    )
                ):
                    continue
            announced = (
                self._as_date(record.get(announced_column))
                if announced_column is not None
                else ex_date
            )
            if announced is None:
                if request.strict_permissions:
                    raise DataContractError(
                        "JQData implemented action lacks announcement date"
                    )
                announced = ex_date
            source_id = (
                str(record.get(id_column))
                if id_column is not None and pd.notna(record.get(id_column))
                else hash_payload({
                    "code": code,
                    "ex_date": ex_date.isoformat(),
                    "row": index,
                })[:20]
            )
            common = {
                "instrument_id": code,
                "record_date": (
                    self._as_date(record.get(record_column))
                    if record_column is not None else None
                ),
                "ex_date": ex_date,
                "pay_date": (
                    self._as_date(record.get(pay_column))
                    if pay_column is not None else None
                ),
                "status": "active",
                "announced_at": announced,
                "source_record_id": source_id,
                "source_batch_id": batch_id,
            }
            cash = self._ratio(record, self.CASH_FIELDS)
            shares = (
                self._ratio(record, self.BONUS_FIELDS)
                + self._ratio(record, self.TRANSFER_FIELDS)
            )
            if cash > 0:
                rows.append({
                    **common,
                    "event_id": hash_payload(
                        {"source": "JQData", "id": source_id, "type": "cash_dividend"}
                    ),
                    "event_type": "cash_dividend",
                    "cash_per_share": cash,
                    "share_ratio": 0.0,
                })
            if shares > 0:
                rows.append({
                    **common,
                    "event_id": hash_payload(
                        {"source": "JQData", "id": source_id, "type": "bonus"}
                    ),
                    "event_type": "bonus",
                    "cash_per_share": 0.0,
                    "share_ratio": shares,
                })
            fund_split = record.get("fund_split_ratio")
            if pd.notna(fund_split):
                split_ratio = float(fund_split)
                if split_ratio > 0 and not np.isclose(split_ratio, 1.0):
                    event_type = "split" if split_ratio > 1 else "merge"
                    rows.append({
                        **common,
                        "event_id": hash_payload({
                            "source": "JQData",
                            "id": source_id,
                            "type": event_type,
                        }),
                        "event_type": event_type,
                        "cash_per_share": 0.0,
                        "share_ratio": split_ratio,
                    })
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _normalize_action_sources(
        source: pd.DataFrame,
        instruments: Sequence[str],
    ) -> pd.DataFrame:
        if "source_table" not in source:
            return source
        stock = source[source["source_table"] != "FUND_DIVIDEND"].copy()
        fund = source[source["source_table"] == "FUND_DIVIDEND"].copy()
        if fund.empty:
            return stock
        by_symbol = {
            code.split(".", 1)[0]: code for code in instruments
        }
        normalized = pd.DataFrame({
            "id": fund["id"].map(lambda value: f"FUND_DIVIDEND:{value}"),
            "code": fund["code"].astype(str).map(by_symbol),
            "a_registration_date": fund.get("record_date"),
            "a_xr_date": fund.get("ex_date"),
            "a_bonus_amount_rmb_date": fund.get("fund_paid_date").combine_first(
                fund.get("pay_date")
            ),
            "implementation_pub_date": fund.get(
                "dividend_implement_date"
            ).combine_first(fund.get("pub_date")),
            "plan_progress": fund.get("process"),
            # FUND_DIVIDEND.proportion is already cash per fund share,
            # while STK_XR_XD.bonus_ratio_rmb is quoted per ten shares.
            "bonus_ratio_rmb": pd.to_numeric(
                fund.get("proportion"), errors="coerce"
            ) * 10.0,
            "bonus_ratio": 0.0,
            "transfer_ratio": 0.0,
            "fund_split_ratio": fund.get("split_ratio"),
        })
        normalized = normalized[normalized["code"].notna()]
        return pd.concat([stock, normalized], ignore_index=True, sort=False)

    def _universe(
        self,
        client: JQDataClientProtocol,
        request: JQDataRequest,
        calendar: pd.DataFrame,
        batch_id: str,
    ) -> pd.DataFrame:
        days = sorted(set(calendar["trade_date"]))
        open_intervals: dict[str, date] = {}
        rows: list[dict[str, Any]] = []
        previous: date | None = None
        for day in days:
            members = set(self._call(
                request,
                f"get_index_stocks[{day}]",
                client.get_index_stocks,
                request.universe_id,
                day,
            ))
            for code in sorted(set(open_intervals) - members):
                started = open_intervals.pop(code)
                rows.append({
                    "universe_id": request.universe_id,
                    "instrument_id": code,
                    "effective_from": started,
                    "effective_to": previous,
                    "known_at": started,
                    "source_batch_id": batch_id,
                })
            for code in sorted(members - set(open_intervals)):
                open_intervals[code] = day
            previous = day
        for code, started in sorted(open_intervals.items()):
            rows.append({
                "universe_id": request.universe_id,
                "instrument_id": code,
                "effective_from": started,
                "effective_to": None,
                "known_at": started,
                "source_batch_id": batch_id,
            })
        if not rows:
            raise DataContractError(
                f"JQData returned no members for universe {request.universe_id}"
            )
        columns = list(REQUIRED_COLUMNS["universe_membership"]) + ["source_batch_id"]
        return pd.DataFrame(rows, columns=columns)

    def _industry(
        self,
        client: JQDataClientProtocol,
        request: JQDataRequest,
        instruments: pd.DataFrame,
        calendar: pd.DataFrame,
        batch_id: str,
    ) -> pd.DataFrame:
        stock_codes = tuple(sorted(
            instruments.loc[
                instruments["asset_type"] == "stock", "instrument_id"
            ].astype(str)
        ))
        try:
            source = self._call(
                request,
                "get_history_industry",
                client.get_history_industry,
                request.industry_classification,
                stock_codes,
            )
        except DataContractError:
            self._warnings.append(
                "get_history_industry unavailable; reconstructed membership "
                "from date-bound get_industry queries"
            )
            return self._industry_by_day(
                client, request, instruments, calendar, batch_id
            )
        if not isinstance(source, pd.DataFrame):
            raise DataContractError("JQData industry history must return DataFrame")
        columns = list(REQUIRED_COLUMNS["industry_membership"]) + [
            "source_record_id", "source_batch_id",
        ]
        if source.empty:
            return pd.DataFrame(columns=columns)
        code_column = self._column(
            source, ("code", "security", "instrument_id"), "industry security"
        )
        industry_column = self._column(
            source,
            ("industry_code", "industry_id", "code_industry"),
            "industry code",
        )
        start_column = self._column(
            source,
            ("start_date", "in_date", "effective_from"),
            "industry effective-from",
        )
        end_column = self._column(
            source,
            ("end_date", "out_date", "effective_to"),
            "industry effective-to",
            required=False,
        )
        source_id_column = self._column(
            source, ("id", "record_id", "source_record_id"),
            "industry source id", required=False,
        )
        rows = []
        for index, record in source.reset_index(drop=True).iterrows():
            code = str(record[code_column])
            if code not in stock_codes:
                continue
            started = self._as_date(record[start_column])
            ended = (
                self._as_date(record.get(end_column))
                if end_column is not None else None
            )
            if started is None:
                continue
            if ended is not None and ended < request.start_date:
                continue
            if started > request.end_date:
                continue
            effective_from = max(started, request.start_date)
            effective_to = (
                min(ended, request.end_date) if ended is not None else None
            )
            source_id = (
                str(record.get(source_id_column))
                if source_id_column is not None
                and pd.notna(record.get(source_id_column))
                else hash_payload({
                    "classification": request.industry_classification,
                    "instrument": code,
                    "industry": str(record[industry_column]),
                    "start": started.isoformat(),
                    "row": index,
                })[:20]
            )
            rows.append({
                "classification": request.industry_classification,
                "industry_id": str(record[industry_column]),
                "instrument_id": code,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "known_at": effective_from,
                "source_record_id": source_id,
                "source_batch_id": batch_id,
            })
        return pd.DataFrame(rows, columns=columns)

    def _industry_by_day(
        self,
        client: JQDataClientProtocol,
        request: JQDataRequest,
        instruments: pd.DataFrame,
        calendar: pd.DataFrame,
        batch_id: str,
    ) -> pd.DataFrame:
        stocks = instruments.loc[
            instruments["asset_type"] == "stock",
            ["instrument_id", "list_date", "delist_date"],
        ].copy()
        stock_codes = tuple(stocks["instrument_id"].astype(str))
        if not stock_codes:
            return pd.DataFrame(columns=[
                *REQUIRED_COLUMNS["industry_membership"],
                "source_record_id", "source_batch_id",
            ])
        days = sorted(set(calendar["trade_date"]))
        current: dict[str, tuple[str, date]] = {}
        rows: list[dict[str, Any]] = []
        previous_day: date | None = None
        for day in days:
            active_codes = tuple(
                row.instrument_id
                for row in stocks.itertuples(index=False)
                if row.list_date <= day
                and (row.delist_date is None or row.delist_date >= day)
            )
            payload = self._call(
                request,
                f"get_industry[{day}]",
                client.get_industry,
                active_codes,
                day,
            )
            if not isinstance(payload, Mapping):
                raise DataContractError("JQData get_industry must return a mapping")
            observed: dict[str, str] = {}
            for code in active_codes:
                security = payload.get(code)
                if not isinstance(security, Mapping):
                    continue
                classification = security.get(request.industry_classification)
                if isinstance(classification, Mapping):
                    industry_id = (
                        classification.get("industry_code")
                        or classification.get("industry_id")
                        or classification.get("code")
                    )
                    if industry_id:
                        observed[code] = str(industry_id)
            if request.strict_permissions:
                missing = set(active_codes) - set(observed)
                if missing:
                    raise DataContractError(
                        f"JQData get_industry did not cover {sorted(missing)} on {day}"
                    )
            for code in stock_codes:
                new_industry = observed.get(code)
                existing = current.get(code)
                if existing is not None and existing[0] != new_industry:
                    rows.append(self._industry_interval(
                        request,
                        code,
                        existing[0],
                        existing[1],
                        previous_day,
                        batch_id,
                    ))
                    current.pop(code)
                if new_industry is not None and code not in current:
                    current[code] = (new_industry, day)
            previous_day = day
        for code, (industry_id, started) in sorted(current.items()):
            rows.append(self._industry_interval(
                request,
                code,
                industry_id,
                started,
                None,
                batch_id,
            ))
        columns = list(REQUIRED_COLUMNS["industry_membership"]) + [
            "source_record_id", "source_batch_id",
        ]
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _industry_interval(
        request: JQDataRequest,
        code: str,
        industry_id: str,
        started: date,
        ended: date | None,
        batch_id: str,
    ) -> dict[str, Any]:
        source_id = hash_payload({
            "classification": request.industry_classification,
            "instrument": code,
            "industry": industry_id,
            "start": started.isoformat(),
        })[:20]
        return {
            "classification": request.industry_classification,
            "industry_id": industry_id,
            "instrument_id": code,
            "effective_from": started,
            "effective_to": ended,
            "known_at": started,
            "source_record_id": source_id,
            "source_batch_id": batch_id,
        }

    @staticmethod
    def _demo_market_rules(
        request: JQDataRequest,
        batch_id: str,
    ) -> pd.DataFrame:
        rows = []
        for exchange in ("XSHG", "XSHE"):
            for asset_type in ("stock", "etf"):
                rows.append({
                    "rule_id": (
                        f"demo-2025-{exchange.lower()}-{asset_type}"
                    ),
                    "exchange": exchange,
                    "asset_type": asset_type,
                    "effective_from": request.start_date,
                    "effective_to": request.end_date,
                    "commission_bps": 2.5,
                    "minimum_commission": 5.0,
                    "sell_tax_bps": 5.0 if asset_type == "stock" else 0.0,
                    "buy_tax_bps": 0.0,
                    "transfer_fee_bps": 0.1 if asset_type == "stock" else 0.0,
                    "currency": "CNY",
                    "source": "zyquant-demo-scenario",
                    "rule_version": "2025.1",
                    "scenario": True,
                    "source_batch_id": batch_id,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _column(
        frame: pd.DataFrame,
        aliases: Sequence[str],
        label: str,
        required: bool = True,
    ) -> str | None:
        for name in aliases:
            if name in frame:
                return name
        if required:
            raise DataContractError(
                f"JQData response missing {label}; expected one of {list(aliases)}"
            )
        return None

    @staticmethod
    def _ratio(
        record: pd.Series,
        candidates: Sequence[tuple[str, float]],
    ) -> float:
        for name, divisor in candidates:
            if name in record.index and pd.notna(record[name]):
                value = float(record[name])
                if value < 0 or not np.isfinite(value):
                    raise DataContractError(
                        f"JQData corporate action has invalid {name}"
                    )
                return value / divisor
        return 0.0

    @staticmethod
    def _as_date(value: Any) -> date | None:
        if value is None or pd.isna(value):
            return None
        converted = pd.Timestamp(value)
        return converted.date()

    @staticmethod
    def _delist_date(value: Any) -> date | None:
        converted = JQDataAdapter._as_date(value)
        if converted is not None and converted.year >= 2200:
            return None
        return converted

    @staticmethod
    def _frame_hash(frame: pd.DataFrame) -> str:
        if frame.empty:
            return hash_payload({"columns": list(frame.columns), "rows": []})
        canonical = frame.copy()
        canonical = canonical.reindex(sorted(canonical.columns), axis=1)
        row_hashes = pd.util.hash_pandas_object(
            canonical, index=False, categorize=True
        ).astype(str).tolist()
        return hash_payload({"columns": list(canonical.columns), "rows": row_hashes})

    @staticmethod
    def _safe_metadata(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): JQDataAdapter._safe_metadata(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [JQDataAdapter._safe_metadata(item) for item in value]
        return str(value)


def canonical_empty_table(name: str) -> pd.DataFrame:
    """Return a correctly shaped empty canonical source table."""

    if name not in REQUIRED_COLUMNS:
        raise DataContractError(f"unknown canonical table: {name}")
    ordered = [
        column for column in FIELD_SPECS[name]
        if column in REQUIRED_COLUMNS[name]
    ]
    return pd.DataFrame(columns=ordered)


def create_adapter(
    request: Mapping[str, Any] | None = None,
) -> JQDataAdapter:
    return JQDataAdapter()


setattr(create_adapter, "plugin_metadata", PluginMetadata(
    name="jqdata",
    version="2.0.0",
    kind="data",
    input_schema="data.source/jqdata@1",
    output_schema="data.canonical_batch@1",
    optional_dependencies=("jqdatasdk>=1.9.8,<2",),
))

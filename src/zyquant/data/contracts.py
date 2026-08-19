from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pyarrow as pa


@dataclass(frozen=True)
class FieldSpec:
    """Machine-readable definition for one canonical data field."""

    arrow_type: pa.DataType
    nullable: bool = False
    unit: str | None = None
    enum: frozenset[str] | None = None
    minimum: float | None = None
    description: str = ""
    required: bool = True

    def arrow_field(self, name: str) -> pa.Field:
        metadata = {
            key: value.encode("utf-8")
            for key, value in {
                "unit": self.unit,
                "description": self.description or None,
                "enum": ",".join(sorted(self.enum)) if self.enum else None,
            }.items()
            if value is not None
        }
        return pa.field(
            name,
            self.arrow_type,
            nullable=self.nullable,
            metadata=metadata or None,
        )


STRING = pa.string()
DATE = pa.date32()
UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")
FLOAT = pa.float64()
INTEGER = pa.int64()
BOOLEAN = pa.bool_()

EXCHANGES: Final = frozenset({"XSHG", "XSHE", "XBEI"})
ASSET_TYPES: Final = frozenset({"stock", "etf"})
ACTION_TYPES: Final = frozenset({
    "cash_dividend", "bonus", "split", "merge", "rights_issue",
})
ACTION_STATUS: Final = frozenset({"active", "cancelled"})
CURRENCIES: Final = frozenset({"CNY"})
STATEMENT_TYPES: Final = frozenset({"balance", "income", "cash_flow"})
REPORT_KINDS: Final = frozenset({"current", "comparative"})
FACT_BASES: Final = frozenset({"instant", "ytd", "per_share", "ratio"})
METRIC_BASES: Final = frozenset({"instant", "ytd", "single_quarter", "ttm"})
FINANCIAL_UNITS: Final = frozenset({
    "CNY", "CNY/share", "shares", "ratio", "percent",
})
METRIC_QUALITY: Final = frozenset({
    "complete", "not_applicable", "source_missing",
})

BASE_TABLES: Final = (
    "instruments",
    "trade_calendar",
    "daily_raw",
    "daily_post_adjusted",
    "corporate_actions",
    "universe_membership",
    "industry_membership",
    "market_rules",
)
FINANCIAL_TABLES: Final = (
    "financial_reports",
    "financial_facts",
    "fundamental_metrics",
    "daily_valuation",
    "share_capital",
)
# Readable when a snapshot carries them, never required to publish one, so a
# snapshot published before they existed stays valid.
OPTIONAL_TABLES: Final = (
    "special_treatment",
    "daily_money_flow",
)
TABLES: Final = BASE_TABLES + FINANCIAL_TABLES + OPTIONAL_TABLES

SOURCE_FIELDS: Final = {
    "source_record_id": FieldSpec(
        STRING, nullable=True, required=False,
        description="Stable record identifier supplied by the source",
    ),
    "source_batch_id": FieldSpec(
        STRING, nullable=True, required=False,
        description="Deterministic identifier of the ingestion batch",
    ),
    "source_updated_at": FieldSpec(
        UTC_TIMESTAMP, nullable=True, required=False,
        unit="UTC",
        description="Last update timestamp reported by the source",
    ),
}

FIELD_SPECS: Final[dict[str, dict[str, FieldSpec]]] = {
    "instruments": {
        "instrument_id": FieldSpec(STRING, description="Canonical security identifier"),
        "symbol": FieldSpec(STRING, description="Exchange-local security code"),
        "exchange": FieldSpec(STRING, enum=EXCHANGES),
        "asset_type": FieldSpec(STRING, enum=ASSET_TYPES),
        "list_date": FieldSpec(DATE),
        "delist_date": FieldSpec(DATE, nullable=True),
        "lot_size": FieldSpec(INTEGER, unit="shares", minimum=1),
        "sell_delay_days": FieldSpec(INTEGER, unit="trading_days", minimum=0),
        "name": FieldSpec(STRING, nullable=True, required=False),
        "currency": FieldSpec(
            STRING, nullable=True, enum=CURRENCIES, required=False,
        ),
        **SOURCE_FIELDS,
    },
    "trade_calendar": {
        "trade_date": FieldSpec(DATE),
        "exchange": FieldSpec(STRING, enum=EXCHANGES),
        **SOURCE_FIELDS,
    },
    "daily_raw": {
        "trade_date": FieldSpec(DATE),
        "instrument_id": FieldSpec(STRING),
        "open": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "high": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "low": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "close": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "pre_close": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "volume": FieldSpec(INTEGER, unit="shares", minimum=0),
        "amount": FieldSpec(FLOAT, unit="CNY", minimum=0),
        "paused": FieldSpec(BOOLEAN),
        "limit_up": FieldSpec(FLOAT, unit="CNY/share", minimum=0, nullable=True),
        "limit_down": FieldSpec(FLOAT, unit="CNY/share", minimum=0, nullable=True),
        **SOURCE_FIELDS,
    },
    "daily_post_adjusted": {
        "trade_date": FieldSpec(DATE),
        "instrument_id": FieldSpec(STRING),
        "open_post": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "high_post": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "low_post": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "close_post": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "pre_close_post": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "adjustment_factor": FieldSpec(FLOAT, unit="ratio", minimum=0),
        "factor_source": FieldSpec(
            STRING, enum=frozenset({"vendor", "corporate_action"}),
        ),
        "adjustment_version": FieldSpec(STRING),
    },
    "corporate_actions": {
        "event_id": FieldSpec(STRING),
        "instrument_id": FieldSpec(STRING),
        "event_type": FieldSpec(STRING, enum=ACTION_TYPES),
        "record_date": FieldSpec(DATE, nullable=True),
        "ex_date": FieldSpec(DATE),
        "pay_date": FieldSpec(DATE, nullable=True),
        "cash_per_share": FieldSpec(FLOAT, unit="CNY/share", minimum=0),
        "share_ratio": FieldSpec(FLOAT, unit="shares/share", minimum=0),
        "subscription_price": FieldSpec(
            FLOAT, nullable=True, required=False,
            unit="CNY/share", minimum=0,
            description="Subscription price for a rights issue",
        ),
        "status": FieldSpec(STRING, enum=ACTION_STATUS),
        "announced_at": FieldSpec(DATE),
        **SOURCE_FIELDS,
    },
    "universe_membership": {
        "universe_id": FieldSpec(STRING),
        "instrument_id": FieldSpec(STRING),
        "effective_from": FieldSpec(DATE),
        "effective_to": FieldSpec(DATE, nullable=True),
        "known_at": FieldSpec(DATE),
        **SOURCE_FIELDS,
    },
    # Faithful transcription of the vendor's special-treatment state log: one
    # window per recorded transition, carrying the short name that took effect.
    # Whether a given name counts as special treatment is a research decision
    # and stays with the strategy, not with the data.
    "special_treatment": {
        "instrument_id": FieldSpec(STRING),
        "name": FieldSpec(
            STRING, description="Exchange short name in force over the window",
        ),
        "state_code": FieldSpec(
            INTEGER, nullable=True,
            description="Vendor company-state code at the transition",
        ),
        "reason_code": FieldSpec(
            INTEGER, nullable=True,
            description="Vendor reason code, populated when entering a state",
        ),
        "effective_from": FieldSpec(DATE),
        "effective_to": FieldSpec(DATE, nullable=True),
        "known_at": FieldSpec(
            DATE, description="Announcement date of the transition",
        ),
        **SOURCE_FIELDS,
    },
    "daily_money_flow": {
        "trade_date": FieldSpec(DATE),
        "instrument_id": FieldSpec(STRING),
        "inflow": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "outflow": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "net_inflow": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY",
        ),
        "inflow_s": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "inflow_m": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "inflow_l": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "inflow_xl": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "outflow_s": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "outflow_m": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "outflow_l": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "outflow_xl": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "net_inflow_s": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY",
        ),
        "net_inflow_m": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY",
        ),
        "net_inflow_l": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY",
        ),
        "net_inflow_xl": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY",
        ),
        "main_net_inflow": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY",
        ),
        "retail_net_inflow": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY",
        ),
        "net_in_open": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY",
        ),
        "net_in_close": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY",
        ),
        "turnover_value": FieldSpec(
            FLOAT, nullable=True, required=False, unit="CNY", minimum=0,
        ),
        "available_at": FieldSpec(
            DATE,
            description="First Asia/Shanghai calendar date the row was known",
        ),
        **SOURCE_FIELDS,
    },
    "industry_membership": {
        "classification": FieldSpec(STRING),
        "industry_id": FieldSpec(STRING),
        "instrument_id": FieldSpec(STRING),
        "effective_from": FieldSpec(DATE),
        "effective_to": FieldSpec(DATE, nullable=True),
        "known_at": FieldSpec(DATE),
        **SOURCE_FIELDS,
    },
    "market_rules": {
        "rule_id": FieldSpec(STRING),
        "exchange": FieldSpec(STRING, enum=EXCHANGES),
        "asset_type": FieldSpec(STRING, enum=ASSET_TYPES),
        "effective_from": FieldSpec(DATE),
        "effective_to": FieldSpec(DATE, nullable=True),
        "commission_bps": FieldSpec(
            FLOAT, nullable=True, unit="bps", minimum=0,
        ),
        "minimum_commission": FieldSpec(
            FLOAT, nullable=True, unit="CNY", minimum=0,
        ),
        "sell_tax_bps": FieldSpec(
            FLOAT, nullable=True, unit="bps", minimum=0,
        ),
        "buy_tax_bps": FieldSpec(
            FLOAT, nullable=True, unit="bps", minimum=0,
        ),
        "transfer_fee_bps": FieldSpec(
            FLOAT, nullable=True, unit="bps", minimum=0,
        ),
        "currency": FieldSpec(STRING, enum=CURRENCIES),
        "source": FieldSpec(STRING, nullable=True, required=False),
        "rule_version": FieldSpec(STRING, nullable=True, required=False),
        "scenario": FieldSpec(BOOLEAN, nullable=True, required=False),
        **SOURCE_FIELDS,
    },
    "financial_reports": {
        "report_id": FieldSpec(STRING),
        "instrument_id": FieldSpec(STRING),
        "statement_type": FieldSpec(STRING, enum=STATEMENT_TYPES),
        "fiscal_period_start": FieldSpec(DATE),
        "fiscal_period_end": FieldSpec(DATE),
        "filing_period_end": FieldSpec(DATE),
        "record_kind": FieldSpec(STRING, enum=REPORT_KINDS),
        "published_at": FieldSpec(DATE),
        "available_at": FieldSpec(DATE),
        "revision_sequence": FieldSpec(INTEGER, minimum=1),
        "currency": FieldSpec(STRING, enum=CURRENCIES),
        "source_report_type": FieldSpec(STRING),
        **SOURCE_FIELDS,
    },
    "financial_facts": {
        "report_id": FieldSpec(STRING),
        "item_code": FieldSpec(STRING),
        "instrument_id": FieldSpec(STRING),
        "statement_type": FieldSpec(STRING, enum=STATEMENT_TYPES),
        "fiscal_period_start": FieldSpec(DATE),
        "fiscal_period_end": FieldSpec(DATE),
        "filing_period_end": FieldSpec(DATE),
        "available_at": FieldSpec(DATE),
        "value": FieldSpec(FLOAT),
        "unit": FieldSpec(STRING, enum=FINANCIAL_UNITS),
        "value_basis": FieldSpec(STRING, enum=FACT_BASES),
        "source_field": FieldSpec(STRING),
        **SOURCE_FIELDS,
    },
    "fundamental_metrics": {
        "metric_id": FieldSpec(STRING),
        "instrument_id": FieldSpec(STRING),
        "metric_code": FieldSpec(STRING),
        "fiscal_period_end": FieldSpec(DATE),
        "basis": FieldSpec(STRING, enum=METRIC_BASES),
        "value": FieldSpec(FLOAT),
        "unit": FieldSpec(STRING, enum=FINANCIAL_UNITS),
        "available_at": FieldSpec(DATE),
        "calculation_version": FieldSpec(STRING),
        "source_report_ids": FieldSpec(STRING),
        "quality_status": FieldSpec(STRING, enum=METRIC_QUALITY),
        **SOURCE_FIELDS,
    },
    "daily_valuation": {
        "trade_date": FieldSpec(DATE),
        "instrument_id": FieldSpec(STRING),
        "pe_ttm": FieldSpec(FLOAT, nullable=True),
        "pe_lyr": FieldSpec(FLOAT, nullable=True),
        "pb": FieldSpec(FLOAT, nullable=True),
        "ps_ttm": FieldSpec(FLOAT, nullable=True),
        "pcf_ttm": FieldSpec(FLOAT, nullable=True),
        "pcf_operating_ttm": FieldSpec(FLOAT, nullable=True),
        "dividend_yield": FieldSpec(FLOAT, nullable=True, unit="ratio"),
        "turnover_rate": FieldSpec(FLOAT, nullable=True, unit="ratio"),
        "total_shares": FieldSpec(FLOAT, nullable=True, unit="shares", minimum=0),
        "market_cap": FieldSpec(FLOAT, nullable=True, unit="CNY", minimum=0),
        "circulating_shares": FieldSpec(
            FLOAT, nullable=True, unit="shares", minimum=0,
        ),
        "circulating_market_cap": FieldSpec(
            FLOAT, nullable=True, unit="CNY", minimum=0,
        ),
        "free_float_shares": FieldSpec(
            FLOAT, nullable=True, unit="shares", minimum=0,
        ),
        "free_float_market_cap": FieldSpec(
            FLOAT, nullable=True, unit="CNY", minimum=0,
        ),
        "a_shares": FieldSpec(FLOAT, nullable=True, unit="shares", minimum=0),
        "a_market_cap": FieldSpec(FLOAT, nullable=True, unit="CNY", minimum=0),
        "available_at": FieldSpec(DATE),
        **SOURCE_FIELDS,
    },
    "share_capital": {
        "capital_event_id": FieldSpec(STRING),
        "instrument_id": FieldSpec(STRING),
        "effective_from": FieldSpec(DATE),
        "announced_at": FieldSpec(DATE),
        "available_at": FieldSpec(DATE),
        "change_reason_code": FieldSpec(STRING, nullable=True),
        "change_reason": FieldSpec(STRING, nullable=True),
        "total_shares": FieldSpec(FLOAT, unit="shares", minimum=0),
        "nontradable_shares": FieldSpec(
            FLOAT, nullable=True, unit="shares", minimum=0,
        ),
        "restricted_shares": FieldSpec(
            FLOAT, nullable=True, unit="shares", minimum=0,
        ),
        "tradable_shares": FieldSpec(
            FLOAT, nullable=True, unit="shares", minimum=0,
        ),
        "a_shares": FieldSpec(FLOAT, nullable=True, unit="shares", minimum=0),
        "b_shares": FieldSpec(FLOAT, nullable=True, unit="shares", minimum=0),
        "h_shares": FieldSpec(FLOAT, nullable=True, unit="shares", minimum=0),
        **SOURCE_FIELDS,
    },
}

REQUIRED_COLUMNS: Final = {
    table: {name for name, spec in fields.items() if spec.required}
    for table, fields in FIELD_SPECS.items()
}

ARROW_SCHEMAS: Final = {
    table: pa.schema([spec.arrow_field(name) for name, spec in fields.items()])
    for table, fields in FIELD_SPECS.items()
}

DATE_COLUMNS: Final = {
    table: tuple(
        name for name, spec in fields.items() if pa.types.is_date(spec.arrow_type)
    )
    for table, fields in FIELD_SPECS.items()
}

TIMESTAMP_COLUMNS: Final = {
    table: tuple(
        name for name, spec in fields.items() if pa.types.is_timestamp(spec.arrow_type)
    )
    for table, fields in FIELD_SPECS.items()
}

PRIMARY_KEYS: Final = {
    "instruments": ("instrument_id",),
    "trade_calendar": ("exchange", "trade_date"),
    "daily_raw": ("trade_date", "instrument_id"),
    "daily_post_adjusted": ("trade_date", "instrument_id"),
    "corporate_actions": ("event_id",),
    "universe_membership": ("universe_id", "instrument_id", "effective_from"),
    "industry_membership": (
        "classification", "instrument_id", "effective_from",
    ),
    "market_rules": ("rule_id",),
    "special_treatment": ("instrument_id", "effective_from"),
    "daily_money_flow": ("trade_date", "instrument_id"),
    "financial_reports": ("report_id",),
    "financial_facts": ("report_id", "item_code"),
    "fundamental_metrics": ("metric_id",),
    "daily_valuation": ("trade_date", "instrument_id"),
    "share_capital": ("capital_event_id",),
}

PRICE_COLUMNS: Final = ("open", "high", "low", "close", "pre_close")
POST_PRICE_COLUMNS: Final = (
    "open_post", "high_post", "low_post", "close_post", "pre_close_post",
)

DYNAMIC_TABLES: Final = {
    "daily_raw", "daily_post_adjusted", "corporate_actions",
    "universe_membership", "industry_membership", "market_rules",
    "financial_reports", "financial_facts", "fundamental_metrics",
    "daily_valuation", "share_capital", "special_treatment",
    "daily_money_flow",
}

VISIBILITY_FIELDS: Final = {
    "corporate_actions": "announced_at",
    "universe_membership": "known_at",
    "industry_membership": "known_at",
    "financial_reports": "available_at",
    "financial_facts": "available_at",
    "fundamental_metrics": "available_at",
    "daily_valuation": "available_at",
    "share_capital": "available_at",
    "special_treatment": "known_at",
    "daily_money_flow": "available_at",
}

from __future__ import annotations

from typing import Mapping

import pandas as pd

from zyquant.core.exceptions import AccountingError
from zyquant.core.versioning import LEDGER_SCHEMA_VERSION

LEDGER_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "sleeve_demands": (
        "strategy_id", "instrument_id", "side", "quantity", "reference_price",
        "lot_size", "demand_id", "execution_phase", "target_quantity",
    ),
    "demand_residuals": (
        "demand_id", "execution_date", "execution_phase", "strategy_id",
        "instrument_id", "side", "quantity", "reason",
    ),
    "internal_crosses": (
        "execution_date", "instrument_id", "seller_strategy_id",
        "buyer_strategy_id", "quantity", "price", "cross_id", "execution_phase",
    ),
    "orders": (
        "order_id", "execution_date", "execution_phase", "instrument_id",
        "side", "quantity", "reference_price", "lot_size",
    ),
    "fills": (
        "order_id", "execution_date", "execution_phase", "instrument_id", "side",
        "requested_quantity", "filled_quantity", "price", "commission", "tax",
        "slippage_bps", "impact_bps", "status", "reject_reason",
    ),
    "fill_allocations": (
        "order_id", "execution_date", "strategy_id", "instrument_id", "side",
        "quantity", "price", "commission", "tax", "slippage_cost", "allocation_id",
    ),
    "cashflows": (
        "event_id", "date", "account_id", "flow_type", "amount",
        "instrument_id", "upstream_event_id",
    ),
    "positions": (
        "date", "strategy_id", "instrument_id", "quantity",
        "last_price", "market_value", "position_status", "valuation_source",
        "last_observed_date", "stale_sessions",
    ),
    "position_lots": (
        "date", "strategy_id", "instrument_id", "cohort_id", "quantity",
        "acquisition_date", "sellable_date", "unit_cost",
    ),
    "master_positions": (
        "date", "account_id", "instrument_id", "quantity",
        "last_price", "market_value", "position_status", "valuation_source",
        "last_observed_date", "stale_sessions",
    ),
    "nav": ("date", "strategy_id", "cash", "receivables", "nav"),
    "master_nav": ("date", "account_id", "cash", "receivables", "nav"),
    "reconciliations": ("date", "phase", "event", "status"),
}


def enforce_ledger_schemas(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for name, columns in LEDGER_COLUMNS.items():
        frame = frames.get(name)
        if frame is None or frame.empty:
            frames[name] = pd.DataFrame(columns=columns)
            continue
        missing = set(columns) - set(frame.columns)
        # Residual records may carry optional order/cross references, but core
        # columns are never optional.
        if missing:
            raise AccountingError(
                f"ledger {name} missing v{LEDGER_SCHEMA_VERSION} columns: "
                f"{sorted(missing)}"
            )
        frames[name] = frame[list(columns) + [
            column for column in frame.columns if column not in columns
        ]]
    return frames

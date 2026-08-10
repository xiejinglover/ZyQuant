from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from zyquant.config import ExecutionConfig

from .types import Fill, MasterOrder

COST_FIELDS: tuple[str, ...] = (
    "commission_bps", "minimum_commission", "sell_tax_bps",
    "buy_tax_bps", "transfer_fee_bps",
)

# Fallback rates, in basis points, used only where neither the configuration
# nor the snapshot supplies a value. They keep a cost model resolvable on a
# snapshot whose market_rules carry `source_missing` placeholders, so no cost
# can silently evaluate to NaN. They are a single flat assumption, not a
# historical series: real commission is broker-specific and the regulated
# rates have changed over time, so a run that falls back here records
# `default` provenance in its manifest.
DEFAULT_MARKET_COSTS: dict[str, dict[str, float]] = {
    "stock": {
        "commission_bps": 2.5,
        "minimum_commission": 5.0,
        "sell_tax_bps": 5.0,
        "buy_tax_bps": 0.0,
        "transfer_fee_bps": 0.1,
    },
    "etf": {
        "commission_bps": 2.5,
        "minimum_commission": 5.0,
        "sell_tax_bps": 0.0,
        "buy_tax_bps": 0.0,
        "transfer_fee_bps": 0.0,
    },
}

# Legacy scalar overrides on ExecutionConfig and the cost field each sets.
_SCALAR_OVERRIDES: tuple[tuple[str, str, bool], ...] = (
    ("commission_bps", "commission_bps", False),
    ("minimum_commission", "minimum_commission", False),
    ("stock_sell_tax_bps", "sell_tax_bps", True),
)


class CostModel(Protocol):
    def resolve(
        self, config: ExecutionConfig, asset_type: str, rule=None,
    ) -> tuple[Mapping[str, float], Mapping[str, str]]: ...


class ExecutionModel(Protocol):
    def execute(
        self, order: MasterOrder, market_row, asset_type: str, rule=None,
    ) -> Fill: ...


def _finite(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def resolve_costs(
    config: ExecutionConfig, asset_type: str, rule=None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Resolve the five cost components and where each one came from.

    Precedence per field: an explicit scalar override, then a per-asset-type
    override, then the snapshot rule when it carries a finite value, then the
    built-in default. Every field therefore resolves to a finite number, which
    is what keeps a `source_missing` rule row from poisoning a ledger with NaN.
    """
    defaults = DEFAULT_MARKET_COSTS.get(
        asset_type, DEFAULT_MARKET_COSTS["stock"]
    )
    override = (config.cost_overrides or {}).get(asset_type)
    scalars: dict[str, float] = {}
    for attribute, field, stock_only in _SCALAR_OVERRIDES:
        if stock_only and asset_type != "stock":
            continue
        value = _finite(getattr(config, attribute, None))
        if value is not None:
            scalars[field] = value
    values: dict[str, float] = {}
    provenance: dict[str, str] = {}
    for field in COST_FIELDS:
        if field in scalars:
            values[field] = scalars[field]
            provenance[field] = "config_scalar"
            continue
        chosen = _finite(getattr(override, field, None)) if override else None
        if chosen is not None:
            values[field] = chosen
            provenance[field] = "config_override"
            continue
        chosen = _finite(getattr(rule, field, None)) if rule is not None else None
        if chosen is not None:
            values[field] = chosen
            provenance[field] = "market_rules"
            continue
        values[field] = float(defaults[field])
        provenance[field] = "default"
    return values, provenance


class HistoricalCostModel:
    """Default historical-rule cost model with explicit fallbacks."""

    def resolve(
        self, config: ExecutionConfig, asset_type: str, rule=None,
    ) -> tuple[Mapping[str, float], Mapping[str, str]]:
        return resolve_costs(config, asset_type, rule)


class MarketExecutor:
    def __init__(
        self,
        config: ExecutionConfig,
        cost_model: CostModel | None = None,
    ):
        self.config = config
        self.cost_model = cost_model or HistoricalCostModel()

    def execute(self, order: MasterOrder, market_row, asset_type: str, rule=None) -> Fill:
        # A missing rule row is no longer fatal: resolve_costs falls back
        # through configuration to the built-in defaults, so the executor and
        # the engine always price an order the same way.
        rule = SimpleNamespace(**dict(
            self.cost_model.resolve(self.config, asset_type, rule)[0]
        ))
        if market_row is None:
            return self._rejected(order, "missing_market_row")
        paused = bool(market_row.paused) if pd.notna(market_row.paused) else False
        if paused or float(market_row.volume) <= 0:
            return self._rejected(order, "paused_or_zero_volume")
        limit_up = float(market_row.limit_up) if pd.notna(market_row.limit_up) else None
        limit_down = float(market_row.limit_down) if pd.notna(market_row.limit_down) else None
        if order.side == "buy" and limit_up is not None:
            if float(market_row.low) >= limit_up - 1e-12:
                return self._rejected(order, "one_price_limit_up")
        if order.side == "sell" and limit_down is not None:
            if float(market_row.high) <= limit_down + 1e-12:
                return self._rejected(order, "one_price_limit_down")
        capacity = math.floor(
            float(market_row.volume) * self.config.max_participation / order.lot_size
        ) * order.lot_size
        quantity = min(order.quantity, capacity)
        quantity = math.floor(quantity / order.lot_size) * order.lot_size
        if quantity <= 0:
            return self._rejected(order, "capacity_below_one_lot")
        amount = max(float(market_row.amount), 1e-12)
        participation = quantity * order.reference_price / amount
        impact = min(
            self.config.max_impact_bps,
            self.config.impact_coefficient_bps * math.sqrt(max(0.0, participation)),
        )
        total_slippage = self.config.slippage_bps + impact
        sign = 1 if order.side == "buy" else -1
        modeled = order.reference_price * (1 + sign * total_slippage / 10000)
        lower = max(float(market_row.low), limit_down if limit_down is not None else -np.inf)
        upper = min(float(market_row.high), limit_up if limit_up is not None else np.inf)
        price = min(max(modeled, lower), upper)
        notional = quantity * price
        commission_bps = (
            self.config.commission_bps
            if self.config.commission_bps is not None
            else float(rule.commission_bps)
        )
        minimum_commission = (
            self.config.minimum_commission
            if self.config.minimum_commission is not None
            else float(rule.minimum_commission)
        )
        sell_tax_bps = (
            self.config.stock_sell_tax_bps
            if self.config.stock_sell_tax_bps is not None and asset_type == "stock"
            else float(rule.sell_tax_bps)
        )
        buy_tax_bps = float(rule.buy_tax_bps)
        transfer_fee_bps = float(rule.transfer_fee_bps)
        commission = max(
            minimum_commission,
            notional * (commission_bps + transfer_fee_bps) / 10000,
        ) if quantity else 0.0
        tax = (
            notional * (sell_tax_bps if order.side == "sell" else buy_tax_bps) / 10000
        )
        return Fill(
            order.order_id, order.execution_date, order.instrument_id, order.side,
            order.quantity, quantity, price, commission, tax,
            self.config.slippage_bps, impact,
            "filled" if quantity == order.quantity else "partial", None,
        )

    @staticmethod
    def _rejected(order, reason):
        return Fill(
            order.order_id, order.execution_date, order.instrument_id, order.side,
            order.quantity, 0, order.reference_price, 0.0, 0.0, 0.0, 0.0,
            "rejected", reason,
        )

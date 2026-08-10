from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from zyquant.core.exceptions import AccountingError

@dataclass
class PositionLot:
    instrument_id: str
    quantity: int
    acquisition_date: date
    sellable_date: date
    unit_cost: float


@dataclass
class SleeveAccount:
    strategy_id: str
    cash: float
    lots: dict[str, list[PositionLot]] = field(default_factory=dict)
    receivables: dict[str, float] = field(default_factory=dict)

    def quantity(self, instrument_id: str) -> int:
        return sum(item.quantity for item in self.lots.get(instrument_id, ()))

    def sellable_quantity(self, instrument_id: str, day: date) -> int:
        return sum(
            item.quantity for item in self.lots.get(instrument_id, ())
            if item.sellable_date <= day
        )

    def nav(self, prices: Mapping[str, float]) -> float:
        market = sum(self.quantity(code) * prices.get(code, 0.0) for code in self.lots)
        return self.cash + sum(self.receivables.values()) + market

    def add_lot(self, lot: PositionLot) -> None:
        self.lots.setdefault(lot.instrument_id, []).append(lot)

    def remove(self, instrument_id: str, quantity: int, day: date) -> int:
        remaining = quantity
        lots = self.lots.get(instrument_id, [])
        for lot in sorted(lots, key=lambda item: (item.sellable_date, item.acquisition_date)):
            if lot.sellable_date > day or remaining <= 0:
                continue
            taken = min(lot.quantity, remaining)
            lot.quantity -= taken
            remaining -= taken
        self.lots[instrument_id] = [item for item in lots if item.quantity > 0]
        return quantity - remaining

    def remove_any(self, instrument_id: str, quantity: int) -> int:
        remaining = quantity
        lots = self.lots.get(instrument_id, [])
        for lot in sorted(lots, key=lambda item: (item.acquisition_date, item.sellable_date)):
            if remaining <= 0:
                continue
            taken = min(lot.quantity, remaining)
            lot.quantity -= taken
            remaining -= taken
        self.lots[instrument_id] = [item for item in lots if item.quantity > 0]
        return quantity - remaining


@dataclass
class MasterAccount(SleeveAccount):
    def reconcile(
        self,
        sleeves: Mapping[str, SleeveAccount],
        prices: Mapping[str, float],
        tolerance: float = 1e-7,
    ) -> None:
        sleeve_cash = sum(item.cash for item in sleeves.values())
        if abs(self.cash - sleeve_cash) > tolerance:
            raise AccountingError(
                f"master/sleeve cash mismatch: {self.cash} vs {sleeve_cash}"
            )
        codes = set(self.lots)
        for sleeve in sleeves.values():
            codes.update(sleeve.lots)
        for code in sorted(codes):
            master_quantity = self.quantity(code)
            sleeve_quantity = sum(item.quantity(code) for item in sleeves.values())
            if master_quantity != sleeve_quantity:
                raise AccountingError(
                    f"master/sleeve position mismatch for {code}: "
                    f"{master_quantity} vs {sleeve_quantity}"
                )
        master_receivables = sum(self.receivables.values())
        sleeve_receivables = sum(
            sum(item.receivables.values()) for item in sleeves.values()
        )
        if abs(master_receivables - sleeve_receivables) > tolerance:
            raise AccountingError("master/sleeve receivables mismatch")
        sleeve_nav = sum(item.nav(prices) for item in sleeves.values())
        if abs(self.nav(prices) - sleeve_nav) > tolerance:
            raise AccountingError(
                f"master/sleeve NAV mismatch: {self.nav(prices)} vs {sleeve_nav}"
            )


@dataclass(frozen=True)
class SleeveDemand:
    strategy_id: str
    instrument_id: str
    side: str
    quantity: int
    reference_price: float
    lot_size: int
    demand_id: str | None = None
    execution_phase: str = "open"
    target_quantity: int | None = None


@dataclass(frozen=True)
class InternalCross:
    execution_date: date
    instrument_id: str
    seller_strategy_id: str
    buyer_strategy_id: str
    quantity: int
    price: float
    cross_id: str | None = None
    execution_phase: str = "open"


@dataclass(frozen=True)
class MasterOrder:
    order_id: str
    execution_date: date
    execution_phase: str
    instrument_id: str
    side: str
    quantity: int
    reference_price: float
    lot_size: int


@dataclass(frozen=True)
class Fill:
    order_id: str
    execution_date: date
    instrument_id: str
    side: str
    requested_quantity: int
    filled_quantity: int
    price: float
    commission: float
    tax: float
    slippage_bps: float
    impact_bps: float
    status: str
    reject_reason: str | None = None


@dataclass(frozen=True)
class FillAllocation:
    order_id: str
    execution_date: date
    strategy_id: str
    instrument_id: str
    side: str
    quantity: int
    price: float
    commission: float
    tax: float
    slippage_cost: float
    allocation_id: str | None = None


@dataclass(frozen=True)
class CashFlow:
    event_id: str
    date: date
    account_id: str
    flow_type: str
    amount: float
    instrument_id: str | None = None
    upstream_event_id: str | None = None


@dataclass(frozen=True)
class BacktestResult:
    run_id: str
    metrics: Mapping[str, Any]
    frames: Mapping[str, Any]
    run_path: Path | None = None
    # Rates actually charged, per asset_type, and where each one came from
    # ("config_scalar", "config_override", "market_rules" or "default"). A run
    # priced off built-in defaults stays auditable after the fact.
    cost_model: Mapping[str, Any] = field(default_factory=dict)

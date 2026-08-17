from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from typing import Any, Iterable

from zyquant.backtest.types import (
    FillAllocation, InternalCross, MasterOrder, SleeveDemand,
)


def net_sleeve_demands(
    demands: Iterable[SleeveDemand],
    execution_date: date,
    phase: str,
) -> tuple[list[InternalCross], list[MasterOrder], dict[tuple[str, str], list[SleeveDemand]]]:
    grouped: dict[str, list[SleeveDemand]] = defaultdict(list)
    for demand in demands:
        if demand.quantity > 0:
            grouped[demand.instrument_id].append(demand)
    crosses: list[InternalCross] = []
    orders: list[MasterOrder] = []
    residual_by_order: dict[tuple[str, str], list[SleeveDemand]] = {}
    for code in sorted(grouped):
        buys: list[list[Any]] = [
            [item, item.quantity]
            for item in sorted(grouped[code], key=lambda x: x.strategy_id)
            if item.side == "buy"
        ]
        sells: list[list[Any]] = [
            [item, item.quantity]
            for item in sorted(grouped[code], key=lambda x: x.strategy_id)
            if item.side == "sell"
        ]
        buy_index = sell_index = 0
        while buy_index < len(buys) and sell_index < len(sells):
            buyer, buy_qty = buys[buy_index]
            seller, sell_qty = sells[sell_index]
            quantity = min(buy_qty, sell_qty)
            if quantity:
                crosses.append(InternalCross(
                    execution_date, code, seller.strategy_id, buyer.strategy_id,
                    quantity, buyer.reference_price,
                    f"{execution_date}:{phase}:{code}:cross:{seller.strategy_id}:{buyer.strategy_id}",
                    phase,
                    seller.cohort_id,
                    buyer.cohort_id,
                ))
            buys[buy_index][1] -= quantity
            sells[sell_index][1] -= quantity
            if buys[buy_index][1] == 0:
                buy_index += 1
            if sells[sell_index][1] == 0:
                sell_index += 1
        for side, values in (("sell", sells), ("buy", buys)):
            residual = [
                SleeveDemand(
                    item.strategy_id, item.instrument_id, item.side, int(quantity),
                    item.reference_price, item.lot_size, item.demand_id, phase,
                    item.target_quantity, item.cohort_id,
                )
                for item, quantity in values if quantity > 0
            ]
            quantity = sum(item.quantity for item in residual)
            if not quantity:
                continue
            sample = residual[0]
            order_id = f"{execution_date}:{phase}:{code}:{side}"
            orders.append(MasterOrder(
                order_id, execution_date, phase, code, side, quantity,
                sample.reference_price, sample.lot_size,
            ))
            residual_by_order[(code, side)] = residual
    return crosses, orders, residual_by_order


def allocate_fill_quantities(
    order: MasterOrder,
    filled_quantity: int,
    demands: list[SleeveDemand],
) -> dict[str, int]:
    if filled_quantity <= 0 or not demands:
        return {}
    total = sum(item.quantity for item in demands)
    lot = order.lot_size
    raw = {item.strategy_id: filled_quantity * item.quantity / total for item in demands}
    allocated = {key: math.floor(value / lot) * lot for key, value in raw.items()}
    remaining = filled_quantity - sum(allocated.values())
    remainders = sorted(
        raw,
        key=lambda key: (-(raw[key] - allocated[key]), key),
    )
    demand_by_strategy = {item.strategy_id: item.quantity for item in demands}
    for strategy_id in remainders:
        if remaining < lot:
            break
        if allocated[strategy_id] + lot <= demand_by_strategy[strategy_id]:
            allocated[strategy_id] += lot
            remaining -= lot
    return {key: value for key, value in allocated.items() if value > 0}


def allocate_fill_demands(
    order: MasterOrder,
    filled_quantity: int,
    demands: list[SleeveDemand],
) -> dict[tuple[str, str | None, str | None], int]:
    """Allocate a fill without collapsing distinct cohorts of one strategy."""
    if filled_quantity <= 0 or not demands:
        return {}
    keys = [
        (item.strategy_id, item.cohort_id, item.demand_id)
        for item in demands
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate sleeve demand allocation key")
    total = sum(item.quantity for item in demands)
    lot = order.lot_size
    requested = {key: item.quantity for key, item in zip(keys, demands)}
    raw = {key: filled_quantity * requested[key] / total for key in keys}
    allocated = {
        key: math.floor(value / lot) * lot for key, value in raw.items()
    }
    remaining = filled_quantity - sum(allocated.values())
    for key in sorted(
        keys, key=lambda item: (-(raw[item] - allocated[item]), item),
    ):
        if remaining < lot:
            break
        if allocated[key] + lot <= requested[key]:
            allocated[key] += lot
            remaining -= lot
    return {key: value for key, value in allocated.items() if value > 0}


def cost_demand_allocations(
    order: MasterOrder,
    quantity_by_demand: dict[tuple[str, str | None, str | None], int],
    price: float,
    commission: float,
    tax: float,
    slippage_bps: float,
) -> list[FillAllocation]:
    """Split order costs over cohort-aware allocations deterministically."""
    total = sum(quantity_by_demand.values())
    if total <= 0:
        return []
    result = []
    assigned_commission = assigned_tax = assigned_slippage = 0.0
    keys = sorted(quantity_by_demand)
    total_slippage = total * abs(price - order.reference_price)
    for index, key in enumerate(keys):
        strategy_id, cohort_id, demand_id = key
        quantity = quantity_by_demand[key]
        if index == len(keys) - 1:
            current_commission = commission - assigned_commission
            current_tax = tax - assigned_tax
            current_slippage = total_slippage - assigned_slippage
        else:
            ratio = quantity / total
            current_commission = commission * ratio
            current_tax = tax * ratio
            current_slippage = total_slippage * ratio
            assigned_commission += current_commission
            assigned_tax += current_tax
            assigned_slippage += current_slippage
        allocation_suffix = demand_id or cohort_id or "default"
        result.append(FillAllocation(
            order.order_id, order.execution_date, strategy_id,
            order.instrument_id, order.side, quantity, price,
            current_commission, current_tax, current_slippage,
            f"{order.order_id}:allocation:{strategy_id}:{allocation_suffix}",
            cohort_id,
        ))
    return result


def cost_allocations(
    order: MasterOrder,
    quantity_by_strategy: dict[str, int],
    price: float,
    commission: float,
    tax: float,
    slippage_bps: float,
    cohort_by_strategy: dict[str, str | None] | None = None,
) -> list[FillAllocation]:
    total = sum(quantity_by_strategy.values())
    if total <= 0:
        return []
    result = []
    assigned_commission = assigned_tax = assigned_slippage = 0.0
    keys = sorted(quantity_by_strategy)
    total_slippage = total * abs(price - order.reference_price)
    for index, strategy_id in enumerate(keys):
        quantity = quantity_by_strategy[strategy_id]
        if index == len(keys) - 1:
            current_commission = commission - assigned_commission
            current_tax = tax - assigned_tax
            current_slippage = total_slippage - assigned_slippage
        else:
            ratio = quantity / total
            current_commission = commission * ratio
            current_tax = tax * ratio
            current_slippage = total_slippage * ratio
            assigned_commission += current_commission
            assigned_tax += current_tax
            assigned_slippage += current_slippage
        result.append(FillAllocation(
            order.order_id, order.execution_date, strategy_id, order.instrument_id, order.side,
            quantity, price, current_commission, current_tax, current_slippage,
            f"{order.order_id}:allocation:{strategy_id}",
            (cohort_by_strategy or {}).get(strategy_id),
        ))
    return result

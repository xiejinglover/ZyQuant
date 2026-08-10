from __future__ import annotations

import math
import inspect
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Mapping

import numpy as np
import pandas as pd

from zyquant.analysis import attribution_report, performance_metrics
from zyquant.config import ExecutionConfig
from zyquant.core.exceptions import BacktestError, DataContractError
from zyquant.core.hashing import canonical_json, hash_payload
from zyquant.core.versioning import LEDGER_SCHEMA_VERSION, derive_seed
from zyquant.data import DataSnapshot
from zyquant.portfolio.sleeve import (
    allocate_fill_quantities, cost_allocations, net_sleeve_demands,
)
from zyquant.portfolio.capital import CapitalAllocator
from zyquant.strategy.types import (
    PortfolioView, PositionView, StrategyContext, StrategyState, TargetPortfolio,
)

from .market import (
    CostModel, ExecutionModel, HistoricalCostModel, MarketExecutor,
)
from .ledger import enforce_ledger_schemas
from .types import (
    BacktestResult, MasterAccount, PositionLot, SleeveAccount, SleeveDemand,
)


@dataclass(frozen=True)
class StrategyBinding:
    strategy: Any
    capital_weight: float
    parameters: Mapping[str, Any] | None = None


class _ActionEvent:
    """One corporate action, with the per-row conversions already done."""

    __slots__ = ("row", "event_id", "code", "event_type")

    def __init__(self, row, event_id, code, event_type):
        self.row = row
        self.event_id = event_id
        self.code = code
        self.event_type = event_type


_CANCELLED_STATUS = {"cancelled", "canceled", "取消"}
_DIVIDEND_TYPES = {"cash_dividend", "dividend", "分红"}

# Every bar field the engine or the matcher reads. Naming them lets the parquet
# layer skip the rest (pre_close, source_*, year).
_BAR_FIELDS = (
    "open", "high", "low", "close", "volume", "amount",
    "paused", "limit_up", "limit_down",
)


class _DayBars:
    """One trading day's bars, held as column slices rather than row objects."""

    __slots__ = ("codes", "columns", "position")

    def __init__(self, codes, columns):
        self.codes = codes
        self.columns = columns
        self.position = None       # built lazily, only on days that trade

    def price_map(self, field):
        # `.tolist()` converts through the same C path as `float(np.float64)`,
        # so the values are bit-identical to the previous per-row conversion.
        return dict(zip(self.codes.tolist(), self.columns[field].tolist()))

    def row(self, code):
        position = self.position
        if position is None:
            position = self.position = {
                item: index for index, item in enumerate(self.codes.tolist())
            }
        index = position.get(code)
        if index is None:
            return None
        return _MarketRow(self.columns, index)


class _MarketRow:
    """A single bar, exposing the same attributes an `itertuples` row did.

    Values are taken as `array[index]`, which yields numpy scalars exactly as
    `itertuples` did over the same arrays. `market.py` therefore sees the same
    types it always has and needs no changes.
    """

    __slots__ = _BAR_FIELDS

    def __init__(self, columns, index):
        for field in _BAR_FIELDS:
            setattr(self, field, columns[field][index])


def _index_bars(raw, calendar):
    """Group the raw bars by trading day, as column slices.

    Replaces a dict keyed by `(day, code)` over the whole range — roughly 13M
    entries and most of the run's memory — that three call sites then scanned in
    full every day to pick out one day's prices. That scan is what made the run
    quadratic in the number of trading days.

    `factorize` matters for memory as much as speed: taking the instrument
    column to numpy directly materialises one distinct `str` per row, while
    factorizing leaves a few thousand shared objects that the per-day slices
    merely point at. The slices themselves are views, not copies.
    """
    dates = raw["trade_date"].to_numpy()
    labels, uniques = pd.factorize(raw["instrument_id"], sort=False)
    unique_codes = np.asarray([str(item) for item in uniques], dtype=object)
    codes = unique_codes[labels]
    columns = {name: np.asarray(raw[name]) for name in _BAR_FIELDS}

    market: dict[Any, _DayBars] = {}
    if raw["trade_date"].is_monotonic_increasing:
        for day in calendar:
            low = int(np.searchsorted(dates, day, side="left"))
            high = int(np.searchsorted(dates, day, side="right"))
            market[day] = _DayBars(
                codes[low:high],
                {name: column[low:high] for name, column in columns.items()},
            )
    else:
        # The snapshot contract sorts by (trade_date, instrument_id), so this
        # should not happen; group explicitly rather than slice silently wrong.
        for day, positions in raw.groupby("trade_date", sort=False).indices.items():
            market[day] = _DayBars(
                codes[positions],
                {name: column[positions] for name, column in columns.items()},
            )
    return market


def _price_map(market, day, field):
    bars = market.get(day)
    return bars.price_map(field) if bars is not None else {}


def _index_actions(actions):
    """Group corporate actions by the day each one acts on.

    The loop used to walk the whole table on every trading day and test
    `ex_date == day` inside the body, re-running four `str()` conversions per
    row per day over Arrow-backed columns. Parsing once and indexing by date
    turns that from O(days x events) into O(days + events).

    Ordering is the delicate part. The original is a single pass in which each
    event is tested for its ex date and then for its pay date, so a day that is
    one event's ex date and another's pay date interleaves them in table order.
    Cash is accumulated with `+=`, and float addition is not associative, so the
    sequence has to be preserved exactly: build ONE list per day, appending each
    event's ex entry before its pay entry, walking the table in order. Splitting
    this into separate by-ex-date and by-pay-date passes would reorder the
    additions and move the last bits of the final NAV.

    Numeric fields are deliberately left unconverted: `float(share_ratio)` and
    the `cash_per_share` notna branch only run on the day an event fires, and
    hoisting them would turn "bad data raises when it is used" into "bad data
    raises at startup", where `float(None)` is a TypeError rather than the
    domain error the caller expects.
    """
    by_day: dict[Any, list[tuple[str, _ActionEvent]]] = defaultdict(list)
    by_record: dict[Any, list[_ActionEvent]] = defaultdict(list)
    if actions.empty:
        return by_day, by_record
    for row in actions.itertuples(index=False):
        event = _ActionEvent(
            row, str(row.event_id), str(row.instrument_id),
            str(row.event_type).lower(),
        )
        record_date = row.record_date
        # Deliberately unfiltered by status: the original selected record-date
        # rows with a plain mask, so cancelled events also landed in
        # `entitlements` (where nothing ever reads them again).
        if record_date is not None and pd.notna(record_date):
            by_record[record_date].append(event)
        if str(row.status).lower() in _CANCELLED_STATUS:
            continue
        by_day[row.ex_date].append(("ex", event))
        if event.event_type in _DIVIDEND_TYPES:
            by_day[row.pay_date].append(("pay", event))
    return by_day, by_record


class BacktestEngine:
    def __init__(
        self,
        execution: ExecutionConfig | None = None,
        factor_engine=None,
        execution_model: ExecutionModel | None = None,
        cost_model: CostModel | None = None,
        capital_allocator: CapitalAllocator | None = None,
    ):
        self.execution_config = execution or ExecutionConfig()
        self.cost_model = cost_model or HistoricalCostModel()
        self.executor = execution_model or MarketExecutor(
            self.execution_config, self.cost_model
        )
        self.factor_engine = factor_engine
        self.capital_allocator = capital_allocator

    def run(
        self,
        snapshot: DataSnapshot,
        start: date,
        end: date,
        strategies: list[StrategyBinding],
        initial_cash: float,
        seed: int = 20260722,
        compute_attribution: bool = False,
    ) -> BacktestResult:
        # A snapshot without a historical fee series no longer blocks a run:
        # every cost component resolves through configuration to a built-in
        # default, and the resolved rates plus their provenance are recorded
        # on the result so a run is never silently priced off assumptions.
        self.cost_provenance: dict[str, dict[str, str]] = {}
        self.resolved_costs: dict[str, dict[str, float]] = {}
        self._validate_bindings(strategies)
        run_fingerprint = self._run_fingerprint(
            snapshot, start, end, strategies, seed
        )
        full_calendar = sorted(set(
            snapshot.table("trade_calendar")["trade_date"]
        ))
        calendar = [day for day in full_calendar if start <= day <= end]
        if len(calendar) < 2:
            raise BacktestError("backtest requires at least two trading days")
        prior_sessions = [day for day in full_calendar if day < calendar[0]]
        bootstrap_day = prior_sessions[-1] if prior_sessions else None
        planning_calendar = (
            [bootstrap_day, *calendar] if bootstrap_day is not None else calendar
        )
        self._calendar_index = {day: index for index, day in enumerate(calendar)}
        self._rule_cache: dict[tuple[Any, ...], Any] = {}
        instruments = snapshot.table("instruments")
        instrument_rows = {str(row.instrument_id): row for row in instruments.itertuples(index=False)}
        self._instrument_ids = frozenset(instrument_rows)
        market_start = bootstrap_day or start
        raw = snapshot.trading(end).bars(
            market_start, end, fields=_BAR_FIELDS
        )
        market = _index_bars(raw, planning_calendar)
        del raw
        allocations = (
            dict(self.capital_allocator.allocate(initial_cash))
            if self.capital_allocator is not None
            else {
                item.strategy.strategy_id: initial_cash * item.capital_weight
                for item in strategies
            }
        )
        expected_ids = {item.strategy.strategy_id for item in strategies}
        if set(allocations) != expected_ids:
            raise BacktestError("capital allocator returned unexpected strategy ids")
        if (
            any(not math.isfinite(float(value)) or float(value) <= 0 for value in allocations.values())
            or abs(sum(map(float, allocations.values())) - initial_cash) > 1e-7
        ):
            raise BacktestError("capital allocator returned invalid cash allocations")
        sleeves = {
            strategy_id: SleeveAccount(strategy_id, float(cash))
            for strategy_id, cash in allocations.items()
        }
        master = MasterAccount("__master__", initial_cash)
        states = {item.strategy.strategy_id: StrategyState() for item in strategies}
        previous_targets: dict[str, TargetPortfolio | None] = {
            item.strategy.strategy_id: None for item in strategies
        }
        schedules = {
            item.strategy.strategy_id: set(
                item.strategy.schedule.decision_dates(planning_calendar)
            )
            for item in strategies
        }
        bootstrap_strategy_ids = {
            item.strategy.strategy_id
            for item in strategies
            if bool(getattr(item.strategy, "bootstrap_on_start", False))
        }
        if bootstrap_strategy_ids:
            if self.execution_config.timing != "next_open":
                raise BacktestError(
                    "bootstrap_on_start requires execution timing='next_open'"
                )
            if bootstrap_day is None:
                raise BacktestError(
                    "bootstrap_on_start requires a trading session before start"
                )
            for strategy_id in bootstrap_strategy_ids:
                schedules[strategy_id].add(bootstrap_day)
        pending: dict[tuple[date, str], list[TargetPortfolio]] = defaultdict(list)
        entitlements: dict[tuple[str, str], int] = {}
        frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
        actions = snapshot.table("corporate_actions", cutoff=end)
        self._has_actions = not actions.empty
        action_days, action_records = _index_actions(actions)
        del actions
        if bootstrap_strategy_ids:
            bootstrap_bindings = [
                item for item in strategies
                if item.strategy.strategy_id in bootstrap_strategy_ids
            ]
            bootstrap_schedules = {
                strategy_id: {bootstrap_day}
                for strategy_id in bootstrap_strategy_ids
            }
            self._generate_decisions(
                bootstrap_day, bootstrap_day, calendar[0], "open", -1,
                bootstrap_bindings, bootstrap_schedules, sleeves, states,
                previous_targets, pending, frames, snapshot,
                _price_map(market, bootstrap_day, "close"), seed,
                is_bootstrap=True,
            )

        # Built after the resume branch, which rebinds `sleeves` and `master`.
        # Membership is fixed for the run and the values are mutated in place,
        # so the corporate-action loop can reuse one mapping instead of
        # rebuilding it per event.
        accounts = {**sleeves, "__master__": master}

        for day_index in range(len(calendar)):
            day = calendar[day_index]
            self._process_actions_before(
                day, action_days, sleeves, master, entitlements,
                instrument_rows, calendar, frames, accounts,
            )
            self._execute_targets(
                day, "open", pending.pop((day, "open"), []), sleeves, market,
                instrument_rows, calendar, frames, master, snapshot,
            )
            if self.execution_config.timing == "same_close" and day_index > 0:
                previous_day = calendar[day_index - 1]
                previous_close_prices = _price_map(market, previous_day, "close")
                self._generate_decisions(
                    day, previous_day, day, "close", day_index, strategies,
                    schedules, sleeves, states, previous_targets, pending, frames,
                    snapshot, previous_close_prices, seed,
                )
            self._execute_targets(
                day, "close", pending.pop((day, "close"), []), sleeves, market,
                instrument_rows, calendar, frames, master, snapshot,
            )
            # Deliberately the whole day's cross-section, not just held codes.
            # Every consumer (`_record_daily`, `_portfolio_view`, `nav`,
            # `reconcile`) only reads the codes it holds, so restricting this
            # would be correct today — but it would bake "valuation only needs
            # holdings" into the engine, and a future consumer would silently
            # read 0.0 instead of failing. Building it per day is cheap.
            close_prices = _price_map(market, day, "close")
            if self.execution_config.timing != "same_close" and day_index + 1 < len(calendar):
                execution_day = calendar[day_index + 1]
                phase = "open" if self.execution_config.timing == "next_open" else "close"
                self._generate_decisions(
                    day, day, execution_day, phase, day_index, strategies,
                    schedules, sleeves, states, previous_targets, pending, frames,
                    snapshot, close_prices, seed,
                )
            self._record_entitlements(
                day, action_records, sleeves, master, entitlements
            )
            self._record_daily(day, sleeves, master, close_prices, frames)
        materialized = {name: pd.DataFrame(rows) for name, rows in frames.items()}
        for required in (
            "targets", "signals", "strategy_states", "orders", "fills",
            "fill_allocations", "internal_crosses", "nav", "positions",
            "corporate_actions", "sleeve_demands", "demand_residuals",
            "cashflows", "master_nav", "master_positions", "reconciliations",
            "master_corporate_actions",
            "target_events", "candidate_targets", "universe_exclusions",
        ):
            materialized.setdefault(required, pd.DataFrame())
        materialized = enforce_ledger_schemas(materialized)
        metrics = performance_metrics(
            materialized["nav"], materialized["fills"], initial_cash,
            materialized["positions"],
        )
        if compute_attribution:
            materialized["attribution"] = attribution_report(
                materialized["nav"], materialized["fill_allocations"],
                materialized["positions"], materialized["corporate_actions"],
                snapshot.table("industry_membership", cutoff=end),
            )
        else:
            materialized["attribution"] = pd.DataFrame(
                columns=["date", "dimension", "component", "pnl"]
            )
        run_id = run_fingerprint[:20]
        return BacktestResult(
            run_id, metrics, materialized,
            cost_model={
                "rates": self.resolved_costs,
                "provenance": self.cost_provenance,
            },
        )

    def _run_fingerprint(self, snapshot, start, end, strategies, seed):
        try:
            engine_source = inspect.getsource(type(self))
        except (OSError, TypeError):
            engine_source = repr(type(self))
        try:
            executor_source = inspect.getsource(type(self.executor))
        except (OSError, TypeError):
            executor_source = repr(type(self.executor))
        try:
            allocator_source = (
                inspect.getsource(type(self.capital_allocator))
                if self.capital_allocator is not None else "binding_weights"
            )
        except (OSError, TypeError):
            allocator_source = repr(type(self.capital_allocator))
        strategy_fingerprints = []
        for item in strategies:
            try:
                source = inspect.getsource(type(item.strategy))
            except (OSError, TypeError):
                source = repr(type(item.strategy))
            strategy_fingerprints.append({
                "strategy_id": item.strategy.strategy_id,
                "capital_weight": item.capital_weight,
                "parameters": item.parameters or {},
                "code": hash_payload(source),
            })
        return hash_payload({
            "dataset": snapshot.metadata.fingerprint,
            "start": start, "end": end,
            "strategies": strategy_fingerprints,
            "execution": self.execution_config.model_dump(mode="json"),
            "engine_code": hash_payload(engine_source),
            "executor_code": hash_payload(executor_source),
            "capital_allocator_code": hash_payload(allocator_source),
            "ledger_schema": LEDGER_SCHEMA_VERSION,
            "seed": seed,
        })

    def _generate_decisions(
        self, signal_day, cutoff, execution_day, phase, day_index, strategies,
        schedules, sleeves, states, previous_targets, pending, frames,
        snapshot, valuation_prices, seed, is_bootstrap=False,
    ):
        for binding in strategies:
            strategy_id = binding.strategy.strategy_id
            if signal_day not in schedules[strategy_id]:
                continue
            view = self._portfolio_view(sleeves[strategy_id], valuation_prices, cutoff)
            context = StrategyContext(
                strategy_id, signal_day, cutoff, execution_day, phase, snapshot, None,
                view, previous_targets[strategy_id], states[strategy_id],
                binding.parameters or {}, np.random.default_rng(
                    derive_seed(seed, strategy_id, signal_day, day_index)
                ),
                self.factor_engine,
                is_bootstrap,
            )
            decision = binding.strategy.decide(context)
            self._validate_decision(context, decision)
            states[strategy_id] = decision.next_state
            if decision.target is not None:
                previous_targets[strategy_id] = decision.target
                pending[(decision.target.execution_date, decision.target.execution_phase)].append(decision.target)
                # Serialised once: the same diagnostics object is written to the
                # target_events row and to every per-instrument targets row, and
                # it carries the whole constraint report.
                diagnostics_json = canonical_json(decision.target.diagnostics)
                frames["target_events"].append({
                    "strategy_id": strategy_id, "signal_date": signal_day,
                    "execution_date": decision.target.execution_date,
                    "execution_phase": decision.target.execution_phase,
                    "cash_weight": decision.target.cash_weight,
                    "universe_fingerprint": decision.target.universe_fingerprint,
                    "signal_fingerprint": decision.target.signal_fingerprint,
                    "state_before_hash": decision.target.state_before_hash,
                    "state_after_hash": decision.target.state_after_hash,
                    "diagnostics": diagnostics_json,
                })
                report = decision.target.diagnostics.get("constraint_report")
                if report is not None:
                    for code, weight in report.before.items():
                        frames["candidate_targets"].append({
                            "strategy_id": strategy_id,
                            "signal_date": signal_day,
                            "execution_date": decision.target.execution_date,
                            "instrument_id": code, "weight": weight,
                        })
                for code, weight in decision.target.weights.items():
                    frames["targets"].append({
                        "strategy_id": strategy_id, "signal_date": signal_day,
                        "execution_date": decision.target.execution_date,
                        "instrument_id": code, "weight": weight,
                        "cash_weight": decision.target.cash_weight,
                        "execution_phase": decision.target.execution_phase,
                        "universe_fingerprint": decision.target.universe_fingerprint,
                        "signal_fingerprint": decision.target.signal_fingerprint,
                        "state_before_hash": decision.target.state_before_hash,
                        "state_after_hash": decision.target.state_after_hash,
                        "diagnostics": diagnostics_json,
                    })
            frames["strategy_states"].append({
                "strategy_id": strategy_id, "date": signal_day,
                "state_hash": hash_payload(decision.next_state),
                "state_json": canonical_json(dict(decision.next_state.payload)),
            })
            if decision.signals is not None:
                for record in decision.signals.frame.to_dict("records"):
                    frames["signals"].append({"strategy_id": strategy_id, **record})
            for exclusion in decision.diagnostics.get("excluded", []):
                frames["universe_exclusions"].append({
                    "strategy_id": strategy_id, "signal_date": signal_day,
                    **exclusion,
                })

    def _execute_targets(
        self, day, phase, targets, sleeves, market, instrument_rows, calendar,
        frames, master, snapshot,
    ):
        if not targets:
            return
        prices = _price_map(market, day, "open" if phase == "open" else "close")
        for target in targets:
            sleeve = sleeves[target.strategy_id]
            missing_held = [
                code for code in sleeve.lots
                if sleeve.quantity(code) > 0 and code not in prices
            ]
            if missing_held:
                raise BacktestError(
                    f"cannot value held instruments on {day}: {missing_held[:10]}"
                )
        demands = []
        remaining_cash = {
            strategy_id: sleeve.cash for strategy_id, sleeve in sleeves.items()
        }
        # A same-phase buy may be funded by that sleeve's planned sells. Sells are
        # executed before crosses and buys, while the batch cash invariant is
        # checked again before any internal transfer.
        for target in targets:
            sleeve = sleeves[target.strategy_id]
            nav = sleeve.nav(prices)
            liquidation = target.diagnostics.get(
                "liquidate_only_instruments"
            )
            codes = (
                set(map(str, liquidation))
                if liquidation is not None
                else set(target.weights) | set(sleeve.lots)
            )
            for code in sorted(codes):
                price = prices.get(code)
                if price is None or price <= 0 or code not in instrument_rows:
                    continue
                lot = int(instrument_rows[code].lot_size)
                desired = (
                    0 if liquidation is not None else math.floor(
                        nav * target.weights.get(code, 0.0) / price / lot
                    ) * lot
                )
                difference = desired - sleeve.quantity(code)
                if difference < 0:
                    sellable = min(
                        -difference, sleeve.sellable_quantity(code, day)
                    )
                    executable = (
                        sellable if liquidation is not None
                        else math.floor(sellable / lot) * lot
                    )
                    remaining_cash[target.strategy_id] += (
                        executable * price
                    )
        for target in targets:
            sleeve = sleeves[target.strategy_id]
            nav = sleeve.nav(prices)
            liquidation = target.diagnostics.get(
                "liquidate_only_instruments"
            )
            codes = (
                set(map(str, liquidation))
                if liquidation is not None
                else set(target.weights) | set(sleeve.lots)
            )
            for code in sorted(codes):
                price = prices.get(code)
                demand_id = f"{day}:{phase}:{target.strategy_id}:{code}"
                if code not in instrument_rows:
                    raise BacktestError(f"target references unknown instrument: {code}")
                if price is None or price <= 0:
                    frames["demand_residuals"].append({
                        "demand_id": demand_id, "execution_date": day,
                        "execution_phase": phase, "strategy_id": target.strategy_id,
                        "instrument_id": code, "side": "unknown", "quantity": None,
                        "reason": "missing_reference_price",
                    })
                    continue
                info = instrument_rows[code]
                lot = int(info.lot_size)
                desired = (
                    0 if liquidation is not None else math.floor(
                        nav * target.weights.get(code, 0.0) / price / lot
                    ) * lot
                )
                current = sleeve.quantity(code)
                difference = desired - current
                if difference < 0:
                    requested = -difference
                    quantity = min(requested, sleeve.sellable_quantity(code, day))
                    if liquidation is None:
                        quantity = math.floor(quantity / lot) * lot
                    if quantity < requested:
                        frames["demand_residuals"].append({
                            "demand_id": demand_id, "execution_date": day,
                            "execution_phase": phase, "strategy_id": target.strategy_id,
                            "instrument_id": code, "side": "sell",
                            "quantity": requested - quantity,
                            "reason": "insufficient_sellable_quantity",
                        })
                    if quantity:
                        demand_lot = 1 if liquidation is not None else lot
                        demand = SleeveDemand(
                            target.strategy_id, code, "sell", quantity, price,
                            demand_lot,
                            demand_id, phase, desired,
                        )
                        demands.append(demand)
                        frames["sleeve_demands"].append(asdict(demand))
                elif difference > 0:
                    requested = math.floor(difference / lot) * lot
                    affordable = math.floor(
                        remaining_cash[target.strategy_id] / price / lot
                    ) * lot
                    quantity = min(requested, affordable)
                    remaining_cash[target.strategy_id] -= quantity * price
                    if quantity < requested:
                        frames["demand_residuals"].append({
                            "demand_id": demand_id, "execution_date": day,
                            "execution_phase": phase, "strategy_id": target.strategy_id,
                            "instrument_id": code, "side": "buy",
                            "quantity": requested - quantity,
                            "reason": "insufficient_cash_for_reference_notional",
                        })
                    if quantity:
                        demand = SleeveDemand(
                            target.strategy_id, code, "buy", quantity, price, lot,
                            demand_id, phase, desired,
                        )
                        demands.append(demand)
                        frames["sleeve_demands"].append(asdict(demand))
        crosses, orders, residuals = net_sleeve_demands(demands, day, phase)
        fills_by_side = {"sell": [], "buy": []}
        for order in orders:
            fills_by_side[order.side].append(order)
            frames["orders"].append(asdict(order))
        for order in fills_by_side["sell"]:
            self._execute_order(
                order, residuals, sleeves, market, instrument_rows, calendar,
                frames, master, snapshot,
            )
        self._apply_internal_crosses(
            crosses, day, sleeves, instrument_rows, calendar, frames
        )
        master.reconcile(sleeves, prices)
        frames["reconciliations"].append({
            "date": day, "phase": phase, "event": "after_internal_cross",
            "status": "passed",
        })
        for order in fills_by_side["buy"]:
            self._execute_order(
                order, residuals, sleeves, market, instrument_rows, calendar,
                frames, master, snapshot,
            )

    def _apply_internal_crosses(
        self, crosses, day, sleeves, instrument_rows, calendar, frames
    ):
        if not crosses:
            return
        accepted = list(crosses)
        while accepted:
            cash_delta = {strategy_id: 0.0 for strategy_id in sleeves}
            for cross in accepted:
                notional = cross.quantity * cross.price
                cash_delta[cross.seller_strategy_id] += notional
                cash_delta[cross.buyer_strategy_id] -= notional
            negative = {
                strategy_id for strategy_id, delta in cash_delta.items()
                if sleeves[strategy_id].cash + delta < -1e-7
            }
            if not negative:
                break
            removed = [
                cross for cross in accepted
                if cross.buyer_strategy_id in negative
            ]
            accepted = [
                cross for cross in accepted
                if cross.buyer_strategy_id not in negative
            ]
            for cross in removed:
                frames["demand_residuals"].append({
                    "demand_id": None, "execution_date": day,
                    "execution_phase": cross.execution_phase,
                    "strategy_id": cross.buyer_strategy_id,
                    "instrument_id": cross.instrument_id, "side": "buy",
                    "quantity": cross.quantity,
                    "reason": "internal_cross_insufficient_cash",
                    "cross_id": cross.cross_id,
                })
                frames["demand_residuals"].append({
                    "demand_id": None, "execution_date": day,
                    "execution_phase": cross.execution_phase,
                    "strategy_id": cross.seller_strategy_id,
                    "instrument_id": cross.instrument_id, "side": "sell",
                    "quantity": cross.quantity,
                    "reason": "internal_cross_counterparty_rejected",
                    "cross_id": cross.cross_id,
                })
        crosses = accepted
        if not crosses:
            return
        cash_delta = {strategy_id: 0.0 for strategy_id in sleeves}
        required: dict[tuple[str, str], int] = defaultdict(int)
        for cross in crosses:
            notional = cross.quantity * cross.price
            cash_delta[cross.seller_strategy_id] += notional
            cash_delta[cross.buyer_strategy_id] -= notional
            required[(cross.seller_strategy_id, cross.instrument_id)] += cross.quantity
        for (strategy_id, code), quantity in required.items():
            if sleeves[strategy_id].sellable_quantity(code, day) < quantity:
                raise BacktestError("internal crosses exceed a sleeve's sellable position")
        for cross in crosses:
            seller = sleeves[cross.seller_strategy_id]
            removed = seller.remove(cross.instrument_id, cross.quantity, day)
            if removed != cross.quantity:
                raise BacktestError("internal cross allocation changed after validation")
        for strategy_id, delta in cash_delta.items():
            sleeves[strategy_id].cash += delta
        for cross in crosses:
            buyer = sleeves[cross.buyer_strategy_id]
            sellable_date = self._sellable_date(
                day, int(instrument_rows[cross.instrument_id].sell_delay_days), calendar
            )
            buyer.add_lot(PositionLot(
                cross.instrument_id, cross.quantity, day, sellable_date, cross.price
            ))
            frames["internal_crosses"].append(asdict(cross))
            frames["cashflows"].extend([
                {
                    "event_id": f"{cross.cross_id}:seller", "date": day,
                    "account_id": cross.seller_strategy_id,
                    "flow_type": "internal_cross_sale",
                    "amount": cross.quantity * cross.price,
                    "instrument_id": cross.instrument_id,
                    "upstream_event_id": cross.cross_id,
                },
                {
                    "event_id": f"{cross.cross_id}:buyer", "date": day,
                    "account_id": cross.buyer_strategy_id,
                    "flow_type": "internal_cross_purchase",
                    "amount": -cross.quantity * cross.price,
                    "instrument_id": cross.instrument_id,
                    "upstream_event_id": cross.cross_id,
                },
            ])

    def _execute_order(
        self, order, residuals, sleeves, market, instrument_rows, calendar,
        frames, master, snapshot,
    ):
        bars = market.get(order.execution_date)
        row = bars.row(order.instrument_id) if bars is not None else None
        info = instrument_rows[order.instrument_id]
        asset_type = str(info.asset_type)
        # A snapshot may carry no rule for this day at all (Hermes writes only
        # `source_missing` placeholders). That is not fatal now: cost
        # resolution falls back to configuration and defaults.
        # Cached per (day, exchange, asset_type): `market_rule` re-reads the whole
        # `market_rules` table on every call, and a rebalance day issues dozens
        # of orders that all resolve to the same rule. The date is part of the
        # key, so the effective-window semantics are untouched.
        rule_key = (order.execution_date, str(info.exchange), asset_type)
        if rule_key in self._rule_cache:
            rule = self._rule_cache[rule_key]
        else:
            try:
                rule = snapshot.market_rule(*rule_key)
            except DataContractError:
                rule = None
            self._rule_cache[rule_key] = rule
        fill = self.executor.execute(order, row, asset_type, rule)
        self._validate_fill(order, fill)
        demands = residuals.get((order.instrument_id, order.side), [])
        quantities = allocate_fill_quantities(order, fill.filled_quantity, demands)
        if order.side == "buy":
            # Resolved once per order: the rule and asset type are loop
            # invariants, and `_cost_rule` was previously re-resolved on every
            # lot as the affordability walk stepped down.
            commission_bps, minimum_commission, _, buy_tax_bps, transfer = (
                self._cost_rule(rule, asset_type)
            )
            for strategy_id in list(quantities):
                quantity = quantities[strategy_id]
                lot = order.lot_size
                while quantity > 0:
                    notional = quantity * fill.price
                    estimated_commission = max(
                        minimum_commission,
                        notional * (commission_bps + transfer) / 10000,
                    ) if quantity else 0.0
                    estimated_tax = notional * buy_tax_bps / 10000
                    if (
                        notional + estimated_commission + estimated_tax
                        <= sleeves[strategy_id].cash + 1e-9
                    ):
                        break
                    quantity -= lot
                quantities[strategy_id] = quantity
            quantities = {key: value for key, value in quantities.items() if value > 0}
        actual = sum(quantities.values())
        actual_notional = actual * fill.price
        # Unconditional: this call is what records the resolved rates and their
        # provenance into `result.cost_model` for every order. Do not fold it
        # into the buy-side resolution above.
        commission_bps, minimum_commission, sell_tax_bps, buy_tax_bps, transfer = (
            self._cost_rule(rule, asset_type)
        )
        commission = max(
            minimum_commission,
            actual_notional * (commission_bps + transfer) / 10000,
        ) if actual else 0.0
        tax_rate = sell_tax_bps if order.side == "sell" else buy_tax_bps
        tax = actual_notional * tax_rate / 10000 if actual else 0.0
        allocations = cost_allocations(
            order, quantities, fill.price, commission, tax,
            fill.slippage_bps + fill.impact_bps,
        )
        for allocation in allocations:
            sleeve = sleeves[allocation.strategy_id]
            notional = allocation.quantity * allocation.price
            if allocation.side == "sell":
                removed = sleeve.remove(allocation.instrument_id, allocation.quantity, order.execution_date)
                if removed != allocation.quantity:
                    raise BacktestError("allocated sell exceeds sellable sleeve quantity")
                sleeve.cash += notional - allocation.commission - allocation.tax
            else:
                total_cost = notional + allocation.commission + allocation.tax
                if total_cost > sleeve.cash + 1e-7:
                    raise BacktestError("allocated buy exceeds sleeve cash")
                sleeve.cash -= total_cost
                sellable_date = self._sellable_date(
                    order.execution_date, int(instrument_rows[allocation.instrument_id].sell_delay_days), calendar
                )
                sleeve.add_lot(PositionLot(
                    allocation.instrument_id, allocation.quantity, order.execution_date,
                    sellable_date, allocation.price,
                ))
            frames["fill_allocations"].append(asdict(allocation))
            signed_notional = notional if allocation.side == "sell" else -notional
            frames["cashflows"].append({
                "event_id": f"{allocation.allocation_id}:trade",
                "date": order.execution_date, "account_id": allocation.strategy_id,
                "flow_type": f"external_{allocation.side}",
                "amount": signed_notional, "instrument_id": allocation.instrument_id,
                "upstream_event_id": allocation.order_id,
            })
            for kind, amount in (
                ("commission", allocation.commission),
                ("tax", allocation.tax),
            ):
                if amount:
                    frames["cashflows"].append({
                        "event_id": f"{allocation.allocation_id}:{kind}",
                        "date": order.execution_date,
                        "account_id": allocation.strategy_id,
                        "flow_type": kind, "amount": -amount,
                        "instrument_id": allocation.instrument_id,
                        "upstream_event_id": allocation.order_id,
                    })
        if actual:
            if order.side == "sell":
                removed = master.remove(
                    order.instrument_id, actual, order.execution_date
                )
                if removed != actual:
                    raise BacktestError("master sell exceeds sellable quantity")
                master.cash += actual_notional - commission - tax
            else:
                total_cost = actual_notional + commission + tax
                if total_cost > master.cash + 1e-7:
                    raise BacktestError("master buy exceeds cash")
                master.cash -= total_cost
                sellable_date = self._sellable_date(
                    order.execution_date, int(info.sell_delay_days), calendar
                )
                master.add_lot(PositionLot(
                    order.instrument_id, actual, order.execution_date,
                    sellable_date, fill.price,
                ))
        allocated_by_strategy = {
            allocation.strategy_id: allocation.quantity for allocation in allocations
        }
        for demand in demands:
            unfilled = demand.quantity - allocated_by_strategy.get(demand.strategy_id, 0)
            if unfilled:
                frames["demand_residuals"].append({
                    "demand_id": demand.demand_id,
                    "execution_date": order.execution_date,
                    "execution_phase": order.execution_phase,
                    "strategy_id": demand.strategy_id,
                    "instrument_id": demand.instrument_id,
                    "side": demand.side, "quantity": unfilled,
                    "reason": fill.reject_reason or (
                        "insufficient_cash" if order.side == "buy" and actual == 0
                        else "partial_or_capacity"
                    ),
                    "order_id": order.order_id,
                })
        frames["fills"].append({
            **asdict(fill), "filled_quantity": actual,
            "commission": commission, "tax": tax,
            "status": "rejected" if actual == 0 else ("filled" if actual == order.quantity else "partial"),
            "reject_reason": fill.reject_reason or (
                "insufficient_cash" if actual == 0 and order.side == "buy" else None
            ),
            "execution_phase": order.execution_phase,
        })
        master.reconcile(sleeves, {order.instrument_id: fill.price})
        frames["reconciliations"].append({
            "date": order.execution_date, "phase": order.execution_phase,
            "event": order.order_id, "status": "passed",
        })

    def _cost_rule(self, rule, asset_type):
        values, provenance = self.cost_model.resolve(
            self.execution_config, asset_type, rule
        )
        values = dict(values)
        provenance = dict(provenance)
        required = {
            "commission_bps", "minimum_commission", "sell_tax_bps",
            "buy_tax_bps", "transfer_fee_bps",
        }
        if set(values) != required:
            raise BacktestError("cost model returned an invalid field set")
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in values.values()):
            raise BacktestError("cost model returned invalid rates")
        self.resolved_costs[asset_type] = values
        self.cost_provenance[asset_type] = provenance
        return (
            values["commission_bps"], values["minimum_commission"],
            values["sell_tax_bps"], values["buy_tax_bps"],
            values["transfer_fee_bps"],
        )

    @staticmethod
    def _validate_fill(order, fill):
        if (
            fill.order_id != order.order_id
            or fill.instrument_id != order.instrument_id
            or fill.side != order.side
            or fill.execution_date != order.execution_date
        ):
            raise BacktestError("execution model returned a fill for another order")
        if fill.requested_quantity != order.quantity:
            raise BacktestError("execution model changed the requested quantity")
        if (
            fill.filled_quantity < 0
            or fill.filled_quantity > order.quantity
            or fill.filled_quantity % order.lot_size != 0
        ):
            raise BacktestError("execution model returned an invalid fill quantity")
        numeric = (
            fill.price, fill.commission, fill.tax,
            fill.slippage_bps, fill.impact_bps,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise BacktestError("execution model returned non-finite fill values")
        if fill.price < 0 or fill.commission < 0 or fill.tax < 0:
            raise BacktestError("execution model returned negative price or costs")

    def _validate_decision(self, context, decision):
        if not isinstance(decision.next_state, StrategyState):
            raise BacktestError(
                f"{context.strategy_id}: next_state must be StrategyState"
            )
        try:
            canonical_json(dict(decision.next_state.payload))
        except (TypeError, ValueError) as exc:
            raise BacktestError(
                f"{context.strategy_id}: strategy state is not serializable"
            ) from exc
        target = decision.target
        if target is None:
            return
        if target.strategy_id != context.strategy_id:
            raise BacktestError("target strategy_id does not match its binding")
        if target.signal_date != context.signal_date:
            raise BacktestError("target signal_date does not match the decision context")
        if target.execution_date != context.execution_date:
            raise BacktestError("target execution_date does not match the decision context")
        if target.execution_phase != context.execution_phase:
            raise BacktestError("target execution_phase does not match the decision context")
        weights = {str(code): float(value) for code, value in target.weights.items()}
        values = [*weights.values(), float(target.cash_weight)]
        if not all(math.isfinite(value) for value in values):
            raise BacktestError("target weights must be finite")
        if any(value < 0 for value in values):
            raise BacktestError("target weights must be non-negative")
        if abs(sum(weights.values()) + float(target.cash_weight) - 1.0) > 1e-12:
            raise BacktestError("target weights and cash must sum to one")
        unknown = sorted(set(weights) - self._instrument_ids)
        if unknown:
            raise BacktestError(
                f"target contains instruments outside the snapshot: {unknown[:10]}"
            )

    @staticmethod
    def _portfolio_view(sleeve, prices, day):
        positions = tuple(
            PositionView(code, sleeve.quantity(code), sleeve.sellable_quantity(code, day), prices.get(code, 0.0))
            for code in sorted(sleeve.lots) if sleeve.quantity(code)
        )
        return PortfolioView(sleeve.nav(prices), sleeve.cash, positions)

    def _sellable_date(self, day, delay, calendar):
        # `calendar.index(day)` is a linear scan and this runs on every buy fill
        # and every internal cross; the positions are fixed for the whole run.
        index = self._calendar_index[day]
        return calendar[min(len(calendar) - 1, index + delay)]

    @staticmethod
    def _record_entitlements(day, action_records, sleeves, master, entitlements):
        for event in action_records.get(day, ()):
            for strategy_id, sleeve in sleeves.items():
                quantity = sleeve.quantity(event.code)
                if quantity:
                    entitlements[(strategy_id, event.event_id)] = quantity
            master_quantity = master.quantity(event.code)
            if master_quantity:
                entitlements[("__master__", event.event_id)] = master_quantity

    def _process_actions_before(
        self, day, action_days, sleeves, master, entitlements, instruments,
        calendar, frames, accounts,
    ):
        # Preserved from the pre-index version: a snapshot with no corporate
        # actions at all skipped this method entirely, so it never reached the
        # reconcile below. Keep that, otherwise an empty-actions run gains a
        # daily invariant check it did not have before.
        if not self._has_actions:
            return
        for kind, item in action_days.get(day, ()):
            event = item.row
            event_id, code, event_type = item.event_id, item.code, item.event_type
            if kind == "ex":
                for strategy_id, sleeve in accounts.items():
                    quantity = entitlements.get((strategy_id, event_id), 0)
                    if not quantity:
                        continue
                    if event_type in {"cash_dividend", "dividend", "分红"}:
                        cash_per_share = (
                            float(event.cash_per_share)
                            if pd.notna(event.cash_per_share) else 0.0
                        )
                        amount = quantity * cash_per_share
                        sleeve.receivables[event_id] = amount
                        destination = (
                            "master_corporate_actions"
                            if strategy_id == "__master__" else "corporate_actions"
                        )
                        frames[destination].append({
                            "date": day, "strategy_id": strategy_id, "event_id": event_id,
                            "instrument_id": code, "type": "dividend_receivable", "amount": amount,
                        })
                    elif event_type in {"split", "merge", "拆分", "合并", "bonus", "送股", "转增"}:
                        ratio = float(event.share_ratio)
                        multiplier = 1 + ratio if event_type in {"bonus", "送股", "转增"} else ratio
                        if strategy_id == "__master__":
                            delta = sum(
                                int(round(
                                    entitlements.get((sid, event_id), 0)
                                    * (multiplier - 1.0)
                                ))
                                for sid in sleeves
                            )
                        else:
                            delta = int(round(quantity * (multiplier - 1.0)))
                        if delta > 0:
                            sleeve.add_lot(PositionLot(code, delta, day, day, 0.0))
                        elif delta < 0 and sleeve.remove_any(code, -delta) != -delta:
                            raise BacktestError("company action exceeds entitled sleeve position")
                        destination = (
                            "master_corporate_actions"
                            if strategy_id == "__master__" else "corporate_actions"
                        )
                        frames[destination].append({
                            "date": day, "strategy_id": strategy_id, "event_id": event_id,
                            "instrument_id": code, "type": "share_adjustment", "ratio": multiplier,
                        })
            else:
                for strategy_id, sleeve in accounts.items():
                    amount = sleeve.receivables.pop(event_id, 0.0)
                    if amount:
                        sleeve.cash += amount
                        destination = (
                            "master_corporate_actions"
                            if strategy_id == "__master__" else "corporate_actions"
                        )
                        frames[destination].append({
                            "date": day, "strategy_id": strategy_id, "event_id": event_id,
                            "instrument_id": code, "type": "dividend_paid", "amount": amount,
                        })
                        frames["cashflows"].append({
                            "event_id": f"{event_id}:{strategy_id}:paid",
                            "date": day, "account_id": strategy_id,
                            "flow_type": "dividend_paid", "amount": amount,
                            "instrument_id": code, "upstream_event_id": event_id,
                        })
        master.reconcile(sleeves, {})

    @staticmethod
    def _record_daily(day, sleeves, master, prices, frames):
        for strategy_id, sleeve in sleeves.items():
            nav = sleeve.nav(prices)
            if sleeve.cash < -1e-7 or nav < -1e-7:
                raise BacktestError("negative cash or NAV accounting invariant")
            frames["nav"].append({
                "date": day, "strategy_id": strategy_id, "cash": sleeve.cash,
                "receivables": sum(sleeve.receivables.values()), "nav": nav,
            })
            for code in sorted(sleeve.lots):
                quantity = sleeve.quantity(code)
                if quantity:
                    price = prices.get(code, 0.0)
                    frames["positions"].append({
                        "date": day, "strategy_id": strategy_id,
                        "instrument_id": code, "quantity": quantity,
                        "last_price": price, "market_value": quantity * price,
                    })
        master.reconcile(sleeves, prices)
        frames["master_nav"].append({
            "date": day, "account_id": "__master__", "cash": master.cash,
            "receivables": sum(master.receivables.values()),
            "nav": master.nav(prices),
        })
        for code in sorted(master.lots):
            quantity = master.quantity(code)
            if quantity:
                price = prices.get(code, 0.0)
                frames["master_positions"].append({
                    "date": day, "account_id": "__master__",
                    "instrument_id": code, "quantity": quantity,
                    "last_price": price, "market_value": quantity * price,
                })
        frames["reconciliations"].append({
            "date": day, "phase": "close", "event": "daily_close",
            "status": "passed",
        })

    @staticmethod
    def _validate_bindings(strategies):
        if not strategies:
            raise BacktestError("at least one strategy is required")
        ids = [item.strategy.strategy_id for item in strategies]
        if len(ids) != len(set(ids)):
            raise BacktestError("strategy ids must be unique")
        if abs(sum(item.capital_weight for item in strategies) - 1.0) > 1e-12:
            raise BacktestError("strategy capital weights must sum to one")
        if any(item.capital_weight <= 0 for item in strategies):
            raise BacktestError("strategy capital weights must be positive")

from __future__ import annotations

import math
import inspect
from bisect import bisect_left
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
    allocate_fill_demands, cost_demand_allocations, net_sleeve_demands,
)
from zyquant.portfolio.capital import CapitalAllocator
from zyquant.strategy.types import (
    PortfolioView, PositionView, ScheduledTargetPortfolio, StrategyContext,
    StrategyState, TargetPortfolio,
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


@dataclass(frozen=True)
class _ValuationMark:
    price: float
    observed_date: date
    paused: bool = False


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


class _LazyMarket:
    """Bounded annual cache backed by partition-pushed parquet reads."""

    def __init__(self, snapshot, cutoff, instruments, fields):
        self.snapshot = snapshot
        self.cutoff = cutoff
        self.instruments = instruments
        self.fields = fields
        self.cache: dict[int, dict[date, _DayBars]] = {}

    def get(self, day):
        year = day.year
        if year not in self.cache:
            year_start = date(year, 1, 1)
            year_end = min(date(year, 12, 31), self.cutoff)
            raw = self.snapshot.trading(self.cutoff).bars(
                year_start, year_end, instruments=self.instruments,
                fields=self.fields,
            )
            days = sorted(set(raw["trade_date"]))
            self.cache[year] = _index_bars(raw, days)
            # A year boundary may need the prior session. Two annual partitions
            # remain far below the previous 12-year resident frame.
            while len(self.cache) > 2:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[year].get(day)


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
        self._calendar = calendar
        self._rule_cache: dict[tuple[Any, ...], Any] = {}
        self._market_rules = snapshot.table("market_rules", cutoff=end)
        instruments = snapshot.table("instruments")
        instrument_rows = {str(row.instrument_id): row for row in instruments.itertuples(index=False)}
        self._instrument_ids = frozenset(instrument_rows)
        scopes = [
            getattr(item.strategy, "market_instruments", None)
            for item in strategies
        ]
        # A prepared strategy may declare the exact instruments it can trade.
        # Only apply the pushdown when every sleeve supplies a scope; otherwise
        # preserve the framework's unrestricted legacy behaviour.
        market_instruments = None
        if scopes and all(scope is not None for scope in scopes):
            scoped_codes: set[str] = set()
            for scope in scopes:
                if scope is not None:
                    scoped_codes.update(map(str, scope))
            market_instruments = sorted(scoped_codes)
        market = _LazyMarket(
            snapshot, end, market_instruments, _BAR_FIELDS,
        )
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
        valued_codes: set[str] = set()
        # The causal valuation book is updated only from observed bars. Missing
        # bars may use it only after the instrument reaches its disposal day.
        valuation_marks: dict[str, _ValuationMark] = {}
        delisting_days: dict[str, date] = {}
        for code, row in instrument_rows.items():
            raw_date = getattr(row, "delist_date", None)
            if raw_date is None or pd.isna(raw_date):
                continue
            delist_date = raw_date.date() if hasattr(raw_date, "date") else raw_date
            index = bisect_left(calendar, delist_date)
            if index < len(calendar):
                delisting_days[code] = calendar[index]
        self._delisting_days = delisting_days
        self._delisted_codes: set[str] = set()
        self._valuation_marks = valuation_marks
        self._delisting_migrations: set[str] = set()
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
                valuation_marks,
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
                valuation_marks,
            )
            # Strategy contexts and daily accounting only value held names.
            # Execution reads its own full day slice from `market`, so building
            # a several-thousand-name dict twice per session is unnecessary.
            valued_codes.clear()
            for sleeve in sleeves.values():
                valued_codes.update(
                    code for code in sleeve.lots if sleeve.quantity(code)
                )
            close_bars = market.get(day)
            close_prices: dict[str, float] = {}
            if close_bars is not None:
                for code in valued_codes:
                    if code in self._delisted_codes:
                        continue
                    bar = close_bars.row(code)
                    if bar is not None:
                        # Match `_DayBars.price_map()`'s Python-scalar
                        # conversion exactly, including floating addition order.
                        price = bar.close.item()
                        close_prices[code] = price
                        valuation_marks[code] = _ValuationMark(
                            price, day, bool(bar.paused),
                        )
            missing_active = sorted(
                code for code in valued_codes
                if code not in close_prices
                and not self._is_delisting_effective(code, day)
            )
            if missing_active:
                raise BacktestError(
                    f"active held instruments have no bar on {day}: "
                    f"{missing_active[:10]}"
                )
            for code in valued_codes:
                if code in close_prices:
                    continue
                mark = valuation_marks.get(code)
                if mark is None:
                    raise BacktestError(
                        f"cannot value delisted held instrument on {day}: {code}"
                    )
                close_prices[code] = mark.price
            self._record_entitlements(
                day, action_records, sleeves, master, entitlements
            )
            self._process_delistings(
                day, sleeves, master, close_prices, frames
            )
            if self.execution_config.timing != "same_close" and day_index + 1 < len(calendar):
                execution_day = calendar[day_index + 1]
                phase = "open" if self.execution_config.timing == "next_open" else "close"
                self._generate_decisions(
                    day, day, execution_day, phase, day_index, strategies,
                    schedules, sleeves, states, previous_targets, pending, frames,
                    snapshot, close_prices, seed,
                )
            self._record_daily(day, sleeves, master, close_prices, frames)
        materialized = {name: pd.DataFrame(rows) for name, rows in frames.items()}
        for required in (
            "targets", "signals", "strategy_states", "orders", "fills",
            "fill_allocations", "internal_crosses", "nav", "positions",
            "position_lots",
            "corporate_actions", "sleeve_demands", "demand_residuals",
            "cashflows", "master_nav", "master_positions", "reconciliations",
            "master_corporate_actions",
            "target_events", "candidate_targets", "universe_exclusions",
            "scheduled_target_residuals",
        ):
            materialized.setdefault(required, pd.DataFrame())
        materialized = enforce_ledger_schemas(materialized)
        metrics = performance_metrics(
            materialized["nav"], materialized["fills"], initial_cash,
            materialized["positions"],
        )
        metrics.update(self._delisting_metrics(materialized["positions"]))
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
                self._enqueue_target(decision.target, pending, frames)
            by_cohort: dict[str, list[ScheduledTargetPortfolio]] = defaultdict(list)
            for item in decision.scheduled_targets:
                by_cohort[item.cohort_id].append(item)
            for cohort_id, scheduled in by_cohort.items():
                indexes = [day_index + item.session_offset for item in scheduled]
                if any(index < 0 or index >= len(self._calendar) for index in indexes):
                    frames["scheduled_target_residuals"].append({
                        "strategy_id": strategy_id,
                        "signal_date": signal_day,
                        "cohort_id": cohort_id,
                        "reason": "outside_backtest_range",
                    })
                    continue
                for item, index in zip(scheduled, indexes, strict=True):
                    target = TargetPortfolio(
                        item.strategy_id, item.signal_date, self._calendar[index],
                        item.execution_phase, item.weights, item.cash_weight,
                        item.universe_fingerprint, item.signal_fingerprint,
                        item.state_before_hash, item.state_after_hash,
                        item.diagnostics, item.cohort_id,
                    )
                    self._enqueue_target(target, pending, frames)
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

    @staticmethod
    def _enqueue_target(target, pending, frames):
        pending[(target.execution_date, target.execution_phase)].append(target)
        diagnostics_json = canonical_json(target.diagnostics)
        common = {
            "strategy_id": target.strategy_id,
            "signal_date": target.signal_date,
            "execution_date": target.execution_date,
            "execution_phase": target.execution_phase,
            "cash_weight": target.cash_weight,
            "universe_fingerprint": target.universe_fingerprint,
            "signal_fingerprint": target.signal_fingerprint,
            "state_before_hash": target.state_before_hash,
            "state_after_hash": target.state_after_hash,
            "cohort_id": target.cohort_id,
            "diagnostics": diagnostics_json,
        }
        frames["target_events"].append(common)
        report = target.diagnostics.get("constraint_report")
        if report is not None:
            for code, weight in report.before.items():
                frames["candidate_targets"].append({
                    "strategy_id": target.strategy_id,
                    "signal_date": target.signal_date,
                    "execution_date": target.execution_date,
                    "instrument_id": code,
                    "weight": weight,
                    "cohort_id": target.cohort_id,
                })
        for code, weight in target.weights.items():
            frames["targets"].append({
                **common, "instrument_id": code, "weight": weight,
            })

    def _execute_targets(
        self, day, phase, targets, sleeves, market, instrument_rows, calendar,
        frames, master, snapshot, valuation_marks,
    ):
        if not targets:
            return
        daily_bars = market.get(day)
        field = "open" if phase == "open" else "close"
        required_codes: set[str] = set()
        for target in targets:
            sleeve = sleeves[target.strategy_id]
            liquidation = target.diagnostics.get("liquidate_only_instruments")
            required_codes.update(
                map(str, liquidation)
                if liquidation is not None
                else set(target.weights) | sleeve.instruments(target.cohort_id)
            )
            required_codes.update(sleeve.lots)
        prices = {}
        if daily_bars is not None:
            for code in required_codes:
                bar = daily_bars.row(code)
                if bar is not None:
                    prices[code] = getattr(bar, field).item()
        valuation_prices = dict(prices)
        for code in self._delisted_codes:
            mark = valuation_marks.get(code)
            if mark is not None:
                valuation_prices[code] = mark.price
        for target in targets:
            sleeve = sleeves[target.strategy_id]
            for code in sleeve.instruments(target.cohort_id):
                mark = valuation_marks.get(code)
                if (
                    code not in valuation_prices
                    and mark is not None
                    and self._is_delisting_effective(code, day)
                ):
                    valuation_prices[code] = mark.price
        for target in targets:
            sleeve = sleeves[target.strategy_id]
            missing_held = [
                code for code in sleeve.instruments(target.cohort_id)
                if code not in valuation_prices
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
            nav = sleeve.nav(valuation_prices)
            liquidation = target.diagnostics.get(
                "liquidate_only_instruments"
            )
            codes = (
                set(map(str, liquidation))
                if liquidation is not None
                else set(target.weights) | sleeve.instruments(target.cohort_id)
            )
            for code in sorted(codes):
                if code in self._delisted_codes:
                    continue
                price = prices.get(code)
                if price is None or price <= 0 or code not in instrument_rows:
                    continue
                lot = int(instrument_rows[code].lot_size)
                desired = (
                    0 if liquidation is not None else math.floor(
                        nav * target.weights.get(code, 0.0) / price / lot
                    ) * lot
                )
                difference = desired - sleeve.quantity(code, target.cohort_id)
                if difference < 0:
                    sellable = min(
                        -difference,
                        sleeve.sellable_quantity(code, day, target.cohort_id),
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
            nav = sleeve.nav(valuation_prices)
            liquidation = target.diagnostics.get(
                "liquidate_only_instruments"
            )
            codes = (
                set(map(str, liquidation))
                if liquidation is not None
                else set(target.weights) | sleeve.instruments(target.cohort_id)
            )
            for code in sorted(codes):
                is_frozen = code in self._delisted_codes
                price = (
                    valuation_prices.get(code) if is_frozen else prices.get(code)
                )
                cohort_suffix = f":{target.cohort_id}" if target.cohort_id else ""
                demand_id = (
                    f"{day}:{phase}:{target.strategy_id}:{code}{cohort_suffix}"
                )
                if code not in instrument_rows:
                    raise BacktestError(f"target references unknown instrument: {code}")
                if price is None or price <= 0:
                    held = sleeve.quantity(code, target.cohort_id)
                    if self._is_delisting_effective(code, day):
                        side = "sell" if held else "buy"
                        frames["demand_residuals"].append({
                            "demand_id": demand_id, "execution_date": day,
                            "execution_phase": phase,
                            "strategy_id": target.strategy_id,
                            "instrument_id": code, "side": side,
                            "quantity": held or None,
                            "reason": "delisted_illiquid",
                        })
                        continue
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
                current = sleeve.quantity(code, target.cohort_id)
                difference = desired - current
                delist_day = self._delisting_days.get(code)
                if difference > 0 and delist_day is not None and day >= delist_day:
                    frames["demand_residuals"].append({
                        "demand_id": demand_id, "execution_date": day,
                        "execution_phase": phase, "strategy_id": target.strategy_id,
                        "instrument_id": code, "side": "buy",
                        "quantity": math.floor(difference / lot) * lot,
                        "reason": "delisted_illiquid",
                    })
                    continue
                if difference < 0 and is_frozen:
                    frames["demand_residuals"].append({
                        "demand_id": demand_id, "execution_date": day,
                        "execution_phase": phase, "strategy_id": target.strategy_id,
                        "instrument_id": code, "side": "sell",
                        "quantity": -difference,
                        "reason": "delisted_illiquid",
                    })
                    continue
                if difference < 0:
                    requested = -difference
                    quantity = min(
                        requested,
                        sleeve.sellable_quantity(code, day, target.cohort_id),
                    )
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
                            demand_id, phase, desired, target.cohort_id,
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
                            demand_id, phase, desired, target.cohort_id,
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
        required: dict[tuple[str, str, str | None], int] = defaultdict(int)
        for cross in crosses:
            notional = cross.quantity * cross.price
            cash_delta[cross.seller_strategy_id] += notional
            cash_delta[cross.buyer_strategy_id] -= notional
            required[
                (cross.seller_strategy_id, cross.instrument_id,
                 cross.seller_cohort_id)
            ] += cross.quantity
        for (strategy_id, code, cohort_id), quantity in required.items():
            if sleeves[strategy_id].sellable_quantity(
                code, day, cohort_id,
            ) < quantity:
                raise BacktestError("internal crosses exceed a sleeve's sellable position")
        for cross in crosses:
            seller = sleeves[cross.seller_strategy_id]
            removed = seller.remove(
                cross.instrument_id, cross.quantity, day,
                cross.seller_cohort_id,
            )
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
                cross.instrument_id, cross.quantity, day, sellable_date,
                cross.price, cross.buyer_cohort_id,
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
                rule_day, rule_exchange, rule_asset_type = rule_key
                rules = self._market_rules
                current = rules[
                    (rules["exchange"].astype(str) == rule_exchange)
                    & (rules["asset_type"].astype(str) == rule_asset_type)
                    & (rules["effective_from"] <= rule_day)
                    & (
                        rules["effective_to"].isna()
                        | (rules["effective_to"] >= rule_day)
                    )
                ]
                if len(current) != 1:
                    raise DataContractError("market rule is unavailable")
                rule = current.iloc[0]
            except DataContractError:
                rule = None
            self._rule_cache[rule_key] = rule
        fill = self.executor.execute(order, row, asset_type, rule)
        self._validate_fill(order, fill)
        demands = residuals.get((order.instrument_id, order.side), [])
        demand_by_key = {
            (demand.strategy_id, demand.cohort_id, demand.demand_id): demand
            for demand in demands
        }
        if len(demand_by_key) != len(demands):
            raise BacktestError("duplicate same-phase cohort demand")
        quantities = allocate_fill_demands(order, fill.filled_quantity, demands)
        if order.side == "buy":
            # Resolved once per order: the rule and asset type are loop
            # invariants, and `_cost_rule` was previously re-resolved on every
            # lot as the affordability walk stepped down.
            commission_bps, minimum_commission, _, buy_tax_bps, transfer = (
                self._cost_rule(rule, asset_type)
            )
            for demand_key in list(quantities):
                strategy_id = demand_key[0]
                quantity = quantities[demand_key]
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
                quantities[demand_key] = quantity
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
        allocations = cost_demand_allocations(
            order, quantities, fill.price, commission, tax,
            fill.slippage_bps + fill.impact_bps,
        )
        for allocation in allocations:
            sleeve = sleeves[allocation.strategy_id]
            notional = allocation.quantity * allocation.price
            if allocation.side == "sell":
                removed = sleeve.remove(
                    allocation.instrument_id, allocation.quantity,
                    order.execution_date, allocation.cohort_id,
                )
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
                    sellable_date, allocation.price, allocation.cohort_id,
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
        allocated_by_key = {
            (allocation.strategy_id, allocation.cohort_id): allocation.quantity
            for allocation in allocations
        }
        for demand in demands:
            unfilled = demand.quantity - allocated_by_key.get(
                (demand.strategy_id, demand.cohort_id), 0,
            )
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
        if target is not None:
            if target.strategy_id != context.strategy_id:
                raise BacktestError("target strategy_id does not match its binding")
            if target.signal_date != context.signal_date:
                raise BacktestError("target signal_date does not match the decision context")
            if target.execution_date != context.execution_date:
                raise BacktestError("target execution_date does not match the decision context")
            if target.execution_phase != context.execution_phase:
                raise BacktestError("target execution_phase does not match the decision context")
            self._validate_target_weights(target)
        seen: set[tuple[str, int, str]] = set()
        for scheduled in decision.scheduled_targets:
            if not isinstance(scheduled, ScheduledTargetPortfolio):
                raise BacktestError("scheduled target has an invalid type")
            if scheduled.strategy_id != context.strategy_id:
                raise BacktestError(
                    "scheduled target strategy_id does not match its binding"
                )
            if scheduled.signal_date != context.signal_date:
                raise BacktestError(
                    "scheduled target signal_date does not match the decision context"
                )
            if scheduled.session_offset < 1:
                raise BacktestError("scheduled target session_offset must be positive")
            if scheduled.execution_phase not in {"open", "close"}:
                raise BacktestError("scheduled target phase must be open or close")
            if not scheduled.cohort_id:
                raise BacktestError("scheduled target cohort_id must not be empty")
            identity = (
                scheduled.cohort_id, scheduled.session_offset,
                scheduled.execution_phase,
            )
            if identity in seen:
                raise BacktestError("duplicate scheduled target for cohort and phase")
            seen.add(identity)
            self._validate_target_weights(scheduled)

    def _validate_target_weights(self, target):
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

    def _portfolio_view(self, sleeve, prices, day):
        positions = []
        for code in sorted(sleeve.lots):
            quantity = sleeve.quantity(code)
            if not quantity:
                continue
            if code not in prices:
                raise BacktestError(f"cannot value held instrument on {day}: {code}")
            cohort_quantities: dict[str, int] = {}
            cohort_sellable: dict[str, int] = {}
            for lot in sleeve.lots[code]:
                if lot.quantity <= 0 or lot.cohort_id is None:
                    continue
                cohort_quantities[lot.cohort_id] = (
                    cohort_quantities.get(lot.cohort_id, 0) + lot.quantity
                )
                if lot.sellable_date <= day:
                    cohort_sellable[lot.cohort_id] = (
                        cohort_sellable.get(lot.cohort_id, 0) + lot.quantity
                    )
            if code in self._delisted_codes:
                cohort_sellable = {}
            audit = self._valuation_audit(code, day, prices[code])
            positions.append(PositionView(
                code,
                quantity,
                0 if code in self._delisted_codes else sleeve.sellable_quantity(code, day),
                prices[code],
                cohort_quantities,
                cohort_sellable,
                audit["position_status"],
                audit["valuation_source"],
                audit["last_observed_date"],
                audit["stale_sessions"],
            ))
        return PortfolioView(sleeve.nav(prices), sleeve.cash, tuple(positions))

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
                        if sleeve.adjust_shares(code, delta, day) != delta:
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

    def _is_delisting_effective(self, code, day):
        delist_day = self._delisting_days.get(code)
        return delist_day is not None and day >= delist_day

    def _valuation_audit(self, code, day, fallback_price=None):
        mark = self._valuation_marks.get(code)
        if mark is None:
            if fallback_price is None:
                raise BacktestError(
                    f"valuation book has no causal mark for {code} on {day}"
                )
            mark = _ValuationMark(float(fallback_price), day)
        observed_index = bisect_left(self._calendar, mark.observed_date)
        day_index = self._calendar_index.get(day, bisect_left(self._calendar, day))
        stale_sessions = max(0, day_index - observed_index)
        if code in self._delisted_codes:
            status = "delisted_illiquid"
        elif mark.paused and mark.observed_date == day:
            status = "suspended"
        else:
            status = "active"
        return {
            "position_status": status,
            "valuation_source": (
                "market_close" if mark.observed_date == day else "last_observed"
            ),
            "last_observed_date": mark.observed_date,
            "stale_sessions": stale_sessions,
        }

    def _process_delistings(self, day, sleeves, master, prices, frames):
        due_codes = sorted(
            code for code, disposal_day in self._delisting_days.items()
            if disposal_day == day and code not in self._delisted_codes
        )
        if not due_codes:
            return
        policy = self.execution_config.delisting_policy
        recovery = self.execution_config.delisting_recovery_rate
        accounts = {**sleeves, "__master__": master}
        for code in due_codes:
            held_quantity = sum(sleeve.quantity(code) for sleeve in sleeves.values())
            if held_quantity:
                self._delisting_migrations.add(code)
                if code not in prices:
                    raise BacktestError(
                        f"delisted held instrument has no causal mark on {day}: {code}"
                    )
            reference_price = prices.get(code)
            for account_id, account in accounts.items():
                quantity = account.quantity(code)
                if not quantity:
                    continue
                assert reference_price is not None
                carrying_value = quantity * reference_price
                proceeds = (
                    carrying_value * recovery
                    if policy == "cash_settle_last_close" else 0.0
                )
                pnl = (
                    proceeds - carrying_value
                    if policy in {"cash_settle_last_close", "write_off_zero"}
                    else 0.0
                )
                if policy != "carry_last_mark":
                    removed = account.remove_any(code, quantity)
                    if removed != quantity:
                        raise BacktestError("delisting disposal did not remove all lots")
                    account.cash += proceeds
                event_id = f"delisting:{code}:{day}:{account_id}"
                destination = (
                    "master_corporate_actions"
                    if account_id == "__master__" else "corporate_actions"
                )
                frames[destination].append({
                    "date": day,
                    "strategy_id": account_id,
                    "event_id": event_id,
                    "instrument_id": code,
                    "type": "delisting_disposal",
                    "policy": policy,
                    "quantity": quantity,
                    "reference_price": reference_price,
                    "recovery_rate": recovery if policy == "cash_settle_last_close" else None,
                    "amount": proceeds,
                    "pnl": pnl,
                })
                if policy == "cash_settle_last_close":
                    frames["cashflows"].append({
                        "event_id": f"{event_id}:settlement",
                        "date": day,
                        "account_id": account_id,
                        "flow_type": "delisting_cash_settlement",
                        "amount": proceeds,
                        "instrument_id": code,
                        "upstream_event_id": event_id,
                    })
            self._delisted_codes.add(code)
        master.reconcile(sleeves, prices)
        frames["reconciliations"].append({
            "date": day, "phase": "close", "event": "after_delisting",
            "status": "passed",
        })

    def _record_daily(self, day, sleeves, master, prices, frames):
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
                    if code not in prices:
                        raise BacktestError(
                            f"cannot value held instrument on {day}: {code}"
                        )
                    price = prices[code]
                    audit = self._valuation_audit(code, day, price)
                    frames["positions"].append({
                        "date": day, "strategy_id": strategy_id,
                        "instrument_id": code, "quantity": quantity,
                        "last_price": price, "market_value": quantity * price,
                        **audit,
                    })
                for lot in sleeve.lots[code]:
                    if lot.quantity:
                        frames["position_lots"].append({
                            "date": day, "strategy_id": strategy_id,
                            "instrument_id": code,
                            "cohort_id": lot.cohort_id,
                            "quantity": lot.quantity,
                            "acquisition_date": lot.acquisition_date,
                            "sellable_date": lot.sellable_date,
                            "unit_cost": lot.unit_cost,
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
                if code not in prices:
                    raise BacktestError(
                        f"cannot value master held instrument on {day}: {code}"
                    )
                price = prices[code]
                audit = self._valuation_audit(code, day, price)
                frames["master_positions"].append({
                    "date": day, "account_id": "__master__",
                    "instrument_id": code, "quantity": quantity,
                    "last_price": price, "market_value": quantity * price,
                    **audit,
                })
        frames["reconciliations"].append({
            "date": day, "phase": "close", "event": "daily_close",
            "status": "passed",
        })

    def _delisting_metrics(self, positions):
        metrics = {
            "delisting_policy": self.execution_config.delisting_policy,
            "delisting_recovery_rate": self.execution_config.delisting_recovery_rate,
            "delisted_migrated_instruments": len(self._delisting_migrations),
            "terminal_delisted_frozen_positions": 0,
            "terminal_delisted_frozen_market_value": 0.0,
            "maximum_stale_sessions": 0,
        }
        if positions.empty:
            return metrics
        metrics["maximum_stale_sessions"] = int(positions["stale_sessions"].max())
        terminal = positions[positions["date"] == positions["date"].max()]
        frozen = terminal[
            terminal["position_status"] == "delisted_illiquid"
        ]
        metrics["terminal_delisted_frozen_positions"] = int(
            frozen["instrument_id"].nunique()
        )
        metrics["terminal_delisted_frozen_market_value"] = float(
            frozen["market_value"].sum()
        )
        return metrics

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

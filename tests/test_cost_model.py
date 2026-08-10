from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from zyquant.backtest.market import (
    COST_FIELDS, DEFAULT_MARKET_COSTS, resolve_costs,
)
from zyquant.config import CostOverride, ExecutionConfig


def _rule(**values):
    complete = {field: float("nan") for field in COST_FIELDS}
    complete.update(values)
    return SimpleNamespace(**complete)


def test_source_missing_rule_never_yields_a_nan_cost():
    # Hermes writes a placeholder rule row whose every fee column is NaN.
    # Pricing off it used to poison the whole ledger with NaN.
    values, provenance = resolve_costs(ExecutionConfig(), "stock", _rule())

    assert set(values) == set(COST_FIELDS)
    assert all(math.isfinite(value) for value in values.values())
    assert set(provenance.values()) == {"default"}
    assert values == DEFAULT_MARKET_COSTS["stock"]


def test_absent_rule_resolves_the_same_way_as_an_empty_one():
    without_rule, _ = resolve_costs(ExecutionConfig(), "stock", None)
    with_empty_rule, _ = resolve_costs(ExecutionConfig(), "stock", _rule())

    assert without_rule == with_empty_rule


def test_real_rule_values_win_over_defaults():
    values, provenance = resolve_costs(
        ExecutionConfig(), "stock",
        _rule(commission_bps=3.0, minimum_commission=5.0, sell_tax_bps=10.0,
              buy_tax_bps=0.0, transfer_fee_bps=0.2),
    )

    assert values["commission_bps"] == 3.0
    assert values["sell_tax_bps"] == 10.0
    assert set(provenance.values()) == {"market_rules"}


def test_partially_null_rule_falls_back_field_by_field():
    values, provenance = resolve_costs(
        ExecutionConfig(), "stock", _rule(commission_bps=3.0),
    )

    assert values["commission_bps"] == 3.0
    assert provenance["commission_bps"] == "market_rules"
    # transfer_fee_bps has no scalar override, so it used to be NaN here.
    assert values["transfer_fee_bps"] == (
        DEFAULT_MARKET_COSTS["stock"]["transfer_fee_bps"]
    )
    assert provenance["transfer_fee_bps"] == "default"


def test_asset_type_override_reaches_the_fields_no_scalar_covers():
    config = ExecutionConfig(cost_overrides={
        "stock": CostOverride(buy_tax_bps=1.0, transfer_fee_bps=0.5),
    })
    values, provenance = resolve_costs(config, "stock", _rule())

    assert values["buy_tax_bps"] == 1.0
    assert values["transfer_fee_bps"] == 0.5
    assert provenance["buy_tax_bps"] == "config_override"
    assert provenance["transfer_fee_bps"] == "config_override"


def test_scalar_override_outranks_both_rule_and_asset_type_override():
    config = ExecutionConfig(
        commission_bps=1.0,
        cost_overrides={"stock": CostOverride(commission_bps=7.0)},
    )
    values, provenance = resolve_costs(
        config, "stock", _rule(commission_bps=3.0),
    )

    assert values["commission_bps"] == 1.0
    assert provenance["commission_bps"] == "config_scalar"


def test_stock_only_scalar_does_not_leak_into_etf_costs():
    config = ExecutionConfig(stock_sell_tax_bps=5.0)
    stock, _ = resolve_costs(config, "stock", _rule())
    etf, provenance = resolve_costs(config, "etf", _rule())

    assert stock["sell_tax_bps"] == 5.0
    assert etf["sell_tax_bps"] == DEFAULT_MARKET_COSTS["etf"]["sell_tax_bps"]
    assert provenance["sell_tax_bps"] == "default"


def test_etf_defaults_exempt_stamp_duty_and_transfer_fee():
    values, _ = resolve_costs(ExecutionConfig(), "etf", None)

    assert values["sell_tax_bps"] == 0.0
    assert values["transfer_fee_bps"] == 0.0


def test_unknown_asset_type_falls_back_to_stock_defaults():
    values, _ = resolve_costs(ExecutionConfig(), "warrant", None)

    assert values == DEFAULT_MARKET_COSTS["stock"]


def test_negative_override_is_rejected_by_configuration():
    with pytest.raises(ValueError):
        ExecutionConfig(cost_overrides={
            "stock": CostOverride(transfer_fee_bps=-1.0),
        })


def test_backtest_runs_on_a_snapshot_whose_fee_series_is_missing():
    """The Hermes case end to end: placeholder rules, real trades, real costs."""
    import tempfile

    from zyquant.backtest import BacktestEngine, StrategyBinding
    from zyquant.data import SnapshotPublisher
    from tests.support import canonical_tables, signal_frame
    from tests.test_golden_market import strategy_for

    with tempfile.TemporaryDirectory() as temporary:
        tables, days = canonical_tables()
        rules = tables["market_rules"]
        for field in COST_FIELDS:
            rules[field] = float("nan")
        rules["source"] = "source_missing"
        snapshot = SnapshotPublisher(temporary).publish("no-fees-v1", tables)

        result = BacktestEngine(ExecutionConfig(
            max_participation=1, slippage_bps=0, impact_coefficient_bps=0,
        )).run(
            snapshot, days[0], days[-1],
            [StrategyBinding(strategy_for(signal_frame(days)), 1.0)],
            100_000,
        )

        fills = result.frames["fills"]
        traded = fills[fills["filled_quantity"] > 0]
        assert not traded.empty
        assert traded["commission"].notna().all()
        assert traded["tax"].notna().all()
        # The minimum commission is the default, not a NaN-poisoned zero.
        assert (traded["commission"] >= 5.0).all()
        assert result.frames["nav"]["nav"].notna().all()
        assert result.cost_model["provenance"]["stock"] == {
            field: "default" for field in COST_FIELDS
        }
        assert result.cost_model["rates"]["stock"] == (
            DEFAULT_MARKET_COSTS["stock"]
        )

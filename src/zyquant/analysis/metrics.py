from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def performance_metrics(
    nav: pd.DataFrame,
    fills: pd.DataFrame,
    initial_cash: float,
    positions: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    if nav.empty:
        return {}
    daily = nav.groupby("date", as_index=False).agg(
        nav=("nav", "sum"), cash=("cash", "sum")
    ).sort_values("date")
    returns = daily["nav"].pct_change().dropna()
    total = float(daily["nav"].iloc[-1] / initial_cash - 1)
    periods = max(1, len(returns))
    annual = float((1 + total) ** (252 / periods) - 1) if total > -1 else -1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    excess_daily = returns - risk_free_rate / 252
    sharpe = (
        float(excess_daily.mean() / returns.std(ddof=1) * np.sqrt(252))
        if len(returns) > 1 and returns.std(ddof=1) > 0 else None
    )
    downside_values = np.minimum(returns.to_numpy(dtype=float), 0.0)
    downside = float(np.sqrt(np.mean(downside_values**2))) if len(returns) else 0.0
    sortino = (
        float(excess_daily.mean() / downside * np.sqrt(252))
        if downside > 0 else None
    )
    drawdown = daily["nav"] / daily["nav"].cummax() - 1
    maximum_drawdown = float(drawdown.min())
    calmar = annual / abs(maximum_drawdown) if maximum_drawdown < 0 else None
    drawdown_duration = _maximum_duration(drawdown < 0)
    recovery_days = _recovery_days(drawdown)

    commission = float(fills["commission"].sum()) if not fills.empty else 0.0
    tax = float(fills["tax"].sum()) if not fills.empty else 0.0
    traded = (
        float((fills["filled_quantity"] * fills["price"]).sum())
        if not fills.empty else 0.0
    )
    requested = (
        float(fills["requested_quantity"].sum()) if not fills.empty else 0.0
    )
    filled_quantity = (
        float(fills["filled_quantity"].sum()) if not fills.empty else 0.0
    )
    metrics: dict[str, Any] = {
        "start_date": daily["date"].iloc[0],
        "end_date": daily["date"].iloc[-1],
        "initial_cash": initial_cash,
        "final_nav": float(daily["nav"].iloc[-1]),
        "total_return": total,
        "annualized_return": annual,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": maximum_drawdown,
        "max_drawdown_duration_days": drawdown_duration,
        "recovery_days": recovery_days,
        "commission": commission,
        "tax": tax,
        "turnover": traded / float(daily["nav"].mean()) / 2 if len(daily) else 0.0,
        "fills": int((fills["filled_quantity"] > 0).sum()) if not fills.empty else 0,
        "rejections": int((fills["filled_quantity"] == 0).sum()) if not fills.empty else 0,
        "fill_rate": filled_quantity / requested if requested else 0.0,
        "average_cash_weight": float((daily["cash"] / daily["nav"]).mean()),
    }
    if positions is not None and not positions.empty:
        holdings = positions.groupby("date")["instrument_id"].nunique()
        concentration = positions.assign(
            weight=positions["market_value"]
            / positions.groupby("date")["market_value"].transform("sum")
        ).groupby("date")["weight"].max()
        metrics["average_holdings"] = float(holdings.mean())
        metrics["maximum_security_concentration"] = float(concentration.max())
    if benchmark is not None and not benchmark.empty:
        benchmark_returns = benchmark.set_index("date")["return"]
        aligned = pd.DataFrame(
            {"portfolio": returns.to_numpy()},
            index=daily["date"].iloc[1:],
        ).join(benchmark_returns.rename("benchmark"), how="inner")
        if not aligned.empty:
            active = aligned["portfolio"] - aligned["benchmark"]
            tracking = float(active.std(ddof=1) * np.sqrt(252)) if len(active) > 1 else 0.0
            metrics.update({
                "benchmark_return": float(
                    (1 + aligned["benchmark"]).prod() - 1
                ),
                "excess_return": float((1 + active).prod() - 1),
                "tracking_error": tracking,
                "information_ratio": (
                    float(active.mean() / active.std(ddof=1) * np.sqrt(252))
                    if len(active) > 1 and active.std(ddof=1) > 0 else None
                ),
            })
    return metrics


def _maximum_duration(mask: pd.Series) -> int:
    maximum = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        maximum = max(maximum, current)
    return maximum


def _recovery_days(drawdown: pd.Series) -> int | None:
    if drawdown.empty:
        return None
    trough = int(drawdown.to_numpy().argmin())
    if drawdown.iloc[trough] >= 0:
        return 0
    for index in range(trough + 1, len(drawdown)):
        if drawdown.iloc[index] >= -1e-12:
            return index - trough
    return None

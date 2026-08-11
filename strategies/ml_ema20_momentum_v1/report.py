from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from zyquant.core.plugins import PluginMetadata


def _table(frame: pd.DataFrame, *, tail: int = 200) -> str:
    if frame.empty:
        return "<p>无数据</p>"
    return frame.tail(tail).to_html(index=False, border=0, escape=True)


def _daily_decisions(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if states.empty or "state_json" not in states:
        return pd.DataFrame()
    for record in states.itertuples(index=False):
        payload = json.loads(str(record.state_json))
        diagnostic = dict(payload.get("last_decision", {}))
        if diagnostic:
            rows.append(diagnostic)
    return pd.DataFrame(rows)


def _open_cohorts(
    states: pd.DataFrame,
    positions: pd.DataFrame,
    fills: pd.DataFrame,
    allocations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()
    payload = json.loads(str(states.iloc[-1]["state_json"]))
    cohorts = dict(payload.get("cohorts", {}))
    if not cohorts:
        return pd.DataFrame()
    last_date = pd.to_datetime(states.iloc[-1]["date"]).date()
    latest_positions = positions
    if not positions.empty and "date" in positions:
        latest_positions = positions[positions["date"] == positions["date"].max()]
    position_map = {
        str(row.instrument_id): {
            "quantity": int(getattr(row, "quantity", 0)),
            "price": float(getattr(row, "last_price", 0.0)),
        }
        for row in latest_positions.itertuples(index=False)
    }
    position_date = (
        pd.to_datetime(latest_positions["date"].max()).date()
        if not latest_positions.empty and "date" in latest_positions else None
    )
    later_sells: dict[tuple[str, str], int] = {}
    if allocations is not None and not allocations.empty and position_date is not None:
        working = allocations.copy()
        working["execution_date"] = pd.to_datetime(
            working["execution_date"]
        ).dt.date
        working = working[
            (working["side"] == "sell")
            & (working["execution_date"] > position_date)
        ]
        later_sells = {
            (str(cohort_id), str(code)): int(group["quantity"].sum())
            for (cohort_id, code), group in working.groupby(
                ["cohort_id", "instrument_id"], dropna=False,
            )
        }
    failure_map: dict[str, str] = {}
    if not fills.empty:
        failed = fills[
            (fills["side"] == "sell") & fills["reject_reason"].notna()
        ].sort_values("execution_date", kind="mergesort")
        failure_map = {
            str(row.instrument_id): str(row.reject_reason)
            for row in failed.itertuples(index=False)
        }
    rows = []
    for cohort_id, raw in sorted(cohorts.items()):
        signal_date = pd.Timestamp(raw["signal_date"]).date()
        values = {}
        for code in map(str, raw.get("symbols", [])):
            position = position_map.get(code, {"quantity": 0, "price": 0.0})
            quantity = max(
                0, int(position["quantity"])
                - later_sells.get((str(cohort_id), code), 0),
            )
            if quantity:
                values[code] = quantity * float(position["price"])
        symbols = sorted(values)
        if not symbols:
            continue
        rows.append({
            "cohort_id": cohort_id,
            "signal_date": signal_date,
            "status": raw.get("status"),
            "holding_days": (last_date - signal_date).days,
            "symbols": ",".join(symbols),
            "market_value": sum(values.values()),
            "retry_count": int(raw.get("retry_count", 0)),
            "last_failure_reason": ",".join(sorted({
                failure_map[code] for code in symbols if code in failure_map
            })),
        })
    return pd.DataFrame(rows)


def _yearly_performance(nav: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if nav.empty or not {"date", "nav"} <= set(nav):
        return pd.DataFrame(), {}
    working = nav[["date", "nav"]].copy().sort_values("date", kind="mergesort")
    working["date"] = pd.to_datetime(working["date"])
    working["return"] = working["nav"].pct_change().fillna(0.0)
    rows = []
    for year, group in working.groupby(working["date"].dt.year, sort=True):
        returns = group["return"]
        curve = (1.0 + returns).cumprod()
        drawdown = curve / curve.cummax() - 1.0
        volatility = float(returns.std(ddof=0))
        rows.append({
            "year": int(year),
            "sessions": len(group),
            "return": float(curve.iloc[-1] - 1.0),
            "sharpe": (
                float(returns.mean() / volatility * math.sqrt(252.0))
                if volatility > 0 else 0.0
            ),
            "max_drawdown": float(drawdown.min()),
        })
    all_returns = working["return"]
    trimmed = {}
    for count in (10, 20):
        keep = all_returns.drop(index=all_returns.nlargest(min(count, len(all_returns))).index)
        trimmed[f"return_without_best_{count}_days"] = float((1.0 + keep).prod() - 1.0)
    return pd.DataFrame(rows), trimmed


class MomentumReport:
    def write(
        self,
        path: str | Path,
        metrics: Mapping[str, Any],
        frames: Mapping[str, pd.DataFrame],
        manifest: Mapping[str, Any],
    ) -> Path:
        destination = Path(path)
        states = frames.get("strategy_states", pd.DataFrame())
        decisions = _daily_decisions(states)
        fills = frames.get("fills", pd.DataFrame())
        positions = frames.get("positions", pd.DataFrame())
        nav = frames.get("nav", pd.DataFrame())
        allocations = frames.get("fill_allocations", pd.DataFrame())
        open_cohorts = _open_cohorts(states, positions, fills, allocations)
        yearly_performance, trimmed_performance = _yearly_performance(nav)

        fill_summary = pd.DataFrame()
        if not fills.empty:
            working = fills.copy()
            working["reason"] = working["reject_reason"].fillna("filled")
            fill_summary = (
                working.groupby(
                    ["execution_phase", "side", "status", "reason"],
                    dropna=False,
                ).size().rename("orders").reset_index()
            )
        residuals = frames.get("demand_residuals", pd.DataFrame())
        residual_summary = pd.DataFrame()
        if not residuals.empty:
            residual_summary = (
                residuals.groupby(
                    ["execution_phase", "side", "reason"], dropna=False,
                )["quantity"].agg(["count", "sum"]).reset_index()
            )
        utilization = pd.DataFrame()
        if not nav.empty and {"date", "nav", "cash"} <= set(nav):
            utilization = nav[["date", "nav", "cash"]].copy()
            utilization["capital_utilization"] = (
                1.0 - utilization["cash"] / utilization["nav"]
            )

        execution_metrics: dict[str, Any] = {}
        slippage_cost = (
            float(allocations["slippage_cost"].sum())
            if not allocations.empty else 0.0
        )
        if not fills.empty:
            open_buys = fills[
                (fills["execution_phase"] == "open") & (fills["side"] == "buy")
            ]
            execution_metrics.update({
                "open_buy_orders": len(open_buys),
                "open_rejection_rate": (
                    float((open_buys["filled_quantity"] == 0).mean())
                    if len(open_buys) else 0.0
                ),
                "limit_up_deferrals": int(
                    (fills["reject_reason"] == "strategy_hold_limit_up").sum()
                ),
                "fees": float(fills["commission"].sum() + fills["tax"].sum()),
                "slippage_cost": slippage_cost,
                "actual_open_entries": int((open_buys["filled_quantity"] > 0).sum()),
            })
        if {"initial_cash", "final_nav"} <= set(metrics):
            net_pnl = float(metrics["final_nav"]) - float(metrics["initial_cash"])
            costs = float(execution_metrics.get("fees", 0.0)) + slippage_cost
            execution_metrics.update({
                "net_pnl": net_pnl,
                "gross_pnl_before_execution_costs": net_pnl + costs,
                "gross_return_before_execution_costs": (
                    (net_pnl + costs) / float(metrics["initial_cash"])
                ),
            })
        execution_metrics.update(trimmed_performance)
        targets = frames.get("target_events", pd.DataFrame())
        retry_distribution = pd.DataFrame()
        if not targets.empty:
            decoded = targets["diagnostics"].apply(json.loads)
            initial = targets[decoded.apply(lambda item: item.get("kind") == "initial_exit")]
            retries = targets[decoded.apply(lambda item: item.get("kind") == "retry_exit")]
            retry_ids = set(retries["cohort_id"].dropna().astype(str))
            normal = ~initial["cohort_id"].astype(str).isin(retry_ids)
            execution_metrics.update({
                "initial_exit_cohorts": len(initial),
                "normal_t_plus_2_exit_rate": float(normal.mean()) if len(initial) else 0.0,
                "retry_close_targets": len(retries),
            })
            if not retries.empty:
                retry_distribution = (
                    retries.groupby("cohort_id").size().rename("delay_sessions")
                    .value_counts().sort_index().rename("cohorts").reset_index()
                )
        position_exposure = pd.DataFrame()
        if not positions.empty:
            position_exposure = positions.copy()
            totals = position_exposure.groupby("date")["market_value"].transform("sum")
            position_exposure["invested_weight"] = (
                position_exposure["market_value"] / totals.where(totals != 0)
            )
        execution_rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in execution_metrics.items()
        )

        metric_rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in metrics.items()
        )
        destination.write_text(
            "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<title>ml_ema20_momentum_v1</title><style>"
            "body{font-family:system-ui;margin:2rem;max-width:1400px}"
            "table{border-collapse:collapse;width:100%;font-size:13px}"
            "th,td{padding:.35rem;border-bottom:1px solid #ddd;text-align:right}"
            "th:first-child,td:first-child{text-align:left}</style></head><body>"
            "<h1>ml_ema20_momentum_v1 回测报告</h1>"
            f"<p>运行 ID：{html.escape(str(manifest.get('run_id', '')))}</p>"
            f"<h2>绩效指标</h2><table>{metric_rows}</table>"
            f"<h2>策略执行指标</h2><table>{execution_rows}</table>"
            f"<h2>2015–2026逐年绩效</h2>{_table(yearly_performance)}"
            f"<h2>日度 EMA20 池与预测覆盖</h2>{_table(decisions)}"
            f"<h2>执行与延期原因</h2>{_table(fill_summary)}"
            f"<h2>未成交与部分成交原因</h2>{_table(residual_summary)}"
            f"<h2>延期交易日分布</h2>{_table(retry_distribution)}"
            f"<h2>资金利用率</h2>{_table(utilization)}"
            f"<h2>单股持仓暴露</h2>{_table(position_exposure)}"
            f"<h2>期末未关闭 cohort</h2>{_table(open_cohorts)}"
            "</body></html>",
            encoding="utf-8",
        )
        return destination


def create_report(**parameters: Any) -> MomentumReport:
    del parameters
    return MomentumReport()


create_report.plugin_metadata = PluginMetadata(  # type: ignore[attr-defined]
    name="ml_ema20_momentum_report_v1",
    version="1.0.0",
    kind="reports",
    minimum_framework_version="2.0.0",
    deterministic=True,
)

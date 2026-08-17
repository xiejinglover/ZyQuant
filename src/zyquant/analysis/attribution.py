from __future__ import annotations

import pandas as pd

from zyquant.core.exceptions import AccountingError


def attribution_report(
    nav: pd.DataFrame,
    allocations: pd.DataFrame,
    positions: pd.DataFrame | None = None,
    corporate_actions: pd.DataFrame | None = None,
    industry_membership: pd.DataFrame | None = None,
    tolerance: float = 1e-7,
) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame(columns=["date", "dimension", "component", "pnl"])
    daily = nav.groupby("date", as_index=False)["nav"].sum().sort_values("date")
    daily["account_pnl"] = daily["nav"].diff()
    security = _security_attribution(positions)
    security_by_day: dict[object, float] = {}
    for row in security:
        security_by_day[row["date"]] = security_by_day.get(row["date"], 0.0) + row["pnl"]
    action_by_day: dict[object, float] = {}
    if corporate_actions is not None and not corporate_actions.empty:
        receivables = corporate_actions[
            corporate_actions["type"] == "dividend_receivable"
        ]
        for item in receivables.itertuples(index=False):
            action_by_day[item.date] = action_by_day.get(item.date, 0.0) + float(item.amount)
        if "pnl" in corporate_actions.columns:
            delisting = corporate_actions[
                corporate_actions["type"] == "delisting_disposal"
            ].dropna(subset=["pnl"])
            for item in delisting.itertuples(index=False):
                action_by_day[item.date] = (
                    action_by_day.get(item.date, 0.0) + float(item.pnl)
                )

    rows: list[dict] = []
    for record in daily.dropna(subset=["account_pnl"]).itertuples(index=False):
        day_allocations = (
            allocations[allocations["execution_date"] == record.date]
            if not allocations.empty else allocations
        )
        commission_tax = float(
            day_allocations[["commission", "tax"]].sum().sum()
        ) if not day_allocations.empty else 0.0
        slippage = float(
            day_allocations["slippage_cost"].sum()
        ) if not day_allocations.empty else 0.0
        price_pnl = security_by_day.get(record.date, 0.0)
        action_pnl = action_by_day.get(record.date, 0.0)
        account_pnl = float(record.account_pnl)
        residual = (
            account_pnl - price_pnl - action_pnl
            + commission_tax + slippage
        )
        components = {
            "security_price": price_pnl,
            "corporate_action": action_pnl,
            "cash": 0.0,
            "commission_tax": -commission_tax,
            "slippage_impact": -slippage,
            "execution_residual": residual,
        }
        error = sum(components.values()) - account_pnl
        if abs(error) > tolerance:
            raise AccountingError(
                f"attribution does not reconcile on {record.date}: {error}"
            )
        rows.extend(
            {
                "date": record.date, "dimension": "pnl_component",
                "component": name, "pnl": value,
            }
            for name, value in components.items()
        )
        # Compatibility bridge retained as a report dimension, not as a second P&L.
        actual_cost = -(commission_tax + slippage)
        rows.extend([
            {
                "date": record.date, "dimension": "pnl_bridge",
                "component": "gross_pnl", "pnl": account_pnl - actual_cost,
            },
            {
                "date": record.date, "dimension": "pnl_bridge",
                "component": "actual_cost", "pnl": actual_cost,
            },
            {
                "date": record.date, "dimension": "pnl_bridge",
                "component": "account_pnl", "pnl": account_pnl,
            },
            {
                "date": record.date, "dimension": "reconciliation",
                "component": "conservation_error", "pnl": error,
            },
        ])

    for strategy_id, group in nav.sort_values("date").groupby("strategy_id"):
        values = group[["date", "nav"]].copy()
        values["pnl"] = values["nav"].diff()
        for item in values.dropna(subset=["pnl"]).itertuples(index=False):
            rows.append({
                "date": item.date, "dimension": "strategy",
                "component": f"strategy:{strategy_id}", "pnl": float(item.pnl),
            })
    strategy_daily = nav.sort_values("date").copy()
    strategy_daily["strategy_pnl"] = strategy_daily.groupby("strategy_id")["nav"].diff()
    summed = strategy_daily.groupby("date")["strategy_pnl"].sum(min_count=1)
    account = daily.set_index("date")["account_pnl"]
    common = summed.dropna().index.intersection(account.dropna().index)
    difference = (summed.loc[common] - account.loc[common]).abs()
    if not difference.empty and difference.max() > tolerance:
        day = difference.idxmax()
        raise AccountingError(
            f"strategy P&L does not reconcile to master P&L on {day}"
        )
    rows.extend(security)
    if industry_membership is not None and not industry_membership.empty:
        membership = industry_membership.copy()
        for row in security:
            code = row["component"].split(":", 1)[1]
            current = membership[
                (membership["instrument_id"].astype(str) == code)
                & (membership["effective_from"] <= row["date"])
                & (
                    membership["effective_to"].isna()
                    | (membership["effective_to"] >= row["date"])
                )
                & (membership["known_at"] <= row["date"])
            ]
            industry = (
                str(current.sort_values("effective_from").iloc[-1]["industry_id"])
                if not current.empty else "UNKNOWN"
            )
            rows.append({
                "date": row["date"], "dimension": "industry",
                "component": f"industry:{industry}", "pnl": row["pnl"],
            })
    return pd.DataFrame(rows)


def _security_attribution(positions: pd.DataFrame | None) -> list[dict]:
    if positions is None or positions.empty:
        return []
    aggregated = positions.groupby(
        ["date", "instrument_id"], as_index=False
    ).agg(quantity=("quantity", "sum"), last_price=("last_price", "first"))
    dates = sorted(aggregated["date"].unique())
    codes = sorted(aggregated["instrument_id"].astype(str).unique())
    quantity = aggregated.pivot(
        index="date", columns="instrument_id", values="quantity"
    ).reindex(index=dates, columns=codes).fillna(0)
    prices = aggregated.pivot(
        index="date", columns="instrument_id", values="last_price"
    ).reindex(index=dates, columns=codes).ffill()
    contribution = quantity.shift(1).fillna(0) * prices.diff().fillna(0)
    rows = []
    for day in dates[1:]:
        for code in codes:
            value = float(contribution.loc[day, code])
            if value:
                rows.append({
                    "date": day, "dimension": "security",
                    "component": f"security:{code}", "pnl": value,
                })
    return rows

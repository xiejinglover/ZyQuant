from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from zyquant.core.exceptions import DataContractError

ADJUSTMENT_VERSION = "1.2"
# JQData publishes daily factors rounded to six decimal places. Comparing a
# normalized ratio of two such values needs a slightly wider tolerance than
# machine precision, while remaining far below a market price tick.
VENDOR_FACTOR_RTOL = 5e-7


@dataclass(frozen=True)
class AdjustmentDiagnostics:
    instruments: int
    rows: int
    factor_source: str
    event_adjustments: int
    vendor_factors: Mapping[str, object]


@dataclass(frozen=True)
class AdjustedBarsResult:
    daily_post_adjusted: pd.DataFrame
    diagnostics: AdjustmentDiagnostics
    algorithm_version: str = ADJUSTMENT_VERSION


class AdjustmentProcessor:
    """Materialize post-adjusted bars exactly once during snapshot publication."""

    def build(
        self,
        raw_bars: pd.DataFrame,
        corporate_actions: pd.DataFrame,
        vendor_factors: pd.DataFrame | None = None,
        vendor_factor_mode: str = "use",
        vendor_factor_rtol: float = VENDOR_FACTOR_RTOL,
    ) -> AdjustedBarsResult:
        if vendor_factor_mode not in {"off", "validate", "use"}:
            raise ValueError(
                "vendor_factor_mode must be 'off', 'validate', or 'use'"
            )
        if (
            not np.isfinite(vendor_factor_rtol)
            or vendor_factor_rtol < 0
        ):
            raise ValueError("vendor_factor_rtol must be finite and non-negative")
        raw = raw_bars.sort_values(
            ["instrument_id", "trade_date"], ignore_index=True
        ).copy()
        factors, source, event_count, vendor_diagnostics = self._factors(
            raw,
            corporate_actions,
            vendor_factors,
            vendor_factor_mode,
            vendor_factor_rtol,
        )
        result = raw[["trade_date", "instrument_id"]].copy()
        result["adjustment_factor"] = factors.to_numpy(dtype=float)
        for raw_name, post_name in (
            ("open", "open_post"), ("high", "high_post"),
            ("low", "low_post"), ("close", "close_post"),
        ):
            result[post_name] = (
                pd.to_numeric(raw[raw_name], errors="coerce").to_numpy(dtype=float)
                * result["adjustment_factor"].to_numpy(dtype=float)
            )
        previous = result.groupby("instrument_id", sort=False)["close_post"].shift(1)
        first_pre = (
            pd.to_numeric(raw["pre_close"], errors="coerce").to_numpy(dtype=float)
            * result["adjustment_factor"].to_numpy(dtype=float)
        )
        result["pre_close_post"] = previous.where(previous.notna(), first_pre)
        result["factor_source"] = source
        result["adjustment_version"] = ADJUSTMENT_VERSION
        columns = [
            "trade_date", "instrument_id", "open_post", "high_post", "low_post",
            "close_post", "pre_close_post", "adjustment_factor", "factor_source",
            "adjustment_version",
        ]
        result = result[columns].sort_values(
            ["trade_date", "instrument_id"], ignore_index=True
        )
        return AdjustedBarsResult(
            result,
            AdjustmentDiagnostics(
                instruments=int(raw["instrument_id"].nunique()),
                rows=len(raw),
                factor_source=source,
                event_adjustments=event_count,
                vendor_factors=vendor_diagnostics,
            ),
        )

    def _factors(
        self,
        raw: pd.DataFrame,
        actions: pd.DataFrame,
        vendor: pd.DataFrame | None,
        mode: str,
        rtol: float,
    ) -> tuple[pd.Series, str, int, Mapping[str, object]]:
        expected, event_count = self._event_factors(raw, actions)
        if mode == "off":
            return expected, "corporate_action", event_count, {
                "mode": mode,
                "status": "not_checked",
                "rtol": rtol,
                "rows": 0,
                "mismatches": 0,
                "mismatch_rate": 0.0,
            }
        if vendor is not None:
            required = {"trade_date", "instrument_id", "adjustment_factor"}
            if required - set(vendor.columns):
                raise DataContractError("vendor factors missing canonical columns")
            right = vendor[list(required)].copy()
            right["trade_date"] = pd.to_datetime(right["trade_date"]).dt.date
            merged = raw[["trade_date", "instrument_id"]].merge(
                right, on=["trade_date", "instrument_id"], how="left", validate="one_to_one"
            )
            values = pd.to_numeric(merged["adjustment_factor"], errors="coerce")
            if values.isna().any() or (values <= 0).any() or not np.isfinite(values).all():
                raise DataContractError("vendor adjustment factors must be finite and positive")
            normalized = values / values.groupby(raw["instrument_id"].to_numpy()).transform("first")
            vendor_values = normalized.to_numpy(dtype=float)
            expected_values = expected.to_numpy(dtype=float)
            mismatched = ~np.isclose(
                vendor_values, expected_values, rtol=rtol, atol=1e-10,
            )
            relative = np.abs(vendor_values / expected_values - 1.0)
            mismatch_count = int(mismatched.sum())
            diagnostics: Mapping[str, object] = {
                "mode": mode,
                "status": (
                    "deviation_observed" if mismatch_count else "within_tolerance"
                ),
                "rtol": rtol,
                "rows": len(relative),
                "mismatches": mismatch_count,
                "mismatch_rate": mismatch_count / len(relative) if len(relative) else 0.0,
                "relative_deviation": {
                    "median": float(np.quantile(relative, 0.50)),
                    "p95": float(np.quantile(relative, 0.95)),
                    "p99": float(np.quantile(relative, 0.99)),
                    "max": float(relative.max()),
                },
            }
            if mode == "use" and mismatch_count:
                mismatch = raw.loc[
                    mismatched,
                    ["trade_date", "instrument_id"],
                ].head(5)
                raise DataContractError(
                    "vendor factors conflict with canonical corporate actions: "
                    f"{mismatch.to_dict('records')}"
                )
            if mode == "use":
                return (
                    normalized.reset_index(drop=True), "vendor", event_count,
                    diagnostics,
                )
            return expected, "corporate_action", event_count, diagnostics

        return expected, "corporate_action", event_count, {
            "mode": mode,
            "status": "not_available",
            "rtol": rtol,
            "rows": 0,
            "mismatches": 0,
            "mismatch_rate": 0.0,
        }

    def _event_factors(
        self,
        raw: pd.DataFrame,
        actions: pd.DataFrame,
    ) -> tuple[pd.Series, int]:
        active = actions.copy()
        if not active.empty:
            active = active[~active["status"].astype(str).str.lower().isin({"cancelled", "canceled", "取消"})]
        by_key = {
            (str(code), day): group
            for (code, day), group in active.groupby(["instrument_id", "ex_date"], dropna=True)
        }
        output = np.ones(len(raw), dtype=float)
        event_count = 0
        for _, indexes in raw.groupby("instrument_id", sort=False).groups.items():
            ordered = list(indexes)
            factor = 1.0
            previous_close: float | None = None
            code = str(raw.loc[ordered[0], "instrument_id"])
            for index in ordered:
                day = raw.loc[index, "trade_date"]
                events = by_key.get((code, day))
                if events is not None and previous_close is not None:
                    cash = 0.0
                    share_multiplier = 1.0
                    rights_value = 0.0
                    for event in events.itertuples(index=False):
                        event_type = str(event.event_type).lower()
                        cash_value = getattr(event, "cash_per_share", 0.0)
                        ratio_value = getattr(event, "share_ratio", 0.0)
                        cash_value = 0.0 if pd.isna(cash_value) else float(cash_value)
                        ratio_value = 0.0 if pd.isna(ratio_value) else float(ratio_value)
                        if event_type in {"cash_dividend", "dividend", "分红"}:
                            cash += cash_value
                        elif event_type in {"bonus", "送股", "转增"}:
                            share_multiplier *= 1.0 + ratio_value
                        elif event_type in {"split", "merge", "拆分", "合并"}:
                            if ratio_value <= 0:
                                raise DataContractError(
                                    f"invalid split/merge ratio for {code} on {day}"
                                )
                            share_multiplier *= ratio_value
                        elif event_type == "rights_issue":
                            price_value = getattr(
                                event, "subscription_price", np.nan
                            )
                            if (
                                ratio_value <= 0
                                or pd.isna(price_value)
                                or float(price_value) < 0
                            ):
                                raise DataContractError(
                                    f"invalid rights issue for {code} on {day}"
                                )
                            rights_value += ratio_value * float(price_value)
                            share_multiplier += ratio_value
                        else:
                            raise DataContractError(
                                f"unsupported corporate action {event.event_type!r}"
                            )
                    theoretical = (
                        previous_close - cash + rights_value
                    ) / share_multiplier
                    if theoretical <= 0 or not np.isfinite(theoretical):
                        raise DataContractError(
                            f"invalid theoretical ex price for {code} on {day}"
                        )
                    market_reference = float(raw.loc[index, "pre_close"])
                    # Chinese cash-equity ex-rights reference prices are rounded
                    # to the exchange price tick. Preserve the market-published
                    # reference when it reconciles to the exact action formula
                    # within one cent; materially different values still fail
                    # the vendor-factor cross-check below.
                    if (
                        np.isfinite(market_reference)
                        and market_reference > 0
                        and abs(market_reference - theoretical) <= 0.0100001
                    ):
                        theoretical = market_reference
                    factor *= previous_close / theoretical
                    event_count += len(events)
                output[index] = factor
                previous_close = float(raw.loc[index, "close"])
        return pd.Series(output, index=raw.index).reset_index(drop=True), event_count

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from .models import EvidenceRecipe


@dataclass(frozen=True)
class EvidenceResult:
    factor_health: pd.DataFrame
    daily_metrics: pd.DataFrame
    factor_summary: pd.DataFrame
    period_stability: pd.DataFrame
    exposure_diagnostics: pd.DataFrame
    failure_records: pd.DataFrame
    redundancy_pairs: pd.DataFrame


def _hac_mean(values: pd.Series, lag: int, confidence: float) -> dict[str, float]:
    clean = values.dropna().astype(float).to_numpy()
    count = len(clean)
    if count < 2 or lag >= count:
        return {"mean": float(np.mean(clean)) if count else np.nan, "se": np.nan,
                "p_value": np.nan, "lower": np.nan, "upper": np.nan}
    centered = clean - clean.mean()
    long_variance = float(np.dot(centered, centered) / count)
    for current in range(1, lag + 1):
        covariance = float(np.dot(centered[current:], centered[:-current]) / count)
        long_variance += 2 * (1 - current / (lag + 1)) * covariance
    if long_variance <= 0:
        return {"mean": float(clean.mean()), "se": np.nan, "p_value": np.nan,
                "lower": np.nan, "upper": np.nan}
    standard_error = math.sqrt(long_variance / count)
    if standard_error == 0:
        return {"mean": float(clean.mean()), "se": np.nan, "p_value": np.nan,
                "lower": np.nan, "upper": np.nan}
    z_score = float(clean.mean() / standard_error)
    critical = float(norm.ppf(0.5 + confidence / 2))
    return {
        "mean": float(clean.mean()),
        "se": standard_error,
        "p_value": float(2 * norm.sf(abs(z_score))),
        "lower": float(clean.mean() - critical * standard_error),
        "upper": float(clean.mean() + critical * standard_error),
    }


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return pd.Series(np.nan, index=p_values.index, dtype=float)
    count = len(valid)
    adjusted = pd.Series(index=valid.index, dtype=float)
    running = 1.0
    for rank in range(count, 0, -1):
        index = valid.index[rank - 1]
        running = min(running, float(valid.iloc[rank - 1]) * count / rank)
        adjusted.loc[index] = running
    return adjusted.reindex(p_values.index)


class RollingEvidenceEngine:
    """Compute fixed-denominator marginal evidence on standard observation tables."""

    OBSERVATION_COLUMNS = {
        "observation_id", "pool_id", "decision_time", "instrument_id", "label_id",
        "label_value", "label_exit_time",
    }
    FACTOR_COLUMNS = {"observation_id", "factor_id", "factor_value", "factor_known_at"}

    def __init__(self, recipe: EvidenceRecipe):
        self.recipe = recipe

    def evaluate(
        self,
        observations: pd.DataFrame,
        factor_values: pd.DataFrame,
        factor_metadata: pd.DataFrame,
        label_metadata: pd.DataFrame,
        exposures: pd.DataFrame | None = None,
    ) -> EvidenceResult:
        observations = observations.copy()
        factor_values = factor_values.copy()
        self._validate(observations, factor_values, factor_metadata, label_metadata, exposures)
        observations["decision_time"] = pd.to_datetime(observations["decision_time"], utc=True)
        observations["label_exit_time"] = pd.to_datetime(
            observations["label_exit_time"], utc=True
        )
        factor_values["factor_known_at"] = pd.to_datetime(
            factor_values["factor_known_at"], utc=True
        )
        joined = factor_values.merge(observations, on="observation_id", how="left", validate="many_to_one")
        if (joined["factor_known_at"] > joined["decision_time"]).any():
            raise ValueError("factor_known_at exceeds decision_time")
        metadata = factor_metadata.set_index("factor_id")
        health_rows: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        theoretical = observations.groupby("pool_id")["decision_time"].nunique().to_dict()
        for factor_id in metadata.index:
            current = joined[joined["factor_id"] == factor_id].copy()
            factor_type = str(metadata.loc[factor_id, "factor_type"])
            total = len(observations)
            non_null = int(current["factor_value"].notna().sum())
            coverage = non_null / total if total else 0.0
            health_rows.append({
                "factor_id": factor_id,
                "factor_type": factor_type,
                "observation_count": total,
                "non_null_count": non_null,
                "non_null_ratio": coverage,
                "finite": bool(np.isfinite(current["factor_value"].dropna()).all()),
            })
            if factor_type == "nominal":
                failures.append({"factor_id": factor_id, "stage": "economic",
                                 "reason": "nominal_factor_has_no_ordered_economic_test"})
            for (pool_id, decision_time), group in current.groupby(
                ["pool_id", "decision_time"], sort=True
            ):
                metrics, reason = self._daily_metrics(group, factor_type)
                if reason:
                    failures.append({"factor_id": factor_id, "pool_id": pool_id,
                                     "decision_time": decision_time, "stage": "daily",
                                     "reason": reason})
                    continue
                daily_rows.append({"factor_id": factor_id, "pool_id": pool_id,
                                   "decision_time": decision_time, **metrics})
        health = pd.DataFrame(health_rows)
        daily = pd.DataFrame(daily_rows)
        summary_rows: list[dict[str, Any]] = []
        stability_rows: list[dict[str, Any]] = []
        if not daily.empty:
            for (factor_id, pool_id), group in daily.groupby(["factor_id", "pool_id"]):
                rank_stats = _hac_mean(
                    group["rank_ic"], self.recipe.hac_primary_lag,
                    self.recipe.confidence_level,
                )
                spread_stats = _hac_mean(
                    group["spread"], self.recipe.hac_primary_lag,
                    self.recipe.confidence_level,
                )
                direction = 1 if rank_stats["mean"] >= 0 else -1
                adjusted_spread_lower = (
                    spread_stats["lower"] * direction
                    if direction > 0 else -spread_stats["upper"]
                )
                by_year = group.assign(
                    year=pd.to_datetime(group["decision_time"]).dt.year,
                    adjusted_metric=group["rank_ic"] * direction,
                ).groupby("year")["adjusted_metric"].mean()
                positive_share = float((by_year > 0).mean()) if len(by_year) else 0.0
                for year, value in by_year.items():
                    stability_rows.append({"factor_id": factor_id, "pool_id": pool_id,
                                           "year": int(year), "adjusted_rank_ic": value})
                required = min(
                    self.recipe.valid_decision_cap,
                    max(
                        self.recipe.minimum_valid_decisions,
                        math.ceil(self.recipe.valid_decision_coverage * theoretical[pool_id]),
                    ),
                )
                health_row = health.set_index("factor_id").loc[factor_id]
                model_candidate = bool(
                    health_row["non_null_ratio"] >= self.recipe.factor_non_null_ratio
                    and len(group) >= required
                )
                summary_rows.append({
                    "factor_id": factor_id,
                    "pool_id": pool_id,
                    "valid_decisions": len(group),
                    "required_valid_decisions": required,
                    "rank_ic_mean": rank_stats["mean"],
                    "rank_ic_se": rank_stats["se"],
                    "rank_ic_p_value": rank_stats["p_value"],
                    "rank_ic_lower": rank_stats["lower"],
                    "rank_ic_upper": rank_stats["upper"],
                    "spread_mean": spread_stats["mean"],
                    "spread_lower": spread_stats["lower"],
                    "spread_upper": spread_stats["upper"],
                    "adjusted_spread_lower": adjusted_spread_lower,
                    "direction": direction,
                    "positive_year_share": positive_share,
                    "mean_monotonicity": float(group["monotonicity"].mean()),
                    "model_candidate": model_candidate,
                })
        summary = pd.DataFrame(summary_rows)
        summary = self._classify_summary(summary)
        exposure_frame = self._exposures(joined, exposures)
        redundancy = self._redundancy(factor_values)
        return EvidenceResult(
            factor_health=health,
            daily_metrics=daily,
            factor_summary=summary,
            period_stability=pd.DataFrame(stability_rows),
            exposure_diagnostics=exposure_frame,
            failure_records=pd.DataFrame(failures),
            redundancy_pairs=redundancy,
        )

    def combine(
        self,
        results: list[EvidenceResult],
        redundancy_values: pd.DataFrame | None = None,
    ) -> EvidenceResult:
        """Combine independently streamed factors into one statistical family."""
        if not results:
            empty = pd.DataFrame()
            return EvidenceResult(empty, empty, empty, empty, empty, empty, empty)
        summary = self._classify_summary(pd.concat(
            [item.factor_summary for item in results], ignore_index=True
        ))
        redundancy = (
            self._redundancy(redundancy_values)
            if redundancy_values is not None and not redundancy_values.empty
            else pd.DataFrame(columns=["factor_id_a", "factor_id_b", "spearman"])
        )
        return EvidenceResult(
            factor_health=pd.concat(
                [item.factor_health for item in results], ignore_index=True
            ),
            daily_metrics=pd.concat(
                [item.daily_metrics for item in results], ignore_index=True
            ),
            factor_summary=summary,
            period_stability=pd.concat(
                [item.period_stability for item in results], ignore_index=True
            ),
            exposure_diagnostics=pd.concat(
                [item.exposure_diagnostics for item in results], ignore_index=True
            ),
            failure_records=pd.concat(
                [item.failure_records for item in results], ignore_index=True
            ),
            redundancy_pairs=redundancy,
        )

    def _classify_summary(self, summary: pd.DataFrame) -> pd.DataFrame:
        summary = summary.copy()
        if summary.empty:
            return summary
        summary["rank_ic_q_value"] = summary.groupby("pool_id", group_keys=False)[
            "rank_ic_p_value"
        ].transform(_bh_adjust)
        summary["economic_eligible"] = (
            summary["model_candidate"]
            & (summary["rank_ic_q_value"] <= self.recipe.fdr_alpha)
            & (summary["adjusted_spread_lower"] >= self.recipe.minimum_effect)
            & (summary["positive_year_share"] >= self.recipe.positive_period_share)
            & (
                summary["mean_monotonicity"] * summary["direction"]
                >= self.recipe.monotonicity_threshold
            )
        )
        summary["classification"] = np.select(
            [summary["economic_eligible"], summary["model_candidate"]],
            ["economic_eligible", "model_candidate"],
            default="evidence_only",
        )
        return summary

    def _daily_metrics(self, group: pd.DataFrame, factor_type: str):
        valid = group.dropna(subset=["factor_value", "label_value"]).copy()
        if len(group) < self.recipe.minimum_pool_size:
            return {}, "pool_below_minimum_size"
        if len(valid) / len(group) < self.recipe.factor_non_null_ratio:
            return {}, "factor_coverage_below_threshold"
        if valid["factor_value"].nunique() < 2:
            return {}, "factor_has_no_cross_sectional_variation"
        rank_ic = float(spearmanr(valid["factor_value"], valid["label_value"]).statistic)
        if not np.isfinite(rank_ic):
            return {}, "rank_ic_not_finite"
        if factor_type == "nominal":
            return {"rank_ic": np.nan, "spread": np.nan, "monotonicity": np.nan,
                    "group_count": 0}, None
        groups = self._groups(valid["factor_value"])
        if groups is None:
            return {}, "groups_not_structurally_feasible"
        valid["factor_group"] = groups
        means = valid.groupby("factor_group", observed=True)["label_value"].mean()
        if len(means) != int(groups.max()) or len(means) < 2:
            return {}, "endpoint_groups_missing"
        spread = float(means.iloc[-1] - means.iloc[0])
        monotonicity = float(spearmanr(means.index.astype(float), means.values).statistic)
        return {"rank_ic": rank_ic, "spread": spread, "monotonicity": monotonicity,
                "group_count": int(groups.max())}, None

    def _groups(self, values: pd.Series) -> pd.Series | None:
        for count in (self.recipe.preferred_group_count, self.recipe.fallback_group_count):
            if len(values) < count * self.recipe.minimum_group_size:
                continue
            ranks = values.rank(method="average", pct=True)
            groups = np.ceil(ranks * count).clip(1, count).astype(int)
            sizes = groups.value_counts()
            if len(sizes) == count and int(sizes.min()) >= self.recipe.minimum_group_size:
                return groups
        return None

    def _redundancy(self, factor_values: pd.DataFrame) -> pd.DataFrame:
        pivot = factor_values.pivot(index="observation_id", columns="factor_id", values="factor_value")
        if pivot.shape[1] < 2:
            return pd.DataFrame(columns=["factor_id_a", "factor_id_b", "spearman"])
        correlation = pivot.corr(method="spearman", min_periods=10)
        rows = []
        columns = list(correlation.columns)
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1:]:
                value = correlation.loc[left, right]
                if pd.notna(value) and abs(value) >= self.recipe.redundancy_correlation:
                    rows.append({"factor_id_a": left, "factor_id_b": right,
                                 "spearman": float(value)})
        return pd.DataFrame(rows)

    @staticmethod
    def _exposures(joined: pd.DataFrame, exposures: pd.DataFrame | None) -> pd.DataFrame:
        if exposures is None or exposures.empty:
            return pd.DataFrame(columns=["factor_id", "exposure_id", "mean_spearman"])
        rows = []
        for (exposure_id, exposure_type), exposure in exposures.groupby(
            ["exposure_id", "exposure_type"]
        ):
            frame = joined.merge(exposure, on="observation_id", how="inner")
            frame["exposure_known_at"] = pd.to_datetime(
                frame["exposure_known_at"], utc=True
            )
            if (frame["exposure_known_at"] > frame["decision_time"]).any():
                raise ValueError("exposure_known_at exceeds decision_time")
            for factor_id, group in frame.groupby("factor_id"):
                rows.extend(RollingEvidenceEngine._exposure_rows(
                    factor_id, exposure_id, exposure_type, group
                ))
        return pd.DataFrame(rows)

    @staticmethod
    def _exposure_rows(factor_id, exposure_id, exposure_type, group):
        values = []
        for _, daily in group.groupby("decision_time"):
            valid = daily.dropna(subset=["factor_value", "exposure_value"])
            if len(valid) < 3 or valid["factor_value"].nunique() <= 1:
                continue
            if exposure_type == "continuous":
                values.append(
                    spearmanr(valid["factor_value"], valid["exposure_value"]).statistic
                )
            else:
                ranked = valid["factor_value"].rank(pct=True).to_numpy(dtype=float)
                categories = valid["exposure_value"].fillna("UNKNOWN").astype(str)
                design = pd.get_dummies(categories, drop_first=True, dtype=float)
                matrix = np.column_stack([np.ones(len(design)), design.to_numpy()])
                if matrix.shape[1] >= len(ranked):
                    continue
                fitted = matrix @ np.linalg.lstsq(matrix, ranked, rcond=None)[0]
                total = float(np.sum((ranked - ranked.mean()) ** 2))
                residual = float(np.sum((ranked - fitted) ** 2))
                r_squared = 1 - residual / total if total > 0 else np.nan
                n, parameters = len(ranked), matrix.shape[1]
                adjusted = 1 - (1 - r_squared) * (n - 1) / (n - parameters)
                values.append(adjusted)
        row = {"factor_id": factor_id, "exposure_id": exposure_id,
               "exposure_type": exposure_type, "valid_decisions": len(values)}
        if exposure_type == "continuous":
            row["mean_spearman"] = float(np.nanmean(values)) if values else np.nan
            row["mean_adjusted_r2"] = np.nan
        else:
            row["mean_spearman"] = np.nan
            row["mean_adjusted_r2"] = float(np.nanmean(values)) if values else np.nan
        return [row]

    def _validate(self, observations, factors, factor_metadata, label_metadata, exposures):
        missing = self.OBSERVATION_COLUMNS - set(observations)
        if missing:
            raise ValueError(f"observations missing columns: {sorted(missing)}")
        missing = self.FACTOR_COLUMNS - set(factors)
        if missing:
            raise ValueError(f"factor_values missing columns: {sorted(missing)}")
        if observations.duplicated(["observation_id", "label_id"]).any():
            raise ValueError("observations contain duplicate observation_id/label_id")
        if factors.duplicated(["observation_id", "factor_id"]).any():
            raise ValueError("factor_values contain duplicate observation_id/factor_id")
        if not np.isfinite(observations["label_value"].dropna()).all():
            raise ValueError("label_value contains infinity")
        if not np.isfinite(factors["factor_value"].dropna()).all():
            raise ValueError("factor_value contains infinity")
        allowed_types = {"continuous", "ordinal", "binary", "nominal"}
        if not set(factor_metadata["factor_type"]).issubset(allowed_types):
            raise ValueError("factor_metadata contains unsupported factor_type")
        if set(factors["observation_id"]) - set(observations["observation_id"]):
            raise ValueError("factor_values reference unknown observations")
        if set(factors["factor_id"]) - set(factor_metadata["factor_id"]):
            raise ValueError("factor_values reference unknown factors")
        required_label = {
            "label_id", "decision_schedule", "decision_interval_observations",
            "entry_rule", "exit_rule", "holding_horizon_observations",
            "execution_delay_observations", "cost_model", "label_is_net_of_costs",
            "overlapping_positions",
        }
        if required_label - set(label_metadata):
            raise ValueError("label_metadata is incomplete")
        if exposures is not None and not exposures.empty:
            required_exposure = {
                "observation_id", "exposure_id", "exposure_type", "exposure_value",
                "exposure_known_at",
            }
            if required_exposure - set(exposures):
                raise ValueError("exposures table is incomplete")
            if not set(exposures["exposure_type"]).issubset({"continuous", "categorical"}):
                raise ValueError("exposures contain unsupported exposure_type")

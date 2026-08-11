"""Strategy-specific active-bar technical factors for EMA20 momentum."""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from zyquant.core.hashing import hash_file
from zyquant.factors import BaseFactor, FactorContext


ANNUALIZATION_DAYS = 250.0
VOLATILITY_DAYS = 252.0
MOMENTUM_BARS = 21
MAX_HISTORY_BARS = 10_000

FEATURE_NAMES = (
    "annualized_returns", "r2", "slope", "score", "vol_ratio", "lookback",
    "score_dynamic", "has_recent_drop", "decay_days", "is_decaying",
    "over_return_cap", "score_ratio", "score_diff",
    "ret_1d", "ret_3d", "short_term_return_5d", "ret_10d", "ret_20d",
    "ret_60d", "close_over_ma20", "close_over_ma60", "consecutive_up_days",
    "consecutive_down_days", "bollinger_z_20", "rsi_14",
    "macd_norm_12_26", "ma_cross_5_20", "ma_cross_20_60",
    "realized_vol_5", "realized_vol_20", "downside_vol_20",
    "drawdown_from_high_20", "max_dd_5", "atr_ratio_5", "atr_ratio_14",
    "sharpe_like_5", "vol_ratio_recent_hist", "is_high_level_volume_spike",
    "vol_ratio_5", "vol_ratio_20", "volume_zscore_20", "money_ratio_5_20",
    "amihud_illiquidity_20", "price_volume_corr_20", "upper_shadow_ratio",
    "lower_shadow_ratio", "body_ratio", "intraday_return", "overnight_gap",
)

EXCLUDED_CROSS_SECTIONAL_FEATURES = (
    "score_rank_pct", "ret_5d_rank_pct", "ret_20d_rank_pct", "is_new_top1",
)
EXCLUDED_CONSTANT_FEATURES = ("premium_rate", "over_score_cap")


def _spec(formula: str, bars: int, category: str, policy: str = "active_bar"):
    return MappingProxyType({
        "formula": formula,
        "required_bars": bars,
        "category": category,
        "bar_policy": policy,
    })


FEATURE_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "annualized_returns": _spec("exp(weighted_log_slope_21*250)-1", 21, "momentum"),
    "r2": _spec("weighted_log_regression_r2_21", 21, "momentum"),
    "slope": _spec("weighted_log_regression_slope_21", 21, "momentum"),
    "score": _spec("annualized_returns*r2", 21, "momentum"),
    "vol_ratio": _spec("std(return,10)/std(return,30)", 31, "momentum"),
    "lookback": _spec("int(clip(10+50*(1-min(vol_ratio,0.9)),10,60))", 31, "momentum"),
    "score_dynamic": _spec("weighted_momentum_score(dynamic_lookback)", 60, "momentum"),
    "has_recent_drop": _spec("reference_recent_drop_rule", 5, "momentum"),
    "decay_days": _spec("consecutive_score_declines_capped_3", 24, "momentum"),
    "is_decaying": _spec("decay_days>=3", 24, "momentum"),
    "over_return_cap": _spec("annualized_returns>20", 21, "momentum"),
    "score_ratio": _spec("score/previous_active_score", 22, "momentum"),
    "score_diff": _spec("score-previous_active_score", 22, "momentum"),
    "ret_1d": _spec("close/close_active_lag_1-1", 2, "return"),
    "ret_3d": _spec("close/close_active_lag_3-1", 4, "return"),
    "short_term_return_5d": _spec("close/close_active_lag_5-1", 6, "return"),
    "ret_10d": _spec("close/close_active_lag_10-1", 11, "return"),
    "ret_20d": _spec("close/close_active_lag_20-1", 21, "return"),
    "ret_60d": _spec("close/close_active_lag_60-1", 61, "return"),
    "close_over_ma20": _spec("close/mean(previous_20_active_close)-1", 21, "trend"),
    "close_over_ma60": _spec("close/mean(previous_60_active_close)-1", 61, "trend"),
    "consecutive_up_days": _spec("active_close_up_streak_capped_10", 11, "trend"),
    "consecutive_down_days": _spec("active_close_down_streak_capped_10", 11, "trend"),
    "bollinger_z_20": _spec("zscore(active_close,20,ddof=0)", 20, "trend"),
    "rsi_14": _spec("mean(gain,14)/(mean(gain,14)+mean(loss,14))*100", 15, "trend"),
    "macd_norm_12_26": _spec("(EMA12-EMA26)/close_from_first_active_bar", 26, "trend"),
    "ma_cross_5_20": _spec("mean(close,5)/mean(close,20)-1", 20, "trend"),
    "ma_cross_20_60": _spec("mean(close,20)/mean(close,60)-1", 60, "trend"),
    "realized_vol_5": _spec("std(active_return,5,ddof=0)*sqrt(252)", 6, "risk"),
    "realized_vol_20": _spec("std(active_return,20,ddof=0)*sqrt(252)", 21, "risk"),
    "downside_vol_20": _spec("rms(min(active_return,0),20)*sqrt(252)", 21, "risk"),
    "drawdown_from_high_20": _spec("close/max(previous_20_active_high)-1", 21, "risk"),
    "max_dd_5": _spec("max_drawdown(active_close,5)", 5, "risk"),
    "atr_ratio_5": _spec("mean(active_true_range,5)/close", 6, "risk"),
    "atr_ratio_14": _spec("mean(active_true_range,14)/close", 15, "risk"),
    "sharpe_like_5": _spec("mean(active_return,5)/std(active_return,5,ddof=0)", 6, "risk"),
    "vol_ratio_recent_hist": _spec("mean(volume,3)/mean(previous_volume,7)", 10, "volume"),
    "is_high_level_volume_spike": _spec("annualized_returns>5 and vol_ratio_recent_hist>2", 21, "volume"),
    "vol_ratio_5": _spec("volume/mean(previous_5_active_volume)", 6, "volume"),
    "vol_ratio_20": _spec("volume/mean(previous_20_active_volume)", 21, "volume"),
    "volume_zscore_20": _spec("zscore(log1p(active_volume),20,ddof=0)", 20, "volume"),
    "money_ratio_5_20": _spec("mean(amount,5)/mean(amount,20)-1", 20, "liquidity"),
    "amihud_illiquidity_20": _spec("mean(abs(return)/positive_amount,20)*1e8", 21, "liquidity"),
    "price_volume_corr_20": _spec("corr(return,diff(log1p(volume)),20)", 21, "volume"),
    "upper_shadow_ratio": _spec("(high-max(open,close))/(high-low)", 1, "candle", "current_bar"),
    "lower_shadow_ratio": _spec("(min(open,close)-low)/(high-low)", 1, "candle", "current_bar"),
    "body_ratio": _spec("abs(close-open)/(high-low)", 1, "candle", "current_bar"),
    "intraday_return": _spec("close/open-1", 1, "candle", "current_bar"),
    "overnight_gap": _spec("open/previous_active_close-1", 2, "candle"),
})


_FEATURE_CACHE: dict[tuple[Any, ...], pd.DataFrame] = {}


def clear_momentum_feature_cache() -> None:
    _FEATURE_CACHE.clear()


def _rolling_dot(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    if len(values) >= len(weights):
        result[len(weights) - 1:] = np.convolve(values, weights[::-1], mode="valid")
    return result


def _weighted_momentum(values: np.ndarray, window: int) -> tuple[np.ndarray, ...]:
    y = np.log(values.astype(float))
    x = np.arange(window, dtype=float)
    weights = np.linspace(1.0, 2.0, window)
    squared = weights**2
    squared_sum = squared.sum()
    squared_x_sum = np.dot(squared, x)
    squared_x2_sum = np.dot(squared, x**2)
    denominator = squared_x2_sum - squared_x_sum**2 / squared_sum
    squared_y = _rolling_dot(y, squared)
    squared_xy = _rolling_dot(y, squared * x)
    slope = (squared_xy - squared_x_sum * squared_y / squared_sum) / denominator
    intercept = squared_y / squared_sum - slope * squared_x_sum / squared_sum

    residual = np.full(len(values), np.nan)
    total = np.full(len(values), np.nan)
    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(y, window)
        valid_slope = slope[window - 1:]
        valid_intercept = intercept[window - 1:]
        fitted = valid_slope[:, None] * x + valid_intercept[:, None]
        residual[window - 1:] = np.sum(
            weights * (windows - fitted) ** 2, axis=1
        )
        means = windows.mean(axis=1)
        total[window - 1:] = np.sum(
            weights * (windows - means[:, None]) ** 2, axis=1
        )
    r2 = np.where(total > 0, 1.0 - residual / total, 0.0)
    exponent = slope * ANNUALIZATION_DAYS
    annualized = np.where(exponent < 709, np.expm1(exponent), np.nan)
    score = annualized * r2
    return annualized, r2, score, slope


def _streaks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.zeros(len(values), dtype=float)
    down = np.zeros(len(values), dtype=float)
    for index in range(1, len(values)):
        if values[index] > values[index - 1]:
            up[index] = min(10, up[index - 1] + 1)
        elif values[index] < values[index - 1]:
            down[index] = min(10, down[index - 1] + 1)
    return up, down


def _maximum_drawdown_5(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    for index in range(4, len(values)):
        block = values[index - 4:index + 1]
        peaks = np.maximum.accumulate(block)
        result[index] = np.max((peaks - block) / peaks)
    return result


def _one_instrument(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    close = group["close_post"].to_numpy(dtype=float)
    opened = group["open_post"].to_numpy(dtype=float)
    high = group["high_post"].to_numpy(dtype=float)
    low = group["low_post"].to_numpy(dtype=float)
    volume = group["volume"].to_numpy(dtype=float)
    amount = group["amount"].to_numpy(dtype=float)
    close_series = pd.Series(close)
    returns = close_series.pct_change(fill_method=None)
    returns5 = returns.rolling(5, min_periods=5)
    returns20 = returns.rolling(20, min_periods=20)
    output = group[["trade_date", "instrument_id"]].copy()

    annualized, r2, score, slope = _weighted_momentum(close, MOMENTUM_BARS)
    output["annualized_returns"] = annualized
    output["r2"] = r2
    output["slope"] = slope
    output["score"] = score
    short_vol = returns.rolling(10, min_periods=10).std(ddof=0)
    long_vol = returns.rolling(30, min_periods=30).std(ddof=0)
    vol_ratio = short_vol / long_vol
    vol_ratio = vol_ratio.where(long_vol > 0, 1.0)
    output["vol_ratio"] = vol_ratio.to_numpy()
    lookback = np.full(len(group), np.nan)
    valid_ratio = np.isfinite(vol_ratio.to_numpy())
    lookback[valid_ratio] = np.clip(
        10 + 50 * (1 - np.minimum(vol_ratio.to_numpy()[valid_ratio], 0.9)),
        10, 60,
    ).astype(int)
    output["lookback"] = lookback
    dynamic = np.full(len(group), np.nan)
    for window in range(10, 61):
        selected = lookback == window
        if selected.any():
            dynamic[selected] = _weighted_momentum(close, window)[2][selected]
    output["score_dynamic"] = dynamic

    recent_drop = np.full(len(group), np.nan, dtype=float)
    for index in range(4, len(group)):
        recent_drop[index] = float(
            min(close[index] / close[index - 1], close[index - 1] / close[index - 2],
                close[index - 2] / close[index - 3]) < 0.95
            or (close[index] < close[index - 1] < close[index - 2] < close[index - 3]
                and close[index] / close[index - 3] < 0.95)
            or (close[index - 1] < close[index - 2] < close[index - 3] < close[index - 4]
                and close[index - 1] / close[index - 4] < 0.95)
        )
    output["has_recent_drop"] = recent_drop
    decay = np.full(len(group), np.nan, dtype=float)
    counter = 0.0
    for index in range(1, len(group)):
        if not np.isfinite(score[index]):
            continue
        if np.isfinite(score[index - 1]) and score[index] < score[index - 1]:
            counter = min(3, counter + 1)
        else:
            counter = 0.0
        decay[index] = counter
    output["decay_days"] = decay
    output["is_decaying"] = np.where(np.isfinite(decay), (decay >= 3).astype(float), np.nan)
    output["over_return_cap"] = np.where(
        np.isfinite(annualized), (annualized > 20).astype(float), np.nan
    )
    previous_score = pd.Series(score).shift(1).to_numpy()
    output["score_ratio"] = np.where(
        ~np.isfinite(score), np.nan,
        np.where(~np.isfinite(previous_score) | (previous_score == 0), 1.0,
                 score / previous_score),
    )
    output["score_diff"] = np.where(
        ~np.isfinite(score), np.nan,
        np.where(~np.isfinite(previous_score), 0.0, score - previous_score),
    )

    for name, lag in (("ret_1d", 1), ("ret_3d", 3),
                      ("short_term_return_5d", 5), ("ret_10d", 10),
                      ("ret_20d", 20), ("ret_60d", 60)):
        output[name] = close_series.pct_change(lag, fill_method=None).to_numpy()
    output["close_over_ma20"] = (
        close_series / close_series.shift(1).rolling(20, min_periods=20).mean() - 1
    ).to_numpy()
    output["close_over_ma60"] = (
        close_series / close_series.shift(1).rolling(60, min_periods=60).mean() - 1
    ).to_numpy()
    output["consecutive_up_days"], output["consecutive_down_days"] = _streaks(close)
    close_mean20 = close_series.rolling(20, min_periods=20).mean()
    close_std20 = close_series.rolling(20, min_periods=20).std(ddof=0)
    output["bollinger_z_20"] = ((close_series - close_mean20) / close_std20).to_numpy()
    gains = returns.clip(lower=0).rolling(14, min_periods=14).mean()
    losses = (-returns.clip(upper=0)).rolling(14, min_periods=14).mean()
    rsi_denominator = gains + losses
    output["rsi_14"] = (100 * gains / rsi_denominator).where(
        rsi_denominator > 1e-15, 50.0
    ).to_numpy()
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    output["macd_norm_12_26"] = ((ema12 - ema26) / close_series).to_numpy()
    output["ma_cross_5_20"] = (
        close_series.rolling(5, min_periods=5).mean() / close_mean20 - 1
    ).to_numpy()
    output["ma_cross_20_60"] = (
        close_mean20 / close_series.rolling(60, min_periods=60).mean() - 1
    ).to_numpy()

    output["realized_vol_5"] = (returns5.std(ddof=0) * math.sqrt(VOLATILITY_DAYS)).to_numpy()
    output["realized_vol_20"] = (returns20.std(ddof=0) * math.sqrt(VOLATILITY_DAYS)).to_numpy()
    downside_squared = returns.clip(upper=0) ** 2
    output["downside_vol_20"] = (
        downside_squared.rolling(20, min_periods=20).mean().pow(0.5)
        * math.sqrt(VOLATILITY_DAYS)
    ).to_numpy()
    output["drawdown_from_high_20"] = (
        close_series / pd.Series(high).shift(1).rolling(20, min_periods=20).max() - 1
    ).to_numpy()
    output["max_dd_5"] = _maximum_drawdown_5(close)
    previous_close = close_series.shift(1).to_numpy()
    true_range = np.maximum.reduce([
        high - low, np.abs(high - previous_close), np.abs(low - previous_close),
    ])
    true_range_series = pd.Series(true_range)
    output["atr_ratio_5"] = (
        true_range_series.rolling(5, min_periods=5).mean() / close_series
    ).to_numpy()
    output["atr_ratio_14"] = (
        true_range_series.rolling(14, min_periods=14).mean() / close_series
    ).to_numpy()
    output["sharpe_like_5"] = (
        returns5.mean() / returns5.std(ddof=0)
    ).to_numpy()

    volume_series = pd.Series(volume)
    amount_series = pd.Series(amount)
    historical_volume = volume_series.shift(3).rolling(7, min_periods=7).mean()
    recent_volume = volume_series.rolling(3, min_periods=3).mean()
    output["vol_ratio_recent_hist"] = (recent_volume / historical_volume).where(
        historical_volume > 0, 0.0
    ).to_numpy()
    output["is_high_level_volume_spike"] = np.where(
        output["annualized_returns"].notna(),
        ((output["annualized_returns"] > 5)
         & (output["vol_ratio_recent_hist"] > 2)).astype(float),
        np.nan,
    )
    prior_volume5 = volume_series.shift(1).rolling(5, min_periods=5).mean()
    prior_volume20 = volume_series.shift(1).rolling(20, min_periods=20).mean()
    output["vol_ratio_5"] = (volume_series / prior_volume5).where(
        prior_volume5 > 0, 0.0
    ).to_numpy()
    output["vol_ratio_20"] = (volume_series / prior_volume20).where(
        prior_volume20 > 0, 0.0
    ).to_numpy()
    log_volume = np.log1p(volume_series)
    log_mean = log_volume.rolling(20, min_periods=20).mean()
    log_std = log_volume.rolling(20, min_periods=20).std(ddof=0)
    output["volume_zscore_20"] = ((log_volume - log_mean) / log_std).to_numpy()
    amount_mean20 = amount_series.rolling(20, min_periods=20).mean()
    output["money_ratio_5_20"] = (
        amount_series.rolling(5, min_periods=5).mean() / amount_mean20 - 1
    ).where(amount_mean20 > 0).to_numpy()
    amihud_daily = returns.abs() / amount_series.where(amount_series > 0)
    output["amihud_illiquidity_20"] = (
        amihud_daily.rolling(20, min_periods=1).mean() * 1e8
    ).where(returns.rolling(20, min_periods=20).count() == 20).to_numpy()
    volume_change = log_volume.diff()
    output["price_volume_corr_20"] = returns.rolling(
        20, min_periods=20
    ).corr(volume_change).to_numpy()

    spread = high - low
    valid_spread = np.isfinite(spread) & (spread > 0)
    upper_shadow = np.full(len(group), np.nan)
    lower_shadow = np.full(len(group), np.nan)
    body = np.full(len(group), np.nan)
    upper_shadow[spread <= 0] = 0.0
    lower_shadow[spread <= 0] = 0.0
    body[spread <= 0] = 0.0
    np.divide(
        high - np.maximum(opened, close), spread,
        out=upper_shadow, where=valid_spread,
    )
    np.divide(
        np.minimum(opened, close) - low, spread,
        out=lower_shadow, where=valid_spread,
    )
    np.divide(
        np.abs(close - opened), spread, out=body, where=valid_spread,
    )
    output["upper_shadow_ratio"] = upper_shadow
    output["lower_shadow_ratio"] = lower_shadow
    output["body_ratio"] = body
    intraday = np.full(len(group), np.nan)
    overnight = np.full(len(group), np.nan)
    np.divide(
        close, opened, out=intraday,
        where=np.isfinite(opened) & (opened != 0),
    )
    np.divide(
        opened, previous_close, out=overnight,
        where=np.isfinite(previous_close) & (previous_close != 0),
    )
    output["intraday_return"] = intraday - 1.0
    output["overnight_gap"] = overnight - 1.0
    output[list(FEATURE_NAMES)] = output[list(FEATURE_NAMES)].replace(
        [np.inf, -np.inf], np.nan
    )
    return output


def _a_share_codes(context: FactorContext) -> list[str]:
    instruments = context.snapshot.table("instruments")
    required = {"instrument_id", "symbol", "exchange", "asset_type"}
    if required - set(instruments):
        raise ValueError("instruments table cannot identify mainland A shares")
    symbol = instruments["symbol"].astype(str)
    accepted = (
        instruments["exchange"].isin(["XSHG", "XSHE"])
        & instruments["asset_type"].eq("stock")
        & ~((instruments["exchange"].eq("XSHG") & symbol.str.startswith("900"))
            | (instruments["exchange"].eq("XSHE") & symbol.str.startswith("200")))
    )
    if "currency" in instruments:
        accepted &= instruments["currency"].isna() | instruments["currency"].eq("CNY")
    codes = set(instruments.loc[accepted, "instrument_id"].astype(str))
    if context.instruments is not None:
        codes &= set(context.instruments)
    return sorted(codes)


def _feature_frame(context: FactorContext) -> pd.DataFrame:
    codes = _a_share_codes(context)
    key = (
        context.snapshot.metadata.fingerprint, context.start, context.end,
        context.cutoff, tuple(codes),
    )
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]
    post = context.post_adjusted(
        ["open_post", "high_post", "low_post", "close_post"], MAX_HISTORY_BARS,
    )
    raw = context.raw(["volume", "amount", "paused"], MAX_HISTORY_BARS)
    joined = post.merge(
        raw, on=["trade_date", "instrument_id"], how="inner",
        validate="one_to_one",
    )
    joined = joined[
        joined["instrument_id"].astype(str).isin(codes)
        & ~joined["paused"].fillna(True).astype(bool)
    ].copy()
    parts = [
        _one_instrument(group)
        for _, group in joined.groupby("instrument_id", sort=True)
    ]
    frame = (
        pd.concat(parts, ignore_index=True)
        if parts else pd.DataFrame(columns=["trade_date", "instrument_id", *FEATURE_NAMES])
    )
    _FEATURE_CACHE.clear()
    _FEATURE_CACHE[key] = frame
    return frame


class MomentumTechnicalFactor(BaseFactor):
    version = "1"
    lookback = MAX_HISTORY_BARS
    inputs = (
        "daily_post_adjusted.open_post", "daily_post_adjusted.high_post",
        "daily_post_adjusted.low_post", "daily_post_adjusted.close_post",
        "daily_raw.volume", "daily_raw.amount", "daily_raw.paused",
        "instruments.exchange", "instruments.asset_type", "instruments.symbol",
    )

    def __init__(self, feature_name: str):
        if feature_name not in FEATURE_SPECS:
            raise ValueError(f"unknown EMA20 momentum feature: {feature_name}")
        self.feature_name = feature_name
        self.name = f"ml_ema20_{feature_name}"

    def definition(self) -> Mapping[str, Any]:
        spec = FEATURE_SPECS[self.feature_name]
        return {
            **super().definition(),
            "feature_name": self.feature_name,
            "formula": spec["formula"],
            "required_bars": spec["required_bars"],
            "bar_policy": spec["bar_policy"],
            "paused_signal_policy": "omit",
            "non_paused_zero_volume_policy": "retain",
            "annualization_days": ANNUALIZATION_DAYS,
            "volatility_days": VOLATILITY_DAYS,
            "ddof": 0,
            "zero_denominator_policy": "feature_specific_nan_or_zero",
            "universe": "XSHG_XSHE_A_shares",
            "implementation_sha256": hash_file(Path(__file__)),
        }

    def compute(
        self, context: FactorContext, dependencies: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        del dependencies
        frame = _feature_frame(context)
        result = frame.loc[
            frame["trade_date"].between(context.start, context.end),
            ["trade_date", "instrument_id", self.feature_name],
        ].rename(columns={self.feature_name: "value"})
        return result.reset_index(drop=True)


def momentum_factor_catalog() -> Mapping[str, MomentumTechnicalFactor]:
    factors = tuple(MomentumTechnicalFactor(name) for name in FEATURE_NAMES)
    catalog = {factor.name: factor for factor in factors}
    if len(catalog) != 49:
        raise ValueError("EMA20 momentum factor catalog must contain 49 factors")
    return MappingProxyType(catalog)

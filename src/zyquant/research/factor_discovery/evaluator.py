from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .models import ExpressionSpec, Operator


_INDEX = ["trade_date", "instrument_id"]


class ExpressionEvaluator:
    """Evaluate a validated expression over sparse pool-local long frames."""

    def __init__(self, allowed_fields: set[str] | frozenset[str]):
        self.allowed_fields = set(allowed_fields)

    def evaluate(
        self,
        spec: ExpressionSpec,
        fields: Mapping[str, pd.DataFrame],
        index: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        spec.validate_contract(self.allowed_fields)
        required = spec.fields()
        available = set(fields)
        missing = required - available
        if missing:
            raise ValueError(f"missing field values: {sorted(missing)}")
        # Normalizing every registered field made wide research panels needlessly
        # materialize hundreds of long frames for a two-leaf expression.  Restrict
        # work to the expression leaves so adapters may expose lazy mappings over
        # large PIT panels without changing the public Mapping contract.
        normalized = {
            name: self._series(name, fields[name])
            for name in sorted(required)
        }
        if index is None:
            indexes = [series.index for series in normalized.values()]
            if not indexes:
                raise ValueError("an index or at least one field frame is required")
            base_index = indexes[0]
            for current in indexes[1:]:
                base_index = base_index.union(current)
        else:
            self._require_columns(index, set(_INDEX), "index")
            if index.duplicated(_INDEX).any():
                raise ValueError("index contains duplicate trade_date/instrument_id rows")
            base_index = pd.MultiIndex.from_frame(index[_INDEX]).sort_values()
            base_index.names = _INDEX
        calculation_index = base_index
        for series in normalized.values():
            calculation_index = calculation_index.union(series.index)
        calculation_index = calculation_index.sort_values()
        value = self._eval(spec, normalized, calculation_index)
        value = value.reindex(base_index).replace([np.inf, -np.inf], np.nan)
        output = value.rename("value").reset_index()
        return output.sort_values(_INDEX).reset_index(drop=True)

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    def _series(self, name: str, frame: pd.DataFrame) -> pd.Series:
        if name not in self.allowed_fields:
            raise ValueError(f"field is outside whitelist: {name}")
        self._require_columns(frame, {*_INDEX, "value"}, name)
        if frame.duplicated(_INDEX).any():
            raise ValueError(f"field {name} has duplicate date/instrument rows")
        result = frame.set_index(_INDEX)["value"].astype(float).sort_index()
        result.index.names = _INDEX
        return result.replace([np.inf, -np.inf], np.nan)

    def _eval(
        self,
        node: ExpressionSpec,
        fields: Mapping[str, pd.Series],
        base_index: pd.MultiIndex,
    ) -> pd.Series:
        op = node.op
        if op == Operator.CONST:
            if node.value is None:  # guarded by ExpressionSpec, retained for type safety
                raise ValueError("const node is missing value")
            return pd.Series(float(node.value), index=base_index, dtype=float)
        if op == Operator.FIELD:
            return fields[node.leaf_key].reindex(base_index)
        args = [self._eval(arg, fields, base_index) for arg in node.args]
        if op == Operator.NEG:
            return -args[0]
        if op == Operator.ABS:
            return args[0].abs()
        if op == Operator.SIGN:
            return np.sign(args[0])
        if op == Operator.LOG_ABS:
            return np.log1p(args[0].abs())
        if op == Operator.SQRT_ABS:
            return np.sqrt(args[0].abs())
        if op == Operator.ADD:
            return args[0] + args[1]
        if op == Operator.SUB:
            return args[0] - args[1]
        if op == Operator.MUL:
            return args[0] * args[1]
        if op == Operator.SAFE_DIV:
            denominator = args[1].where(args[1].abs() > 1e-12)
            return args[0] / denominator
        if op == Operator.MIN:
            return pd.concat(args, axis=1).min(axis=1, skipna=False)
        if op == Operator.MAX:
            return pd.concat(args, axis=1).max(axis=1, skipna=False)

        window = int(node.window or 1)
        if op == Operator.LAG:
            return self._group_transform(args[0], lambda x: x.shift(window))
        if op == Operator.DELTA:
            return self._group_transform(args[0], lambda x: x.diff(window))
        if op == Operator.RETURN:
            return self._group_transform(
                args[0], lambda x: x / x.shift(window).where(x.shift(window).abs() > 1e-12) - 1
            )
        if op == Operator.MEAN:
            return self._rolling(args[0], window, "mean")
        if op == Operator.STD:
            return self._rolling(args[0], window, "std")
        if op == Operator.SUM:
            return self._rolling(args[0], window, "sum")
        if op == Operator.ROLLING_MIN:
            return self._rolling(args[0], window, "min")
        if op == Operator.ROLLING_MAX:
            return self._rolling(args[0], window, "max")
        if op == Operator.RANK:
            return self._group_transform(
                args[0],
                lambda x: x.rolling(window, min_periods=window).apply(
                    lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=False
                ),
            )
        if op == Operator.EMA:
            return self._group_transform(
                args[0], lambda x: x.ewm(span=window, min_periods=window, adjust=False).mean()
            )
        if op == Operator.CORR:
            pair = pd.concat({"left": args[0], "right": args[1]}, axis=1)
            pieces = []
            for _, group in pair.groupby(level="instrument_id", sort=False):
                left = group["left"].reset_index(level="instrument_id", drop=True)
                right = group["right"].reset_index(level="instrument_id", drop=True)
                values = left.rolling(window, min_periods=window).corr(right)
                instrument = group.index.get_level_values("instrument_id")[0]
                values.index = pd.MultiIndex.from_arrays(
                    [values.index, [instrument] * len(values)], names=_INDEX
                )
                pieces.append(values)
            return pd.concat(pieces).sort_index() if pieces else args[0] * np.nan
        raise ValueError(f"unsupported operator: {op}")

    @staticmethod
    def _group_transform(series: pd.Series, function) -> pd.Series:
        return series.groupby(level="instrument_id", group_keys=False, sort=False).transform(
            function
        )

    def _rolling(self, series: pd.Series, window: int, method: str) -> pd.Series:
        return self._group_transform(
            series,
            lambda x: getattr(x.rolling(window, min_periods=window), method)(),
        )

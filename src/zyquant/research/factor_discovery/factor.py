from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import pandas as pd

from zyquant.factors import BaseFactor
from zyquant.factors.base import FactorContext

from .evaluator import ExpressionEvaluator
from .models import ExpressionSpec


class ExpressionFactor(BaseFactor):
    """A promoted expression that can use ZyQuant's canonical factor cache."""

    version = "1"

    def __init__(
        self,
        expression: ExpressionSpec,
        *,
        allowed_fields: set[str],
        dependency_factors: Mapping[str, BaseFactor] | None = None,
        name: str | None = None,
    ) -> None:
        expression.validate_contract(allowed_fields)
        self.expression = expression
        self.allowed_fields = frozenset(allowed_fields)
        self.name = name or f"expr_{expression.expression_id[:16]}"
        self.lookback = expression.lookback
        self._dependency_factors = dict(dependency_factors or {})
        required_dependencies = {
            item.split(".", 1)[1] for item in expression.fields()
            if item.startswith("factor.")
        }
        missing = required_dependencies - set(self._dependency_factors)
        if missing:
            raise ValueError(f"missing dependency factors: {sorted(missing)}")
        self.dependencies = tuple(
            self._dependency_factors[key] for key in sorted(required_dependencies)
        )
        self.inputs = tuple(sorted(
            field for field in expression.fields() if not field.startswith("factor.")
        ))

    def definition(self) -> Mapping[str, Any]:
        return {
            **super().definition(),
            "expression": self.expression.canonical_payload(),
            "expression_id": self.expression.expression_id,
        }

    def compute(
        self,
        context: FactorContext,
        dependencies: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        requested: dict[str, list[str]] = defaultdict(list)
        for leaf in self.expression.fields():
            source, field = leaf.split(".", 1)
            requested[source].append(field)
        fields: dict[str, pd.DataFrame] = {}
        for source in ("raw", "post_adjusted", "valuation"):
            names = sorted(set(requested.get(source, ())))
            if not names:
                continue
            if source == "raw":
                frame = context.raw(names, lookback=self.lookback)
            elif source == "post_adjusted":
                frame = context.post_adjusted(names, lookback=self.lookback)
            else:
                frame = context.valuation(names, lookback=self.lookback)
            for field in names:
                fields[f"{source}.{field}"] = frame[
                    ["trade_date", "instrument_id", field]
                ].rename(columns={field: "value"})
        for name in requested.get("factor", ()):
            factor = self._dependency_factors[name]
            fields[f"factor.{name}"] = dependencies[factor.name][
                ["trade_date", "instrument_id", "value"]
            ]
        output = ExpressionEvaluator(self.allowed_fields).evaluate(self.expression, fields)
        return output[
            (output["trade_date"] >= context.start) & (output["trade_date"] <= context.end)
        ].reset_index(drop=True)


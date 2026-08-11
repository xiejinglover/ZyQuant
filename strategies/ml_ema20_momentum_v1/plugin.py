from __future__ import annotations

from typing import Any

from zyquant.core.plugins import PluginMetadata

from .strategy import Ema20MomentumStrategy


PLUGIN_NAME = "ml_ema20_momentum_v1"


def create_strategy(**parameters: Any) -> Ema20MomentumStrategy:
    return Ema20MomentumStrategy(**parameters)


create_strategy.plugin_metadata = PluginMetadata(  # type: ignore[attr-defined]
    name=PLUGIN_NAME,
    version="1.0.0",
    kind="strategies",
    minimum_framework_version="2.0.0",
    input_schema="strategy.parameters/ml_ema20_momentum_v1@1",
    output_schema="strategy.scheduled_target_portfolio@1",
    deterministic=True,
)

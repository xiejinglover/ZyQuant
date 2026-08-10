from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


class CapitalAllocator(Protocol):
    def allocate(self, total_cash: float) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class FixedCapitalAllocator:
    weights: Mapping[str, float]

    def allocate(self, total_cash: float) -> Mapping[str, float]:
        if abs(sum(self.weights.values()) - 1.0) > 1e-12:
            raise ValueError("capital weights must sum to one")
        if any(value <= 0 for value in self.weights.values()):
            raise ValueError("capital weights must be positive")
        return {key: total_cash * value for key, value in self.weights.items()}


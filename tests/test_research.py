from __future__ import annotations

import tempfile
import unittest

from zyquant.data import SnapshotPublisher
from zyquant.factors import FactorEngine, MomentumFactor
from zyquant.portfolio import (
    ConstraintEngine, PortfolioConstraints, TopKEqualWeightConstructor,
)
from zyquant.strategy import (
    DailySchedule, ExternalSignalGenerator, PipelineStrategy, StandardUniverseSelector,
)
from zyquant.strategy.types import PortfolioView, StrategyContext, StrategyState

from tests.support import CODE_A, CODE_B, canonical_tables, signal_frame


class ResearchTests(unittest.TestCase):
    def test_factor_cache_and_pipeline_constraints(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("sample-v1", tables)
            engine = FactorEngine(f"{temporary}/cache")
            factor = MomentumFactor(2)
            first = engine.compute(factor, snapshot, days[2], days[-1], [CODE_A, CODE_B])
            second = engine.compute(factor, snapshot, days[2], days[-1], [CODE_A, CODE_B])
            self.assertFalse(first.from_cache)
            self.assertTrue(second.from_cache)
            self.assertEqual(first.cache_key, second.cache_key)

            strategy = PipelineStrategy(
                "alpha",
                DailySchedule(),
                StandardUniverseSelector("TEST", median_amount_window=1),
                ExternalSignalGenerator(signal_frame(days)),
                TopKEqualWeightConstructor(2),
                ConstraintEngine(PortfolioConstraints(max_instrument_weight=0.6)),
            )
            context = StrategyContext(
                "alpha", days[2], days[2], days[3], "open", snapshot, None,
                PortfolioView(1_000_000, 1_000_000), None, StrategyState(), {},
                __import__("numpy").random.default_rng(7), engine,
            )
            decision = strategy.decide(context)
            self.assertIsNotNone(decision.target)
            self.assertAlmostEqual(sum(decision.target.weights.values()), 1.0)
            self.assertLessEqual(max(decision.target.weights.values()), 0.6)
            self.assertEqual(set(decision.target.weights), {CODE_A, CODE_B})


if __name__ == "__main__":
    unittest.main()

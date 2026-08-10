from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from zyquant.experiment import ExperimentStore
from zyquant.optimize import GridSampler, IntRange, SearchEngine, SearchSpace


def objective(parameters):
    value = parameters["x"]
    return {"score": -(value - 3) ** 2, "drawdown": -0.1}


class ExperimentSearchTests(unittest.TestCase):
    def test_search_persists_and_reuses_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ExperimentStore(Path(temporary) / "runs.sqlite")
            parameters = GridSampler().sample(SearchSpace({"x": IntRange(1, 5)}))
            engine = SearchEngine(store, workers=1)
            first = engine.run(
                "search-1", "data-v1", "code-v1", parameters,
                objective, "score", maximize=True,
                constraints={"drawdown": (">=", -0.2)},
            )
            self.assertEqual(first.best.parameters["x"], 3)
            second = engine.run(
                "search-1", "data-v1", "code-v1", parameters,
                objective, "score", maximize=True,
            )
            self.assertTrue(all(item.reused for item in second.trials))
            store.close()


if __name__ == "__main__":
    unittest.main()


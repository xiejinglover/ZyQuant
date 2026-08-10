"""Fixed local search benchmark.

Run with: python benchmarks/benchmark_search.py
The benchmark reports a warning rather than failing because CI hardware varies.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

from zyquant.experiment import ExperimentStore
from zyquant.optimize import SearchEngine


def cpu_objective(parameters):
    value = float(parameters["value"])
    accumulator = 0.0
    for index in range(5_000_000):
        accumulator += math.sin(value + index * 1e-5) ** 2
    return {"score": -abs(accumulator)}


def run(workers: int, database: Path):
    parameters = [{"value": index / 10} for index in range(24)]
    with ExperimentStore(database) as store:
        started = time.perf_counter()
        SearchEngine(store, workers=workers).run(
            f"benchmark-{workers}", "benchmark-data-v1", "benchmark-code-v1",
            parameters, cpu_objective, "score",
        )
        return time.perf_counter() - started


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        serial = run(1, root / "serial.sqlite")
        parallel = run(4, root / "parallel.sqlite")
    speedup = serial / parallel
    status = "PASS" if speedup >= 3.0 else "WARN"
    print({
        "serial_seconds": serial,
        "parallel_seconds": parallel,
        "speedup": speedup,
        "target": 3.0,
        "status": status,
    })


if __name__ == "__main__":
    main()

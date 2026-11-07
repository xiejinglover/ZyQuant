# LLM-assisted factor discovery

ZyQuant's discovery workflow separates idea generation from execution. The LLM
can propose an economic hypothesis and a strictly validated expression tree; it
cannot execute code, access files, change statistical thresholds, or see holdout
feedback. Exact duplicates are rejected. Highly correlated candidates remain in
the fixed denominator and are reported as diagnostics.

Install the optional OpenAI client:

```bash
python -m pip install 'zyquant[ai]'
```

Keep the API key outside configuration and source control:

```bash
export OPENAI_API_KEY='configured-on-the-host'
```

The server-local adapter named by `adapter: package.module:create_adapter` must
implement:

```python
class Adapter:
    def metadata(self) -> dict: ...
    def load_period(self, period: str, start: date, end: date) -> DiscoveryDataset: ...
```

`metadata()` identifies the PIT dataset and pool. `load_period()` returns the
standard observations, label metadata, optional exposures, and one long value
frame per whitelisted field. Adapters map existing production semantics; they
must not reimplement or approximate the strategy label.

Create and execute an immutable plan:

```bash
zyq factors discover plan --config campaign.yaml --output-dir runs/plan
zyq factors discover run --plan-dir runs/plan --output-dir runs/campaign
zyq factors discover resume --plan-dir runs/plan --run-dir runs/campaign
zyq factors discover report --run-dir runs/campaign
```

The config is separate from a normal backtest config. Important fields are the
dataset fingerprint, pool adapter, field whitelist, three disjoint time windows,
label contract, LLM settings, candidate budget, evidence thresholds, and free
disk guard. Use `provider: replay` with a JSON replay file for CI and offline
validation.

When a wheel is validated outside its source checkout, set the non-secret
`ZYQUANT_BUILD_COMMIT` environment variable to the exact commit used to build
the wheel. The frozen plan records it instead of an unrelated surrounding
checkout revision.

Candidate values are stored as campaign-local Parquet shards. Only manually
promoted `ExpressionFactor` objects enter the canonical ZyQuant factor cache.
The report distinguishes `evidence_only`, `model_candidate`, and
`economic_eligible`; none of these states changes a production strategy.

# RD-Agent research without Qlib

ZyQuant 2.2 uses the upstream RD-Agent research loop for hypothesis, reflection,
memory, and feedback while keeping all data access and experiment execution inside
ZyQuant. The integration is deliberately process-isolated:

```text
RD-Agent worker (Python 3.11)
        | immutable JSON request/result contracts
        v
zyq research agent worker
        | PIT data, safe expression evaluator, frozen model/account adapters
        v
ZyQuant run directory
```

The worker is pinned to RD-Agent `v1.0.0` at commit
`484776c211e4fbbeef03e0ec00d6bbee7362a4f4`. It must not import `qlib` or any
`rdagent.app.qlib_*` module. RD-Agent is not a dependency of the public ZyQuant
wheel; install it only in the dedicated worker environment.

## Commands

```bash
zyq research agent plan --config campaign.yaml --output-dir runs/plan
zyq research agent run --plan-dir runs/plan --output-dir runs/campaign
zyq research agent resume --run-dir runs/campaign
zyq research agent report --run-dir runs/campaign
zyq research agent verify --run-dir runs/campaign
```

Planning freezes the dataset fingerprint, strategy and label contracts, fields,
annual vintages, fixed seeds, model recipe bounds, adapter argv, prompt/schema hashes,
and resource guards. Completed directories are immutable; resumption only settles
unfinished entries from the frozen denominator.

## Research boundaries

- Factor proposals are `ExpressionSpec` trees. Python source, imports, file access,
  network access, custom objectives, and dynamic execution are not accepted.
- Model proposals are validated `XGBRanker` or `XGBRegressor` recipes. Seeds, labels,
  universes, execution rules, and account adapters are frozen by the campaign plan.
- Each annual vintage may use only factors known at its information cutoff and labels
  whose actual exit time is no later than that cutoff.
- Candidate-level development evidence may be returned to the agent. Diagnostic and
  reference results are written only after routes are locked and are never included
  in agent feedback.
- Monotonicity, HAC, confidence intervals, and FDR remain reported evidence; they are
  not short-period hard gates for model candidacy.

## Worker environment

Create a separate Python 3.11 environment and install the exact upstream revision.
Do not install `pyqlib`:

```bash
python3.11 -m venv /absolute/path/zyquant-rdagent-1.0
/absolute/path/zyquant-rdagent-1.0/bin/pip install \
  "git+https://github.com/microsoft/RD-Agent.git@484776c211e4fbbeef03e0ec00d6bbee7362a4f4"
/absolute/path/zyquant-rdagent-1.0/bin/pip install zyquant-2.2.0-py3-none-any.whl
/absolute/path/zyquant-rdagent-1.0/bin/python -c \
  "import importlib.util; assert importlib.util.find_spec('qlib') is None"
```

`OPENAI_API_KEY` is read from the worker environment only. It is never serialized to
the frozen plan, trace, logs, reports, or manifests.

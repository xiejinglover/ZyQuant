# Changelog

## 2.2.0

- Add a structured RD-Agent research bridge for joint factor and model experiments.
- Keep RD-Agent in an isolated Python 3.11 worker and communicate through immutable
  JSON/Parquet contracts; Qlib is neither installed nor imported.
- Add resumable `zyq research agent` planning, execution, reporting, and verification.
- Preserve strict PIT cutoffs: only development folds may be returned to the agent;
  diagnostic and previously observed reference periods are evaluated after route lock.
- Add fixed-seed XGBRanker/XGBRegressor recipes and safe, declarative factor expressions.

## 2.1.0

- Add a closed, validated factor-expression AST and sparse pool-local evaluator.
- Add promotion of reviewed expressions into the existing `BaseFactor` cache workflow.
- Add OpenAI Responses API structured output plus deterministic replay support.
- Add immutable discovery plans, crash-safe resumability, and development-only feedback.
- Add rolling RankIC, spread, HAC, BH-FDR, stability, exposure, and redundancy diagnostics.
- Add deterministic factor-level process parallelism for candidate and locked-period evidence.
- Add `zyq factors discover plan|run|resume|report` without changing existing commands.
- Keep OpenAI optional through the `ai` extra and read credentials only from the host environment.

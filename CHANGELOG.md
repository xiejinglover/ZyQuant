# Changelog

## 2.1.0

- Add a closed, validated factor-expression AST and sparse pool-local evaluator.
- Add promotion of reviewed expressions into the existing `BaseFactor` cache workflow.
- Add OpenAI Responses API structured output plus deterministic replay support.
- Add immutable discovery plans, crash-safe resumability, and development-only feedback.
- Add rolling RankIC, spread, HAC, BH-FDR, stability, exposure, and redundancy diagnostics.
- Add deterministic factor-level process parallelism for candidate and locked-period evidence.
- Add `zyq factors discover plan|run|resume|report` without changing existing commands.
- Keep OpenAI optional through the `ai` extra and read credentials only from the host environment.

# ML System Test Plan — ChatDFT

Senior ML Engineer onboarding checklist: 10 test categories, 60+ individual tests.

## Test Categories

| # | Category | What It Proves | Priority |
|---|---|---|---|
| 1 | **Environment & Setup** | Can a new engineer clone + run in <10 min? | P0 |
| 2 | **Data Integrity** | Are DB schemas correct? Feature stores valid? | P0 |
| 3 | **Model Correctness** | Do algorithms produce mathematically correct results? | P0 |
| 4 | **Pipeline Integration** | Do components chain correctly end-to-end? | P0 |
| 5 | **Determinism & Reproducibility** | Same input → same output across runs? | P1 |
| 6 | **Edge Cases & Robustness** | Does it crash on empty/malformed/adversarial input? | P1 |
| 7 | **Performance & Latency** | Are algorithms fast enough for interactive use? | P1 |
| 8 | **MLOps Infrastructure** | Model registry, experiment tracker, monitoring — do they work? | P1 |
| 9 | **Security & Input Validation** | Prompt injection, SQL injection, file path traversal? | P2 |
| 10 | **Regression & Drift** | Do metrics hold across code changes? | P2 |

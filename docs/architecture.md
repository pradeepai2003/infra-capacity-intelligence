# Architecture

```
 ┌─────────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
 │ 1. Data Generation   │     │ 2. Databricks-style │     │ 3. Forecasting      │
 │ (synthetic, until    │ --> │    Cleansing +      │ --> │    Linear Reg. +    │
 │  Azure Monitor is    │     │    Trend Analysis   │     │    Prophet          │
 │  connected)          │     │    + Time-Series    │     │    (4wk / 12wk)     │
 │                       │     │    Prep              │     │                     │
 └─────────────────────┘     └────────────────────┘     └──────────┬──────────┘
                                                                      │
                                                                      v
 ┌─────────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
 │ 6. Power BI          │ <-- │ 5. Dataset Export    │ <-- │ 4. Recommendation   │
 │    Dashboard         │     │    (CSV for Power BI)│     │    Engine (rules)   │
 │                       │     │                      │     │    + AI Narrative   │
 │                       │     │                      │     │    (Gemini/Ollama)  │
 └─────────────────────┘     └────────────────────┘     └─────────────────────┘
```

## Stage-by-stage

1. **Data Generation** (`src/data_generation/`) -- produces realistic
   synthetic compute/storage/network time series plus three seeded scenarios
   (capacity shortfall, chronic waste, seasonal spike) used to validate the
   recommendation engine deterministically. This stage is designed to be
   swapped for real Azure Monitor data with no changes downstream (see
   `src/pipeline/azure_functions/scheduled_collector/`).

2. **Cleansing / Trend Analysis** (`src/databricks/notebooks/`) -- four
   notebook-style scripts (numbered to preserve execution order) that
   dedupe, validate, aggregate to daily granularity, and compute rolling
   trend indicators (7/30-day moving averages, growth rate). Written to run
   standalone with pandas for local dev/CI, and to import cleanly into an
   actual Databricks workspace.

3. **Forecasting** (`src/forecasting/`) -- runs both a Linear Regression
   baseline and Prophet (weekly/yearly seasonality) for every resource, at
   4-week and 12-week horizons. Prophet gracefully falls back to Linear
   Regression if not installed, so CI never hard-fails on an optional
   dependency.

4. **Recommendation Engine** (`src/recommendation_engine/`) -- a
   threshold-based rule engine (`rule_engine.py`) turns forecasts into
   structured `Recommendation` objects (increase storage / downsize compute /
   decommission / enable autoscaling / no action), then
   `ai_narrative_generator.py` turns each into a human-readable sentence via
   Gemini or Ollama, with a deterministic template fallback.

5. **Dataset Export** (`src/powerbi/dataset_export.py`) -- flattens the
   trend + recommendation data into three CSVs Power BI Desktop can read
   directly via "Get Data -> Folder".

6. **Dashboard** -- built manually once in Power BI Desktop against those
   CSVs (see `src/powerbi/refresh_config.json` for the recommended visual
   layout), then refreshed automatically by the `data-pipeline.yml` /
   `deploy-dashboard.yml` GitHub Actions workflows.

## CI/CD

`src/pipeline/run_pipeline.py` is the single entrypoint that chains stages
1-5. GitHub Actions (`.github/workflows/`) runs the full pytest suite on
every push/PR, and runs the pipeline itself on a daily schedule (plus manual
trigger), committing refreshed Power BI datasets back to the repo.

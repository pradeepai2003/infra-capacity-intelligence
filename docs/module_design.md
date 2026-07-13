# Module Design

| Path | Purpose |
|---|---|
| `config/config.yaml` | Central config: date ranges, thresholds, forecast horizons, provider selection |
| `config/logging.yaml` | Logging format/handlers |
| `config/.env.example` | Template for API keys / connection strings (never commit the real `.env`) |
| `src/data_generation/schema.py` | Shared column definitions + validation for compute/storage/network data |
| `src/data_generation/generate_compute_metrics.py` | Synthetic CPU/memory/cluster utilization, incl. deliberately underutilized clusters |
| `src/data_generation/generate_storage_metrics.py` | Synthetic disk usage with realistic growth trends |
| `src/data_generation/generate_network_metrics.py` | Synthetic bandwidth/throughput/latency with a seasonal spike window |
| `src/data_generation/scenario_seeder.py` | Deterministic seeded scenarios for validating the 3 required cases |
| `src/databricks/notebooks/01_ingest_raw_data.py` | Reads + schema-validates raw CSVs |
| `src/databricks/notebooks/02_data_cleansing.py` | Dedup, missing-value handling, range clipping |
| `src/databricks/notebooks/03_aggregation_trend_analysis.py` | Daily aggregation + rolling trend indicators |
| `src/databricks/notebooks/04_time_series_prep.py` | Reshapes to per-resource `ds`/`y`/`t` series for ML |
| `src/databricks/jobs/databricks_job_config.json` | Databricks Jobs API task graph + daily schedule |
| `src/forecasting/linear_regression_model.py` | Fast linear-trend baseline forecaster |
| `src/forecasting/prophet_model.py` | Seasonality-aware forecaster, falls back to linear regression if Prophet isn't installed |
| `src/forecasting/forecast_runner.py` | Orchestrates both models across all resources/horizons |
| `src/recommendation_engine/rule_engine.py` | Threshold-based logic -> `Recommendation` objects |
| `src/recommendation_engine/recommendation_schema.py` | Dataclass <-> DataFrame/dict serialization |
| `src/recommendation_engine/ai_narrative_generator.py` | Gemini/Ollama narrative generation with template fallback |
| `src/recommendation_engine/prompts/narrative_prompt_template.txt` | Prompt template for the narrative generator |
| `src/pipeline/run_pipeline.py` | End-to-end orchestrator (single entrypoint) |
| `src/pipeline/azure_functions/scheduled_collector/` | Timer-triggered Azure Function stub for future real-data collection |
| `src/powerbi/dataset_export.py` | Writes final CSVs for Power BI Desktop |
| `src/powerbi/refresh_config.json` | Documents data source + recommended visuals for the dashboard |
| `tests/**` | 1:1 mirror of `src/**`; run via `pytest tests/ --cov=src` |
| `.github/workflows/ci.yml` | Lint (flake8/black) + full test suite on every push/PR |
| `.github/workflows/data-pipeline.yml` | Scheduled/manual full pipeline run, commits refreshed Power BI CSVs |
| `.github/workflows/deploy-dashboard.yml` | Optional Power BI Service refresh trigger |

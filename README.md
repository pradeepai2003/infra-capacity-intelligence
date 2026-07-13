# Infrastructure Capacity Intelligence Platform

AI-powered platform that turns reactive infrastructure capacity planning into
proactive, data-driven decisions -- synthetic data generation, Databricks-style
cleansing, ML forecasting (Linear Regression + Prophet), an AI-narrated
recommendation engine (Gemini/Ollama), and Power BI visualization, wired
together with a GitHub Actions CI/CD pipeline.

See `docs/problem_statement.md` and `docs/architecture.md` for background.

## Project layout

See `docs/module_design.md` for a full breakdown of every module.

## Setup (VS Code / local)

```bash
git clone <your-repo-url>
cd infra-capacity-intelligence

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt

cp config/.env.example .env      # fill in GEMINI_API_KEY etc. if using AI narratives
```

Open the folder in VS Code -- `.vscode/settings.json` is already configured to
run pytest and flake8/black through the Python extension (recommended
extensions are listed in `.vscode/extensions.json`).

## Running the pipeline end-to-end

```bash
python -m src.pipeline.run_pipeline
```

This will:
1. Generate synthetic compute/storage/network metrics + the 3 seeded scenarios
2. Run the Databricks-style cleansing/aggregation/time-series-prep steps locally
3. Forecast 4-week and 12-week horizons per resource (Linear Regression + Prophet)
4. Generate rule-based recommendations + AI narratives (falls back to a
   deterministic template if no `GEMINI_API_KEY`/Ollama server is configured)
5. Export `src/powerbi/*.csv` -- open these in Power BI Desktop via
   **Get Data -> Folder** to build the dashboard described in
   `src/powerbi/refresh_config.json`

## Running individual stages

```bash
python -m src.data_generation.generate_compute_metrics
python -m src.data_generation.generate_storage_metrics
python -m src.data_generation.generate_network_metrics
python -m src.data_generation.scenario_seeder
```

Databricks notebooks can be run locally the same way (they're plain `.py`
files with `# Databricks notebook source` / `# COMMAND ----------` markers so
they also import cleanly into an actual Databricks workspace via Repos):

```bash
python src/databricks/notebooks/01_ingest_raw_data.py
python src/databricks/notebooks/02_data_cleansing.py
python src/databricks/notebooks/03_aggregation_trend_analysis.py
python src/databricks/notebooks/04_time_series_prep.py
```

## Running tests

```bash
pytest tests/ --cov=src --cov-report=term-missing -v
```

Every file under `src/` has a matching test file under `tests/` (same
sub-path), so `--cov=src` gives per-module coverage. This mirrors what
`.github/workflows/ci.yml` runs on every push/PR.

## CI/CD (GitHub Actions)

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | every push/PR | flake8 + black + full pytest suite w/ coverage |
| `data-pipeline.yml` | daily cron + manual + push to `src/`/`config/` | runs the full pipeline, commits refreshed Power BI CSVs |
| `deploy-dashboard.yml` | after `data-pipeline.yml` succeeds (or manual) | optional Power BI Service dataset refresh via REST API |

Add these repo secrets if/when you connect real services:
`GEMINI_API_KEY`, `OLLAMA_HOST`, `POWERBI_WORKSPACE_ID`, `POWERBI_DATASET_ID`,
`POWERBI_ACCESS_TOKEN`.

## Moving from synthetic to real data

Everything downstream of `data/raw/*.csv` is agnostic to where that data came
from. To switch to real infrastructure data:
1. Fill in Azure credentials in `.env` (see `config/.env.example`)
2. Implement the commented-out Azure Monitor Query calls in
   `src/pipeline/azure_functions/scheduled_collector/function_app.py`
3. Deploy that function to Azure (timer-triggered, see `function.json`)
4. Point `src/databricks/notebooks/01_ingest_raw_data.py` at the resulting
   Blob Storage path instead of `data/raw/`

No other module needs to change, since they all consume the same schema
defined in `src/data_generation/schema.py`.

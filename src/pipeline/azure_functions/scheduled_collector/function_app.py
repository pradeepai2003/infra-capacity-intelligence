"""
Timer-triggered Azure Function that will eventually replace the synthetic
data generator once real Azure Monitor access is available. Runs on a
schedule (see function.json), pulls metrics via Azure Monitor Query SDK,
and writes them to Blob Storage in the same schema the rest of the pipeline
expects (see src/data_generation/schema.py).

This is a stub: azure.monitor.query calls are commented out since no live
Azure Monitor workspace is connected yet. Swap in real credentials via
config/.env once available.
"""

from __future__ import annotations

import datetime
import logging

import azure.functions as func

app = func.FunctionApp()

logger = logging.getLogger(__name__)


@app.function_name(name="ScheduledMetricsCollector")
@app.timer_trigger(schedule="0 0 * * * *", arg_name="mytimer", run_on_startup=False)
def scheduled_collector(mytimer: func.TimerRequest) -> None:
    """Runs hourly. Collects compute/storage/network metrics from Azure Monitor
    and writes them to Blob Storage as CSV, matching the schema in
    src/data_generation/schema.py so downstream Databricks jobs need no changes.
    """
    utc_now = datetime.datetime.utcnow().isoformat()

    if mytimer.past_due:
        logger.warning("Timer is past due at %s", utc_now)

    logger.info("ScheduledMetricsCollector triggered at %s", utc_now)

    # --- Real implementation sketch (uncomment once Azure Monitor is wired up) ---
    # from azure.identity import DefaultAzureCredential
    # from azure.monitor.query import MetricsQueryClient
    # from azure.storage.blob import BlobServiceClient
    # import os, pandas as pd
    #
    # credential = DefaultAzureCredential()
    # metrics_client = MetricsQueryClient(credential)
    # response = metrics_client.query_resource(
    #     resource_uri="<compute-cluster-resource-id>",
    #     metric_names=["Percentage CPU", "Available Memory Bytes"],
    #     timespan=datetime.timedelta(hours=1),
    # )
    # df = _metrics_response_to_dataframe(response)
    #
    # blob_service = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
    # container = blob_service.get_container_client(os.environ["AZURE_STORAGE_CONTAINER"])
    # container.upload_blob(f"compute_metrics_{utc_now}.csv", df.to_csv(index=False), overwrite=True)

    logger.info("ScheduledMetricsCollector completed (stub - no live Azure Monitor connection configured).")

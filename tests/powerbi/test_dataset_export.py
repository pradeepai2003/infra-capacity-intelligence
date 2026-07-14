import pandas as pd

from src.powerbi.dataset_export import export_for_powerbi


def _sample_trends() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    compute = pd.DataFrame(
        {
            "cluster_id": ["cluster-01"] * 3,
            "date": dates,
            "cluster_utilization_pct": [40.0, 42.0, 45.0],
        }
    )
    storage = pd.DataFrame(
        {
            "storage_id": ["storage-01"] * 3,
            "date": dates,
            "disk_utilization_pct": [60.0, 62.0, 65.0],
        }
    )
    network = pd.DataFrame(
        {
            "link_id": ["network-01"] * 3,
            "date": dates,
            "bandwidth_mbps": [1000] * 3,
            "throughput_mbps": [300.0, 320.0, 340.0],
        }
    )
    return {"compute": compute, "storage": storage, "network": network}


def _sample_recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "resource_id": "storage-01",
                "resource_type": "storage",
                "recommendation_type": "increase_storage_capacity",
                "risk_level": "critical",
                "current_value": 82.0,
                "forecasted_value": 95.0,
                "forecast_horizon_weeks": 10,
                "details": {"threshold": 90},
                "ai_narrative": "Storage is projected to reach capacity soon.",
            },
            {
                "resource_id": "cluster-01",
                "resource_type": "compute",
                "recommendation_type": "no_action",
                "risk_level": "info",
                "current_value": 45.0,
                "forecasted_value": 45.0,
                "forecast_horizon_weeks": 0,
                "details": {},
                "ai_narrative": "Cluster is operating normally.",
            },
        ]
    )


def test_export_for_powerbi_writes_all_three_csvs(tmp_path):
    output_dir = str(tmp_path)
    export_for_powerbi(_sample_trends(), _sample_recommendations(), output_dir=output_dir)

    overview = pd.read_csv(f"{output_dir}/utilization_overview.csv")
    recommendations = pd.read_csv(f"{output_dir}/recommendations.csv")
    risk_summary = pd.read_csv(f"{output_dir}/risk_summary.csv")

    assert set(overview["resource_type"]) == {"compute", "storage", "network"}
    assert len(overview) == 9  # 3 resource types x 3 days each
    assert len(recommendations) == 2
    assert not risk_summary.empty


def test_export_for_powerbi_computes_network_utilization_pct(tmp_path):
    output_dir = str(tmp_path)
    export_for_powerbi(_sample_trends(), _sample_recommendations(), output_dir=output_dir)

    overview = pd.read_csv(f"{output_dir}/utilization_overview.csv")
    network_rows = overview[overview["resource_type"] == "network"]
    # throughput_mbps / bandwidth_mbps * 100 -> e.g. 300/1000*100 = 30.0
    assert network_rows["utilization_pct"].iloc[0] == 30.0


def test_export_for_powerbi_risk_summary_counts_by_level(tmp_path):
    output_dir = str(tmp_path)
    export_for_powerbi(_sample_trends(), _sample_recommendations(), output_dir=output_dir)

    risk_summary = pd.read_csv(f"{output_dir}/risk_summary.csv")
    critical_row = risk_summary[
        (risk_summary["resource_type"] == "storage") & (risk_summary["risk_level"] == "critical")
    ]
    assert critical_row["count"].iloc[0] == 1


def test_export_for_powerbi_handles_empty_recommendations(tmp_path):
    output_dir = str(tmp_path)
    empty_recs = pd.DataFrame(columns=["resource_id", "resource_type", "risk_level"])

    export_for_powerbi(_sample_trends(), empty_recs, output_dir=output_dir)

    risk_summary = pd.read_csv(f"{output_dir}/risk_summary.csv")
    assert risk_summary.empty

import pandas as pd

from src.powerbi.dataset_export import export_for_powerbi


def _sample_trends() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "mac_id": ["mac-01"] * 3 + ["mac-02"] * 3,
            "project_name": ["Project Atlas"] * 3 + ["Project Nova"] * 3,
            "resource_type": ["cpu"] * 3 + ["ram"] * 3,
            "date": list(dates) * 2,
            "utilization_pct": [40.0, 42.0, 45.0, 60.0, 62.0, 65.0],
        }
    )


def _sample_recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mac_id": "mac-01",
                "resource_type": "disk",
                "project_name": "Project Atlas",
                "recommendation_type": "increase_allocation",
                "risk_level": "critical",
                "current_value": 82.0,
                "forecasted_value": 95.0,
                "forecast_horizon_weeks": 10,
                "details": {"threshold": 90},
                "ai_narrative": "Disk nearly full.",
            },
            {
                "mac_id": "mac-02",
                "resource_type": "cpu",
                "project_name": "Project Nova",
                "recommendation_type": "no_action",
                "risk_level": "info",
                "current_value": 45.0,
                "forecasted_value": 45.0,
                "forecast_horizon_weeks": 0,
                "details": {},
                "ai_narrative": "Operating normally.",
            },
        ]
    )


def _sample_equalization() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "resource_type": "cpu",
                "overloaded_mac_id": "mac-01",
                "overloaded_project": "Project Atlas",
                "overloaded_utilization_pct": 90.0,
                "underloaded_mac_id": "mac-05",
                "underloaded_project": "Project Falcon",
                "underloaded_utilization_pct": 12.0,
                "fleet_average_utilization_pct": 50.0,
                "suggested_action": "Shift load.",
                "risk_level": "critical",
                "ai_narrative": "Rebalance recommended.",
            }
        ]
    )


def test_export_for_powerbi_writes_all_four_csvs(tmp_path):
    output_dir = str(tmp_path)
    export_for_powerbi(_sample_trends(), _sample_recommendations(), _sample_equalization(), output_dir=output_dir)

    overview = pd.read_csv(f"{output_dir}/mac_utilization_overview.csv")
    recommendations = pd.read_csv(f"{output_dir}/recommendations.csv")
    equalization = pd.read_csv(f"{output_dir}/equalization_summary.csv")
    risk_summary = pd.read_csv(f"{output_dir}/risk_summary.csv")

    assert set(overview["mac_id"]) == {"mac-01", "mac-02"}
    assert len(overview) == 6
    assert len(recommendations) == 2
    assert len(equalization) == 1
    assert not risk_summary.empty


def test_export_for_powerbi_overview_has_project_name(tmp_path):
    output_dir = str(tmp_path)
    export_for_powerbi(_sample_trends(), _sample_recommendations(), _sample_equalization(), output_dir=output_dir)

    overview = pd.read_csv(f"{output_dir}/mac_utilization_overview.csv")
    assert "project_name" in overview.columns
    assert set(overview["project_name"]) == {"Project Atlas", "Project Nova"}


def test_export_for_powerbi_risk_summary_counts_by_level(tmp_path):
    output_dir = str(tmp_path)
    export_for_powerbi(_sample_trends(), _sample_recommendations(), _sample_equalization(), output_dir=output_dir)

    risk_summary = pd.read_csv(f"{output_dir}/risk_summary.csv")
    critical_row = risk_summary[(risk_summary["resource_type"] == "disk") & (risk_summary["risk_level"] == "critical")]
    assert critical_row["count"].iloc[0] == 1


def test_export_for_powerbi_handles_empty_recommendations_and_equalization(tmp_path):
    output_dir = str(tmp_path)
    empty_recs = pd.DataFrame(columns=["mac_id", "resource_type", "risk_level"])
    empty_equalization = pd.DataFrame(columns=["resource_type", "overloaded_mac_id"])

    export_for_powerbi(_sample_trends(), empty_recs, empty_equalization, output_dir=output_dir)

    risk_summary = pd.read_csv(f"{output_dir}/risk_summary.csv")
    assert risk_summary.empty

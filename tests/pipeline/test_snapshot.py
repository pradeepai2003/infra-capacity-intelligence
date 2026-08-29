import os

import pandas as pd

from src.pipeline.snapshot import create_dated_snapshot


def _make_cfg(tmp_path, onedrive_dir="", test_report_file=None, coverage_xml_file=None):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    scenarios_dir = tmp_path / "scenarios"
    powerbi_dir = tmp_path / "powerbi"
    snapshot_dir = tmp_path / "runs"
    for d in [raw_dir, processed_dir, scenarios_dir, powerbi_dir]:
        d.mkdir()

    pd.DataFrame({"a": [1, 2]}).to_csv(raw_dir / "compute_metrics.csv", index=False)
    pd.DataFrame({"b": [3, 4]}).to_csv(processed_dir / "recommendations.csv", index=False)
    pd.DataFrame({"c": [5, 6]}).to_csv(scenarios_dir / "capacity_shortfall.csv", index=False)
    pd.DataFrame({"d": [7, 8]}).to_csv(powerbi_dir / "utilization_overview.csv", index=False)

    return {
        "topic_id": "S3-P-07",
        "paths": {
            "raw_data_dir": str(raw_dir),
            "processed_data_dir": str(processed_dir),
            "seeded_scenarios_dir": str(scenarios_dir),
            "powerbi_export_dir": str(powerbi_dir),
            "snapshot_dir": str(snapshot_dir),
            "onedrive_dir": onedrive_dir,
            "test_report_file": test_report_file or "test-report-that-does-not-exist.txt",
            "coverage_xml_file": coverage_xml_file or "coverage-that-does-not-exist.xml",
        },
    }


def test_create_dated_snapshot_folder_name_format(tmp_path):
    cfg = _make_cfg(tmp_path)
    snapshot_path = create_dated_snapshot(cfg, run_date="2026-08-29")

    assert os.path.basename(snapshot_path) == "S3-P-07 - 2026-08-29"
    assert os.path.isdir(snapshot_path)


def test_create_dated_snapshot_copies_all_data_subfolders(tmp_path):
    cfg = _make_cfg(tmp_path)
    snapshot_path = create_dated_snapshot(cfg, run_date="2026-08-29")

    assert os.path.exists(os.path.join(snapshot_path, "raw", "compute_metrics.csv"))
    assert os.path.exists(os.path.join(snapshot_path, "processed", "recommendations.csv"))
    assert os.path.exists(os.path.join(snapshot_path, "seeded_scenarios", "capacity_shortfall.csv"))
    assert os.path.exists(os.path.join(snapshot_path, "powerbi", "utilization_overview.csv"))


def test_create_dated_snapshot_skips_test_report_when_absent(tmp_path):
    cfg = _make_cfg(tmp_path)
    snapshot_path = create_dated_snapshot(cfg, run_date="2026-08-29")

    assert not os.path.exists(os.path.join(snapshot_path, "test_report"))


def test_create_dated_snapshot_includes_test_report_when_present(tmp_path):
    report_file = tmp_path / "test-report.txt"
    report_file.write_text("5 passed in 1.2s")
    coverage_file = tmp_path / "coverage.xml"
    coverage_file.write_text("<coverage></coverage>")

    cfg = _make_cfg(tmp_path, test_report_file=str(report_file), coverage_xml_file=str(coverage_file))
    snapshot_path = create_dated_snapshot(cfg, run_date="2026-08-29")

    assert os.path.exists(os.path.join(snapshot_path, "test_report", "test-report.txt"))
    assert os.path.exists(os.path.join(snapshot_path, "test_report", "coverage.xml"))


def test_create_dated_snapshot_mirrors_to_onedrive_when_configured_and_exists(tmp_path):
    onedrive_dir = tmp_path / "onedrive"
    onedrive_dir.mkdir()

    cfg = _make_cfg(tmp_path, onedrive_dir=str(onedrive_dir))
    create_dated_snapshot(cfg, run_date="2026-08-29")

    mirrored_path = onedrive_dir / "S3-P-07 - 2026-08-29"
    assert mirrored_path.is_dir()
    assert (mirrored_path / "raw" / "compute_metrics.csv").exists()


def test_create_dated_snapshot_skips_onedrive_mirror_when_path_missing(tmp_path):
    # onedrive_dir points somewhere that doesn't exist -- should not raise,
    # should simply skip the mirror (this is the expected CI behavior).
    cfg = _make_cfg(tmp_path, onedrive_dir=str(tmp_path / "nonexistent_onedrive_folder"))
    snapshot_path = create_dated_snapshot(cfg, run_date="2026-08-29")

    assert os.path.isdir(snapshot_path)  # local snapshot still created fine


def test_create_dated_snapshot_no_onedrive_configured(tmp_path):
    cfg = _make_cfg(tmp_path, onedrive_dir="")
    snapshot_path = create_dated_snapshot(cfg, run_date="2026-08-29")
    assert os.path.isdir(snapshot_path)


def test_create_dated_snapshot_excludes_non_csv_files(tmp_path):
    cfg = _make_cfg(tmp_path)
    # Simulate the real powerbi export dir, which also contains source code/config
    # files alongside the actual CSV exports.
    powerbi_dir = tmp_path / "powerbi"
    (powerbi_dir / "dataset_export.py").write_text("# not data")
    (powerbi_dir / "refresh_config.json").write_text("{}")

    snapshot_path = create_dated_snapshot(cfg, run_date="2026-08-29")

    snapshot_powerbi = os.path.join(snapshot_path, "powerbi")
    copied_files = os.listdir(snapshot_powerbi)
    assert "dataset_export.py" not in copied_files
    assert "refresh_config.json" not in copied_files
    assert "utilization_overview.csv" in copied_files


def test_create_dated_snapshot_uses_todays_date_by_default(tmp_path):
    from datetime import datetime

    cfg = _make_cfg(tmp_path)
    snapshot_path = create_dated_snapshot(cfg)
    expected_date = datetime.now().strftime("%Y-%m-%d")
    assert f"S3-P-07 - {expected_date}" in snapshot_path

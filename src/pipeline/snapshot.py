"""
Creates a single, dated "snapshot" folder for one pipeline run, named
"<topic_id> - <YYYY-MM-DD>" (e.g. "S3-P-07 - 2026-08-29"). The snapshot
bundles together:

  - raw/          synthetic data generated this run (compute/storage/network + seeded scenarios)
  - processed/    cleaned data, trend indicators, and recommendations
  - powerbi/      the 3 CSVs exported for Power BI
  - test_report/  pytest output + coverage report, IF a prior CI step produced them
                  (skipped locally if you haven't run pytest with report output)

This keeps every run's data + its accompanying test evidence together as one
unit, rather than overwriting the same few files run after run.

If `onedrive_dir` is configured (only meaningful when running locally on a
machine with the OneDrive desktop client installed), the same dated folder
is also copied there. OneDrive then syncs it to the cloud automatically --
no API keys or app registration needed. This does nothing in CI (cloud
runners have no OneDrive access), which is expected and not an error.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


def _copy_dir_contents(src_dir: str, dest_dir: str, extensions: tuple[str, ...] = (".csv",)) -> int:
    """Copy files matching `extensions` from src_dir into dest_dir (non-recursive, flat).
    Returns the number of files copied. Silently does nothing if src_dir doesn't exist
    or has no matching files.

    Defaults to CSV only, since several source directories (notably powerbi_export_dir)
    mix actual data output with source code / config / README files that don't belong
    in a data snapshot.
    """
    if not os.path.isdir(src_dir):
        return 0
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        if os.path.isfile(src_path) and name.lower().endswith(extensions):
            shutil.copy2(src_path, os.path.join(dest_dir, name))
            count += 1
    return count


def create_dated_snapshot(cfg: dict, run_date: str = None) -> str:
    """
    Args:
        cfg: the loaded config dict (see config/config.yaml)
        run_date: override the date used in the folder name (defaults to
            today, "YYYY-MM-DD"); mainly useful for tests

    Returns:
        the path to the created snapshot folder
    """
    topic_id = cfg.get("topic_id", "S3-P-07")
    run_date = run_date or datetime.now().strftime("%Y-%m-%d")
    folder_name = f"{topic_id} - {run_date}"

    snapshot_root = cfg["paths"].get("snapshot_dir", "runs")
    snapshot_path = os.path.join(snapshot_root, folder_name)
    os.makedirs(snapshot_path, exist_ok=True)

    counts = {
        "raw": _copy_dir_contents(cfg["paths"]["raw_data_dir"], os.path.join(snapshot_path, "raw")),
        "seeded_scenarios": _copy_dir_contents(
            cfg["paths"]["seeded_scenarios_dir"], os.path.join(snapshot_path, "seeded_scenarios")
        ),
        "processed": _copy_dir_contents(cfg["paths"]["processed_data_dir"], os.path.join(snapshot_path, "processed")),
        "powerbi": _copy_dir_contents(cfg["paths"]["powerbi_export_dir"], os.path.join(snapshot_path, "powerbi")),
    }

    # Test report + coverage file, if a prior CI/dev step produced them.
    test_report_dest = os.path.join(snapshot_path, "test_report")
    test_report_file = cfg["paths"].get("test_report_file", "test-report.txt")
    coverage_xml_file = cfg["paths"].get("coverage_xml_file", "coverage.xml")
    report_files_found = 0
    if os.path.isfile(test_report_file):
        os.makedirs(test_report_dest, exist_ok=True)
        shutil.copy2(test_report_file, os.path.join(test_report_dest, os.path.basename(test_report_file)))
        report_files_found += 1
    if os.path.isfile(coverage_xml_file):
        os.makedirs(test_report_dest, exist_ok=True)
        shutil.copy2(coverage_xml_file, os.path.join(test_report_dest, os.path.basename(coverage_xml_file)))
        report_files_found += 1

    logger.info(
        "[SNAPSHOT] %s: %d data files + %d test-report file(s) -> %s",
        folder_name,
        sum(counts.values()),
        report_files_found,
        snapshot_path,
    )

    # Optional local OneDrive mirror -- no-op if not configured or path doesn't exist
    onedrive_dir = cfg["paths"].get("onedrive_dir", "")
    if onedrive_dir:
        if os.path.isdir(onedrive_dir):
            onedrive_dest = os.path.join(onedrive_dir, folder_name)
            shutil.copytree(snapshot_path, onedrive_dest, dirs_exist_ok=True)
            logger.info("[SNAPSHOT] Mirrored to OneDrive-synced folder -> %s", onedrive_dest)
        else:
            logger.warning(
                "[SNAPSHOT] onedrive_dir is set to '%s' but that path doesn't exist on this machine -- "
                "skipping OneDrive mirror (expected in CI; check the path if running locally)",
                onedrive_dir,
            )

    return snapshot_path

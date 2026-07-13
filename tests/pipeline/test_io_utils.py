import os

import pandas as pd

from src.pipeline.io_utils import save_and_log


def test_save_and_log_writes_file(tmp_path, capsys):
    df = pd.DataFrame({"a": [1, 2, 3]})
    out_path = str(tmp_path / "nested" / "out.csv")

    returned_path = save_and_log(df, out_path, "Test data")

    assert os.path.exists(out_path)
    assert returned_path == out_path
    loaded = pd.read_csv(out_path)
    pd.testing.assert_frame_equal(loaded, df)


def test_save_and_log_prints_confirmation(tmp_path, caplog):
    df = pd.DataFrame({"a": [1, 2, 3]})
    out_path = str(tmp_path / "out.csv")

    with caplog.at_level("INFO"):
        save_and_log(df, out_path, "Test data")

    assert "[SAVED]" in caplog.text
    assert "Test data" in caplog.text
    assert "3" in caplog.text  # row count
    assert out_path in caplog.text


def test_save_and_log_creates_parent_directories(tmp_path):
    df = pd.DataFrame({"a": [1]})
    out_path = str(tmp_path / "a" / "b" / "c" / "out.csv")

    save_and_log(df, out_path, "Deeply nested data")

    assert os.path.exists(out_path)

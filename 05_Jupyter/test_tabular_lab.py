"""Offline smoke/contract tests for the three self-contained applied notebooks.

Run: python -m unittest discover -s 05_Jupyter -p 'test_tabular_lab.py'
Uses the same numpy/pandas/sklearn/matplotlib prerequisites as the notebooks.
"""
import contextlib
import io
import json
import os
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tabular-lab-mpl"))

import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parent
NAMES = (
    "05k_Forecast_Bike_Demand.ipynb",
    "05l_Detect_Telemetry_Anomalies.ipynb",
    "05n_Clean_Messy_Data.ipynb",
)


def execute(name, cwd):
    notebook = json.loads((LAB_DIR / name).read_text())
    scope = {}
    previous = Path.cwd()
    try:
        os.chdir(cwd)
        # The default run must remain offline, even when requests are possible.
        with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Unexpected network request")):
            with contextlib.redirect_stdout(io.StringIO()):
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), f"{name}:{index}", "exec"), scope)
    finally:
        os.chdir(previous)
        plt.close("all")
    return scope


class TabularNotebookTests(unittest.TestCase):
    def test_offline_root_and_lesson_launch(self):
        for cwd in (LAB_DIR.parent, LAB_DIR):
            for name in NAMES:
                with self.subTest(cwd=cwd.name, notebook=name):
                    scope = execute(name, cwd)
                    self.assertEqual(scope["LAB_DIR"], LAB_DIR)

    def test_bike_forecast_contract(self):
        scope = execute(NAMES[0], LAB_DIR)
        self.assertFalse(scope["USE_REAL_DATA"])
        self.assertNotIn("cnt", scope["X"].columns)
        self.assertNotIn("registered", scope["X"].columns)
        self.assertNotIn("casual", scope["X"].columns)
        scaler = scope["selected_model"].named_steps["standardscaler"]
        self.assertEqual(scaler.n_samples_seen_, scope["n_train"])
        self.assertEqual(len(scope["test_prediction"]), 147)
        # Exercise the optional archive parser with a local fixture, not a request.
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("day.csv", "dteday,cnt\n2011-01-01,42\n")
        class FakeOpener:
            def open(self, url, timeout):
                assert url == "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
                assert timeout == 20
                return io.BytesIO(payload.getvalue())
        with patch("urllib.request.build_opener", return_value=FakeOpener()):
            loaded = scope["load_uci_daily"]()
        self.assertEqual(loaded.cnt.tolist(), [42])

    def test_telemetry_training_and_evaluation(self):
        scope = execute(NAMES[1], LAB_DIR)
        self.assertEqual(int(scope["train"].is_anomaly.sum()), 0)
        self.assertEqual(sum(scope[key] for key in ("tn", "fp", "fn", "tp")), 300)
        self.assertNotIn("is_anomaly", scope["FEATURES"])
        self.assertGreaterEqual(scope["threshold"], scope["validation_score"].min())
        self.assertLessEqual(scope["threshold"], scope["validation_score"].max())

    def test_cleaning_policy(self):
        scope = execute(NAMES[2], LAB_DIR)
        clean = scope["clean"].set_index("order_id")
        with self.assertRaises(sqlite3.ProgrammingError):
            scope["connection"].execute("SELECT 1")  # Connection really closed.
        self.assertEqual(clean.loc["A02", "utc_day"], "2025-06-02")
        self.assertNotIn("A06", clean.index)
        self.assertNotIn("A13", clean.index)
        self.assertEqual(clean.loc["A07", "product"], "unknown")
        self.assertEqual(scope["price_to_cents"]("1.005", "usd"), None)
        self.assertEqual(scope["price_to_cents"]("NaN", "usd"), None)
        self.assertEqual(scope["price_to_cents"]("12.50", "usd"), 1250)


if __name__ == "__main__":
    unittest.main()

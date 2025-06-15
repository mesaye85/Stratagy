import subprocess
import sys
import shutil
from pathlib import Path
import pandas as pd
import os


def test_main_runs_and_creates_merged_csv(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    src_csv = repo_root / "main.csv"
    src_script = repo_root / "main.py"
    # copy needed files into temp dir
    shutil.copy(src_csv, tmp_path / "main.csv")
    shutil.copy(src_script, tmp_path / "main.py")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run([sys.executable, "main.py"], cwd=tmp_path, check=True, env=env)

    merged = tmp_path / "main_merged.csv"
    assert merged.exists(), "Merged CSV not created"
    df = pd.read_csv(merged)
    assert not df.empty

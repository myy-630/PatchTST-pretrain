"""Kernel: A/B test WFDB vs NPY mmap backend.

Runs both experiments sequentially on a Kaggle GPU and writes a comparison
report to /kaggle/working/BACKEND_TRAINING_AB_REPORT.md.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def first_existing(name, candidates):
    for item in candidates:
        path = Path(item)
        if path.exists():
            print(f"{name}: {path}")
            return str(path)
    raise FileNotFoundError(f"{name} not found. Tried: {candidates}")


def find_code_dir():
    candidates = [
        "/kaggle/input/datasets/meiyangyang630/codedataset/patchtst-ecg-code",
        "/kaggle/input/datasets/meiyangyang630/codedataset",
        "/kaggle/input/codedataset/patchtst-ecg-code",
        "/kaggle/input/codedataset",
    ]
    for item in candidates:
        path = Path(item)
        if (
            (path / "run.py").exists()
            and (path / "configs" / "patchtst_ssl_pilot_wfdb.yaml").exists()
            and (path / "src").exists()
        ):
            print(f"CODE_DIR: {path}")
            return str(path)

    for path in Path("/kaggle/input").rglob("run.py"):
        root = path.parent
        if (
            (root / "configs" / "patchtst_ssl_pilot_wfdb.yaml").exists()
            and (root / "configs" / "patchtst_ssl_pilot_npy_mmap.yaml").exists()
            and (root / "src").exists()
        ):
            print(f"CODE_DIR: {root}")
            return str(root)

    raise FileNotFoundError("Could not find code directory under /kaggle/input")


def find_data_dir():
    candidates = [
        "/kaggle/input/datasets/meiyangyang630/ecgdatasetmyy/shd-af-clean-data",
        "/kaggle/input/datasets/meiyangyang630/ecgdatasetmyy",
        "/kaggle/input/ecgdatasetmyy/shd-af-clean-data",
        "/kaggle/input/ecgdatasetmyy",
    ]
    for item in candidates:
        path = Path(item)
        if (path / "raw_wfdb").exists() and (path / "records_npy").exists():
            print(f"DATA_DIR: {path}")
            return str(path)

    for root in Path("/kaggle/input").rglob("*"):
        if root.is_dir() and (root / "raw_wfdb").exists() and (root / "records_npy").exists():
            print(f"DATA_DIR: {root}")
            return str(root)

    raise FileNotFoundError("Could not find data directory with raw_wfdb and records_npy under /kaggle/input")


CODE_DIR = find_code_dir()
DATA_DIR = find_data_dir()
sys.path.insert(0, CODE_DIR)

# Kaggle images usually lack wfdb. Keep this explicit in the log.
os.system("pip install wfdb pyyaml -q 2>&1 | tail -1")

EXPERIMENTS = [
    {
        "name": "wfdb",
        "config": f"{CODE_DIR}/configs/patchtst_ssl_pilot_wfdb.yaml",
        "output": "/kaggle/working/outputs/backend_ab/wfdb",
    },
    {
        "name": "npy_mmap",
        "config": f"{CODE_DIR}/configs/patchtst_ssl_pilot_npy_mmap.yaml",
        "output": "/kaggle/working/outputs/backend_ab/npy_mmap",
    },
]

results = {}

for exp in EXPERIMENTS:
    name = exp["name"]
    print(f"\n{'=' * 60}\n  Running: {name}\n{'=' * 60}")

    os.makedirs(exp["output"], exist_ok=True)
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = CODE_DIR if not env.get("PYTHONPATH") else f"{CODE_DIR}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONUNBUFFERED"] = "1"

    ret = subprocess.run([
        sys.executable, f"{CODE_DIR}/run.py",
        "--config", exp["config"],
        "--data-root", DATA_DIR,
        "--output-dir", exp["output"],
    ], cwd=CODE_DIR, env=env, check=False)

    wall = time.time() - t0
    print(f"\n  {name} finished in {wall:.0f}s (exit {ret.returncode})")

    metrics_path = Path(exp["output"]) / "metrics.csv"
    metrics = {}
    if metrics_path.exists():
        import pandas as pd

        df = pd.read_csv(metrics_path)
        if len(df) > 0:
            row = df.iloc[-1]
            metrics = {
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "epoch_time_s": float(row["elapsed_s"]),
                "windows_per_sec": float(row["windows_per_sec"]),
                "grad_norm": float(row["grad_norm"]),
            }

    results[name] = {
        "exit_code": ret.returncode,
        "wall_time_s": wall,
        **metrics,
    }

w = results.get("wfdb", {})
n = results.get("npy_mmap", {})
wps_w = w.get("windows_per_sec", 0)
wps_n = n.get("windows_per_sec", 0)
speedup = wps_n / wps_w if wps_w > 0 else 0

report = f"""# Backend A/B Report - WFDB vs NPY mmap

## Results

| Metric | WFDB | NPY mmap | Ratio |
|---|---|---|---|
| windows/sec | {wps_w:.1f} | {wps_n:.1f} | {speedup:.1f}x |
| epoch time (s) | {w.get('epoch_time_s', 0):.0f} | {n.get('epoch_time_s', 0):.0f} | - |
| wall time (s) | {w.get('wall_time_s', 0):.0f} | {n.get('wall_time_s', 0):.0f} | - |
| train_loss | {w.get('train_loss', 0):.4f} | {n.get('train_loss', 0):.4f} | - |
| val_loss | {w.get('val_loss', 0):.4f} | {n.get('val_loss', 0):.4f} | - |
| grad_norm | {w.get('grad_norm', 0):.4f} | {n.get('grad_norm', 0):.4f} | - |
| exit_code | {w.get('exit_code', -1)} | {n.get('exit_code', -1)} | - |

## Decision

"""
if speedup >= 1.5 and w.get("exit_code") == 0 and n.get("exit_code") == 0:
    report += "**Switch to NPY mmap as default backend for full training.**\n"
elif speedup >= 1.0:
    report += "NPY mmap is faster but verify loss parity before switching.\n"
else:
    report += "WFDB is faster - keep WFDB for now.\n"

Path("/kaggle/working/BACKEND_TRAINING_AB_REPORT.md").write_text(report, encoding="utf-8")
print(report)
print("Done.")

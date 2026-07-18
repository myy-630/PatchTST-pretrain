#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/root/autodl-tmp/shd-af-clean-data}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="/root/autodl-tmp/outputs/backend_ab/${TS}"

cd "$REPO_ROOT"
source .venv/bin/activate

python cloud/autodl/verify_data.py --data-root "$DATA_ROOT" --repo-root "$REPO_ROOT"

mkdir -p "$OUT_ROOT/wfdb" "$OUT_ROOT/npy_mmap"

echo "Running WFDB backend..."
python run.py \
  --config configs/patchtst_ssl_pilot_wfdb.yaml \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/wfdb" \
  2>&1 | tee "$OUT_ROOT/wfdb.log"

echo "Running NPY mmap backend..."
python run.py \
  --config configs/patchtst_ssl_pilot_npy_mmap.yaml \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/npy_mmap" \
  2>&1 | tee "$OUT_ROOT/npy_mmap.log"

python - "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
rows = []
for name in ["wfdb", "npy_mmap"]:
    metrics_path = out / name / "metrics.csv"
    if metrics_path.exists():
        row = pd.read_csv(metrics_path).iloc[-1].to_dict()
        row["backend"] = name
        rows.append(row)

if not rows:
    raise SystemExit("No metrics.csv files found.")

df = pd.DataFrame(rows)
cols = ["backend", "epoch", "train_loss", "val_loss", "elapsed_s", "windows_per_sec", "grad_norm"]
print(df[cols].to_string(index=False))
df[cols].to_csv(out / "backend_ab_summary.csv", index=False)
print(f"summary: {out / 'backend_ab_summary.csv'}")
PY

echo "Backend A/B complete: $OUT_ROOT"


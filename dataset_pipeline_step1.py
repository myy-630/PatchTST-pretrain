#!/usr/bin/env python3
"""
Dataset Pipeline — Step 1: Build valid_windows.csv
====================================================
Merges window_index.csv + quality_mask_v3.csv, filters keep=True,
runs consistency checks, and outputs the core dataset index.

Output:
  valid_windows.csv   — core fields for downstream Dataset construction
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ===========================================================================
# Paths
# ===========================================================================

WINDOW_INDEX = Path(__file__).resolve().parent / "window_index.csv"
QUALITY_MASK = Path(__file__).resolve().parent / "quality_mask_v3.csv"
OUTPUT = Path(__file__).resolve().parent / "valid_windows.csv"
LOG_FILE = Path(__file__).resolve().parent / "dataset_pipeline_step1.log"

# ===========================================================================
# Logging
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step1")


# ===========================================================================
# Consistency checks
# ===========================================================================

def _check_1to1(left: pd.DataFrame, right: pd.DataFrame, key: str) -> dict:
    """Check whether two DataFrames match 1-to-1 on `key`."""
    left_only = set(left[key]) - set(right[key])
    right_only = set(right[key]) - set(left[key])
    dup_left = left[key].duplicated().sum()
    dup_right = right[key].duplicated().sum()
    return {
        "left_only": len(left_only),
        "right_only": len(right_only),
        "dup_left": int(dup_left),
        "dup_right": int(dup_right),
    }


# ===========================================================================
# Main
# ===========================================================================

def run_step1() -> None:
    log.info("=" * 60)
    log.info("Dataset Pipeline — Step 1: Build valid_windows.csv")
    log.info("=" * 60)

    # --- Load ---
    log.info("Loading window_index.csv …")
    windex = pd.read_csv(WINDOW_INDEX, encoding="utf-8-sig", dtype={"record_id": str})

    log.info("Loading quality_mask_v3.csv …")
    qmask = pd.read_csv(QUALITY_MASK, encoding="utf-8-sig")

    # --- Consistency check: 1-to-1 on window_id ---
    ck = _check_1to1(windex, qmask, "window_id")
    log.info("Consistency check: window_id")
    log.info("  window_index only : %d", ck["left_only"])
    log.info("  quality_mask only : %d", ck["right_only"])
    log.info("  dup in window_index: %d", ck["dup_left"])
    log.info("  dup in quality_mask: %d", ck["dup_right"])

    if ck["left_only"] > 0:
        log.warning("  ⚠ %d window_ids in window_index missing from quality_mask",
                     ck["left_only"])
    if ck["right_only"] > 0:
        log.warning("  ⚠ %d window_ids in quality_mask missing from window_index",
                     ck["right_only"])
    if ck["left_only"] == 0 and ck["right_only"] == 0 and ck["dup_left"] == 0 and ck["dup_right"] == 0:
        log.info("  [OK] Perfect 1-to-1 match, no duplicates.")

    # --- Merge ---
    log.info("Merging on window_id (inner join) …")
    merged = windex.merge(qmask, on="window_id", how="inner")
    log.info("Merged: %d rows", len(merged))

    # --- Missing values check ---
    null_counts = merged.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        log.info("Columns with missing values:")
        for col, cnt in null_cols.items():
            log.info("  %s: %d", col, cnt)
    else:
        log.info("✅ No missing values in merged DataFrame.")

    # --- Filter keep=True ---
    before = len(merged)
    valid = merged[merged["keep"] == True].copy()
    after = len(valid)
    dropped = before - after
    log.info("Filter keep=True: %d → %d (dropped %d, keep rate %.2f%%)",
             before, after, dropped, after / before * 100)

    # --- Fail reason breakdown ---
    fail_reasons = merged.loc[~merged["keep"], "fail_reason"]
    log.info("Fail reason breakdown:")
    for reason, cnt in fail_reasons.value_counts().items():
        log.info("  %s: %d", reason, cnt)

    # --- Select & rename output columns ---
    out_cols = {
        "window_id":     "window_id",
        "subject_id":    "subject_id",
        "record_id":     "record_id",
        "start_sample":  "start_sample",
        "end_sample":    "end_sample",
        "start_second":  "start_second",
        "end_second":    "end_second",
        "label":         "af_label",          # AF / Normal / Unlabeled / Other
        "mixed_label":   "mixed_label",
        "fail_reason":   "v3_fail_reason",
        "keep":          "v3_keep",
    }
    valid_out = valid[list(out_cols.keys())].rename(columns=out_cols)

    # --- Final checks ---
    log.info("Final valid_windows.csv checks:")
    log.info("  Rows: %d", len(valid_out))
    log.info("  Columns: %d", len(valid_out.columns))
    dup_wid = valid_out["window_id"].duplicated().sum()
    log.info("  Duplicate window_id: %d", dup_wid)
    na_rows = valid_out.isnull().any(axis=1).sum()
    log.info("  Rows with any NaN: %d", na_rows)

    # --- Label distribution ---
    label_dist = valid_out["af_label"].value_counts()
    log.info("Label distribution:")
    for lbl, cnt in label_dist.items():
        log.info("  %s: %d (%.1f%%)", lbl, cnt, cnt / len(valid_out) * 100)

    # Fix record_id: ensure zero-padded string (pandas may parse "001" → 1)
    valid_out["record_id"] = valid_out["record_id"].apply(
        lambda x: f"{int(x):03d}" if pd.notna(x) else x
    )

    # --- Write ---
    valid_out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    log.info("Wrote valid_windows.csv → %s", OUTPUT)

    # --- Console summary ---
    print("\n" + "=" * 60)
    print("  Dataset Pipeline Step 1 — Done.")
    print("=" * 60)
    print(f"  Input windows : {before:,}")
    print(f"  Valid windows : {after:,}  ({after / before * 100:.2f}%)")
    print(f"  Dropped       : {dropped:,}")
    print(f"  Output        : {OUTPUT}")
    for lbl, cnt in label_dist.items():
        print(f"  {lbl:12s}: {cnt:,}")
    print("=" * 60)
    log.info("Done.")


if __name__ == "__main__":
    run_step1()

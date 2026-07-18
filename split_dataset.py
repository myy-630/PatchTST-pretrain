#!/usr/bin/env python3
"""
Dataset Split Module — Subject-Level Stratified Split
=======================================================
Splits valid_windows.csv into train / val / test with strict
subject-level separation (no subject appears in >1 set).

Supports:
  - Fixed random seed for reproducibility
  - Configurable ratios
  - Label-stratified subject assignment
  - Extensible interface for future datasets (MIT-BIH, CPSC, …)

Output:
  splits/train.csv
  splits/val.csv
  splits/test.csv
  splits/subject_split.json
  splits/split_statistics.md
"""

import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ===========================================================================
# Configuration
# ===========================================================================

VALID_WINDOWS = Path(__file__).resolve().parent / "valid_windows.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "splits"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# When subject_id is NaN, use this column as proxy
SUBJECT_COLUMN = "record_id"       # each WFDB record = one patient
LABEL_COLUMN = "af_label"

# ===========================================================================
# Logging
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("split")


# ===========================================================================
# Core split logic
# ===========================================================================

def stratified_subject_split(
    df: pd.DataFrame,
    subject_col: str,
    label_col: str,
    train_r: float,
    val_r: float,
    test_r: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Split subjects into train/val/test, then assign all their windows.

    Strategy:
      1. Assign each subject a single label (mode / most frequent).
      2. Stratified split on subject labels → train_subjects, val_subjects, test_subjects.
      3. All windows for a subject go to that subject's split.
    """
    assert abs(train_r + val_r + test_r - 1.0) < 0.001, "Ratios must sum to 1"

    rng = np.random.default_rng(seed)

    # --- Step 1: assign one label per subject ---
    subject_labels = df.groupby(subject_col)[label_col].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else s.iloc[0]
    )
    subjects = subject_labels.index.tolist()
    subj_labels = subject_labels.values

    log.info("Subjects: %d", len(subjects))
    for lb, cnt in Counter(subj_labels).items():
        log.info("  %s: %d", lb, cnt)

    # --- Step 2: split subjects into train+val+test ---
    # First split: train vs rest
    train_subj, rest_subj, train_lbl, rest_lbl = train_test_split(
        subjects, subj_labels,
        test_size=(val_r + test_r),
        stratify=subj_labels,
        random_state=seed,
    )

    # Second split: val vs test (adjust ratio: val_r / (val_r + test_r))
    val_ratio_of_rest = val_r / (val_r + test_r)
    val_subj, test_subj = train_test_split(
        rest_subj,
        test_size=(1 - val_ratio_of_rest),
        stratify=rest_lbl,
        random_state=seed,
    )

    train_set = set(train_subj)
    val_set = set(val_subj)
    test_set = set(test_subj)

    log.info("Subject split: train=%d  val=%d  test=%d",
             len(train_set), len(val_set), len(test_set))

    # --- Step 3: assign windows ---
    train_df = df[df[subject_col].isin(train_set)].copy()
    val_df = df[df[subject_col].isin(val_set)].copy()
    test_df = df[df[subject_col].isin(test_set)].copy()

    # Sanity check: no subject overlap
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    log.info("Subject overlap check: PASSED (0 overlap)")

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "train_subjects": sorted(train_set),
        "val_subjects": sorted(val_set),
        "test_subjects": sorted(test_set),
    }


# ===========================================================================
# Statistics & report
# ===========================================================================

def _build_statistics(result: dict) -> str:
    """Generate split_statistics.md."""
    L: list[str] = []
    L.append("# Dataset Split Statistics")
    L.append("")
    L.append(f"> Generated: {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"> Seed: {RANDOM_SEED}  |  "
             f"Ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    L.append(f"> Subject column: `{SUBJECT_COLUMN}`")
    L.append("")

    # Subject counts
    L.append("## 1. Subject Split")
    L.append("")
    L.append("| Split | Subjects |")
    L.append("|---|---|")
    for name in ["train", "val", "test"]:
        L.append(f"| {name} | {len(result[f'{name}_subjects'])} |")
    L.append("")

    # Overlap check
    train_s = set(result["train_subjects"])
    val_s = set(result["val_subjects"])
    test_s = set(result["test_subjects"])
    ov12 = len(train_s & val_s)
    ov13 = len(train_s & test_s)
    ov23 = len(val_s & test_s)
    L.append("## 2. Subject Overlap Check")
    L.append("")
    L.append("| Check | Overlap |")
    L.append("|---|---|")
    L.append(f"| Train ∩ Val | {ov12} |")
    L.append(f"| Train ∩ Test | {ov13} |")
    L.append(f"| Val ∩ Test | {ov23} |")
    L.append(f"| **Status** | {'✅ PASS' if ov12+ov13+ov23==0 else '❌ FAIL'} |")
    L.append("")

    # Window counts
    L.append("## 3. Window Split")
    L.append("")
    L.append("| Split | Windows | % |")
    L.append("|---|---|---|")
    total_win = 0
    for name in ["train", "val", "test"]:
        n = len(result[name])
        total_win += n
    for name in ["train", "val", "test"]:
        n = len(result[name])
        L.append(f"| {name} | {n:,} | {n/total_win*100:.1f}% |")
    L.append(f"| **Total** | **{total_win:,}** | **100%** |")
    L.append("")

    # Label distribution per split
    L.append("## 4. Label Distribution per Split")
    L.append("")
    labels_order = ["AF", "Normal", "Mixed", "Other", "Unlabeled"]
    L.append("| Label | Train | Train% | Val | Val% | Test | Test% |")
    L.append("|---|---|---|---|---|---|---|")
    for lb in labels_order:
        t = int((result["train"]["af_label"] == lb).sum()) if "train" in result else 0
        v = int((result["val"]["af_label"] == lb).sum()) if "val" in result else 0
        te = int((result["test"]["af_label"] == lb).sum()) if "test" in result else 0
        L.append(
            f"| {lb} | {t:,} | {t/len(result['train'])*100:.1f}% | "
            f"{v:,} | {v/len(result['val'])*100:.1f}% | "
            f"{te:,} | {te/len(result['test'])*100:.1f}% |"
        )
    L.append("")

    # Subject lists (compact)
    L.append("## 5. Subject Assignment")
    L.append("")
    for name in ["train", "val", "test"]:
        subs = result[f"{name}_subjects"]
        L.append(f"**{name}** ({len(subs)} subjects): "
                 f"`{', '.join(subs[:20])}{'...' if len(subs)>20 else ''}`")
        L.append("")
    L.append("")
    L.append("> For the full per-subject assignment, see `subject_split.json`.")
    L.append("")

    L.append("---")
    L.append(f"*Generated on {datetime.now().isoformat(timespec='seconds')}*")
    L.append("")

    return "\n".join(L)


# ===========================================================================
# Save outputs
# ===========================================================================

def _save_outputs(result: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    drop_cols = ["v3_fail_reason", "v3_keep", "subject_id"]
    for name in ["train", "val", "test"]:
        path = OUTPUT_DIR / f"{name}.csv"
        out = result[name].drop(columns=[c for c in drop_cols if c in result[name].columns])
        out.to_csv(path, index=False, encoding="utf-8-sig")
        log.info("Wrote %s.csv → %d windows, %d subjects",
                 name, len(result[name]),
                 result[name][SUBJECT_COLUMN].nunique())

    # subject_split.json
    subj_map = {
        "train_subjects": result["train_subjects"],
        "val_subjects": result["val_subjects"],
        "test_subjects": result["test_subjects"],
        "seed": RANDOM_SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "subject_column": SUBJECT_COLUMN,
        "label_column": LABEL_COLUMN,
    }
    json_path = OUTPUT_DIR / "subject_split.json"
    json_path.write_text(json.dumps(subj_map, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    log.info("Wrote subject_split.json")

    # Statistics
    report = _build_statistics(result)
    (OUTPUT_DIR / "split_statistics.md").write_text(report, encoding="utf-8")
    log.info("Wrote split_statistics.md")


# ===========================================================================
# Main entry point — designed to be importable as a module
# ===========================================================================

def run_split(
    input_csv: Path | str = VALID_WINDOWS,
    output_dir: Path | str = OUTPUT_DIR,
    subject_col: str = SUBJECT_COLUMN,
    label_col: str = LABEL_COLUMN,
    train_r: float = TRAIN_RATIO,
    val_r: float = VAL_RATIO,
    test_r: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run the full split pipeline.  Returns the result dict.

    Can be called programmatically with different parameters for
    LOSO / K-Fold / different datasets in the future.
    """
    log.info("=" * 60)
    log.info("Dataset Split — Subject-Level Stratified")
    log.info("Input: %s", input_csv)
    log.info("Ratios: %.0f/%.0f/%.0f  |  Seed: %d",
             train_r * 100, val_r * 100, test_r * 100, seed)
    log.info("=" * 60)

    df = pd.read_csv(input_csv, encoding="utf-8-sig",
                     dtype={subject_col: str})

    # Auto-detect: if subject_id is all NaN, use record_id as proxy
    if subject_col == "subject_id" and df["subject_id"].isna().all():
        log.warning("subject_id is all NaN — using record_id as subject proxy")
        subject_col = "record_id"

    result = stratified_subject_split(
        df, subject_col, label_col, train_r, val_r, test_r, seed
    )
    _save_outputs(result)

    # Console
    print("\n" + "=" * 60)
    print("  Dataset Split — Done.")
    print("=" * 60)
    for name in ["train", "val", "test"]:
        print(f"  {name:5s}: {len(result[name]):,} windows  "
              f"({result[name][subject_col].nunique()} subjects)")
    print(f"  Output: {output_dir}/")
    print("=" * 60)

    return result


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    run_split()

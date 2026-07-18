#!/usr/bin/env python3
"""
Create Pilot Splits — stratified by record_id, proportional to window count.
=============================================================================
Samples from train.csv → pilot_train.csv (20 000 windows)
       from val.csv   → pilot_val.csv   (4 000 windows)

Fixed seed 42.  Record-level stratification ensures each record contributes
proportionally.  No overlap, no test.csv usage.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create pilot splits from train/val")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--val-size", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sample_by_record(df: pd.DataFrame, target: int, rng: np.random.Generator,
                     source_name: str) -> pd.DataFrame:
    """Sample `target` windows, stratified by record_id proportionally.

    Each record gets floor(target * record_windows / total_windows) as quota.
    Remaining slots are filled by sampling from records with the largest
    fractional remainders (Hare quota method).
    """
    rec_counts = df.groupby("record_id").size()
    total = len(df)

    if target >= total:
        print(f"  [{source_name}] target={target} >= total={total} — using all windows")
        return df.copy()

    # Proportional allocation
    quotas = {}
    for rec, cnt in rec_counts.items():
        quotas[rec] = max(1, int(target * cnt / total))  # at least 1 per record

    # Adjust if sum exceeds target (due to min-1 per record)
    allocated = sum(quotas.values())
    if allocated > target:
        # Reduce from largest records
        excess = allocated - target
        sorted_recs = sorted(quotas, key=lambda r: quotas[r], reverse=True)
        for rec in sorted_recs:
            if excess <= 0:
                break
            if quotas[rec] > 1:
                quotas[rec] -= 1
                excess -= 1

    # Fill remaining slots via largest fractional remainder
    allocated = sum(quotas.values())
    shortfall = target - allocated
    if shortfall > 0:
        remainders = {}
        for rec, cnt in rec_counts.items():
            exact = target * cnt / total
            remainders[rec] = exact - quotas[rec]
        sorted_recs = sorted(remainders, key=lambda r: remainders[r], reverse=True)
        for rec in sorted_recs:
            if shortfall <= 0:
                break
            quotas[rec] += 1
            shortfall -= 1

    # Sample
    pieces = []
    for rec, quota in quotas.items():
        rec_df = df[df["record_id"] == rec]
        n_avail = len(rec_df)
        n_pick = min(quota, n_avail)
        idx = rng.choice(n_avail, size=n_pick, replace=False)
        pieces.append(rec_df.iloc[idx])

    result = pd.concat(pieces, ignore_index=True)
    return result


def build_stats(source_df: pd.DataFrame, pilot_df: pd.DataFrame,
                name: str) -> str:
    """Generate markdown comparison table."""
    lines = [f"### {name}", ""]
    lines.append(f"| Metric | Source | Pilot | Δ |")
    lines.append("|---|---|---|---|")

    # Windows
    s = len(source_df); p = len(pilot_df)
    lines.append(f"| Windows | {s:,} | {p:,} | {p-s:,} |")

    # Records
    s_rec = source_df["record_id"].nunique()
    p_rec = pilot_df["record_id"].nunique()
    lines.append(f"| Records | {s_rec} | {p_rec} | {p_rec-s_rec:+d} |")
    lines.append(f"| Record coverage | 100% | {p_rec/s_rec*100:.1f}% | — |")

    # Label distribution
    for lb in ["AF", "Normal", "Unlabeled", "Mixed", "Other"]:
        s_cnt = int((source_df["af_label"] == lb).sum())
        p_cnt = int((pilot_df["af_label"] == lb).sum())
        s_pct = s_cnt / len(source_df) * 100
        p_pct = p_cnt / len(pilot_df) * 100
        delta = p_pct - s_pct
        flag = " ⚠️" if abs(delta) > 3 else ""
        lines.append(
            f"| {lb} | {s_cnt:,} ({s_pct:.1f}%) | "
            f"{p_cnt:,} ({p_pct:.1f}%) | {delta:+.1f}%{flag} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    # Check overwrite
    out1 = out_dir / "pilot_train.csv"
    out2 = out_dir / "pilot_val.csv"
    if (out1.exists() or out2.exists()) and not args.overwrite:
        print(f"ERROR: output files exist in {out_dir}. Use --overwrite to replace.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("Create Pilot Splits")
    print(f"train={args.train_csv}  val={args.val_csv}")
    print(f"target: train={args.train_size}  val={args.val_size}  seed={args.seed}")
    print("=" * 60)

    # Load
    print("\nLoading source CSVs …")
    train_full = pd.read_csv(args.train_csv, encoding="utf-8-sig",
                             dtype={"record_id": str})
    val_full   = pd.read_csv(args.val_csv, encoding="utf-8-sig",
                             dtype={"record_id": str})
    # Verify no test.csv loaded
    print(f"  train: {len(train_full):,} windows, {train_full['record_id'].nunique()} records")
    print(f"  val:   {len(val_full):,} windows, {val_full['record_id'].nunique()} records")

    # Sample
    print("\nSampling …")
    pilot_train = sample_by_record(train_full, args.train_size, rng, "train")
    pilot_val   = sample_by_record(val_full,   args.val_size,   rng, "val")

    # Sort
    pilot_train = pilot_train.sort_values(["record_id", "start_sample"]).reset_index(drop=True)
    pilot_val   = pilot_val.sort_values(["record_id", "start_sample"]).reset_index(drop=True)

    # Checks
    print("\n--- Consistency Checks ---")
    t_wids = set(pilot_train["window_id"])
    v_wids = set(pilot_val["window_id"])
    full_t = set(train_full["window_id"])
    full_v = set(val_full["window_id"])

    assert len(t_wids) == len(pilot_train), "DUPLICATE in pilot_train!"
    assert len(v_wids) == len(pilot_val), "DUPLICATE in pilot_val!"
    assert len(t_wids & v_wids) == 0, "OVERLAP between pilot_train and pilot_val!"
    assert t_wids.issubset(full_t), "pilot_train has windows NOT in train.csv!"
    assert v_wids.issubset(full_v), "pilot_val has windows NOT in val.csv!"
    assert t_wids.isdisjoint(full_v), "pilot_train contains val windows!"
    assert v_wids.isdisjoint(full_t), "pilot_val contains train windows!"

    print(f"  [OK] pilot_train: {len(pilot_train):,} windows, unique: {len(t_wids)}")
    print(f"  [OK] pilot_val:   {len(pilot_val):,} windows, unique: {len(v_wids)}")
    print(f"  [OK] overlap train ∩ val: {len(t_wids & v_wids)}")
    print(f"  [OK] test.csv not accessed")

    # Write CSVs
    pilot_train.to_csv(out1, index=False, encoding="utf-8-sig")
    pilot_val.to_csv(out2, index=False, encoding="utf-8-sig")
    print(f"\nWrote {out1}")
    print(f"Wrote {out2}")

    # Statistics
    print("\n--- Statistics ---")
    stats_content = [
        "# Pilot Split Statistics",
        "",
        f"> Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"> Seed: {args.seed}  |  Train target: {args.train_size}  |  Val target: {args.val_size}",
        "",
        build_stats(train_full, pilot_train, "Train → Pilot Train"),
        build_stats(val_full, pilot_val, "Val → Pilot Val"),
        "---",
        f"*Generated on {datetime.now().isoformat(timespec='seconds')}*",
    ]
    stats_path = out_dir / "pilot_statistics.md"
    stats_path.write_text("\n".join(stats_content), encoding="utf-8")

    # Config
    config = {
        "random_seed": args.seed,
        "train_source": str(Path(args.train_csv).resolve()),
        "val_source": str(Path(args.val_csv).resolve()),
        "pilot_train_size": int(len(pilot_train)),
        "pilot_val_size": int(len(pilot_val)),
        "sampling_method": "stratified_by_record_id_proportional",
        "stratification_column": "record_id",
        "sort_order": "record_id, start_sample",
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    cfg_path = out_dir / "pilot_sampling_config.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Summary
    print(f"\n  pilot_train: {len(pilot_train):,} windows, {pilot_train['record_id'].nunique()} records")
    print(f"  pilot_val:   {len(pilot_val):,} windows, {pilot_val['record_id'].nunique()} records")
    print(f"  Stats: {stats_path}")
    print(f"  Config: {cfg_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

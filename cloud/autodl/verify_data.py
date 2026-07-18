#!/usr/bin/env python3
"""Verify AutoDL data layout before starting ECG SSL training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def require(path: Path, kind: str) -> None:
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"missing directory: {path}")
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/shd-af-clean-data")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    repo_root = Path(args.repo_root)

    require(data_root, "dir")
    require(data_root / "raw_wfdb", "dir")
    require(data_root / "records_npy", "dir")
    require(data_root / "splits", "dir")
    require(data_root / "splits" / "train.csv", "file")
    require(data_root / "splits" / "val.csv", "file")
    require(data_root / "splits" / "test.csv", "file")
    require(repo_root / "splits" / "pilot" / "pilot_train.csv", "file")
    require(repo_root / "splits" / "pilot" / "pilot_val.csv", "file")

    npy_files = sorted((data_root / "records_npy").glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"no .npy files under {data_root / 'records_npy'}")

    arr = np.load(npy_files[0], mmap_mode="r")
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"unexpected npy shape for {npy_files[0]}: {arr.shape}")

    pilot_train = pd.read_csv(repo_root / "splits" / "pilot" / "pilot_train.csv")
    pilot_val = pd.read_csv(repo_root / "splits" / "pilot" / "pilot_val.csv")

    print("data_root", data_root)
    print("raw_wfdb files", len(list((data_root / "raw_wfdb").glob("*"))))
    print("records_npy files", len(npy_files))
    print("sample_npy", npy_files[0].name, arr.shape, arr.dtype)
    print("pilot_train rows", len(pilot_train))
    print("pilot_val rows", len(pilot_val))
    print("verification ok")


if __name__ == "__main__":
    main()


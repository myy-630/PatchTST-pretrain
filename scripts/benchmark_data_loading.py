#!/usr/bin/env python3
"""Benchmark: wfdb vs npy_mmap backend — pure data loading (no model)."""
import argparse, time, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.ecg_dataset import BaseECGDataset, collate_ssl
from torch.utils.data import DataLoader

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="splits/pilot/pilot_train.csv")
    p.add_argument("--backend", default="wfdb")
    p.add_argument("--npy-dir", default="records_npy")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=5000)
    return p.parse_args()

def benchmark(loader, max_win):
    times = []; t0 = time.time(); first_batch = None; total_win = 0
    for i, batch in enumerate(loader):
        bt = time.time() - t0
        times.append(bt); total_win += len(batch["window_id"])
        if i == 0: first_batch = bt
        t0 = time.time()
        if total_win >= max_win: break
    times = np.array(times)
    return {"total_windows": total_win, "total_time_s": sum(times),
            "windows_per_sec": total_win / sum(times) if sum(times) > 0 else 0,
            "first_batch_s": first_batch, "n_batches": len(times),
            "mean_batch_s": times.mean(), "p50_batch_s": np.median(times),
            "p95_batch_s": np.percentile(times, 95)}

def main():
    args = parse_args()
    print(f"Benchmark: backend={args.backend} batch={args.batch_size} workers={args.num_workers}")
    if args.backend == "npy_mmap":
        ds = BaseECGDataset(args.csv, backend="npy_mmap", npy_dir=args.npy_dir)
    else:
        ds = BaseECGDataset(args.csv, backend="wfdb")
    # Use Subset to limit
    from torch.utils.data import Subset
    sub = Subset(ds, range(min(args.max_windows, len(ds))))
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_ssl)
    result = benchmark(loader, args.max_windows)
    for k, v in result.items():
        if isinstance(v, float): print(f"  {k}: {v:.2f}")
        else: print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

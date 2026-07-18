#!/usr/bin/env python3
"""Run PatchTST SSL batch-size sweep on AutoDL.

The script creates temporary configs under the output root, runs each batch
size sequentially, and writes a CSV summary with throughput, loss, exit code,
and parsed peak GPU memory from the training log.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


GPU_RE = re.compile(r"gpu=([0-9.]+)G")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/patchtst_ssl_pilot_wfdb.yaml")
    parser.add_argument("--data-root", default="/root/autodl-tmp/shd-af-clean-data")
    parser.add_argument("--output-root", default="/root/autodl-tmp/outputs/batch_sweep")
    parser.add_argument("--backend", default="wfdb", choices=["wfdb", "npy_mmap"])
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 64, 96, 128, 192, 256, 384, 512])
    parser.add_argument("--mixed-precision", default="fp32", choices=["fp32", "fp16"])
    return parser.parse_args()


def write_config(base_config: Path, out_path: Path, batch_size: int, output_dir: Path, args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["data"]["backend"] = args.backend
    if args.backend == "npy_mmap":
        cfg["data"]["npy_dir"] = "records_npy"
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["epochs"] = 1
    cfg["training"]["mixed_precision"] = args.mixed_precision
    cfg["checkpoint"]["save_dir"] = str(output_dir)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def parse_gpu_gb(log_text: str) -> float | None:
    matches = GPU_RE.findall(log_text)
    if not matches:
        return None
    return max(float(item) for item in matches)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    base_config = repo_root / args.base_config
    output_root = Path(args.output_root) / f"{args.backend}_{args.mixed_precision}_{time.strftime('%Y%m%d_%H%M%S')}"
    config_root = repo_root / ".tmp_batch_sweep" / f"{args.backend}_{args.mixed_precision}_{time.strftime('%Y%m%d_%H%M%S')}"
    config_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for bs in args.batches:
        run_dir = output_root / f"bs{bs}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = config_root / f"bs{bs}.yaml"
        log_path = run_dir / "run.log"
        write_config(base_config, cfg_path, bs, run_dir, args)

        cmd = [
            sys.executable,
            "run.py",
            "--config",
            str(cfg_path),
            "--data-root",
            args.data_root,
            "--output-dir",
            str(run_dir),
        ]

        print(f"\n=== batch_size={bs} backend={args.backend} amp={args.mixed_precision} ===", flush=True)
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(cmd, cwd=repo_root, stdout=log_f, stderr=subprocess.STDOUT, check=False)
        wall_s = time.time() - start

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        row = {
            "backend": args.backend,
            "mixed_precision": args.mixed_precision,
            "batch_size": bs,
            "exit_code": proc.returncode,
            "wall_s": wall_s,
            "gpu_gb": parse_gpu_gb(log_text),
            "run_dir": str(run_dir),
        }

        metrics_path = run_dir / "metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path).iloc[-1].to_dict()
            row.update(metrics)
            print(
                f"ok bs={bs} wps={row.get('windows_per_sec', 0):.1f} "
                f"train={row.get('train_loss', 0):.4f} val={row.get('val_loss', 0):.4f} "
                f"gpu={row.get('gpu_gb')}G",
                flush=True,
            )
        else:
            tail = "\n".join(log_text.splitlines()[-20:])
            row["error_tail"] = tail
            print(f"failed bs={bs} exit={proc.returncode}", flush=True)
            print(tail, flush=True)

        rows.append(row)

        if proc.returncode != 0 and "out of memory" in log_text.lower():
            print("Stopping sweep after out-of-memory failure.", flush=True)
            break

    df = pd.DataFrame(rows)
    summary_path = output_root / "batch_sweep_summary.csv"
    df.to_csv(summary_path, index=False)

    display_cols = [
        "backend",
        "mixed_precision",
        "batch_size",
        "exit_code",
        "train_loss",
        "val_loss",
        "elapsed_s",
        "windows_per_sec",
        "gpu_gb",
        "wall_s",
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    print("\nSummary:")
    print(df[display_cols].to_string(index=False))
    print(f"\nsummary_path={summary_path}")


if __name__ == "__main__":
    main()

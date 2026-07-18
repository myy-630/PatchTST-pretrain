#!/usr/bin/env python3
"""Run PatchTST SSL mask-strategy sweep on AutoDL."""

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
    parser.add_argument("--output-root", default="/root/autodl-tmp/outputs/mask_sweep")
    parser.add_argument("--mask-types", nargs="+", default=["random", "random_synchronized", "block"])
    parser.add_argument("--patch-len", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--backend", default="wfdb", choices=["wfdb", "npy_mmap"])
    parser.add_argument("--mixed-precision", default="fp32", choices=["fp32", "fp16"])
    return parser.parse_args()


def patch_num(seq_len: int, patch_len: int, stride: int) -> int:
    return (seq_len - patch_len) // stride + 1


def parse_gpu_gb(log_text: str) -> float | None:
    matches = GPU_RE.findall(log_text)
    if not matches:
        return None
    return max(float(item) for item in matches)


def safe_name(value: str) -> str:
    return value.lower().replace("-", "_")


def write_config(base_config: Path, out_path: Path, mask_type: str, run_dir: Path, args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["data"]["backend"] = args.backend
    cfg["pretrain"]["patch_len"] = args.patch_len
    cfg["pretrain"]["stride"] = args.patch_len
    cfg["pretrain"]["patch_num"] = patch_num(cfg["data"]["seq_len"], args.patch_len, args.patch_len)
    cfg["pretrain"]["mask_type"] = mask_type
    cfg["training"]["batch_size"] = args.batch_size
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["mixed_precision"] = args.mixed_precision
    cfg["checkpoint"]["save_dir"] = str(run_dir)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    base_config = repo_root / args.base_config
    run_id = f"{args.backend}_{args.mixed_precision}_patch{args.patch_len}_bs{args.batch_size}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_root) / run_id
    config_root = repo_root / "configs"
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for mask_type in args.mask_types:
        name = safe_name(mask_type)
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = config_root / f"_tmp_mask_sweep_{run_id}_{name}.yaml"
        log_path = run_dir / "run.log"
        write_config(base_config, cfg_path, mask_type, run_dir, args)

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

        print(f"\n=== mask_type={mask_type} patch={args.patch_len} bs={args.batch_size} ===", flush=True)
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(cmd, cwd=repo_root, stdout=log_f, stderr=subprocess.STDOUT, check=False)
        wall_s = time.time() - start

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        row = {
            "backend": args.backend,
            "mixed_precision": args.mixed_precision,
            "batch_size": args.batch_size,
            "patch_len": args.patch_len,
            "patch_num": patch_num(6000, args.patch_len, args.patch_len),
            "mask_type": mask_type,
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
                f"ok mask={mask_type} wps={row.get('windows_per_sec', 0):.1f} "
                f"train={row.get('train_loss', 0):.4f} val={row.get('val_loss', 0):.4f} "
                f"gpu={row.get('gpu_gb')}G",
                flush=True,
            )
        else:
            tail = "\n".join(log_text.splitlines()[-20:])
            row["error_tail"] = tail
            print(f"failed mask={mask_type} exit={proc.returncode}", flush=True)
            print(tail, flush=True)

        rows.append(row)
        if proc.returncode != 0 and "out of memory" in log_text.lower():
            print("Stopping sweep after out-of-memory failure.", flush=True)
            break

    df = pd.DataFrame(rows)
    summary_path = output_root / "mask_sweep_summary.csv"
    df.to_csv(summary_path, index=False)

    display_cols = [
        "backend",
        "batch_size",
        "patch_len",
        "mask_type",
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


#!/usr/bin/env python3
"""
Convert all WFDB records to .npy (chunked, memory-safe).
One .npy per record: (channels, samples) float32.
Reads 1M samples at a time, writes via memmap.
"""

import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd, wfdb
from tqdm import tqdm

CHUNK = 1_000_000

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wfdb-dir", default="AF")
    p.add_argument("--output-dir", default="records_npy")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--records", default=None, help="Comma-sep. record IDs to convert; omit for all")
    return p.parse_args()

def convert_one(record_id, wfdb_dir, out_dir, overwrite):
    npy_path = out_dir / f"{record_id}.npy"
    wfdb_path = wfdb_dir / record_id

    result = {"record_id": record_id, "wfdb_record_path": str(wfdb_path),
              "npy_path": str(npy_path), "n_channels": 2, "dtype": "float32",
              "sampling_rate": None, "npy_size_bytes": 0,
              "conversion_status": "pending", "error_message": ""}

    if npy_path.exists() and not overwrite:
        try:
            s = np.load(npy_path, allow_pickle=False)
            result["n_samples"] = s.shape[1]; result["npy_size_bytes"] = npy_path.stat().st_size
            result["conversion_status"] = "exists_ok"; return result
        except:
            pass

    try:
        header = wfdb.rdheader(str(wfdb_path))
        sig_len = header.sig_len; fs = header.fs
        result["n_samples"] = sig_len; result["sampling_rate"] = fs

        # memmap write → then convert to proper .npy
        tmp_path = npy_path.with_suffix(".raw")
        mmap = np.memmap(str(tmp_path), dtype="float32", mode="w+", shape=(2, sig_len))
        for start in range(0, sig_len, CHUNK):
            end = min(start + CHUNK, sig_len)
            sig, _ = wfdb.rdsamp(str(wfdb_path), sampfrom=start, sampto=end, channels=[0, 1])
            mmap[:, start:end] = sig.astype(np.float32).T
            del sig
        mmap.flush(); del mmap
        # Convert raw memmap → proper .npy (read-only mmap, save as .npy)
        data = np.memmap(str(tmp_path), dtype="float32", mode="r", shape=(2, sig_len))
        np.save(str(npy_path), data)
        del data
        tmp_path.unlink()  # remove raw tmp
        result["npy_size_bytes"] = npy_path.stat().st_size
        result["conversion_status"] = "success"
    except Exception as e:
        result["conversion_status"] = "failed"; result["error_message"] = str(e)
        if npy_path.exists(): npy_path.unlink()
    return result

def main():
    args = parse_args()
    wfdb_dir = Path(args.wfdb_dir).resolve(); out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover records
    all_records = sorted(set(p.stem for p in wfdb_dir.glob("*.hea")))
    if args.records:
        selected = [r.strip() for r in args.records.split(",")]
        records = [r for r in all_records if r in selected]
    else:
        records = all_records
    print(f"Records: {len(records)} (of {len(all_records)} available)")

    if args.dry_run:
        for r in records: print(f"  {r}")
        return

    original_cwd = os.getcwd(); os.chdir(str(wfdb_dir.parent))

    manifest, errors = [], []
    for rid in tqdm(records, desc="Converting", unit="rec"):
        row = convert_one(rid, wfdb_dir, out_dir, args.overwrite)
        manifest.append(row)
        if row["conversion_status"] == "failed": errors.append(row)

    os.chdir(original_cwd)

    df = pd.DataFrame(manifest)
    df.to_csv(out_dir / "records_npy_manifest.csv", index=False)

    ok = int((df["conversion_status"].isin(["success","exists_ok"])).sum())
    failed = int((df["conversion_status"]=="failed").sum())
    tb = df["npy_size_bytes"].sum()
    print(f"\nDone: {ok} ok, {failed} failed, {tb/1e9:.2f} GB total")
    print(f"Manifest: {out_dir / 'records_npy_manifest.csv'}")
    if errors:
        for e in errors: print(f"  FAIL {e['record_id']}: {e['error_message']}")
        sys.exit(1)

if __name__ == "__main__":
    main()

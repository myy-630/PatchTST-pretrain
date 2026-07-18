# AutoDL Runbook

This directory contains scripts for running the PatchTST ECG SSL project on
AutoDL. The expected cloud layout is:

```text
/root/PatchTST-pretrain
/root/autodl-tmp/shd-af-clean-data
  ├─ raw_wfdb/
  ├─ records_npy/
  ├─ records_npy_manifest.csv
  ├─ splits/
  ├─ valid_windows.csv
  └─ window_index.csv
```

## 1. Clone Code On AutoDL

```bash
cd /root
git clone https://github.com/myy-630/PatchTST-pretrain.git
cd /root/PatchTST-pretrain
```

## 2. Set Up Python Environment

Choose an AutoDL PyTorch image with CUDA already installed. Then run:

```bash
bash cloud/autodl/setup_env.sh
```

The script intentionally does not reinstall `torch`, so it will not replace the
AutoDL CUDA-enabled PyTorch package with a CPU wheel.

## 3. Upload Data

Do not put ECG data in Git. Upload a data archive or directory to AutoDL, then
extract/copy it to:

```text
/root/autodl-tmp/shd-af-clean-data
```

If you upload `shd-af-clean-data.tar.gz`, unpack it with:

```bash
bash cloud/autodl/unpack_data.sh /root/autodl-tmp/shd-af-clean-data.tar.gz
```

## 4. Verify Data

```bash
source .venv/bin/activate
python cloud/autodl/verify_data.py --data-root /root/autodl-tmp/shd-af-clean-data
```

## 5. Run Backend A/B Pilot

```bash
bash cloud/autodl/run_backend_ab.sh /root/autodl-tmp/shd-af-clean-data
```

Outputs are written under:

```text
/root/autodl-tmp/outputs/backend_ab/<timestamp>/
```

The A/B comparison uses the same pilot CSV files from this repository and only
changes the backend:

- `wfdb`: reads `raw_wfdb`
- `npy_mmap`: reads `records_npy`


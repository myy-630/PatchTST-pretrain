# Kaggle Upload Manifest

> Generated: 2026-07-14

---

## 1. Code Dataset: `patchtst-ecg-code`

```
patchtst-ecg-code/
├── run.py
├── requirements.txt
├── MASKED_PRETRAINING_READINESS.md
├── PATCHTST_SSL_REPRODUCTION_PLAN.md
├── configs/
│   ├── patchtst_ssl_pilot.yaml
│   └── patchtst_ssl_full.yaml
└── src/
    ├── __init__.py
    ├── config.py
    ├── trainer.py
    ├── dataset/
    │   ├── __init__.py
    │   └── ecg_dataset.py
    └── models/
        ├── __init__.py
        ├── encoder.py
        ├── loss.py
        └── reconstruction.py
```

**Files**: 15 | **Size**: ~87 KB

### Excluded from code upload

| Excluded | Reason |
|---|---|
| `debug_*.py` × 5, `train_ssl_smoke_test.py` | Development-only; not needed at runtime |
| `dataset_pipeline_step1.py`, `split_dataset.py` | Preprocessing; already done |
| `PROJECT_LOG.md` | Local project log |
| `quality_mask_v3.csv`, `window_index.csv`, `valid_windows.csv` | Data files go in data dataset |
| `archive/`, `outputs/`, `debug_output/` | Old artifacts |
| `venv/`, `__pycache__/` | Environment |
| `visualization*`, `review_v3/` | Viz outputs |
| `configs/patchtst_ssl_full.yaml` (original root) | Only copy in kaggle_upload |

---

## 2. Data Dataset: `shdb-af-clean-data`

```
shd-af-clean-data/
├── metadata.csv
├── window_index.csv
├── quality_mask_v3.csv
├── valid_windows.csv
├── splits/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── subject_split.json
└── raw_wfdb/
    ├── RECORDS.txt
    ├── *.hea × 128
    ├── *.dat × 128
    ├── *.atr × 98
    └── *.qrs × 128
```

**Files**: 490 | **Size**: ~8.4 GB

| Subfolder | Files | Size |
|---|---|---|
| `splits/` | 4 | ~46 MB |
| `raw_wfdb/` | 482 (128 hea + 128 dat + 98 atr + 128 qrs + RECORDS) | ~8.3 GB |
| Root CSVs | 4 (metadata, window_index, quality_mask_v3, valid_windows) | ~91 MB |

### Excluded from data upload

| Excluded | Reason |
|---|---|
| `archive/` | Old preprocessing outputs |
| `quality_mask_v1.csv`, `quality_mask_v2.csv` | Superseded by quality_mask_v3.csv |
| `visualization*/`, `review_v3/` | Viz outputs |
| `artifact_labels.csv`, `analysis_report.md` | Manual annotation intermediates |
| `feature_analysis/`, `v3_validation/` | Development analysis |
| `dataset_pipeline_step1.py`, `split_dataset.py` | Code not needed in data folder |
| `*.log` files | Runtime logs |

---

## 3. Path Checks

| Check | Result |
|---|---|
| YAML contains absolute paths | **None** |
| CSV contains absolute paths (`E:\`, `C:\`) | **None** |
| `record_id` + `start_sample` + `end_sample` present in CSV | ✅ |
| YAML `data_dir` points to `raw_wfdb` | ✅ |
| `--data-root` CLI flag overrides all paths | ✅ |
| `requirements.txt` excludes torch/torchvision/torchaudio | ✅ |

---

## 4. Verification Results

| # | Check | Result |
|---|---|---|
| 1 | `run.py` exists | ✅ |
| 2 | Both YAML configs exist | ✅ |
| 3 | `src/` module tree complete | ✅ |
| 4 | Python imports resolve | ✅ |
| 5 | `requirements.txt` clean | ✅ (wfdb + pyyaml only) |
| 6 | `train.csv` / `val.csv` / `test.csv` exist | ✅ |
| 7 | `raw_wfdb/` exists | ✅ |
| 8 | WFDB files: 128 hea, 128 dat, 98 atr, 128 qrs | ✅ |
| 9 | CSV fields correct (window_id, record_id, start_sample, end_sample, af_label) | ✅ |
| 10 | No absolute paths in YAML or CSV | ✅ |
| 11 | No torch/torchvision/torchaudio in requirements | ✅ |
| 12 | No API keys, tokens, or credentials | ✅ |

---

## 5. Ready for Upload?

**Yes.** Both folders are clean, self-contained, and path-portable.

### Before uploading, manually confirm:

- [ ] `wfdb` is not in Kaggle's pre-installed packages (may need `!pip install wfdb pyyaml` in notebook)
- [ ] Data fits within Kaggle Dataset size limit (8.4 GB — OK for Dataset, too large for Notebook upload alone)
- [ ] Choose Dataset visibility: Private for now, until paper submission

### Kaggle Notebook startup cell

```python
!pip install wfdb pyyaml -q

import sys
sys.path.insert(0, "/kaggle/input/patchtst-ecg-code")

!python /kaggle/input/patchtst-ecg-code/run.py \
    --config /kaggle/input/patchtst-ecg-code/configs/patchtst_ssl_pilot.yaml \
    --data-root /kaggle/input/shd-af-clean-data \
    --output-dir /kaggle/working/outputs/pilot
```

---

*Generated 2026-07-14*

# Masked Pretraining — Readiness Check

> 2026-07-14

---

## 1. Completed Modules

| Phase | Module | Status | File |
|---|---|---|---|
| 1 | `MaskedPretrainDataset` | ✅ | `src/dataset/ecg_dataset.py` |
| 1 | `collate_ssl` | ✅ | `src/dataset/ecg_dataset.py` |
| 2 | `PatchTSTEncoder` | ✅ | `src/models/encoder.py` |
| 2 | `PatchTSTBackbone` | ✅ | `src/models/encoder.py` |
| 3 | `ReconstructionHead` | ✅ | `src/models/reconstruction.py` |
| 4 | `masked_reconstruction_loss` | ✅ | `src/models/loss.py` |
| 5 | Smoke test (A–D) | ✅ | `train_ssl_smoke_test.py` |
| — | Configs | ✅ | `configs/patchtst_ssl_pilot.yaml` |
| — | Configs | ✅ | `configs/patchtst_ssl_full.yaml` |

---

## 2. Hyperparameter Table

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `seq_len` | 6000 | ECG | 30s × 200Hz, fixed |
| `n_channels` | 2 | ECG | Lead I + II, fixed |
| `patch_len` | 12 | Official | 60ms at 200Hz. **Tune candidate**: 24–30 may better capture QRS |
| `stride` | 12 | Official | SSL requires non-overlap (= patch_len) |
| `patch_num` | 500 | Derived | (6000−12)/12 + 1 |
| `mask_ratio` | 0.4 | Official | **Tune candidate**: 0.3 may be better for structured ECG |
| `d_model` | 128 | Official | **Tune candidate**: 256 if VRAM allows |
| `n_heads` | 16 | Official | Must divide d_model evenly |
| `n_layers` | 3 | Official | **Tune candidate**: 4–6 |
| `d_ff` | 256 | Official | Usually 4× d_model |
| `dropout` | 0.1 | Official | |
| `pos_encoding` | Learnable | Official | `nn.Parameter(1, max_patches, d_model)` |
| `RevIN` | Per-instance | Official | Applied in Dataset (vs Backbone in official code) |
| `batch_size` | Pilot: 32, Full: 64 | Tune | GPU VRAM dependent |
| `epochs` | Pilot: 2, Full: 100 | Tune | Early stopping may reduce |
| `optimizer` | AdamW | Official | |
| `lr` | 1e-4 | Official | Pretrain default |
| `weight_decay` | Pilot: 0, Full: 0.05 | Tune | |
| `grad_clip` | 1.0 | Official | |
| `scheduler` | Full: cosine | Tune | Pilot: none |
| `warmup` | Full: 5 epochs | Tune | Pilot: none |
| `seed` | 42 | Fixed | |
| `mixed_precision` | Pilot: fp32, Full: fp16 | Tune | Verify fp16 stability first |

---

## 3. Pilot Run Acceptance Criteria

| # | Check | Threshold |
|---|---|---|
| 1 | No OOM | GPU fits batch_size=32 |
| 2 | Throughput | ≥ 50 windows/sec on GPU |
| 3 | Loss trend | Train loss decreases across 2 epochs |
| 4 | No NaN/Inf | Throughout 2 epochs |
| 5 | Checkpoint save | `last.pt` + `best.pt` written |
| 6 | Checkpoint load | Resumable from `last.pt` |
| 7 | Val loss stable | Not diverging |
| 8 | Grad norm | 0.01–10.0 (no vanishing/exploding) |

If all 8 pass → proceed to Full.

---

## 4. Before Full Training: Issues to Resolve

| # | Issue | Severity | Action |
|---|---|---|---|
| 1 | No formal Trainer class | Medium | Write `src/trainer.py` wrapping train/val/checkpoint logic |
| 2 | No YAML config loader | Low | Add `--config` CLI to training script |
| 3 | Random masking each epoch | ✅ **DONE** | `mode="train"` + `set_epoch()` → dynamic; `mode="val"` → deterministic. 7/7 tests pass |
| 4 | `patch_len=12` (60ms) too short for QRS | Medium | Run a 1-epoch ablation: p=12 vs p=24 vs p=30 |
| 5 | No fp16 verification | Low | Pilot runs fp32; test fp16 → fp32 loss difference |
| 6 | No multi-GPU / DDP | Low | Single GPU sufficient for 360k windows |
| 7 | test.csv excluded from pretrain | ✅ | Already guaranteed by design |

---

## 5. Expected Output Files (after Full Training)

```
checkpoints/full/
├── best.pt                  # best val_loss Encoder weights (no recon head)
├── last.pt                  # most recent state (for resume)
├── epoch_005.pt             # periodic snapshot
├── ...
└── config.yaml              # saved config for reproducibility

logs/
└── pretrain_YYYYMMDD_HHMMSS/
    ├── events.out.tfevents  # TensorBoard
    └── metrics.csv          # per-epoch train/val loss
```

---

## 6. Decision

```
Pilot Run prerequisite:  Write src/trainer.py (train/val/checkpoint/save).
                         Fix Dataset seed=None for training.
                         Add --config YAML support.

Status:  CODE READY for Pilot.
         Trainer scaffolding needed before GPU run.
         Do NOT start full training until Pilot passes.
```

---

*Generated 2026-07-14*

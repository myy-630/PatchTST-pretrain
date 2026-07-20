# ECG Self-Supervised Pretraining Experiment Protocol

## Purpose

This document fixes the comparison protocol for ECG self-supervised pretraining
experiments. The goal is to compare pretraining strategies fairly, without
letting differences in data split, model size, training budget, or downstream
evaluation protocol dominate the result.

## Current Leakage Status

The current dataset split is patient-level because `record_id` has been manually
confirmed to represent the patient/subject identifier.

Checked split properties:

| Item | Result |
|---|---:|
| train patients | 89 |
| val patients | 19 |
| test patients | 20 |
| train-val patient overlap | 0 |
| train-test patient overlap | 0 |
| val-test patient overlap | 0 |
| cross-split `window_id` overlap | 0 |

Therefore, the current train/val/test split has no observed patient-level or
window-level leakage.

The test split must remain unused for pretraining, hyperparameter selection,
early stopping, model selection, and protocol decisions.

## Current Main Baseline

The current strongest completed baseline is masked reconstruction with PatchTST:

```text
backend = wfdb
patch_len = 50
stride = 50
patch_num = 120
mask_type = random
mask_ratio = 0.4
batch_size = 128
epochs = 100
optimizer = AdamW
lr = 1e-3
scheduler = cosine
warmup_epochs = 5
mixed_precision = fp32
```

Model architecture:

```text
d_model = 128
n_heads = 16
n_layers = 3
d_ff = 256
dropout = 0.1
n_channels = 2
seq_len = 6000
```

Completed results:

| Stage | Metric | Result |
|---|---|---:|
| full pretrain | best masked val MSE | 0.2697 |
| frozen linear probe | best val AUROC | 0.9971 |
| frozen linear probe | best val AUPRC | about 0.9906 |
| frozen linear probe | best val F1 | about 0.9587 |

These results are validation results only.

## Strategies To Compare

Three pretraining strategies should be compared under the same protocol.

| Strategy ID | Pretraining objective | Status |
|---|---|---|
| S1 | Masked reconstruction | implemented |
| S2 | Contrastive learning with two augmented views of the same ECG window | pending |
| S3 | Temporal predictive / forecasting objective | pending |

S2 definition:

```text
For each ECG window, generate two independently augmented views.
The two views from the same window are positives.
Other windows in the batch are negatives.
Train the PatchTST encoder with a contrastive objective.
```

S3 definition:

```text
Given visible context patches from an ECG window, predict future/held-out
temporal patch representations or signals.
Train the PatchTST encoder to learn temporal dynamics rather than reconstructing
randomly masked patches.
```

## Fixed Parameters For Fair Comparison

The following must be identical across strategies unless there is a documented
technical reason why a strategy cannot use one of them.

### Data

```text
train_csv = splits/train.csv
val_csv = splits/val.csv
test_csv = splits/test.csv
data split = fixed patient-level split
pretraining data = train only
validation data = val only
test usage = final evaluation only
label usage in pretraining = none
```

For linear probe:

```text
positive class = AF
negative class = Normal
excluded labels = Unlabeled, Mixed, Other
```

### Encoder

```text
backbone = PatchTST encoder
seq_len = 6000
n_channels = 2
patch_len = 50
stride = 50
patch_num = 120
d_model = 128
n_heads = 16
n_layers = 3
d_ff = 256
dropout = 0.1
RevIN = true
```

Rationale: current sweeps show `patch_len=50` is both faster and stronger than
`patch_len=25` for masked reconstruction linear-probe performance. Fixing it
across strategies prevents a patch-size advantage from being confused with a
pretraining-objective advantage.

### Training Budget

Use the same budget for each strategy:

```text
epochs = 100
batch_size = 128
optimizer = AdamW
base lr = 1e-3
scheduler = cosine
warmup_epochs = 5
seed = 42
num_workers = 2
device = cuda
mixed_precision = fp32
```

If a strategy cannot use exactly the same learning rate or schedule stably, the
change must be documented and the strategy must receive the same tuning budget
as the others.

## Strategy-Specific Parameters

Some parameters belong to a specific pretraining objective and should not be
forced to match across objectives.

Examples:

| Strategy type | Strategy-specific parameters |
|---|---|
| masked reconstruction | `mask_ratio`, `mask_type` |
| contrastive objective | augmentations, temperature, positive-pair definition |
| predictive objective | context length, prediction horizon |

Fair handling rule:

1. Give each strategy the same number of validation-selected trials.
2. Select the best checkpoint using validation-only criteria.
3. Do not use test results to choose strategy-specific parameters.

Recommended small tuning budget:

```text
maximum trials per strategy = 3
```

For masked reconstruction, the already-tested candidates are:

```text
patch25 random mask_ratio 0.2
patch25 random mask_ratio 0.4
patch25 random mask_ratio 0.6
patch50 random mask_ratio 0.4
```

Optional next masked-reconstruction trial:

```text
patch50 random mask_ratio 0.6
```

This should be treated as one additional strategy-specific tuning trial, not as
information available to other strategies unless they receive equal tuning
opportunity.

### S2 Default Contrastive Setup

Default S2 should keep the same encoder and patching scheme as S1:

```text
patch_len = 50
stride = 50
patch_num = 120
encoder = same PatchTST backbone
batch_size = 128
epochs = 100
```

Recommended first contrastive design:

```text
objective = NT-Xent / InfoNCE
positive pair = two augmentations of the same 30 s ECG window
negative pairs = other windows in the batch
projection head = small MLP on pooled encoder representation
representation for contrast = mean-pooled patch embeddings across time and leads
temperature = 0.1
```

Initial ECG augmentations should be label-preserving and conservative:

```text
amplitude scaling
small Gaussian noise
small baseline wander
small time masking / dropout
```

Avoid aggressive augmentations in the first trial:

```text
large time warping
lead permutation
large temporal crop that removes rhythm context
augmentations that can erase AF evidence
```

Allowed S2 tuning candidates, within the same tuning budget:

```text
temperature = 0.05 / 0.1 / 0.2
augmentation strength = weak / medium / strong
```

Use validation linear-probe AUROC/AUPRC to select among S2 candidates.

### S3 Default Temporal Predictive Setup

Default S3 should also keep the same encoder and patching scheme as S1:

```text
patch_len = 50
stride = 50
patch_num = 120
encoder = same PatchTST backbone
batch_size = 128
epochs = 100
```

Recommended first predictive design:

```text
context = earlier patches in the window
prediction target = later patches or later patch embeddings
prediction horizon = fixed number of future patches
loss = MSE for signal prediction, or InfoNCE/MSE for latent prediction
```

For the first implementation, prefer a simple and interpretable target:

```text
Use the first part of the 30 s window as context.
Predict a held-out future block of patches.
Compute loss only on the future target block.
```

Allowed S3 tuning candidates, within the same tuning budget:

```text
context ratio = 0.5 / 0.75
prediction horizon = 10 / 20 / 30 patches
target type = signal patches first; latent target only after signal baseline
```

Use validation linear-probe AUROC/AUPRC to select among S3 candidates.

## Model Selection

For each pretraining run:

1. Train on `splits/train.csv`.
2. Track validation pretraining metric on `splits/val.csv`.
3. Save `best.pt` by validation pretraining metric.
4. Export `pretrained_encoder.pt` from the selected checkpoint.
5. Do not inspect or use `splits/test.csv`.

For downstream validation:

1. Freeze the encoder.
2. Train only a linear AF-vs-Normal head.
3. Use the same train/val split for every strategy.
4. Use the same linear-probe hyperparameters for every strategy.

Current linear-probe protocol:

```text
epochs = 20
batch_size = 256
lr = 1e-3
weight_decay = 0.0
positive class = AF
negative class = Normal
excluded labels = Unlabeled, Mixed, Other
selection metric = val AUROC, with AUPRC/F1 as supporting metrics
```

Primary validation metrics:

```text
AUROC
AUPRC
F1
sensitivity
specificity
confusion matrix
```

## Test-Set Rule

The test set is used only once the following are fixed:

1. pretraining strategy
2. strategy-specific hyperparameters
3. encoder architecture
4. downstream evaluation protocol
5. model-selection rule

No test-set result should be used to revise the above choices.

## Reporting Template

For each strategy, report:

| Field | Required |
|---|---|
| pretraining objective | yes |
| fixed encoder parameters | yes |
| strategy-specific parameters | yes |
| pretraining train metric | yes |
| pretraining val metric | yes |
| selected checkpoint | yes |
| frozen linear probe AUROC | yes |
| frozen linear probe AUPRC | yes |
| frozen linear probe F1 | yes |
| sensitivity/specificity | yes |
| whether test was used | must be "no" before final |

## Current Interpretation

The current masked reconstruction baseline is strong and does not show obvious
patient-level leakage. However, it should be treated as the S1 baseline until
S2 and S3 are run under the same fixed protocol.

The main comparison question should be:

```text
Given the same patient-level split, PatchTST encoder, patching scheme, training
budget, and frozen linear-probe protocol, which pretraining objective produces
the best validation representation?
```

Only after that answer is fixed should the final test-set evaluation be run.

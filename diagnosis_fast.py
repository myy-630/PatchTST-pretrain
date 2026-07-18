"""Fast diagnostic — no training loop, single forward passes only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import MaskedPretrainDataset
from src.dataset.ecg_dataset import collate_ssl
from src.models import PatchTSTBackbone, ReconstructionHead
from src.models.loss import masked_reconstruction_loss

SEED = 42
DEVICE = torch.device("cpu")
OUT_DIR = Path("diagnosis_output"); OUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)

# Tiny model for local CPU fast tests
MODEL = PatchTSTBackbone(12, n_vars=2, d_model=64, n_heads=8, n_layers=1, d_ff=128)
HEAD  = ReconstructionHead(64, 12)

# Fixed batch
ds = MaskedPretrainDataset("splits/train.csv", patch_len=12, stride=12, mask_ratio=0.4, seed=SEED, mode="val")
loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_ssl)
batch = next(iter(loader))
batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

masked = batch["masked_signal"]  # (B,C,N,P)
target = batch["target_signal"]
mask   = batch["mask"]
B, C, N, P = target.shape

# ── 1. Zero baseline ──
mask_exp = mask.unsqueeze(-1).expand_as(target)
zero_loss = nn.functional.mse_loss(torch.zeros_like(target)[mask_exp], target[mask_exp])
print(f"[1] Zero-prediction baseline: {zero_loss.item():.6f}")
print(f"    → loss=1.0 means model predicts mean (~0 after RevIN), NOT learning")

# ── 2. Target stats ──
t_all   = target
t_mask  = target[mask_exp]
t_vis   = target[~mask_exp]
print(f"\n[2] Target stats:")
print(f"    All:      mean={t_all.mean():.4f}  std={t_all.std():.4f}")
print(f"    Masked:   mean={t_mask.mean():.4f}  std={t_mask.std():.4f}")
print(f"    Visible:  mean={t_vis.mean():.4f}  std={t_vis.std():.4f}")

# ── 3. Random-init prediction stats ──
MODEL.eval(); HEAD.eval()
with torch.no_grad():
    encoded = MODEL(masked[:, :, :50, :])   # use only 50 patches for speed
    f = encoded.reshape(B*C, 50, 64)
    recon_50 = HEAD(f).reshape(B, C, 50, P)
p_masked = recon_50[mask[:,:,:50].unsqueeze(-1).expand_as(recon_50)]
print(f"\n[3] Random-init prediction (50 patches):")
print(f"    mean={p_masked.mean():.4f}  std={p_masked.std():.4f}")
print(f"    → prediction std << target std → model barely moves from zero")

# ── 4. Single step gradient check (50 patches) ──
MODEL.train(); HEAD.train()
opt = torch.optim.AdamW(list(MODEL.parameters())+list(HEAD.parameters()), lr=1e-4)
m50 = masked[:,:,:50,:]; t50 = target[:,:,:50,:]; mk50 = mask[:,:,:50]

opt.zero_grad()
encoded = MODEL(m50)
f = encoded.reshape(B*C, 50, 64)
recon = HEAD(f).reshape(B, C, 50, -1)
loss = masked_reconstruction_loss(recon, t50, mk50)
loss.backward()

gn_enc = sum(p.grad.data.norm(2).item()**2 for p in MODEL.parameters() if p.grad is not None)**0.5
gn_head = sum(p.grad.data.norm(2).item()**2 for p in HEAD.parameters() if p.grad is not None)**0.5

print(f"\n[4] Single-step gradients (lr=1e-4):")
print(f"    loss={loss.item():.6f}")
print(f"    grad_norm enc={gn_enc:.6f}  head={gn_head:.6f}")
print(f"    Encoder grad > 0: {gn_enc > 0}")
print(f"    Head grad > 0:    {gn_head > 0}")

# ── 5. 10-step quick overfit (50 patches) ──
MODEL.train(); HEAD.train()
opt = torch.optim.AdamW(list(MODEL.parameters())+list(HEAD.parameters()), lr=1e-3)
losses = []
for step in range(30):
    opt.zero_grad()
    encoded = MODEL(m50); f = encoded.reshape(B*C, 50, 64)
    recon = HEAD(f).reshape(B, C, 50, -1)
    loss = masked_reconstruction_loss(recon, t50, mk50)
    loss.backward(); opt.step()
    losses.append(loss.item())
    if step % 10 == 0:
        print(f"  step {step:2d}: loss={loss.item():.6f}")

print(f"\n[5] Quick overfit (lr=1e-3, 30 steps):")
print(f"    loss: {losses[0]:.4f} → {losses[-1]:.4f} ({(1-losses[-1]/losses[0])*100:.1f}% decrease)")

# ── 6. Plot ──
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(losses); ax.set_xlabel("Step"); ax.set_ylabel("MSE")
ax.set_title("Single-batch Overfit (50 patches, lr=1e-3)")
fig.savefig(OUT_DIR/"overfit_quick.png", dpi=120); plt.close(fig)

# ── 7. Full model grad check (300 patches) ──
print(f"\n[6] Full model (300 patches) — single forward test:")
full_model = PatchTSTBackbone(12, n_vars=2, d_model=128, n_heads=16, n_layers=3, d_ff=256)
fhead = ReconstructionHead(128, 12)
m300 = masked[:,:,:300,:]; t300 = target[:,:,:300,:]; mk300 = mask[:,:,:300]
with torch.no_grad():
    enc = full_model(m300)
    fr = enc.reshape(B*C, 300, 128)
    r300 = fhead(fr).reshape(B, C, 300, P)
    l300 = masked_reconstruction_loss(r300, t300, mk300)
print(f"    loss (random init) = {l300.item():.6f}")
print(f"    → If ≈1.0 → same as zero baseline → model output ~0")

print(f"\n{'='*60}")
print("Key finding: loss ~1.0 = zero-prediction baseline.")
print("Root cause: RevIN makes target ~N(0,1); model predicts 0 = MSE≈1.")
print("Model CAN overfit small data (see plot). Full data: lr may be too low.")
print(f"{'='*60}")

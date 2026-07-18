# PatchTST Self-Supervised Pretraining — Reproduction Plan

> 基于 PatchTST 官方仓库 (yuqinie98/PatchTST) 及 ICLR 2023 论文  
> 目标: SHDB-AF ECG (30s, 200Hz, 2-lead) → 自监督预训练 → AF 分类

---

## 1. 原论文自监督预训练完整数据流

```
原始时间序列 (B, C, L)
    │
    ▼
RevIN norm: 逐实例去均值/除标准差
    │
    ▼
Patching: 非重叠切分 → (B×C, N, P)
    │  N = floor((L - P) / S) + 1,  SSL 用 S = P (无重叠)
    │
    ▼
Patch Embedding: 线性投影 P → d_model
    │
    ▼
+ Learnable Position Embedding (B×C, N, d_model)
    │
    ▼
Random Patch Masking: 随机选 ⌊N×0.4⌋ 个 patch 置零
    │
    ▼
Transformer Encoder (L layers, H heads)
    │
    ▼
Reconstruction Head: 线性层 d_model → P (仅在被 mask 位置)
    │
    ▼
RevIN denorm: 逆归一化还原原始量纲
    │
    ▼
MSE Loss: 仅在被 mask 的 patch 上计算
```

**官方对应文件**: `PatchTST_self_supervised/src/`

---

## 2. 自监督任务本质

**Masked Patch Reconstruction** — 受 BERT (MLM) 和 MAE 启发。

| 对比 | BERT (NLP) | MAE (CV) | PatchTST (TS) |
|---|---|---|---|
| 输入单元 | Token | Image Patch | Time-series Patch |
| Mask 粒度 | Word piece | 16×16 px block | P=12 time steps |
| 被 Mask 替换 | [MASK] token | 0 (zero) | 0 (zero) |
| 目标 | 预测 token | 重建像素 | 重建时间点 |
| Loss | Cross-entropy | MSE | MSE |

模型必须从可见的上下文 patch 中推断被 mask 位置的原始波形 — 这迫使 Encoder 学习有意义的时序表征。

---

## 3. Patch 切分方式

**SSL 预训练**: `stride = patch_len` (非重叠)

```
patch_len = 12, stride = 12  ← 官方默认
patch_len = 12, stride = 12  → N = 6000 / 12 = 500 patches (适配 ECG)
```

**计算**:
- 输入长度 L = 6000 (30s × 200Hz)
- Patch 数: N = floor((L - P) / S) + 1 = 500
- 每 patch 含 P=12 个时间点 (60ms)

**官方代码**: `PatchTST_backbone.py` 中 `create_patch()` 使用 `x.unfold(dimension=-1, size=patch_len, step=stride)`

| | Supervised (Forecasting) | Self-Supervised (Pretrain) |
|---|---|---|
| Stride | S < P (有重叠) | **S = P (无重叠)** |
| 原因 | 增加序列覆盖率 | 防止可见 patch 泄漏 masked patch 信息 |

---

## 4. Mask 粒度

**在 Patch 层面 mask，不是原始时间点。**

```python
# 伪代码 (PatchMaskCB)
patch_indices = random.sample(range(N), int(N * mask_ratio))  # 选 40% 的 patch
x[patch_indices, :] = 0  # 整 patch 置零
```

如果一个 patch 的 12 个时间点中任意一个可见，则 MAE 式的逐点插值就可以重建，自监督任务将退化。因此必须整 patch 一起 mask。

**官方代码**: `src/callback/patch_mask.py` → `PatchMaskCB` 类

---

## 5. 被 Mask 的 Patch 替换值

**替换为 0 (zero-masking)**。

```python
x[mask] = 0  # 官方实现
```

不使用 [MASK] token (像 BERT)，也不使用随机噪声 (像某些 denoising AE)。MAE 论文证明了 zero-masking 在 patch 级重建任务中效果最佳。

---

## 6. Mask Ratio

**官方默认: 40%**

```python
mask_ratio = 0.4  # 500 patches → 200 被 mask, 300 可见
```

| mask_ratio | 含义 | 适用场景 |
|---|---|---|
| 0.4 | 40% mask | 官方默认，验证充分 |
| 0.5–0.6 | 更高难度 | 更长的序列或更简单的数据 |
| 0.2–0.3 | 更简单 | 短序列、高噪声数据 |

ECG 是结构化信号 (QRS 等)，40% 可能偏高 — 建议初始用 0.3，后续实验对比 0.3 vs 0.4。

---

## 7. Encoder 输入/输出 Shape

```
输入:  (B×C, N, d_model)     B=batch, C=channels (2), N=patches (500), d_model=128–512
输出:  (B×C, N, d_model)     每个 patch 位置一个 embedding
```

- Channel-independent: 2 个 ECG 导联在 batch 维度拼接，共享同一 Encoder 权重
- N = 500 个 patch token，ATTN 复杂度 O(N²) = 250K，可接受
- 官方 d_model: 128–512, n_heads: 16, n_layers: 3

**官方代码**: `PatchTST_backbone.py` → TSTEncoder

---

## 8. Reconstruction Head

**结构**: 单层 Linear: `d_model → patch_len`

```
PretrainHead(nn.Module):
    Linear(d_model, patch_len)    # 128 → 12
```

- 输入: Encoder 输出的 patch embedding `(B×C, N, d_model)`
- 输出: 重建的 patch 值 `(B×C, N, patch_len)`
- 仅在被 mask 的位置计算 loss (类似 MAE)

**官方代码**: `PatchTST_backbone.py` → `PretrainHead` 类

---

## 9. Reconstruction Loss

**MSE — 仅在被 mask 的 patch 上计算**:

```python
loss = F.mse_loss(pred[mask], target[mask])
```

- 不在所有 patch 上计算 (与 denoising AE 不同)
- 不对 mask token 做特殊处理
- 加上 RevIN denorm 后 MSE 在原始量纲计算

---

## 10. Instance Normalization / RevIN

**RevIN (Reversible Instance Normalization)**:

```python
class RevIN(nn.Module):
    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=-1, keepdim=True)       # 逐样本
            self.stdev = x.std(dim=-1, keepdim=True)
            return (x - self.mean) / (self.stdev + eps)
        elif mode == 'denorm':
            return x * self.stdev + self.mean               # 逆运算
```

**在 SSL Pipeline 中的位置**:
```
[revIN norm] → [patching + mask + encoder + head] → [revIN denorm] → [loss]
```

ECG 的 baseline wander 使不同窗口的均值/标准差差异很大 — RevIN 对此类分布偏移尤其有效。

**是否需要 RevIN**: **是**。ECG 信号存在显著的个体间和窗口间 amplitude 差异，RevIN 是 PatchTST 的重要组成部分。

---

## 11. 预训练模型保存

官方使用 `SaveModelCB` callback:

```python
SaveModelCB(monitor='valid_loss', fname='patchtst_pretrained_xxx')
```

保存内容:
- Encoder backbone (TSTEncoder 全参数)
- Patch embedding (投影层权重)
- RevIN affine 参数
- Position embedding
- **不保存** Reconstruction head (pretrain head)

---

## 12. Fine-tuning 权重加载

```python
# 加载预训练权重
pretrained = torch.load('patchtst_pretrained_xxx.pth')

# 加载 backbone (匹配的参数)
model.backbone.load_state_dict(pretrained, strict=False)

# 新建分类 head (替换 reconstruction head)
model.classification_head = nn.Sequential(
    nn.Flatten(),           # (B×C, N, d_model) → (B×C, N*d_model)
    nn.Linear(N*d_model, num_classes)  # → 2 (AF / Non-AF)
)
```

| 模块 | 加载预训练 | 说明 |
|---|---|---|
| Patch embedding | ✅ | 投影权重 |
| Position embedding | ✅ | 位置编码 |
| Transformer Encoder | ✅ | 全部层 |
| RevIN affine | ✅ | γ, β |
| Reconstruction head | ❌ 丢弃 | 替换为分类 head |
| Classification head | ✨ 随机初始化 | 新增 |

---

## 13. 与 ECG AF 分类的适配分析

### ✅ 可以完全沿用的设计

| 设计 | 原因 |
|---|---|
| Channel-independent architecture | 2 导联共享权重，与 PatchTST 一脉相承 |
| Patch-level masking | 防止 QRS 局部插值泄漏信息 |
| RevIN | 消除个体/窗口间幅度差异 |
| MSE loss on masked patches | 标准自监督目标 |
| 非重叠 patch (SSL phase) | 保证 mask 无泄漏 |

### ⚠️ 必须针对 ECG 修改的设计

| 原论文 | ECG 适配 |
|---|---|
| patch_len=12, stride=12 | ECG 200Hz → 12 samples = **60ms** — 太短，QRS 本身宽 ~80–100ms。建议 `patch_len=20–40` (100–200ms) |
| 仅用于 univariate 时序 | ECG 是双导联。Channel-independent 架构天然支持 — 两个导联各自 patch、共享 encoder，无需修改 |
| 目标为长序列预测 | 替换 reconstruction head → classification head，仅 fine-tune 阶段修改 |
| 使用整段连续时序 | 单个 30s window 作为独立样本; 不需要 segment-level 设计 |

### 🔬 需要实验验证的超参数

| 超参数 | 候选值 | 验证方法 |
|---|---|---|
| patch_len | 12, 20, 24, 30, 40 | Pretrain + Linear Probing AUC |
| mask_ratio | 0.3, 0.4, 0.5 | Pretrain val loss + probing AUC |
| d_model | 128, 256, 512 | 模型大小 vs GPU 显存 |
| n_layers | 3, 4, 6, 8 | Deeper 是否带来增益 |
| 是否 freeze backbone | freeze / full fine-tune | 标注量少时 freeze 可能更优 |

---

## 14. 推荐目录结构和实现顺序

### 推荐目录结构

```
暑假/
├── splits/                          # 已有
│   ├── train.csv / val.csv / test.csv
│   └── subject_split.json
│
├── AF/                              # 原始 WFDB (已有)
│
├── src/                             # 【新建】SSL 实现
│   ├── dataset/
│   │   └── shdb_dataset.py          # wfdb → ECG window → tensor
│   ├── models/
│   │   ├── patching.py              # Patch 切分 + embedding
│   │   ├── revin.py                 # RevIN 层
│   │   ├── encoder.py               # Transformer Encoder
│   │   └── patchtst_ssl.py          # PatchTST SSL 完整模型
│   ├── ssl/
│   │   ├── pretrain.py              # 预训练入口
│   │   ├── masking.py               # Patch-level masking
│   │   └── loss.py                  # Reconstruction loss
│   └── finetune/
│       ├── classifier.py            # 分类 head
│       └── finetune.py              # Fine-tune 入口
│
├── configs/
│   └── pretrain_config.yaml         # 预训练超参数
│
├── checkpoints/                     # 预训练权重
│
└── logs/                            # TensorBoard 日志
```

### 最小实现顺序

| 阶段 | 内容 | 产出 |
|---|---|---|
| **Phase 1** | `shdb_dataset.py` — 从 train.csv 读取 window，wfdb 加载 30s ECG → torch.Tensor (B, 2, 6000) | 可用的 DataLoader |
| **Phase 2** | `patching.py` + `revin.py` + `encoder.py` — 核心模型组件 | 前向传播可用 |
| **Phase 3** | `masking.py` + `loss.py` + `pretrain.py` — SSL 预训练循环 | 可 pretrain |
| **Phase 4** | `classifier.py` + `finetune.py` — 线性探测 → 全模型微调 | AF 分类 baseline |
| **Phase 5** | 超参数 search: patch_len, mask_ratio, d_model | 最优配置 |
| **Phase 6** | Test set 最终评估 | Final AUC |

### Phase 1–3 检查点

- [ ] DataLoader 正确输出 `(B, 2, 6000)` tensor
- [ ] Patching 输出 `(B×2, 500, 12)` shape
- [ ] 40% mask 正确置零且不泄漏
- [ ] Encoder 前向通过无 NaN
- [ ] Pretrain loss 持续下降
- [ ] 随机抽取重建样本，可视化对比原始 vs 重建波形

---

## 附录: 关键官方代码索引

| 组件 | 文件 | 类/函数 |
|---|---|---|
| Backbone | `PatchTST_backbone.py` | `TSTEncoder`, `PatchTST_backbone` |
| Patching | `PatchTST_backbone.py` | `create_patch()` |
| RevIN | `src/models/layers/revin.py` | `RevIN` |
| Pretrain head | `PatchTST_backbone.py` | `PretrainHead` |
| Classification head | `PatchTST_backbone.py` | `ClassificationHead` |
| Patch masking | `src/callback/patch_mask.py` | `PatchMaskCB` |
| Pretrain entry | `patchtst_pretrain.py` | CLI |
| Fine-tune entry | `patchtst_finetune.py` | CLI |

---

*Plan generated 2026-07-13. 下一步: Phase 1 — SHDB ECG Dataset.*

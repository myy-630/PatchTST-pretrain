# ECG AF Detection — 项目日志

> 最后更新: 2026-07-10  
> 项目路径: `e:\python练习\暑假\`

---

## 目录结构总览

```
暑假/
├── PROJECT_LOG.md              ← 本文件
│
├── AF/                          ← 原始 WFDB 数据集 (PhysioNet)
│   ├── RECORDS.txt              # 128 record IDs
│   ├── *.hea ×128               # 头文件
│   ├── *.dat ×128               # 信号数据 (24h, 200Hz, 2导联)
│   ├── *.atr ×98                # 心律标注 (AFIB/AFL/N/NOD)
│   └── *.qrs ×128               # QRS 检测标注
│
├── scan_dataset.py              ← Stage 1: 数据集扫描
├── metadata.csv                 ← Stage 1 输出: record 元数据
├── dataset_summary.json         ← Stage 1 输出: 汇总统计
├── scan_dataset.log             ← Stage 1 日志
│
├── generate_window_index.py     ← Stage 2: 窗口索引生成
├── window_index.csv             ← Stage 2 输出: 30s 窗口索引
├── window_summary.json          ← Stage 2 输出: 窗口统计
├── window_index.log             ← Stage 2 日志
│
├── quality_check_v1.py          ← Stage 4: 质量检查 V1
├── quality_mask_v1.csv          ← Stage 4 输出: 每窗口 keep/fail + 信号统计
├── quality_report.txt           ← Stage 4 输出: 质量报告
├── quality_check_v1.log         ← Stage 4 日志
│
├── quality_check_v1_1.py        ← Stage 4.1: Artifact Feature 提取 + 分析
├── analysis_report.md           ← Stage 4.1 输出: 异常 vs 正常比较报告
├── quality_check_v1_1.log       ← Stage 4.1 日志
│
├── random_visualization.py      ← Stage 5: 随机抽样可视化
├── random_visualization.log     ← Stage 5 日志
│
├── visualization/               ← Stage 5 输出目录
│   ├── sample_index.csv         # 100 张抽样的索引
│   └── random_samples/          # sample_0001.png … sample_0100.png
│
├── visualization_round2.py      ← Stage 6: Round 2 分层可视化
├── visualization_round2.log     ← Stage 6 日志
│
└── visualization_round2/        ← Stage 6 输出目录
    ├── round2_index.csv         # 100 张抽样的索引 (含采样来源)
    └── sample_0001.png … sample_0100.png
```

---

## 各文件/目录详细说明

### 原始数据

| 路径 | 类型 | 说明 | 大小 |
|---|---|---|---|
| `AF/` | 文件夹 | PhysioNet Long-Term AF Database (ltafdb)，WFDB 格式 | ~8.6 GB |
| `AF/RECORDS.txt` | 文件 | 128 条 record ID 列表（不连续，001–143 有跳号） | ~1 KB |
| `AF/*.hea` ×128 | 文件 | WFDB 头文件: 采样率、信号长度、导联名 | ~100 B/条 |
| `AF/*.dat` ×128 | 文件 | 原始 ECG 信号 (Format 16, 200 Hz, 2 ch) | ~66 MB/条 |
| `AF/*.atr` ×98 | 文件 | WFDB rhythm annotation: AFIB/AFL/N/NOD | ~200 KB/条 |
| `AF/*.qrs` ×128 | 文件 | QRS 检测标注 | ~500 KB/条 |

### 探索

| 路径 | 类型 | 说明 |
|---|---|---|
| `看图.ipynb` | Notebook | 原始数据探索，wfdb 读取测试 (record 001) |

---

### Stage 1 — 数据集扫描

| 路径 | 说明 | 行/大小 | 创建时间 |
|---|---|---|---|
| `scan_dataset.py` | 主脚本: 解析 .hea 头文件 (不加载 ECG) | 17 KB | 2026-07-09 |
| `metadata.csv` | **输出**: 128 条 record 的元数据 | 128 rows | 2026-07-09 |
| `dataset_summary.json` | **输出**: 时长/采样率/导联/标注统计 | 1 KB | 2026-07-09 |
| `scan_dataset.log` | 运行日志 | 9 KB | 2026-07-09 |

**metadata.csv 字段**: `record_id, sampling_frequency, signal_length, duration_seconds, num_channels, channel_names, has_annotations, annotation_types, dat_exists, dat_size_bytes, dat_size_mb, hea_path, Subject_ID, Study_ID, Annotated, AF_Type`

**关键发现**:
- 128 records, 全部 200 Hz, 2 导联
- 时长: 9–24 h, 平均 23.83 h
- 98 条有 .atr 标注, 30 条缺失 .atr (053–104 号段)

---

### Stage 2 — 窗口索引生成

| 路径 | 说明 | 行/大小 | 创建时间 |
|---|---|---|---|
| `generate_window_index.py` | 主脚本: 30s 非重叠切分 + 标签赋值 | 16 KB | 2026-07-09 |
| `window_index.csv` | **输出**: 所有窗口索引 (未保存 ECG) | 366,044 rows, 23 MB | 2026-07-09 |
| `window_summary.json` | **输出**: 窗口分布统计 | 3 KB | 2026-07-09 |
| `window_index.log` | 运行日志 | 17 KB | 2026-07-09 |

**window_index.csv 字段**: `window_id, subject_id, record_id, start_sample, end_sample, start_second, end_second, label, mixed_label`

**标签分布**:

| label | 窗口数 | 占比 |
|---|---|---|
| Normal | 218,983 | 59.8% |
| Unlabeled | 86,360 | 23.6% |
| AF | 58,394 | 16.0% |
| Mixed | 1,313 | 0.36% |
| Other | 994 | 0.27% |

**逻辑**: `(AFIB/(AFL → AF`, `(N → Normal`, 跨节律 → `Mixed`, 无 .atr → `Unlabeled`

---

### Stage 4 — 质量检查 V1

| 路径 | 说明 | 行/大小 | 创建时间 |
|---|---|---|---|
| `quality_check_v1.py` | 主脚本: NaN/Inf/Flat/Read 检查 + 信号统计 | 11 KB | 2026-07-10 |
| `quality_mask_v1.csv` | **输出**: 每窗口 keep/fail + 6 个统计量 | 366,044 rows, 27 MB | 2026-07-10 |
| `quality_report.txt` | **输出**: 详细质量报告 | 2 KB | 2026-07-10 |
| `quality_check_v1.log` | 运行日志 | 1 KB | 2026-07-10 |

**quality_mask_v1.csv 字段**: `window_id, keep, fail_reason, std_ch1, std_ch2, max_ch1, min_ch1, max_ch2, min_ch2`

**结果**:
- keep=True: 366,041 (100.00%)
- 仅 3 个 flat_signal (record 131 末尾)
- 0 NaN, 0 Inf, 0 read_error
- 阈值: `FLAT_STD_THRESHOLD = 0.005 mV`

---

### Stage 5 — 随机可视化

| 路径 | 说明 | 行/大小 | 创建时间 |
|---|---|---|---|
| `random_visualization.py` | 主脚本: 100 个窗口随机抽样绘图 | 8 KB | 2026-07-10 |
| `visualization/sample_index.csv` | **输出**: 100 张图片的索引 | 100 rows | 2026-07-10 |
| `visualization/random_samples/` | **输出**: PNG 图片 ×100 | ~20 MB total | 2026-07-10 |
| `random_visualization.log` | 运行日志 | 1 KB | 2026-07-10 |

**参数**: seed=42, 14×7 inches, 120 DPI, Lead I (蓝) + Lead II (红)

**抽样标签**: Normal 57, Unlabeled 25, AF 18

---

### Stage 4.1 — Quality Check V1.1 (Artifact Features)

| 路径 | 说明 | 行/大小 | 创建时间 |
|---|---|---|---|
| `quality_check_v1_1.py` | 主脚本: 新增 4 个 artifact features + 分析报告 | 14 KB | 2026-07-10 |
| `analysis_report.md` | **输出**: 异常 vs 正常样本比较分析报告 | ~8 KB | 2026-07-10 |
| `quality_check_v1_1.log` | 运行日志 | 1 KB | 2026-07-10 |

**quality_mask_v1.csv 新增字段**:

| 字段 | 说明 | 参数 |
|---|---|---|
| `max_abs` | max(\|signal\|) — 尖峰检测 | — |
| `spike_count` | \|amplitude\| > 2.0 mV 的采样点数 | SPIKE_THRESHOLD=2.0 |
| `diff_std` | max(std(diff(ch1)), std(diff(ch2))) — 高频噪声 | — |
| `baseline_range` | 移动平均基线 max-min (MA=1s/200samp) | BASELINE_WINDOW_S=1.0 |

**分析结论** (4 异常 vs 96 正常):

| 候选强度 | 特征 | 异常/正常 Median | 说明 |
|---|---|---|---|
| Moderate | `baseline_range` | 2.9x | 2/4 异常超过正常 P95 |
| Weak | 其余 9 个特征 | <1.5x | 单特征不足以分离 |

---

## 变更记录

| 日期 | 变更 | 详情 |
|---|---|---|
| 2026-07-09 | 新建 `scan_dataset.py` | Stage 1: 数据集扫描脚本 |
| 2026-07-09 | 生成 `metadata.csv` | 128 rows, 所有 record 的 header 元数据 |
| 2026-07-09 | 生成 `dataset_summary.json` | 汇总: 时长/采样率/导联/标注 |
| 2026-07-09 | 新建 `generate_window_index.py` | Stage 2: 窗口索引生成 (仅处理有 .atr 的 record) |
| 2026-07-09 | 生成 `window_index.csv` + `window_summary.json` | 279,684 窗口 (98 records) |
| 2026-07-09 | **更新** `generate_window_index.py` | 补充无 .atr 的 30 条 record → Unlabeled |
| 2026-07-09 | **更新** `window_index.csv` + `window_summary.json` | 366,044 窗口 (128 records) |
| 2026-07-10 | 新建 `quality_check_v1.py` | Stage 4: 质量检查 (NaN/Inf/Flat/Read) |
| 2026-07-10 | 生成 `quality_mask_v1.csv` + `quality_report.txt` | 366,044 窗口, 3 flat 被拒, 其余通过 |
| 2026-07-10 | **更新** `quality_check_v1.py` | 修复: pandas record_id 类型丢失前导零 → dtype=str |
| 2026-07-10 | **更新** `quality_check_v1.py` | 重构: 按 record 整条读取 → ~10 分钟完成 |
| 2026-07-10 | **更新** `quality_mask_v1.csv` + `quality_report.txt` | 正确结果: 366,041 keep, 3 flat |
| 2026-07-10 | 新建 `random_visualization.py` | Stage 5: 随机抽样 100 窗口 → PNG |
| 2026-07-10 | 生成 `visualization/` (100 PNG + sample_index.csv) | seed=42 |
| 2026-07-10 | **更新** `random_visualization.py` | 修复 subject_id NaN 显示问题 |
| 2026-07-10 | 新建 `PROJECT_LOG.md` | 本文件: 项目日志 |
| 2026-07-10 | 新建 `quality_check_v1_1.py` | Stage 4.1: 新增 4 个 artifact features |
| 2026-07-10 | **更新** `quality_mask_v1.csv` | 新增 4 列: max_abs, spike_count, diff_std, baseline_range |
| 2026-07-10 | 生成 `analysis_report.md` | 4 异常 vs 96 正常样本的比较分析 |
| 2026-07-10 | **更新** `analysis_report.md` | 修复 spike_count inf ratio 排名逻辑 |
| 2026-07-10 | 新建 `visualization_round2.py` | Stage 6: Round 2 分层抽样可视化 (随机+定向) |
| 2026-07-10 | 生成 `visualization_round2/` (100 PNG + round2_index.csv) | 30 baseline_top + 20 diff_top + 10 spike_top + 10 spikecount_top + 30 random |
| 2026-07-10 | 生成 `artifact_labels.csv` | 100 rows: 人工标注 5 种 artifact (50 positive, 50 clean) |
| 2026-07-10 | 新建 `feature_analysis.py` | Stage 7: Feature discriminability analysis (AUC + Cohen's d + boxplots) |
| 2026-07-10 | 生成 `feature_analysis/` (16 boxplots + artifact_feature_analysis.md) | 4 artifact types × 4 features each |
| 2026-07-10 | 新建 `quality_check_v2.py` | Stage 8: V2 质量筛选 (baseline_wander + hf_noise) |
| 2026-07-10 | 生成 `quality_mask_v2.csv` + `quality_check_v2_report.md` | 366,044 rows; V2 keep=362,177 (98.94%) |
| 2026-07-10 | 生成 `V2 候选规则笔记` (聊天框) | baseline_range #1 + diff_std #2 规则文档 |
| 2026-07-13 | 生成 `V3 候选规则 #3 笔记` (聊天框) | max_local_mad 规则文档 |
| 2026-07-13 | 新建 `v3_feature_validation.py` | Stage 9: V3 候选 Feature 验证 (max_local_mad + spike_ratio) |
| 2026-07-13 | **更新** `quality_mask_v2.csv` | 新增 2 列: max_local_mad, spike_ratio (15 cols total) |
| 2026-07-13 | 生成 `v3_validation/` (plots + report) | max_local_mad ✅ AUC=0.893; spike_ratio ❌ AUC=0.133 |
| 2026-07-13 | 生成 `quality_mask_v3.csv` | V3 最终版: 360,365 keep / 5,679 dropped (3 rules) |
| 2026-07-13 | 生成 `review_v3/` (100 PNG + review_list.csv) | V3 keep 池随机复查 seed=42 |
| 2026-07-13 | 生成 `quality_control_validation_v3.md` | V3 最终验证报告: 建议停止开发规则, 进入 Dataset Construction |

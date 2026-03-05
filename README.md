# HFA-Net Research Entry (Publishable Pipeline)

本项目已升级为可发表版研究入口，核心目标是统一训练/实验接口，支持：

- `R2.2` 可切换 baseline：`FPN baseline` vs `HFA enhanced`
- `R3.1` feature-space DSDF（`P2/P3`）
- `R8.2` 一键跑完整对比 + 消融并自动汇总

## 1. 核心架构

统一入口模型在 [src/models/research_hfa.py](E:/温室无人机巡检系统/杂草/src/models/research_hfa.py)：

- Backbone：ConvNeXt（`C2/C3/C4/C5`）
- FPN：`C2-C5 -> P2-P5`
- 分支模式：
  - `--arch fpn`：标准 FPN baseline
  - `--arch hfa`：FPN + DSDF + TSBR
- DSDF 模式：
  - `--dsdf_mode none`
  - `--dsdf_mode logits`
  - `--dsdf_mode feature`（推荐，SCI 主创新）
- DSDF 层级：
  - `--dsdf_levels p2`
  - `--dsdf_levels p2p3`（推荐）

## 2. 训练入口

训练脚本：[src/train.py](E:/温室无人机巡检系统/杂草/src/train.py)

### 2.1 关键参数

- `--arch`: `fpn|hfa|srdnet|unet|deeplabv3plus|segformerb0`
- `--dsdf_mode`: `none|logits|feature`
- `--dsdf_levels`: `p2|p2p3`
- `--no_fac`: 关闭 FAC（默认开启）
- `--cons_weight` + `--cons_warmup_epochs`: 两阶段一致性（warmup 前自动置 0）
- `--edge_weight`: TSBR 边界损失权重
- `--bf_radius`: boundary F1 评估边界宽度
- `--run_dir`: 标准实验输出目录

### 2.2 产物目录（论文友好）

`--run_dir runs/<exp_name>` 下自动生成：

- `config.yaml`
- `metrics.csv`
- `best.ckpt`
- `model_meta.yaml`（参数量/FPS）
- `pred_vis/`（固定样本可视化）
- `train.log`, `train.err`（由 `scripts/run_all.py` 生成）

## 3. 标签与 FAC 验收点

数据集逻辑在 [src/dataset.py](E:/温室无人机巡检系统/杂草/src/dataset.py)：

- 标签统一到 `{0,1,2}`
- 兼容 `{0,255}`、`{0,128,255}`
- `cofly` / `cofly_` 映射到 `weed=2`
- 类别统计与训练使用同一映射逻辑
- FAC 尝试上限：`--max_fac_tries`（默认 3）
- 每 epoch 输出 FAC 统计：`fac_avg_fg_ratio / fac_avg_attempts / fac_samples`

## 4. 一键实验（对比 + 消融）

配置文件：[scripts/experiments.yaml](E:/温室无人机巡检系统/杂草/scripts/experiments.yaml)

执行：

```bash
python scripts/run_all.py --config scripts/experiments.yaml
```

结果汇总：

- `runs/summary_table.csv`

可选仅跑部分实验：

```bash
python scripts/run_all.py --only fpn_base,hfa_feature
```

## 5. 推荐实验矩阵

配置已包含：

- 对比：`unet`, `deeplabv3plus`, `segformer_b0`, `srdnet`, `fpn_base`, `hfa_full`
- 消融：`fpn_fac`, `fpn_fac_tsbr`, `hfa_logits`, `hfa_feature`, `hfa_full`

## 6. 关键指标

每个 epoch 自动记录到 `metrics.csv`：

- `mIoU`
- `IoU(bg/crop/weed)`
- `weed_recall`
- `boundary_f1`
- `lr`
- `cons_weight`
- `FAC` 统计

## 7. 快速命令

### 7.1 FPN baseline

```bash
python src/train.py --data_dir Combined_Dataset --arch fpn --dsdf_mode none --no_fac --edge_weight 0 --cons_weight 0 --run_dir runs/fpn_base
```

### 7.2 HFA full（推荐）

```bash
python src/train.py --data_dir Combined_Dataset --arch hfa --dsdf_mode feature --dsdf_levels p2p3 --edge_weight 0.5 --cons_weight 0.1 --cons_warmup_epochs 10 --run_dir runs/hfa_full
```

### 7.3 边界评估调优（可选）

当 Boundary F1 偏低时可尝试：

```bash
python src/train.py --data_dir Combined_Dataset --arch hfa --bf_radius 2 --run_dir runs/hfa_bf2
```

## 8. 说明

- 当前已停止旧训练进程，避免实验污染。
- 训练恢复请使用 `--resume <best.ckpt>`，checkpoint 已含 `optimizer/scheduler/epoch/best_iou`。

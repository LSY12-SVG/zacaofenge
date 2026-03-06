# HFA-Net Research Entry (Publishable Pipeline)

本项目已升级为可发表版研究入口，核心目标是统一训练/实验接口，支持：

- `R2.2` 可切换 baseline：`FPN baseline` vs `HFA enhanced`
- `R3.1` feature-space DSDF（`P2/P3`）
- `R8.2` 一键跑完整对比 + 消融并自动汇总

补充资料：
- 汇报稿：`PROJECT_TRAINING_REPORT_PLAN.md`
- 命令清单：`training_commands.md`

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
- `--fps_h/--fps_w/--fps_batch_size/--fps_warmup_iters/--fps_timed_iters`: FPS 复现实验协议
- `--run_dir`: 标准实验输出目录

### 2.2 产物目录（论文友好）

`--run_dir runs/<exp_name>` 下自动生成：

- `config.yaml`
- `metrics.csv`
- `best.ckpt`
- `model_meta.yaml`（参数量/FPS + 复现实验元信息）
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

## 4. 指标定义（投稿必备）

- `weed_recall`（像素级）：
  `TP_weed / (TP_weed + FN_weed)`
- `weed_tp/weed_fn/weed_fp`：
  基于像素级预测类别 `weed=2` 与 GT 的混淆统计
- `boundary_f1`：
  先由语义 mask 生成二值边界；再用 `bf_radius` 做容忍匹配（dilate-tolerant），计算 precision/recall/F1
- `mIoU` 与 `per-class IoU`：
  按像素级交并比计算，类别为 `background/crop/weed`

`metrics.csv` 已记录上述指标及来源统计，可直接用于论文表格与答辩追溯。

## 5. FPS/复杂度复现协议

启用 `--estimate_fps` 后，默认采用：

- 输入分辨率：`480x480`
- batch：`1`
- warmup：`20` 次
- timed：`50` 次

`model_meta.yaml` 会记录：

- `params_million`
- `fps`
- `device/device_type`（GPU 型号或 CPU）
- `input_size`
- `batch_size`
- `warmup_iters/timed_iters`
- `torch_version/cuda_version`

## 6. 一键实验（对比 + 消融）

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

## 7. 推荐实验矩阵

配置已包含：

- `Fair baselines (Table 1)`：
  `fair_unet/fair_deeplabv3plus/fair_segformer_b0/fair_srdnet/fair_fpn/fair_hfa`
  （统一 `no FAC + no consistency + 同 epochs/lr/scheduler`）
- `Enhanced & Ablation`：
  `ab_fpn_base/ab_fpn_fac/ab_fpn_fac_tsbr/ab_hfa_logits/ab_hfa_feature/ab_hfa_full`

## 8. 关键指标

每个 epoch 自动记录到 `metrics.csv`：

- `mIoU`
- `IoU(bg/crop/weed)`
- `weed_recall`
- `weed_tp / weed_fn / weed_fp`
- `boundary_f1`
- `lr`
- `cons_weight`
- `FAC` 统计

## 9. 快速命令

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

### 7.4 Fair baseline 示例

```bash
python src/train.py --data_dir Combined_Dataset --arch deeplabv3plus --backbone resnet34 --no_fac --edge_weight 0 --cons_weight 0 --run_dir runs/fair_deeplabv3plus
```

### 7.5 TensorBoard 实时面板

训练时启用 TB 日志：

```bash
python -u src/train.py --data_dir Combined_Dataset --arch hfa --run_dir runs/hfa_full --tensorboard
```

启动 Web 面板（PowerShell）：

```powershell
./scripts/start_tensorboard.ps1 -Logdir runs -Port 6006
```

浏览器打开：

```text
http://127.0.0.1:6006
```

## 10. 说明

- 当前已停止旧训练进程，避免实验污染。
- 训练恢复请使用 `--resume <best.ckpt>`，checkpoint 已含 `optimizer/scheduler/epoch/best_iou`。

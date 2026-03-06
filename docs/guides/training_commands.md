# 训练命令清单（与当前 `src/train.py` 一致）

## 1) 单实验快速训练

### FPN baseline（公平对比）
```bash
python -u src/train.py --data_dir Combined_Dataset --arch fpn --backbone convnext_tiny --no_fac --edge_weight 0 --cons_weight 0 --run_dir runs/fair_fpn
```

### HFA full（推荐主模型）
```bash
python -u src/train.py --data_dir Combined_Dataset --arch hfa --backbone convnext_tiny --dsdf_mode feature --dsdf_levels p2p3 --edge_weight 0.5 --cons_weight 0.1 --cons_warmup_epochs 10 --run_dir runs/ab_hfa_full
```

### DeepLabV3+（公平对比）
```bash
python -u src/train.py --data_dir Combined_Dataset --arch deeplabv3plus --backbone resnet34 --no_fac --edge_weight 0 --cons_weight 0 --run_dir runs/fair_deeplabv3plus
```

## 2) 一键跑实验矩阵

### 全量（fair + ablation）
```bash
python scripts/run_all.py --config scripts/experiments.yaml
```

### 只跑公平主表（6 组）
```bash
python scripts/run_all.py --only fair_unet,fair_deeplabv3plus,fair_segformer_b0,fair_srdnet,fair_fpn,fair_hfa
```

### 只跑消融（6 组）
```bash
python scripts/run_all.py --only ab_fpn_base,ab_fpn_fac,ab_fpn_fac_tsbr,ab_hfa_logits,ab_hfa_feature,ab_hfa_full
```

### 仅检查命令不执行（推荐先做）
```bash
python scripts/run_all.py --only fair_fpn,ab_hfa_feature --dry_run
```

## 3) 实时监控

### 终端指标监控（CSV）
```bash
python scripts/monitor_training.py --metrics_csv runs/ab_hfa_full/metrics.csv --interval 5 --show_gpu
```

### TensorBoard（Web）
```bash
python -u src/train.py --data_dir Combined_Dataset --arch hfa --run_dir runs/ab_hfa_full --tensorboard --tb_logdir C:\tb_logs\ab_hfa_full
```

```powershell
./scripts/start_tensorboard.ps1 -Logdir C:\tb_logs -Port 6006
```

浏览器：
```text
http://127.0.0.1:6006
```

## 4) 关键产物检查

- 每个实验目录：`runs/<exp>/`
  - `config.yaml`
  - `metrics.csv`
  - `best.ckpt`
  - `model_meta.yaml`
  - `pred_vis/`
- 全局汇总：
  - `runs/summary_table.csv`

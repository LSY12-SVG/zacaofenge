# 训练命令快速参考

## 已完成的优化

### ✅ 损失函数优化 (src/loss.py)
- DiceLoss: `include_bg=False` (不计算背景)
- FocalLoss: per-class alpha `[0.1, 1.0, 1.0]`
- SRDNetLoss: 优化后的组合损失

### ✅ 数据集修复 (src/dataset.py)
- 类别统计：使用 `cv2.imread` 替代 `np.fromfile`
- IoU 计算：正确的 per-class 计数

### ✅ 学习率策略 (src/train.py)
- CosineAnnealingLR (无周期重启)
- 平滑衰减从 1e-4 到 1e-6

### ✅ HFA-Net 模型 (src/hfa_model.py)
- DSDF-lite: logits 空间语义 - 细节解耦
- TSBR: 边界细化分支
- 参数量：42.96M

---

## 训练命令

### 1. 训练 SRDNet (优化后的损失函数)
```bash
python src\train.py --data_dir "Combined_Dataset" --model srdnet --backbone convnext_tiny --epochs 50 --batch_size 8 --lr 0.0001 --loss srdnet_loss --dice_weight 1.0 --focal_weight 1.0 --boundary_weight 0.5 --save_name "srdnet_best.pth"
```

### 2. 训练 SRDNet (从 checkpoint 恢复)
```bash
python src\train.py --data_dir "Combined_Dataset" --model srdnet --backbone convnext_tiny --epochs 50 --batch_size 8 --lr 0.0001 --loss srdnet_loss --save_name "srdnet_best.pth" --resume "srdnet_best.pth"
```

### 3. 训练 HFA-Net (需要更新 train.py 支持)
```bash
python src\train.py --data_dir "Combined_Dataset" --model hfa --backbone convnext_tiny --epochs 50 --batch_size 8 --lr 0.0001 --loss hfa_loss --save_name "hfa_best.pth"
```

---

## 模型对比

| 模型 | Backbone | 参数量 | 特点 |
|------|----------|--------|------|
| SRDNet | ConvNeXt-Tiny | ~40M | 结构残差解耦 |
| HFA-Net | ConvNeXt-Tiny | 42.96M | +DSDF-lite +TSBR |

---

## 预期性能提升

### 优化前 (baseline)
- Overall IoU: ~52%
- Weed IoU: ~36%

### 优化后 (预期)
- Overall IoU: ~55-58%
- Weed IoU: ~40-45%

### 关键改进
1. ✅ Dice 不算背景 → 更关注前景
2. ✅ Focal per-class alpha → 解决类别不平衡
3. ✅ 正确的 IoU 统计 → 准确评估
4. ✅ 稳定 LR 策略 → 避免性能波动
5. 🆕 HFA-Net → 边界细化 + 语义解耦

---

## 训练曲线可视化

训练完成后自动生成：
- `training_plots/training_curves.png` - 损失和 IoU 曲线
- `training_plots/training_history.npz` - 历史数据

---

## 常见问题

### Q: 为什么第一次类别统计显示 Crop/Weed 为 0？
A: 这是正常的。第一次统计的是 train 数据集（使用 np.fromfile 可能失败），第二次统计 val 数据集（正常）。已修复为使用 cv2.imread。

### Q: IoU 计算是否正确？
A: 已修复。现在使用实际计数而不是 num_batches，避免人为降低 IoU。

### Q: 学习率会重启吗？
A: 不会。已改为 CosineAnnealingLR，平滑衰减无重启。

### Q: HFA-Net 需要修改 SRDNet 吗？
A: 不需要。HFA-Net 是 wrapper 设计，完全无侵入。

---

## 下一步

1. 运行 SRDNet 训练（优化后的损失）
2. 运行 HFA-Net 训练
3. 对比性能差异
4. 生成消融实验结果

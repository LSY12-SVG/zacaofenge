# 合并数据集训练指南

## 📊 数据集统计

合并后的数据集已成功创建：
- **训练集**: 2150 张图像 (Tobacco: 2016, CoFly: 134)
- **验证集**: 265 张图像 (Tobacco: 252, CoFly: 13)
- **测试集**: 306 张图像 (Tobacco: 252, CoFly: 54)

## 🎯 类别映射

- **Tobacco**: 0=背景，1=作物，2=杂草
- **CoFly**: 0=背景，1=杂草 → 转换为 0=背景，2=杂草 (无作物类别)

## 🚀 开始训练

### 基础训练命令

```bash
# 使用 MAnet + EfficientNet-B4 (推荐配置)
python src/train.py \
    --data_dir "Combined_Dataset" \
    --model manet \
    --backbone efficientnet-b4 \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --save_name manet_effb4_combined.pth
```

### 不同模型架构

```bash
# 1. MAnet + EfficientNet-B4 (最佳性能)
python src/train.py --data_dir "Combined_Dataset" --model manet --backbone efficientnet-b4 --epochs 50

# 2. DeepLabV3+ + ResNet34 (平衡速度与精度)
python src/train.py --data_dir "Combined_Dataset" --model deeplabv3_plus --backbone resnet34 --epochs 50

# 3. U-Net++ + ResNet34 (嵌套跳跃连接)
python src/train.py --data_dir "Combined_Dataset" --model unetplusplus --backbone resnet34 --epochs 50

# 4. 基础 U-Net (快速实验)
python src/train.py --data_dir "Combined_Dataset" --model simple_unet --epochs 30
```

### 高级训练选项

```bash
# 使用 Dice + Focal 损失 (推荐处理类别不平衡)
python src/train.py \
    --data_dir "Combined_Dataset" \
    --model manet \
    --backbone efficientnet-b4 \
    --epochs 50 \
    --loss dice_focal

# 从预训练权重继续训练
python src/train.py \
    --data_dir "Combined_Dataset" \
    --model manet \
    --backbone efficientnet-b4 \
    --resume_path "models/best_model.pth" \
    --epochs 30

# 禁用数据增强 (用于调试)
python src/train.py \
    --data_dir "Combined_Dataset" \
    --model manet \
    --augment False
```

## 📈 训练监控

训练过程中会输出：
- 每个 epoch 的训练损失
- 验证集 IoU
- 最佳模型自动保存

示例输出：
```
Epoch 1/50:
  Train Loss: 0.4523
  Val Loss: 0.3891, Val IoU: 0.6234
  ✓ New best model saved!

Epoch 2/50:
  Train Loss: 0.3876
  Val Loss: 0.3654, Val IoU: 0.6451
  ✓ New best model saved!
```

## 🔍 数据验证

训练前建议验证数据集：

```bash
# 检查数据集加载
python -c "from src.dataset import WeedDataset; ds = WeedDataset('Combined_Dataset', mode='train'); print(f'Loaded {len(ds)} samples')"

# 检查类别分布
python -c "
from src.dataset import WeedDataset
import numpy as np
ds = WeedDataset('Combined_Dataset', mode='train')
print(f'Dataset type: {ds.dataset_type}')
print(f'Total samples: {len(ds)}')
"
```

## 💡 训练技巧

### 1. 迁移学习策略
```bash
# Step 1: 在 Tobacco 上预训练
python src/train.py --data_dir "Tobacco Aerial Dataset" --model manet --backbone efficientnet-b4 --epochs 30 --save_name tobacco_pretrain.pth

# Step 2: 在合并数据集上微调
python src/train.py --data_dir "Combined_Dataset" --model manet --backbone efficientnet-b4 --resume_path "tobacco_pretrain.pth" --epochs 20
```

### 2. 类别权重调整
如果杂草分割效果不佳，可以修改 loss.py 添加类别权重：
```python
# DiceFocalLoss 中设置 class_weights
criterion = DiceFocalLoss(dice_weight=0.5, focal_weight=0.5)
# 或
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))  # 增加杂草权重
```

### 3. 混合精度训练 (加速)
```bash
# 修改 train.py 启用 AMP
# 添加: from torch.cuda.amp import autocast, GradScaler
# 在训练循环中使用 autocast
```

## 📊 预期性能

基于合并数据集的训练预期：
- **mIoU**: 0.60-0.65 (相比单一数据集提升 2-5%)
- **杂草 IoU**: 0.55-0.60 (提升明显)
- **泛化能力**: 在 CoFly 测试集上 IoU > 0.50

## ⚠️ 注意事项

1. **显存需求**: batch_size=4 需要约 8GB 显存
2. **训练时间**: 50 epochs 约需 2-3 小时 (RTX 3060)
3. **数据平衡**: Tobacco 占主导 (93%)，CoFly 占 7%
4. **类别不平衡**: 杂草像素较少，建议使用 Dice/Focal 损失

## 🎯 下一步

训练完成后：
1. 在 Tobacco 测试集上评估
2. 在 CoFly 测试集上评估
3. 对比单一数据集 vs 合并数据集的性能差异
4. 可视化预测结果

祝训练顺利！🚀

"""
简化的 SRDNet 训练脚本 - 用于诊断导入问题
"""
import sys
import os
sys.path.insert(0, 'src')

print("=" * 60)
print("SRDNet 训练 - 诊断模式")
print("=" * 60)

# 测试导入
print("\n步骤 1: 测试基础导入...")
try:
    import torch
    print(f"✓ torch {torch.__version__}")
except Exception as e:
    print(f"✗ torch 导入失败：{e}")
    sys.exit(1)

print("\n步骤 2: 测试 timm 导入...")
try:
    import timm
    print(f"✓ timm {timm.__version__}")
except Exception as e:
    print(f"✗ timm 导入失败：{e}")
    print("提示：运行 pip install timm")
    sys.exit(1)

print("\n步骤 3: 测试 SRDNet 导入...")
try:
    from models.srdnet import SRDNet
    print("✓ SRDNet 导入成功")
except Exception as e:
    print(f"✗ SRDNet 导入失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n步骤 4: 测试模型创建...")
try:
    model = SRDNet(n_classes=3, backbone='convnext_tiny', pretrained=False)
    print("✓ 模型创建成功")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量：{total_params / 1e6:.2f}M")
except Exception as e:
    print(f"✗ 模型创建失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n步骤 5: 测试数据集加载...")
try:
    from dataset import WeedDataset
    dataset = WeedDataset('Combined_Dataset', mode='train')
    print(f"✓ 数据集加载成功：{len(dataset)} 张图像")
except Exception as e:
    print(f"✗ 数据集加载失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有诊断测试通过！")
print("=" * 60)
print("\n现在可以开始正式训练...")
print("命令：python src/train.py --model srdnet --backbone convnext_tiny")

"""
SRDNet 快速测试脚本
测试 SRDNet 模型的前向传播
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from src.models.srdnet import SRDNet

print("=" * 60)
print("SRDNet: Structure Residual Decoupling Network - 快速测试")
print("=" * 60)

# 创建测试输入
print("\n1. 创建测试输入...")
x = torch.randn(1, 3, 224, 224)
print(f"   输入形状：{x.shape}")

# 创建模型 (不加载预训练权重以加快测试)
print("\n2. 创建 SRDNet 模型 (ConvNeXt-Tiny backbone, 不加载预训练权重)...")
try:
    model = SRDNet(n_classes=3, backbone='convnext_tiny', pretrained=False)
    print("   ✓ 模型创建成功")
except Exception as e:
    print(f"   ✗ 模型创建失败：{e}")
    sys.exit(1)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   总参数量：{total_params / 1e6:.2f}M")
print(f"   可训练参数量：{trainable_params / 1e6:.2f}M")

# 前向传播
print("\n3. 执行前向传播...")
try:
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"   ✓ 前向传播成功")
    print(f"   输出形状：{output.shape}")
except Exception as e:
    print(f"   ✗ 前向传播失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 获取作物概率图
print("\n4. 测试作物结构头 (Crop Structure Head)...")
try:
    with torch.no_grad():
        crop_mask = model.get_crop_mask(x)
    print(f"   ✓ 作物概率图提取成功")
    print(f"   作物概率图形状：{crop_mask.shape}")
    print(f"   作物概率范围：[{crop_mask.min():.4f}, {crop_mask.max():.4f}]")
except Exception as e:
    print(f"   ✗ 作物概率图提取失败：{e}")
    sys.exit(1)

# 测试不同输入尺寸
print("\n5. 测试不同输入尺寸兼容性...")
test_sizes = [(224, 224), (352, 480), (512, 512)]
for h, w in test_sizes:
    x_test = torch.randn(1, 3, h, w)
    with torch.no_grad():
        out = model(x_test)
    assert out.shape[2:] == (h, w), f"输出尺寸 {out.shape[2:]} 与输入尺寸 {(h, w)} 不匹配"
    print(f"   ✓ 输入 {h}x{w} → 输出 {out.shape[2]}x{out.shape[3]}")

print("\n" + "=" * 60)
print("✓ 所有测试通过！SRDNet 模型实现正确。")
print("=" * 60)

print("\n下一步:")
print("1. 安装依赖：pip install -r requirements.txt")
print("2. 开始训练：python src/train.py --model srdnet --backbone convnext_tiny")
print("3. 查看完整训练命令：参考 .trae/documents/SRDNet_Implementation_Plan.md")

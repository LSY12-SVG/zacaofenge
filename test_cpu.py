"""
SRDNet 训练脚本 - CPU 模式 (快速测试)
"""
import os
# 设置环境变量，禁用 CUDA 加速导入
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("SRDNet 训练 - CPU 模式")
print("=" * 60)

print("\n导入 torch...")
import torch
print(f"✓ torch {torch.__version__}")
print(f"  CUDA 可用：{torch.cuda.is_available()}")

print("\n导入 timm...")
import timm
print(f"✓ timm {timm.__version__}")

print("\n导入 SRDNet...")
from models.srdnet import SRDNet
print("✓ SRDNet 导入成功")

print("\n创建模型...")
model = SRDNet(n_classes=3, backbone='convnext_tiny', pretrained=False)
print("✓ 模型创建成功")

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"  参数量：{total_params / 1e6:.2f}M")

print("\n前向传播测试...")
model.eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(x)
print(f"✓ 测试通过：{x.shape} → {output.shape}")

print("\n" + "=" * 60)
print("✓ SRDNet 可以正常使用！")
print("=" * 60)

# 现在可以开始训练了
print("\n提示：如果要使用 GPU 训练，请确保 CUDA 正常配置")
print("如果使用 CPU 训练，添加环境变量：CUDA_VISIBLE_DEVICES=''")

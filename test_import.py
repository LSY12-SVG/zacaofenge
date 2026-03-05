"""
测试 SRDNet 导入和模型创建
"""
import sys
import time
sys.path.insert(0, 'src')

start = time.time()
print(f"开始时间：{time.strftime('%H:%M:%S')}")

print("\n[1/5] 导入 torch...")
t0 = time.time()
import torch
print(f"   ✓ torch 导入成功 (耗时：{time.time()-t0:.2f}s)")
print(f"   版本：{torch.__version__}")

print("\n[2/5] 导入 timm...")
t0 = time.time()
import timm
print(f"   ✓ timm 导入成功 (耗时：{time.time()-t0:.2f}s)")
print(f"   版本：{timm.__version__}")

print("\n[3/5] 导入 SRDNet 模型...")
t0 = time.time()
from models.srdnet import SRDNet
print(f"   ✓ SRDNet 导入成功 (耗时：{time.time()-t0:.2f}s)")

print("\n[4/5] 创建模型 (不加载预训练权重)...")
t0 = time.time()
model = SRDNet(n_classes=3, backbone='convnext_tiny', pretrained=False)
print(f"   ✓ 模型创建成功 (耗时：{time.time()-t0:.2f}s)")

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"   参数量：{total_params / 1e6:.2f}M")

print("\n[5/5] 前向传播测试...")
t0 = time.time()
x = torch.randn(1, 3, 224, 224)
model.eval()
with torch.no_grad():
    output = model(x)
print(f"   ✓ 前向传播成功 (耗时：{time.time()-t0:.2f}s)")
print(f"   输入：{x.shape} → 输出：{output.shape}")

print(f"\n总耗时：{time.time()-start:.2f}s")
print("\n✓ 所有测试通过！SRDNet 可以正常使用")

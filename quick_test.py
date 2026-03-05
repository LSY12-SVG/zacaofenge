import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("SRDNet 快速测试")
print("=" * 60)

print("\n1. 导入依赖...")
import torch
print(f"   ✓ PyTorch 版本：{torch.__version__}")

print("\n2. 导入 SRDNet 模型...")
from models.srdnet import SRDNet
print("   ✓ SRDNet 导入成功")

print("\n3. 创建模型 (ConvNeXt-Tiny, 不加载预训练权重)...")
model = SRDNet(n_classes=3, backbone='convnext_tiny', pretrained=False)
print("   ✓ 模型创建成功")

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"   参数量：{total_params / 1e6:.2f}M")

print("\n4. 执行前向传播测试...")
x = torch.randn(1, 3, 224, 224)
model.eval()
with torch.no_grad():
    output = model(x)
print(f"   ✓ 测试通过！")
print(f"   输入：{x.shape} → 输出：{output.shape}")

print("\n" + "=" * 60)
print("✓ SRDNet 模型测试成功！可以开始训练")
print("=" * 60)

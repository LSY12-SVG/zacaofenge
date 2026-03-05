"""
详细调试导入过程
"""
import sys
import time
import traceback

print("=" * 60)
print("Python 导入调试")
print("=" * 60)
print(f"Python 版本：{sys.version}")
print(f"Python 路径：{sys.executable}")
print(f"当前目录：{sys.path[0]}")
print(f"工作目录：{sys.path}")
print()

# 记录每个导入步骤的时间
start = time.time()

print("[0/10] 开始导入测试...")
print(f"时间：{time.time() - start:.2f}s")

try:
    print("\n[1/10] 导入 sys...")
    t0 = time.time()
    # sys 已经导入
    print(f"✓ 完成 (耗时：{time.time()-t0:.2f}s)")
except Exception as e:
    print(f"✗ 失败：{e}")
    traceback.print_exc()

try:
    print("\n[2/10] 导入 os...")
    t0 = time.time()
    import os
    print(f"✓ 完成 (耗时：{time.time()-t0:.2f}s)")
except Exception as e:
    print(f"✗ 失败：{e}")
    traceback.print_exc()

try:
    print("\n[3/10] 导入 time...")
    t0 = time.time()
    import time
    print(f"✓ 完成 (耗时：{time.time()-t0:.2f}s)")
except Exception as e:
    print(f"✗ 失败：{e}")
    traceback.print_exc()

try:
    print("\n[4/10] 导入 numpy...")
    t0 = time.time()
    import numpy
    print(f"✓ numpy {numpy.__version__} (耗时：{time.time()-t0:.2f}s)")
except Exception as e:
    print(f"✗ 失败：{e}")
    traceback.print_exc()

try:
    print("\n[5/10] 导入 torch (这可能需要几分钟)...")
    t0 = time.time()
    import torch
    print(f"✓ torch {torch.__version__} (耗时：{time.time()-t0:.2f}s)")
    print(f"  CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 版本：{torch.version.cuda}")
        print(f"  GPU 数量：{torch.cuda.device_count()}")
except Exception as e:
    print(f"✗ 失败：{e}")
    traceback.print_exc()
    print("\n提示：这可能是由于 CUDA 初始化问题")
    print("尝试：pip uninstall torch torchvision torchaudio")
    print("然后：pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")

try:
    print("\n[6/10] 导入 timm...")
    t0 = time.time()
    import timm
    print(f"✓ timm {timm.__version__} (耗时：{time.time()-t0:.2f}s)")
except Exception as e:
    print(f"✗ 失败：{e}")
    traceback.print_exc()

try:
    print("\n[7/10] 导入 torchvision...")
    t0 = time.time()
    import torchvision
    print(f"✓ torchvision {torchvision.__version__} (耗时：{time.time()-t0:.2f}s)")
except Exception as e:
    print(f"✗ 失败：{e}")
    traceback.print_exc()

try:
    print("\n[8/10] 导入 segmentation_models_pytorch...")
    t0 = time.time()
    import segmentation_models_pytorch
    print(f"✓ SMP {segmentation_models_pytorch.__version__} (耗时：{time.time()-t0:.2f}s)")
except Exception as e:
    print(f"✗ 失败：{e}")
    traceback.print_exc()

try:
    print("\n[9/10] 导入 albumentations...")
    t0 = time.time()
    import albumentations
    print(f"✓ albumentations {albumentations.__version__} (耗时：{time.time()-t0:.2f}s)")
except Exception as e:
    print(f"✗ 失败：{e}")
    traceback.print_exc()

try:
    print("\n[10/10] 导入 SRDNet...")
    t0 = time.time()
    sys.path.insert(0, 'src')
    from models.srdnet import SRDNet
    print(f"✓ SRDNet 导入成功 (耗时：{time.time()-t0:.2f}s)")
    
    # 创建模型
    print("\n创建 SRDNet 模型...")
    model = SRDNet(n_classes=3, backbone='convnext_tiny', pretrained=False)
    print(f"✓ 模型创建成功")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量：{total_params / 1e6:.2f}M")
except Exception as e:
    print(f"✗ 失败：{e}")
    traceback.print_exc()

print(f"\n总耗时：{time.time() - start:.2f}s")
print("=" * 60)

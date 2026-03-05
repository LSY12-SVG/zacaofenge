import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.srdnet import SRDNet
    SRDNET_AVAILABLE = True
except Exception:
    SRDNET_AVAILABLE = False


class DSDFLite(nn.Module):
    """
    DSDF-lite: 在 logits/feature 空间做语义 - 细节解耦 + 门控融合
    说明：为保证与现有 SRDNet 完全兼容，这里不依赖 backbone 中间特征。
    """
    def __init__(self, channels: int):
        super().__init__()
        # Semantic branch (更平滑 / 语义聚合)
        self.semantic = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # Detail branch (高频 / 边缘细节)
        self.detail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # Gating
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.semantic(x)
        d = self.detail(x)
        g = self.gate(torch.cat([s, d], dim=1))
        return s + g * d


class EdgeHead(nn.Module):
    """TSBR：细结构边界分支（1 通道 edge map）"""
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class HFANet(nn.Module):
    """
    HFA-Net (wrapper):
      - base: SRDNet (输出 logits: [B, C, H, W])
      - dsdf-lite: logits-space decoupling refinement
      - edge head: boundary refinement branch (TSBR)
    forward 返回：
      - logits: [B, C, H, W]
      - edge_logits: [B, 1, H, W]
    """
    def __init__(self, n_classes=3, backbone="convnext_tiny", pretrained=True):
        super().__init__()
        if not SRDNET_AVAILABLE:
            raise ImportError("SRDNet not available. Please ensure models/srdnet.py and dependencies are installed.")

        self.base = SRDNet(n_classes=n_classes, backbone=backbone, pretrained=pretrained)

        # 这里 channels 用 n_classes（logits 通道数），保证无侵入、稳定可跑
        self.dsdf = DSDFLite(channels=n_classes)

        # edge head 用 refinement 后的 logits 作为输入特征（简单但有效）
        self.edge_head = EdgeHead(in_ch=n_classes)

    def forward(self, x: torch.Tensor):
        logits = self.base(x)  # [B,C,H,W]
        refined = self.dsdf(logits)
        edge_logits = self.edge_head(refined)
        return refined, edge_logits


if __name__ == '__main__':
    # 测试
    if SRDNET_AVAILABLE:
        print("=" * 60)
        print("HFA-Net 模型测试")
        print("=" * 60)
        
        x = torch.randn(2, 3, 352, 480)
        model = HFANet(n_classes=3, backbone='convnext_tiny', pretrained=False)
        
        print("\n前向传播测试...")
        model.train()
        logits, edge_logits = model(x)
        
        print(f"✓ 输入：{x.shape}")
        print(f"✓ 输出 logits: {logits.shape}")
        print(f"✓ 输出 edge_logits: {edge_logits.shape}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n参数量统计:")
        print(f"  总参数量：{total_params / 1e6:.2f}M")
        print(f"  可训练参数：{trainable_params / 1e6:.2f}M")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
    else:
        print("SRDNet not available, skipping test")

import torch
import torch.nn as nn


class ResidualFeatureExtractor(nn.Module):
    """
    Residual Feature Extractor: 残差特征提取器
    
    功能:
    - 数学表达：F_r = F_b - α·F_c
    - α 为可学习参数
    - 实现"去作物化"特征增强
    
    动机:
    - 作物和杂草高度相似 → CNN 容易混淆
    - 从原始特征中"减去作物特征"可以增强杂草
    - 语义层面的 subtractive decoupling
    
    创新点:
    - 目前主流分割模型 (DeepLab, SegFormer, Mask2Former) 都没有显式"语义残差抑制"
    - 在农业场景是有意义的创新
    """
    
    def __init__(self, channels, alpha_init=1.0):
        """
        Args:
            channels: 特征通道数
            alpha_init: α 参数初始值
        """
        super().__init__()
        # 可学习参数 α
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        # 将 crop mask 映射到特征空间的卷积
        self.crop_feature_conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合卷积 (可选，增强表达能力)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, F_b, crop_mask, crop_feat=None):
        """
        Args:
            F_b: Backbone 特征，shape: [B, C, H, W]
            crop_mask: 作物概率图，shape: [B, 1, H, W]
            crop_feat (可选): 作物特征，shape: [B, C, H, W]
        
        Returns:
            F_r: 残差特征 (杂草增强特征), shape: [B, C, H, W]
        """
        # 优先使用 crop_feat (如果 crop_head 输出了特征)
        # 否则使用 crop_mask 投影到特征空间
        if crop_feat is not None:
            # 修复：使用 detach 防止 crop_head 学到 trivial solution
            F_c = crop_feat.detach()
        else:
            # 将 crop mask 映射到特征空间
            F_c = self.crop_feature_conv(crop_mask)
        
        # 残差解耦：从原始特征中减去作物特征
        # 修复：F_r = F_b - α * F_c (正确的语义减法)
        F_r = F_b - self.alpha * F_c
        
        # 特征融合增强
        F_r = self.feature_fusion(F_r)
        
        return F_r


if __name__ == '__main__':
    # 测试
    B, C, H, W = 2, 768, 11, 15
    F_b = torch.randn(B, C, H, W)
    crop_mask = torch.rand(B, 1, H, W)
    crop_feat = torch.randn(B, C, H, W)
    
    print("测试 ResidualFeatureExtractor...")
    
    # 测试 1: 使用 crop_mask
    model = ResidualFeatureExtractor(channels=C)
    F_r1 = model(F_b, crop_mask, crop_feat=None)
    print(f"\n使用 crop_mask:")
    print(f"  Backbone feature: {F_b.shape}")
    print(f"  Crop mask: {crop_mask.shape}")
    print(f"  Residual feature: {F_r1.shape}")
    print(f"  Alpha: {model.alpha.item():.4f}")
    
    # 测试 2: 使用 crop_feat (带 detach)
    F_r2 = model(F_b, crop_mask, crop_feat=crop_feat)
    print(f"\n使用 crop_feat (带 detach):")
    print(f"  Residual feature: {F_r2.shape}")

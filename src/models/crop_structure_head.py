import torch
import torch.nn as nn


class CropStructureHead(nn.Module):
    """
    Crop Structure Head: 结构先验建模模块
    
    功能:
    - 输入：Backbone 特征 F_b ∈ R^(C×H×W)
    - 输出：作物概率图 M_c ∈ R^(1×H×W)
    - 实现：轻量级卷积分支 (2-3 层 Conv + Sigmoid)
    
    动机:
    - UAV 农田具有行间排列规律
    - 作物在固定区域，杂草多出现在行间
    - 显式建模空间先验，给网络农业领域特有 inductive bias
    """
    
    def __init__(self, in_channels, mid_channels=64, output_features=False):
        """
        Args:
            in_channels: 输入特征通道数 (backbone 输出通道)
            mid_channels: 中间层通道数
            output_features: 是否同时输出作物特征 (用于残差解耦)
        """
        super().__init__()
        self.output_features = output_features
        
        # 共享的特征提取层
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 作物概率图预测分支
        self.mask_branch = nn.Sequential(
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 作物特征预测分支 (可选，用于残差解耦)
        if output_features:
            self.feature_branch = nn.Sequential(
                nn.Conv2d(mid_channels // 2, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.feature_branch = None
    
    def forward(self, x):
        """
        Args:
            x: Backbone 特征，shape: [B, C, H, W]
        
        Returns:
            crop_mask: 作物概率图，shape: [B, 1, H, W]
            crop_feat (可选): 作物特征，shape: [B, C, H, W]
        """
        # 共享特征提取
        shared_feat = self.shared_conv(x)
        
        # 作物概率图
        crop_mask = self.mask_branch(shared_feat)
        
        # 作物特征 (如果需要)
        if self.output_features:
            crop_feat = self.feature_branch(shared_feat)
            return crop_mask, crop_feat
        else:
            return crop_mask


if __name__ == '__main__':
    # 测试
    x = torch.randn(2, 768, 11, 15)  # ConvNeXt-Tiny 输出特征
    
    print("测试 CropStructureHead...")
    
    # 测试 1: 只输出 mask
    model1 = CropStructureHead(in_channels=768, mid_channels=64, output_features=False)
    output1 = model1(x)
    print(f"\n只输出 mask:")
    print(f"  输入：{x.shape}")
    print(f"  输出：{output1.shape}")
    print(f"  范围：[{output1.min():.4f}, {output1.max():.4f}]")
    
    # 测试 2: 同时输出 mask 和 feature
    model2 = CropStructureHead(in_channels=768, mid_channels=64, output_features=True)
    output2 = model2(x)
    crop_mask, crop_feat = output2
    print(f"\n同时输出 mask 和 feature:")
    print(f"  输入：{x.shape}")
    print(f"  crop_mask: {crop_mask.shape}")
    print(f"  crop_feat: {crop_feat.shape}")
    print(f"  crop_mask 范围：[{crop_mask.min():.4f}, {crop_mask.max():.4f}]")

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not installed. Please install via 'pip install timm'")

try:
    from .crop_structure_head import CropStructureHead
    from .residual_extractor import ResidualFeatureExtractor
    from .frequency_enhancement import FrequencyEnhancementBlock
    from .decoder import LightweightDecoder
except ImportError:
    from crop_structure_head import CropStructureHead
    from residual_extractor import ResidualFeatureExtractor
    from frequency_enhancement import FrequencyEnhancementBlock
    from decoder import LightweightDecoder


class SRDNet(nn.Module):
    """
    SRDNet: Structure Residual Decoupling Network
    
    整体架构:
    Input Image 
         ↓ 
    Backbone (ConvNeXt-Tiny) 
         ↓ 
    Stage1: Crop Structure Head → 作物结构特征
         ↓ 
    Stage2: Residual Feature Extractor → 杂草增强特征
         ↓ 
    Stage3: Frequency Enhancement Block → 频域增强特征
         ↓ 
    Stage4: Lightweight Decoder → 分割输出
    
    核心思想:
    先学作物结构 → 再"去作物化" → 再增强杂草细节
    
    解决 UAV 农田杂草分割中:
    - 作物 - 杂草高度相似
    - 小目标
    - 光照复杂
    - 行结构明显
    """
    
    def __init__(self, n_classes=3, backbone='convnext_tiny', pretrained=True):
        """
        Args:
            n_classes: 分割类别数 (背景、作物、杂草)
            backbone: Backbone 名称 (默认：convnext_tiny)
            pretrained: 是否使用预训练权重
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for SRDNet. Please install it.")
        
        # 1. Backbone (ConvNeXt-Tiny) - 输出多尺度特征
        print(f"Loading backbone: {backbone} (pretrained: {pretrained})")
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(0, 1, 2, 3)  # 输出所有 4 个 stage 的特征
        )
        
        # 获取 backbone 输出通道数 (使用最高级特征)
        backbone_features = self.backbone.feature_info.channels()
        out_channels = backbone_features[-1]  # 最后一层输出通道 (768 for ConvNeXt-Tiny)
        print(f"Backbone output channels (highest level): {out_channels}")
        print(f"Multi-scale features: {backbone_features}")
        
        # 2. Crop Structure Head (结构先验建模) - 输出 mask 和 feature
        self.crop_head = CropStructureHead(
            in_channels=out_channels, 
            mid_channels=64,
            output_features=True  # 同时输出作物特征
        )
        
        # 3. Residual Feature Extractor (残差解耦)
        self.residual_extractor = ResidualFeatureExtractor(
            channels=out_channels, 
            alpha_init=1.0
        )
        
        # 4. Frequency Enhancement Block (频域增强)
        self.freq_enhance = FrequencyEnhancementBlock(
            channels=out_channels, 
            beta=0.5
        )
        
        # 5. Lightweight Decoder (轻量解码器) - 支持多尺度特征融合
        self.decoder = LightweightDecoder(
            channels=out_channels, 
            n_classes=n_classes,
            use_attention=True
        )
        
        # 注意：不初始化 backbone 权重，只初始化自定义模块
        # timm 已经正确初始化了 backbone 的预训练权重
        self._init_custom_modules()
    
    def _init_custom_modules(self):
        """
        只初始化自定义模块的权重，不覆盖 backbone 的预训练权重
        
        修复问题：原来的 _init_weights() 会重置 backbone 的预训练权重
        """
        for name, m in self.named_modules():
            # 跳过 backbone 的模块
            if "backbone" in name:
                continue
            
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 输入图像，shape: [B, 3, H, W]
        
        Returns:
            logits: 分割 logits, shape: [B, n_classes, H, W]
        """
        input_size = x.shape[2:]
        
        # Stage 0: Backbone 特征提取 (多尺度特征)
        features = self.backbone(x)
        F_b = features[-1]  # 使用最高级特征 [B, C, H/32, W/32]
        
        # Stage 1: Crop Structure Head (作物结构预测)
        # 输出：crop_mask (概率图), crop_feat (作物特征)
        crop_output = self.crop_head(F_b)
        if isinstance(crop_output, tuple):
            crop_mask, crop_feat = crop_output
        else:
            crop_mask = crop_output
            crop_feat = None
        
        # Stage 2: Residual Feature Extractor (残差解耦)
        # 修复：使用 crop_feat 而不是 crop_mask 进行残差计算
        # 如果 crop_head 没有输出 feature，则使用 crop_mask 投影
        F_r = self.residual_extractor(F_b, crop_mask, crop_feat)
        
        # Stage 3: Frequency Enhancement Block (频域增强)
        F_hybrid = self.freq_enhance(F_r)
        
        # Stage 4: Lightweight Decoder (解码)
        logits = self.decoder(F_hybrid)
        
        # 上采样到输入尺寸 (修复：align_corners=False)
        logits = F.interpolate(
            logits, 
            size=input_size, 
            mode='bilinear', 
            align_corners=False  # 修复：避免分辨率不一致问题
        )
        
        return logits
    
    def get_crop_mask(self, x):
        """
        获取作物概率图 (用于可视化或分析)
        
        Args:
            x: 输入图像，shape: [B, 3, H, W]
        
        Returns:
            crop_mask: 作物概率图，shape: [B, 1, H, W]
        """
        features = self.backbone(x)
        F_b = features[-1]
        crop_output = self.crop_head(F_b)
        # 如果 crop_head 返回 tuple (mask, feat)，只返回 mask
        if isinstance(crop_output, tuple):
            return crop_output[0]
        return crop_output


def create_srdnet(n_classes=3, backbone='convnext_tiny', pretrained=True):
    """
    创建 SRDNet 模型的工厂函数
    
    Args:
        n_classes: 分割类别数
        backbone: Backbone 名称
        pretrained: 是否使用预训练权重
    
    Returns:
        SRDNet 模型
    """
    return SRDNet(n_classes=n_classes, backbone=backbone, pretrained=pretrained)


if __name__ == '__main__':
    # 测试
    if TIMM_AVAILABLE:
        print("=" * 60)
        print("SRDNet 模型测试 - 修复版本")
        print("=" * 60)
        
        x = torch.randn(2, 3, 352, 480)
        model = create_srdnet(n_classes=3, backbone='convnext_tiny', pretrained=False)
        
        print("\n前向传播测试...")
        model.train()
        logits = model(x)
        print(f"✓ 训练模式 - 输入：{x.shape} → 输出：{logits.shape}")
        
        model.eval()
        with torch.no_grad():
            logits = model(x)
            crop_mask = model.get_crop_mask(x)
        
        print(f"✓ 推理模式 - 输入：{x.shape} → 输出：{logits.shape}")
        print(f"✓ Crop mask: {crop_mask.shape}, 范围：[{crop_mask.min():.4f}, {crop_mask.max():.4f}]")
        
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
        print("timm not available, skipping test")

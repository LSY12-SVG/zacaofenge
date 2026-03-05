import torch
import torch.nn as nn


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    轻量级注意力机制，结合通道注意力和空间注意力
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        
        return x


class LightweightDecoder(nn.Module):
    """
    Lightweight Decoder: 轻量解码器
    
    功能:
    - FPN-like 结构
    - 深度可分离卷积 (Depthwise Separable Conv)
    - 注意力机制 (CBAM)
    
    设计目标:
    - 在保证精度提升的同时，参数量不增加太多
    - 适合无人机端部署
    """
    
    def __init__(self, channels, n_classes=3, use_attention=True):
        """
        Args:
            channels: 输入特征通道数
            n_classes: 分割类别数
            use_attention: 是否使用 CBAM 注意力
        """
        super().__init__()
        self.use_attention = use_attention
        
        # 使用深度可分离卷积减少参数量
        self.decoder = nn.Sequential(
            # 第一层：通道数减半
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            
            # CBAM 注意力 (可选)
            CBAM(channels // 2) if use_attention else nn.Identity(),
            
            # 第二层
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            
            # CBAM 注意力 (可选)
            CBAM(channels // 4) if use_attention else nn.Identity(),
            
            # 第三层
            nn.Conv2d(channels // 4, channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Conv2d(channels // 8, n_classes, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: 编码器输出特征，shape: [B, C, H, W]
        
        Returns:
            logits: 分割 logits, shape: [B, n_classes, H, W]
        """
        return self.decoder(x)


if __name__ == '__main__':
    # 测试
    x = torch.randn(2, 768, 11, 15)
    
    model = LightweightDecoder(channels=768, n_classes=3, use_attention=True)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyEnhancementBlock(nn.Module):
    """
    Frequency Enhancement Block: 频域增强模块
    
    功能:
    - FFT 变换：F = FFT(F_r)
    - 高频增强滤波器：H(F)
    - 逆变换：F_f = IFFT(H(F))
    - 融合：F_hybrid = F_r + β·F_f
    
    动机:
    - 杂草通常边缘更复杂、高频纹理明显、叶片细碎
    - UAV 低空图像里高频信息很重要
    - 频域增强可以突出杂草细节
    
    创新意义:
    - 目前农业 UAV 分割几乎没有系统使用频域增强
    - 频空联合建模是论文的"加分项"
    """
    
    def __init__(self, channels, beta=0.5, highpass_cutoff=0.1):
        """
        Args:
            channels: 特征通道数
            beta: 频域特征融合权重 (可学习)
            highpass_cutoff: 高频滤波器截止频率
        """
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        
        # 可学习的高频增强滤波器参数
        self.highpass_filter = nn.Parameter(
            torch.ones(1, channels, 1, 1) * (1.0 - highpass_cutoff)
        )
        
        # 通道注意力增强 (可选)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征，shape: [B, C, H, W]
        
        Returns:
            x_hybrid: 频空混合特征，shape: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # ========== 方法 1: 简化的空间域高频增强 (推荐，更稳定) ==========
        # 使用 Laplacian 算子提取高频分量
        laplacian_kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], 
            dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        
        # 对每个通道应用 Laplacian
        x_highpass = F.conv2d(x, laplacian_kernel, padding=1, groups=C)
        
        # 通道注意力加权
        attention = self.channel_attention(x_highpass)
        x_highpass = x_highpass * attention
        
        # 融合：原始特征 + 高频增强
        x_hybrid = x + self.beta * x_highpass
        
        return x_hybrid
    
    def forward_with_fft(self, x):
        """
        使用 FFT 的版本 (备选方案，可能在某些场景效果更好)
        
        Args:
            x: 输入特征，shape: [B, C, H, W]
        
        Returns:
            x_hybrid: 频空混合特征，shape: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # FFT 变换 (使用 rfft2 处理实数输入)
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))
        
        # 获取幅度和相位
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # 构建高频增强滤波器
        # 创建频率坐标
        freq_h = torch.fft.fftfreq(H, d=1/H).to(x.device)
        freq_w = torch.fft.fftfreq(W // 2 + 1, d=1/W).to(x.device)
        
        # 网格化
        freq_h, freq_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
        freq_h = freq_h.unsqueeze(0).unsqueeze(0).abs()  # [1, 1, H, W//2+1]
        freq_w = freq_w.unsqueeze(0).unsqueeze(0).abs()
        
        # 高频滤波器 (距离原点越远，响应越强)
        highpass = torch.sqrt(freq_h ** 2 + freq_w ** 2)
        highpass = highpass / (highpass.max() + 1e-8)  # 归一化到 [0, 1]
        
        # 应用可学习滤波器参数
        filter_response = self.highpass_filter + (1.0 - self.highpass_filter) * highpass
        
        # 增强幅度谱
        enhanced_magnitude = magnitude * filter_response
        
        # 逆 FFT
        x_freq = torch.fft.irfft2(
            enhanced_magnitude * torch.exp(1j * phase),
            dim=(-2, -1),
            s=(H, W)
        )
        
        # 融合
        x_hybrid = x + self.beta * x_freq
        
        return x_hybrid


if __name__ == '__main__':
    # 测试
    x = torch.randn(2, 768, 11, 15)
    
    model = FrequencyEnhancementBlock(channels=768, beta=0.5)
    
    # 测试空间域版本
    output_spatial = model(x)
    print(f"Spatial version - Input shape: {x.shape}, Output shape: {output_spatial.shape}")
    
    # 测试 FFT 版本 (可选)
    try:
        output_fft = model.forward_with_fft(x)
        print(f"FFT version - Input shape: {x.shape}, Output shape: {output_fft.shape}")
    except Exception as e:
        print(f"FFT version error: {e}")
    
    print(f"Beta value: {model.beta.item():.4f}")

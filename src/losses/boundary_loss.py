import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """
    Boundary Loss: 边界损失
    
    功能:
    - 增强边缘分割精度
    - 使用 Sobel 算子检测边缘
    - 对预测边缘和真实边缘计算 BCE Loss
    
    动机:
    - UAV 杂草分割最严重的问题：小目标召回率低
    - Boundary Loss 增强边缘，提高小目标分割质量
    - 与 Dice + Focal 组合使用效果更佳
    """
    
    def __init__(self, ignore_index=None):
        """
        Args:
            ignore_index: 忽略的类别索引 (默认：None)
        """
        super().__init__()
        self.ignore_index = ignore_index
        
        # Sobel 算子
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # 初始化 Sobel 权重
        self._init_sobel()
        
        # BCE Loss
        self.bce = nn.BCELoss()
    
    def _init_sobel(self):
        """初始化 Sobel 算子权重"""
        # Sobel 核
        sobel_x = torch.tensor([
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]
        ], dtype=torch.float32)
        
        sobel_y = torch.tensor([
            [-1, -2, -1], 
            [0, 0, 0], 
            [1, 2, 1]
        ], dtype=torch.float32)
        
        # 设置权重
        self.sobel_x.weight.data = sobel_x.view(1, 1, 3, 3)
        self.sobel_y.weight.data = sobel_y.view(1, 1, 3, 3)
        
        # 冻结权重 (不参与梯度更新)
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测 logits, shape: [B, n_classes, H, W]
            target: 真实 mask, shape: [B, H, W]
        
        Returns:
            boundary_loss: 边界损失值
        """
        # 获取预测概率
        pred_prob = torch.softmax(pred, dim=1)
        
        # 获取预测边缘
        pred_edge = self._get_boundary(pred_prob)
        
        # 获取真实边缘
        n_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()
        target_edge = self._get_boundary(target_one_hot)
        
        # 计算 BCE Loss
        boundary_loss = self.bce(pred_edge, target_edge)
        
        return boundary_loss
    
    def _get_boundary(self, mask):
        """
        使用 Sobel 算子检测边缘
        
        Args:
            mask: 概率图或 one-hot mask, shape: [B, n_classes, H, W]
        
        Returns:
            edge: 边缘图，shape: [B, 1, H, W]
        """
        # 确保 Sobel 算子在正确的设备上
        device = mask.device
        if self.sobel_x.weight.device != device:
            self.sobel_x = self.sobel_x.to(device)
            self.sobel_y = self.sobel_y.to(device)
        
        # 合并通道 (对所有类别求和)
        mask_sum = mask.sum(dim=1, keepdim=True)
        
        # 应用 Sobel 算子
        grad_x = self.sobel_x(mask_sum)
        grad_y = self.sobel_y(mask_sum)
        
        # 计算梯度幅值
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # 归一化到 [0, 1]
        edge = edge / (edge.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
        
        return edge


if __name__ == '__main__':
    # 测试
    B, n_classes, H, W = 2, 3, 352, 480
    pred = torch.randn(B, n_classes, H, W)
    target = torch.randint(0, n_classes, (B, H, W))
    
    loss_fn = BoundaryLoss()
    loss = loss_fn(pred, target)
    
    print(f"Input shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Boundary loss: {loss.item():.4f}")

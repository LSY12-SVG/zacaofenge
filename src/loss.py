import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    # mask: [B,H,W] long
    return F.one_hot(mask, num_classes=num_classes).permute(0, 3, 1, 2).float()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None, include_bg=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.include_bg = include_bg

    def forward(self, pred, target):
        # pred: [B, C, H, W] (logits)
        # target: [B, H, W] (indices)
        
        pred = torch.softmax(pred, dim=1)
        
        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 可选：不算背景（class 0）- 对小目标分割几乎是标配
        if not self.include_bg and num_classes > 1:
            pred = pred[:, 1:, ...]
            target_one_hot = target_one_hot[:, 1:, ...]
        
        # Intersection
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average over batch and classes
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        # alpha: None / float / Tensor[C] - 支持每个类别不同
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index if ignore_index is not None else -100, 
                                     reduction='none')

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)          # [B,H,W]
        pt = torch.exp(-ce_loss)                 # pt = exp(logpt)
        
        # alpha 支持每个类别不同（强烈建议对 Weed/Crop 提高）
        if self.alpha is None:
            alpha_t = 1.0
        else:
            if not torch.is_tensor(self.alpha):
                alpha_t = self.alpha
            else:
                # alpha: [C] -> 按 target gather 到 [B,H,W]
                alpha = self.alpha.to(pred.device)
                alpha_t = alpha.gather(0, target.view(-1)).view_as(target).float()
        
        focal = (1 - pt) ** self.gamma
        loss = alpha_t * focal * ce_loss
        return loss.mean()


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weight=None):
        super(DiceCELoss, self).__init__()
        self.dice = DiceLoss(include_bg=False)
        self.ce = nn.CrossEntropyLoss(weight=class_weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        ce_loss = self.ce(pred, target)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0, focal_alpha=None):
        super(DiceFocalLoss, self).__init__()
        self.dice = DiceLoss(include_bg=False)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=2.0)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


class SRDNetLoss(nn.Module):
    """
    SRDNet 组合损失函数（优化版）
    
    功能:
    - Dice Loss + Focal Loss + Boundary Loss
    - 解决类别不平衡 (Focal with per-class alpha)
    - 保证整体分割效果 (Dice without background)
    - 增强边缘精度 (Boundary)
    
    优化:
    - Dice 不算背景 (include_bg=False) - 对小目标分割几乎是标配
    - Focal 用 per-class alpha - 背景 0.1，crop/weed 1.0
    - UAV 杂草分割最严重的问题：小目标召回率低 + 类别不平衡
    - 组合损失可以全面提升分割质量
    """
    
    def __init__(self, lambda_dice=1.0, lambda_focal=1.0, lambda_boundary=0.5):
        """
        Args:
            lambda_dice: Dice Loss 权重
            lambda_focal: Focal Loss 权重
            lambda_boundary: Boundary Loss 权重
        """
        super().__init__()
        # Dice 不算背景 - 避免背景主导优化
        self.dice = DiceLoss(include_bg=False)
        
        # Focal 用 per-class alpha - 背景权重低，前景权重高
        # 建议：背景 0.1，crop 1.0，weed 1.0
        self.focal = FocalLoss(alpha=torch.tensor([0.1, 1.0, 1.0]), gamma=2.0)
        
        # 从 losses 包导入 BoundaryLoss
        try:
            from losses.boundary_loss import BoundaryLoss
            self.boundary = BoundaryLoss()
        except ImportError:
            # 如果不在 src 目录下，尝试直接导入
            try:
                from boundary_loss import BoundaryLoss
                self.boundary = BoundaryLoss()
            except ImportError:
                print("Warning: BoundaryLoss not found, using Dice+Focal only")
                self.boundary = None
        
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.lambda_boundary = lambda_boundary
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测 logits, shape: [B, n_classes, H, W]
            target: 真实 mask, shape: [B, H, W]
        
        Returns:
            total_loss: 总损失值
        """
        loss_dice = self.dice(pred, target)
        loss_focal = self.focal(pred, target)
        
        total_loss = self.lambda_dice * loss_dice + self.lambda_focal * loss_focal
        
        if self.boundary is not None:
            loss_boundary = self.boundary(pred, target)
            total_loss = total_loss + self.lambda_boundary * loss_boundary
        
        return total_loss


def mask_to_edge(mask: torch.Tensor, num_classes: int = 3, radius: int = 1) -> torch.Tensor:
    """
    从语义 mask 生成 edge GT（1 通道，前景边界）
    - 采用形态学梯度：dilate - erode
    - 返回 shape: [B,1,H,W] float(0/1)
    """
    # foreground union (crop or weed)
    fg = (mask > 0).float().unsqueeze(1)  # [B,1,H,W]
    
    # dilation: maxpool
    k = 2 * radius + 1
    dil = F.max_pool2d(fg, kernel_size=k, stride=1, padding=radius)
    ero = -F.max_pool2d(-fg, kernel_size=k, stride=1, padding=radius)
    edge = (dil - ero).clamp(0, 1)
    return edge


class EdgeLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, edge_logits: torch.Tensor, edge_gt: torch.Tensor) -> torch.Tensor:
        # BCE
        bce = self.bce(edge_logits, edge_gt)

        # Dice for binary edge
        probs = torch.sigmoid(edge_logits)
        inter = (probs * edge_gt).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + edge_gt.sum(dim=(2, 3))
        dice = 1.0 - ((2 * inter + 1.0) / (union + 1.0)).mean()

        return self.dice_weight * dice + self.bce_weight * bce


class HFACombinedLoss(nn.Module):
    """
    总损失：
      L = L_seg + λ_edge * L_edge + λ_cons * L_cons
    """
    def __init__(
        self,
        seg_loss: nn.Module,
        edge_weight: float = 0.5,
        cons_weight: float = 0.1,
        edge_radius: int = 1
    ):
        super().__init__()
        self.seg_loss = seg_loss
        self.edge_weight = edge_weight
        self.cons_weight = cons_weight
        self.edge_radius = edge_radius
        self.edge_loss = EdgeLoss(dice_weight=0.5, bce_weight=0.5)

    def consistency_loss(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        # p1, p2: logits [B,C,H,W]
        s1 = torch.softmax(p1, dim=1)
        s2 = torch.softmax(p2, dim=1)
        return (s1 - s2).abs().mean()

    def forward(self, out, target: torch.Tensor, out_aug=None) -> torch.Tensor:
        """
        out: (logits, edge_logits)
        out_aug: (logits_aug, edge_logits_aug)  # 用于一致性
        """
        logits, edge_logits = out
        loss = self.seg_loss(logits, target)

        # edge
        if self.edge_weight > 0 and edge_logits is not None:
            edge_gt = mask_to_edge(target, radius=self.edge_radius).to(edge_logits.device)
            loss = loss + self.edge_weight * self.edge_loss(edge_logits, edge_gt)

        # consistency
        if self.cons_weight > 0 and out_aug is not None:
            logits_aug, _ = out_aug
            loss = loss + self.cons_weight * self.consistency_loss(logits, logits_aug)

        return loss

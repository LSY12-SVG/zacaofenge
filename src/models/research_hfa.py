import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as exc:  # pragma: no cover
    raise ImportError("timm is required for research_hfa model") from exc


class DSDFLogits(nn.Module):
    """Logits-space DSDF (ablation mode)."""

    def __init__(self, channels: int):
        super().__init__()
        self.semantic = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.semantic(x)
        d = self.detail(x)
        g = self.gate(torch.cat([s, d], dim=1))
        return s + g * d


class DSDFFeature(nn.Module):
    """Feature-space DSDF for P2/P3."""

    def __init__(self, channels: int):
        super().__init__()
        self.semantic = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.semantic(x)
        hp = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        d = self.detail(x + hp)
        g = self.gate(torch.cat([s, d], dim=1))
        return s + g * d


class EdgeHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.GELU(),
            nn.Conv2d(in_ch // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FPNDecoder(nn.Module):
    """Standard FPN decoder: C2-C5 -> P2-P5."""

    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(4)]
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c2, c3, c4, c5 = feats["c2"], feats["c3"], feats["c4"], feats["c5"]
        p5 = self.lateral[3](c5)
        p4 = self.lateral[2](c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral[1](c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.lateral[0](c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        p2, p3, p4, p5 = self.smooth[0](p2), self.smooth[1](p3), self.smooth[2](p4), self.smooth[3](p5)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


class ResearchHFANet(nn.Module):
    """
    Unified research entry:
    - arch=fpn: standard FPN baseline
    - arch=hfa: FPN + DSDF + TSBR
    """

    def __init__(
        self,
        n_classes=3,
        backbone="convnext_tiny",
        pretrained=True,
        arch="hfa",
        dsdf_mode="feature",
        dsdf_levels="p2p3",
        fpn_channels=128,
    ):
        super().__init__()
        self.arch = arch
        self.dsdf_mode = dsdf_mode
        self.dsdf_levels = dsdf_levels

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        ch = self.backbone.feature_info.channels()
        self.c_channels = {"c2": ch[0], "c3": ch[1], "c4": ch[2], "c5": ch[3]}
        print(f"C2/C3/C4/C5 channels: {ch[0]}/{ch[1]}/{ch[2]}/{ch[3]}")

        self.fpn = FPNDecoder([ch[0], ch[1], ch[2], ch[3]], out_channels=fpn_channels)
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
            nn.Conv2d(fpn_channels, n_classes, 1),
        )

        self.dsdf_p2 = DSDFFeature(fpn_channels) if dsdf_mode == "feature" and ("p2" in dsdf_levels) else None
        self.dsdf_p3 = DSDFFeature(fpn_channels) if dsdf_mode == "feature" and ("p3" in dsdf_levels) else None
        self.dsdf_logits = DSDFLogits(n_classes) if dsdf_mode == "logits" else None

        self.edge_head = EdgeHead(fpn_channels) if arch == "hfa" else None

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c2, c3, c4, c5 = self.backbone(x)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}

    def decode(self, features: Dict[str, torch.Tensor], input_size: Tuple[int, int]):
        pyr = self.fpn(features)
        p2, p3 = pyr["p2"], pyr["p3"]

        if self.dsdf_mode == "feature":
            if self.dsdf_p2 is not None:
                p2 = self.dsdf_p2(p2)
            if self.dsdf_p3 is not None:
                p3 = self.dsdf_p3(p3)

        logits = self.seg_head(p2)
        if self.dsdf_mode == "logits" and self.dsdf_logits is not None:
            logits = self.dsdf_logits(logits)

        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        if self.arch == "hfa":
            edge_logits = self.edge_head(p2)
            edge_logits = F.interpolate(edge_logits, size=input_size, mode="bilinear", align_corners=False)
            return logits, edge_logits
        return logits

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        feats = self.forward_features(x)
        return self.decode(feats, input_size=input_size)

    def estimate_fps(self, device: str, h: int = 480, w: int = 480, warmup: int = 10, iters: int = 30) -> float:
        self.eval()
        x = torch.randn(1, 3, h, w, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = self(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                _ = self(x)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
        return float(iters / max(dt, 1e-6))

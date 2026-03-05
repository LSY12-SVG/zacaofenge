import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WeedDataset
from hfa_model import HFANet
from loss import SRDNetLoss, HFACombinedLoss
from models.srdnet import SRDNet


def calculate_iou(logits, true_mask, num_classes=3):
    pred = torch.argmax(logits, dim=1)

    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (true_mask == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float("nan"))
        else:
            iou_per_class.append((intersection / union).item())

    return float(np.nanmean(iou_per_class)), iou_per_class


def strong_tensor_augment(x: torch.Tensor) -> torch.Tensor:
    """
    用于一致性正则的轻量"第二视图"增强（不依赖额外库）
    - random flip
    - gaussian noise
    - brightness jitter (tensor space)
    """
    y = x
    if torch.rand(1).item() < 0.5:
        y = torch.flip(y, dims=[3])
    if torch.rand(1).item() < 0.2:
        y = torch.flip(y, dims=[2])

    # brightness jitter
    if torch.rand(1).item() < 0.7:
        b = (torch.rand(1, device=y.device) * 0.2 - 0.1).item()
        y = y + b

    # noise
    if torch.rand(1).item() < 0.7:
        y = y + torch.randn_like(y) * 0.03

    return y.clamp(-5, 5)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_iou):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_iou": best_iou,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_iou = float(ckpt.get("best_iou", 0.0))
        return start_epoch, best_iou
    else:
        # 兼容旧版只存 state_dict
        model.load_state_dict(ckpt)
        return 0, 0.0


def main():
    parser = argparse.ArgumentParser("Train segmentation model (HFA-Net / SRDNet)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="hfanet", choices=["hfanet", "srdnet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--save_name", type=str, default="hfanet_best.pth")
    parser.add_argument("--resume", type=str, default=None)

    # default FAC params
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--fg_prob", type=float, default=0.7)
    parser.add_argument("--no_fac", action="store_true")
    parser.add_argument("--class_stat_mode", type=str, default="random", choices=["random", "full", "none"])
    parser.add_argument("--class_stat_samples", type=int, default=500)

    # loss weights
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--boundary_weight", type=float, default=0.0)  # 默认 0，避免 boundary_loss 缺失
    parser.add_argument("--edge_weight", type=float, default=0.5)
    parser.add_argument("--cons_weight", type=float, default=0.1)

    # aug
    parser.add_argument("--no-aug", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # dataset
    train_set = WeedDataset(
        args.data_dir,
        mode="train",
        augment=not args.no_aug,
        dataset_type="auto",
        crop_size=args.crop_size,
        fg_prob=args.fg_prob,
        enable_fac=(not args.no_fac),
        class_stat_mode=args.class_stat_mode,
        class_stat_samples=args.class_stat_samples,
    )
    val_set = WeedDataset(
        args.data_dir,
        mode="val",
        augment=False,
        dataset_type="auto",
        crop_size=args.crop_size,
        fg_prob=args.fg_prob,
        enable_fac=False,
        class_stat_mode="none",
        class_stat_samples=args.class_stat_samples,
    )

    print(f"Training on {len(train_set)} samples, validating on {len(val_set)} samples")
    print(f"FAC enabled: {not args.no_fac}, Augmentation enabled: {not args.no_aug}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # model
    if args.model == "hfanet":
        model = HFANet(n_classes=3, backbone=args.backbone, pretrained=True).to(device)
    else:
        model = SRDNet(n_classes=3, backbone=args.backbone, pretrained=True).to(device)
    print(f"Model: {args.model}, Backbone: {args.backbone}")

    # optimizer / scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # loss
    seg_loss = SRDNetLoss(
        lambda_dice=args.dice_weight,
        lambda_focal=args.focal_weight,
        lambda_boundary=args.boundary_weight
    )
    if args.model == "hfanet":
        criterion = HFACombinedLoss(
            seg_loss=seg_loss,
            edge_weight=args.edge_weight,
            cons_weight=args.cons_weight,
            edge_radius=1
        )
    else:
        criterion = seg_loss
        if args.edge_weight > 0 or args.cons_weight > 0:
            print("Note: edge_weight/cons_weight are only used by hfanet and will be ignored for srdnet.")

    # resume
    start_epoch = 0
    best_iou = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_iou = load_checkpoint(args.resume, model, optimizer, scheduler, device=device)
        print(f"Resume start_epoch={start_epoch}, best_iou={best_iou:.4f}")

    # training loop
    for epoch in range(start_epoch, args.epochs):
        train_set.set_epoch(epoch)

        model.train()
        running = 0.0

        pbar = tqdm(total=len(train_set), desc=f"Epoch {epoch+1}/{args.epochs}", unit="img")
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)

            # forward
            out = model(images)

            # second view for consistency
            out_aug = None
            if args.model == "hfanet" and args.cons_weight > 0:
                images2 = strong_tensor_augment(images)
                out_aug = model(images2)

            if args.model == "hfanet":
                loss = criterion(out, masks, out_aug=out_aug)
                logits = out[0]
            else:
                loss = criterion(out, masks)
                logits = out
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            pbar.update(images.shape[0])
            pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))
        pbar.close()

        # validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        per_sum = [0.0, 0.0, 0.0]
        per_cnt = [0, 0, 0]

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                if args.model == "hfanet":
                    logits, edge_logits = model(images)
                    loss = criterion((logits, edge_logits), masks, out_aug=None)
                else:
                    logits = model(images)
                    loss = criterion(logits, masks)
                val_loss += loss.item()

                miou, per = calculate_iou(logits, masks, num_classes=3)
                val_iou += miou
                for i in range(3):
                    if not np.isnan(per[i]):
                        per_sum[i] += per[i]
                        per_cnt[i] += 1

        val_loss /= max(len(val_loader), 1)
        val_iou /= max(len(val_loader), 1)
        per_iou = [per_sum[i] / max(per_cnt[i], 1) for i in range(3)]

        print(f"Validation Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}")
        print(f"  - Background IoU: {per_iou[0]:.4f}")
        print(f"  - Crop IoU:       {per_iou[1]:.4f}")
        print(f"  - Weed IoU:       {per_iou[2]:.4f}")

        scheduler.step()

        # save best
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(args.save_name, model, optimizer, scheduler, epoch, best_iou)
            print(f"Saved best checkpoint to {args.save_name} (mIoU: {best_iou:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()

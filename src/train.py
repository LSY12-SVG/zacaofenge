import argparse
import csv
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import WeedDataset
from loss import HFACombinedLoss, SRDNetLoss, mask_to_edge
from model import UNet
from models.research_hfa import ResearchHFANet
from models.srdnet import SRDNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_iou(logits, true_mask, num_classes=3):
    pred = torch.argmax(logits, dim=1)
    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = true_mask == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float("nan"))
        else:
            iou_per_class.append((intersection / union).item())
    return float(np.nanmean(iou_per_class)), iou_per_class


def calculate_weed_recall(logits, true_mask, weed_cls=2):
    pred = torch.argmax(logits, dim=1)
    tp = ((pred == weed_cls) & (true_mask == weed_cls)).sum().float()
    fn = ((pred != weed_cls) & (true_mask == weed_cls)).sum().float()
    denom = tp + fn
    if denom <= 0:
        return float("nan")
    return float((tp / denom).item())


def calculate_weed_stats(logits, true_mask, weed_cls=2):
    pred = torch.argmax(logits, dim=1)
    tp = int(((pred == weed_cls) & (true_mask == weed_cls)).sum().item())
    fn = int(((pred != weed_cls) & (true_mask == weed_cls)).sum().item())
    fp = int(((pred == weed_cls) & (true_mask != weed_cls)).sum().item())
    return tp, fn, fp


def calculate_boundary_f1(logits, true_mask, radius=1):
    pred = torch.argmax(logits, dim=1)
    pred_edge = (mask_to_edge(pred, radius=1) > 0.5).float()
    gt_edge = (mask_to_edge(true_mask, radius=1) > 0.5).float()

    if radius > 0:
        k = 2 * radius + 1
        pred_dil = (torch.nn.functional.max_pool2d(pred_edge, kernel_size=k, stride=1, padding=radius) > 0).float()
        gt_dil = (torch.nn.functional.max_pool2d(gt_edge, kernel_size=k, stride=1, padding=radius) > 0).float()
    else:
        pred_dil = pred_edge
        gt_dil = gt_edge

    matched_pred = pred_edge * gt_dil
    matched_gt = gt_edge * pred_dil
    tp_p = matched_pred.sum().float()
    all_p = pred_edge.sum().float()
    all_g = gt_edge.sum().float()
    precision = tp_p / (all_p + 1e-6)
    recall = matched_gt.sum().float() / (all_g + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    return float(f1.item())


def strong_tensor_augment(x: torch.Tensor) -> torch.Tensor:
    y = x
    if torch.rand(1).item() < 0.5:
        y = torch.flip(y, dims=[3])
    if torch.rand(1).item() < 0.2:
        y = torch.flip(y, dims=[2])
    if torch.rand(1).item() < 0.7:
        y = y + (torch.rand(1, device=y.device) * 0.2 - 0.1).item()
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
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_iou = float(ckpt.get("best_iou", 0.0))
        return start_epoch, best_iou
    model.load_state_dict(ckpt)
    return 0, 0.0


def ensure_csv_header(path):
    header = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_miou",
        "iou_bg",
        "iou_crop",
        "iou_weed",
        "weed_recall",
        "weed_tp",
        "weed_fn",
        "weed_fp",
        "boundary_f1",
        "lr",
        "cons_weight",
        "fac_avg_fg_ratio",
        "fac_avg_attempts",
        "fac_samples",
    ]
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_epoch_metrics(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def dump_config(path, cfg):
    try:
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            import json

            json.dump(cfg, f, ensure_ascii=False, indent=2)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array([[0, 0, 0], [0, 200, 0], [220, 0, 0]], dtype=np.uint8)
    mask = np.clip(mask, 0, 2).astype(np.int64)
    return palette[mask]


def denorm_image(img_t: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def model_forward(model, images, arch):
    out = model(images)
    if arch == "hfa":
        logits, edge_logits = out
        return logits, edge_logits, out
    return out, None, out


def benchmark_fps(model, arch, device, h=480, w=480, batch_size=1, warmup=20, iters=50):
    model.eval()
    x = torch.randn(batch_size, 3, h, w, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_forward(model, x, arch)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model_forward(model, x, arch)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
    return float((iters * batch_size) / max(dt, 1e-6))


def save_prediction_visuals(model, val_set, device, arch, out_dir, epoch, num_samples=4):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    max_n = min(num_samples, len(val_set))
    with torch.no_grad():
        for i in range(max_n):
            image, mask = val_set[i]
            x = image.unsqueeze(0).to(device)
            logits, _, _ = model_forward(model, x, arch)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            gt = mask.cpu().numpy().astype(np.uint8)
            vis_img = denorm_image(image)
            vis_gt = colorize_mask(gt)
            vis_pred = colorize_mask(pred)
            canvas = np.concatenate([vis_img, vis_gt, vis_pred], axis=1)
            save_path = os.path.join(out_dir, f"epoch_{epoch:03d}_sample_{i:02d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser("Train research-ready segmentation models")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--arch",
        type=str,
        default="hfa",
        choices=["fpn", "hfa", "srdnet", "unet", "deeplabv3plus", "segformerb0"],
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--save_name", type=str, default="best.ckpt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--metrics_csv", type=str, default="")
    parser.add_argument("--estimate_fps", action="store_true")
    parser.add_argument("--fps_h", type=int, default=480)
    parser.add_argument("--fps_w", type=int, default=480)
    parser.add_argument("--fps_batch_size", type=int, default=1)
    parser.add_argument("--fps_warmup_iters", type=int, default=20)
    parser.add_argument("--fps_timed_iters", type=int, default=50)

    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--fg_prob", type=float, default=0.7)
    parser.add_argument("--no_fac", action="store_true")
    parser.add_argument("--class_stat_mode", type=str, default="random", choices=["random", "full", "none"])
    parser.add_argument("--class_stat_samples", type=int, default=500)
    parser.add_argument("--max_fac_tries", type=int, default=3)

    parser.add_argument("--dsdf_mode", type=str, default="feature", choices=["none", "logits", "feature"])
    parser.add_argument("--dsdf_levels", type=str, default="p2p3", choices=["p2", "p2p3"])
    parser.add_argument("--fpn_channels", type=int, default=128)

    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--boundary_weight", type=float, default=0.5)
    parser.add_argument("--edge_weight", type=float, default=0.5)
    parser.add_argument("--cons_weight", type=float, default=0.1)
    parser.add_argument("--cons_warmup_epochs", type=int, default=10)
    parser.add_argument("--bf_radius", type=int, default=1)

    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--vis_every", type=int, default=10)
    parser.add_argument("--vis_samples", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_dir = args.run_dir.strip()
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "pred_vis"), exist_ok=True)
        checkpoint_path = os.path.join(run_dir, "best.ckpt")
        metrics_csv = args.metrics_csv or os.path.join(run_dir, "metrics.csv")
        config_path = os.path.join(run_dir, "config.yaml")
        dump_config(config_path, vars(args))
    else:
        checkpoint_path = args.save_name
        metrics_csv = args.metrics_csv or "results/training_metrics.csv"

    os.makedirs(os.path.dirname(metrics_csv) or ".", exist_ok=True)
    ensure_csv_header(metrics_csv)

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
        max_fac_tries=args.max_fac_tries,
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
        max_fac_tries=args.max_fac_tries,
    )

    print(f"Training on {len(train_set)} samples, validating on {len(val_set)} samples")
    print(f"FAC enabled: {not args.no_fac}, Augmentation enabled: {not args.no_aug}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    pretrained = not args.no_pretrained
    if args.arch in ["fpn", "hfa"]:
        model = ResearchHFANet(
            n_classes=3,
            backbone=args.backbone,
            pretrained=pretrained,
            arch=args.arch,
            dsdf_mode=args.dsdf_mode,
            dsdf_levels=args.dsdf_levels,
            fpn_channels=args.fpn_channels,
        ).to(device)
    elif args.arch == "srdnet":
        model = SRDNet(n_classes=3, backbone=args.backbone, pretrained=pretrained).to(device)
    elif args.arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=args.backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=3,
        ).to(device)
    elif args.arch == "segformerb0":
        model = smp.Segformer(
            encoder_name="mit_b0",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=3,
        ).to(device)
    else:
        model = UNet(n_channels=3, n_classes=3).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch}, Backbone: {args.backbone}, Params: {total_params / 1e6:.2f}M")

    fps = None
    if args.estimate_fps:
        fps = benchmark_fps(
            model=model,
            arch=args.arch,
            device=device,
            h=args.fps_h,
            w=args.fps_w,
            batch_size=args.fps_batch_size,
            warmup=args.fps_warmup_iters,
            iters=args.fps_timed_iters,
        )
        print(f"Estimated FPS: {fps:.2f}")

    if run_dir:
        if device == "cuda":
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        else:
            device_name = "cpu"
        meta = {
            "params_million": round(total_params / 1e6, 4),
            "fps": fps if fps is not None else "",
            "device": device_name,
            "device_type": device,
            "input_size": [args.fps_h, args.fps_w],
            "batch_size": args.fps_batch_size,
            "warmup_iters": args.fps_warmup_iters,
            "timed_iters": args.fps_timed_iters,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.version.cuda is not None else "",
        }
        dump_config(os.path.join(run_dir, "model_meta.yaml"), meta)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    seg_loss = SRDNetLoss(
        lambda_dice=args.dice_weight,
        lambda_focal=args.focal_weight,
        lambda_boundary=args.boundary_weight,
    )

    if args.arch == "hfa":
        criterion = HFACombinedLoss(
            seg_loss=seg_loss,
            edge_weight=args.edge_weight,
            cons_weight=args.cons_weight,
            edge_radius=1,
        )
    else:
        criterion = seg_loss
        if args.edge_weight > 0 or args.cons_weight > 0:
            print("Note: edge_weight/cons_weight are only active for arch=hfa.")

    start_epoch = 0
    best_iou = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_iou = load_checkpoint(args.resume, model, optimizer, scheduler, device=device)
        print(f"Resume start_epoch={start_epoch}, best_iou={best_iou:.4f}")

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_set, "set_epoch"):
            train_set.set_epoch(epoch)

        model.train()
        running = 0.0
        active_cons_weight = 0.0
        pbar = tqdm(total=len(train_set), desc=f"Epoch {epoch+1}/{args.epochs}", unit="img")

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, edge_logits, out = model_forward(model, images, args.arch)

            out_aug = None
            if args.arch == "hfa":
                active_cons_weight = args.cons_weight if epoch >= args.cons_warmup_epochs else 0.0
                criterion.cons_weight = active_cons_weight
                if active_cons_weight > 0:
                    images2 = strong_tensor_augment(images)
                    logits2, edge_logits2, out2 = model_forward(model, images2, args.arch)
                    out_aug = (logits2, edge_logits2)

            if args.arch == "hfa":
                loss = criterion((logits, edge_logits), masks, out_aug=out_aug)
            else:
                loss = criterion(logits, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            pbar.update(images.shape[0])
            pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))
        pbar.close()

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        weed_tp_sum = 0
        weed_fn_sum = 0
        weed_fp_sum = 0
        bf1_sum = 0.0
        per_sum = [0.0, 0.0, 0.0]
        per_cnt = [0, 0, 0]

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits, edge_logits, _ = model_forward(model, images, args.arch)

                if args.arch == "hfa":
                    loss = criterion((logits, edge_logits), masks, out_aug=None)
                else:
                    loss = criterion(logits, masks)
                val_loss += loss.item()

                miou, per = calculate_iou(logits, masks, num_classes=3)
                val_iou += miou
                tp, fn, fp = calculate_weed_stats(logits, masks)
                weed_tp_sum += tp
                weed_fn_sum += fn
                weed_fp_sum += fp
                bf1_sum += calculate_boundary_f1(logits, masks, radius=args.bf_radius)

                for i in range(3):
                    if not np.isnan(per[i]):
                        per_sum[i] += per[i]
                        per_cnt[i] += 1

        train_loss = running / max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        val_iou /= max(len(val_loader), 1)
        per_iou = [per_sum[i] / max(per_cnt[i], 1) for i in range(3)]
        weed_recall = float(weed_tp_sum / max(weed_tp_sum + weed_fn_sum, 1))
        boundary_f1 = bf1_sum / max(len(val_loader), 1)
        fac_stats = (
            train_set.get_epoch_stats() if hasattr(train_set, "get_epoch_stats") else {
                "fac_avg_fg_ratio": 0.0,
                "fac_avg_attempts": 0.0,
                "fac_samples": 0,
            }
        )
        lr_now = float(optimizer.param_groups[0]["lr"])

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}")
        print(f"  - Background IoU: {per_iou[0]:.4f}")
        print(f"  - Crop IoU:       {per_iou[1]:.4f}")
        print(f"  - Weed IoU:       {per_iou[2]:.4f}")
        print(f"  - Weed Recall:    {weed_recall:.4f}")
        print(f"  - Weed TP/FN/FP:  {weed_tp_sum}/{weed_fn_sum}/{weed_fp_sum}")
        print(f"  - Boundary F1:    {boundary_f1:.4f}")
        print(f"  - LR:             {lr_now:.6g}")
        if args.arch == "hfa":
            print(f"  - Cons Weight:    {active_cons_weight:.4f}")
        print(
            f"  - FAC avg_fg:     {fac_stats['fac_avg_fg_ratio']:.4f}, "
            f"avg_try: {fac_stats['fac_avg_attempts']:.2f}, samples: {fac_stats['fac_samples']}"
        )

        append_epoch_metrics(
            metrics_csv,
            [
                epoch + 1,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_iou:.6f}",
                f"{per_iou[0]:.6f}",
                f"{per_iou[1]:.6f}",
                f"{per_iou[2]:.6f}",
                f"{weed_recall:.6f}",
                weed_tp_sum,
                weed_fn_sum,
                weed_fp_sum,
                f"{boundary_f1:.6f}",
                f"{lr_now:.8f}",
                f"{active_cons_weight:.6f}",
                f"{fac_stats['fac_avg_fg_ratio']:.6f}",
                f"{fac_stats['fac_avg_attempts']:.6f}",
                fac_stats["fac_samples"],
            ],
        )

        if args.vis_every > 0 and run_dir and (((epoch + 1) % args.vis_every == 0) or (epoch + 1 == args.epochs)):
            save_prediction_visuals(
                model=model,
                val_set=val_set,
                device=device,
                arch=args.arch,
                out_dir=os.path.join(run_dir, "pred_vis"),
                epoch=epoch + 1,
                num_samples=args.vis_samples,
            )

        scheduler.step()

        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_iou)
            print(f"Saved best checkpoint to {checkpoint_path} (mIoU: {best_iou:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()

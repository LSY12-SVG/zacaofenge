import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WeedDataset(Dataset):
    """
    升级版：支持 FAC（Foreground-Aware Curriculum）
    - 前景引导裁剪：训练时优先从 crop/weed 区域采样 patch
    - 课程阈值：epoch 越早，要求 patch 前景比例越高（易→难）
    - 统计读取更稳（cv2.imread）
    """
    def __init__(
        self,
        root_dir,
        mode="train",
        augment=True,
        dataset_type="auto",
        crop_size=512,
        fg_prob=0.7,
        enable_fac=True,
        class_stat_mode="random",
        class_stat_samples=500,
        max_fac_tries=3
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.dataset_type = dataset_type
        self.images = []
        self.masks = []

        # FAC params
        self.crop_size = int(crop_size)
        self.fg_prob = float(fg_prob)
        self.enable_fac = bool(enable_fac and mode == "train")
        self.current_epoch = 0
        self.class_stat_mode = str(class_stat_mode).lower()
        self.class_stat_samples = int(class_stat_samples)
        self.max_fac_tries = int(max_fac_tries)
        self._fac_attempt_sum = 0.0
        self._fac_fg_ratio_sum = 0.0
        self._fac_sample_count = 0

        root_path = Path(root_dir)

        # auto detect
        if dataset_type == "auto":
            if (root_path / "train").exists() and (root_path / "val").exists():
                self.dataset_type = "combined"
            elif (root_path / "Campaign no. 1").exists():
                self.dataset_type = "tobacco"
            elif (root_path / "images").exists() or (root_path / "CoFly-WeedDB").exists():
                self.dataset_type = "cofly"
            else:
                self.dataset_type = "tobacco"

        print(f"Scanning {root_dir} for dataset (type: {self.dataset_type})...")

        if self.dataset_type == "combined":
            self._load_combined_dataset(root_path)
        elif self.dataset_type == "tobacco":
            self._load_tobacco_dataset(root_path)
        elif self.dataset_type == "cofly":
            self._load_cofly_dataset(root_path)

        print(f"Loaded {len(self.images)} image-mask pairs from {self.root_dir}")
        if len(self.images) == 0:
            print(f"Warning: No images found in {root_dir}")

        self._compute_class_distribution()

        # augmentations
        if self.mode == "train" and augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),

                # 用 Affine 替代 ShiftScaleRotate（更推荐）
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.05, 0.05),
                    rotate=(-45, 45),
                    shear=(-10, 10),
                    p=0.5
                ),

                A.OneOf([
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.GaussianBlur(blur_limit=7, p=0.5),
                    A.GaussNoise(p=0.5),
                ], p=0.3),

                A.OneOf([
                    A.RandomBrightnessContrast(p=0.7),
                    A.HueSaturationValue(p=0.7),
                    A.CLAHE(p=0.5),
                ], p=0.3),

                # 训练阶段先不强行 resize 到 480；我们会在裁剪后 resize
            ])
        else:
            self.transform = A.Compose([])

        # 最终统一到网络输入大小（默认 480，与原工程一致）
        self.final_resize = A.Resize(
            height=480, width=480,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        )
        self.normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.to_tensor = ToTensorV2()

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        self._fac_attempt_sum = 0.0
        self._fac_fg_ratio_sum = 0.0
        self._fac_sample_count = 0

    def get_epoch_stats(self):
        if self._fac_sample_count == 0:
            return {
                "fac_avg_attempts": 0.0,
                "fac_avg_fg_ratio": 0.0,
                "fac_samples": 0
            }
        return {
            "fac_avg_attempts": float(self._fac_attempt_sum / self._fac_sample_count),
            "fac_avg_fg_ratio": float(self._fac_fg_ratio_sum / self._fac_sample_count),
            "fac_samples": int(self._fac_sample_count)
        }

    def _fac_min_fg_ratio(self) -> float:
        # 默认课程：易→难
        e = self.current_epoch
        if e < 10:
            return 0.15
        elif e < 30:
            return 0.08
        else:
            return 0.03

    def _compute_class_distribution(self):
        if len(self.masks) == 0:
            self.class_weights = np.ones(3)
            return

        if self.class_stat_mode == "none":
            self.class_weights = np.ones(3)
            print("Skip class distribution statistics (class_stat_mode=none)")
            return

        hist = np.zeros(3, dtype=np.float64)
        total_pixels = 0

        total_masks = len(self.masks)
        if self.class_stat_mode == "full" or self.class_stat_samples >= total_masks:
            sample_idx = np.arange(total_masks)
            stat_desc = f"全量 {total_masks}"
        else:
            n = max(1, min(self.class_stat_samples, total_masks))
            rng = np.random.default_rng(seed=42)
            sample_idx = rng.choice(total_masks, size=n, replace=False)
            stat_desc = f"随机采样 {n}/{total_masks}"

        for idx in sample_idx:
            idx = int(idx)
            mp = self.masks[idx]
            try:
                img_path = self.images[idx]
                mask = self._read_mask(mp)
                mask = self._map_labels(img_path, mask)

                for i in range(3):
                    hist[i] += np.sum(mask == i)
                total_pixels += mask.size
            except Exception:
                continue

        if total_pixels > 0:
            ratio = hist / total_pixels
            print(f"\n{'='*60}")
            print(f"类别分布统计 ({stat_desc} 张图像):")
            print(f"  Background (0): {hist[0]:10.0f} pixels ({ratio[0]*100:5.2f}%)")
            print(f"  Crop (1):       {hist[1]:10.0f} pixels ({ratio[1]*100:5.2f}%)")
            print(f"  Weed (2):       {hist[2]:10.0f} pixels ({ratio[2]*100:5.2f}%)")
            print(f"{'='*60}\n")

            eps = 1e-6
            w = 1.0 / np.log(ratio + eps + 1.02)
            w = w / w.sum() * 3
            self.class_weights = w
            print(f"建议的类别权重：{self.class_weights}")
            print(f"{'='*60}\n")
        else:
            self.class_weights = np.ones(3)

    def _load_combined_dataset(self, root_path: Path):
        mode_dir = root_path / self.mode
        if not mode_dir.exists():
            print(f"Warning: {mode_dir} does not exist")
            return
        img_dir = mode_dir / "images"
        mask_dir = mode_dir / "masks"
        if img_dir.exists() and mask_dir.exists():
            for img_p in img_dir.glob("*.png"):
                mask_p = mask_dir / img_p.name
                if mask_p.exists():
                    self.images.append(str(img_p))
                    self.masks.append(str(mask_p))

    def _load_tobacco_dataset(self, root_path: Path):
        candidates = list(root_path.glob("**/data/*.png")) + list(root_path.glob("**/data/*.jpg"))
        for img_p in candidates:
            parent = img_p.parent.parent
            mask_p = parent / "mask" / img_p.name
            if mask_p.exists():
                self.images.append(str(img_p))
                self.masks.append(str(mask_p))
        if len(self.images) == 0:
            img_dir = root_path / "data"
            mask_dir = root_path / "mask"
            if img_dir.exists() and mask_dir.exists():
                for img_p in img_dir.glob("*.png"):
                    mask_p = mask_dir / img_p.name
                    if mask_p.exists():
                        self.images.append(str(img_p))
                        self.masks.append(str(mask_p))

    def _load_cofly_dataset(self, root_path: Path):
        mode_dir = root_path / self.mode
        if mode_dir.exists():
            self._load_combined_dataset(root_path)
            if len(self.images) > 0:
                return

        img_dir = root_path / "images"
        mask_dir = root_path / "labels"
        if img_dir.exists() and mask_dir.exists():
            split_file = root_path / f"{self.mode}_split1.txt"
            if split_file.exists():
                with open(split_file, "r", encoding="utf-8") as f:
                    img_names = [line.strip() for line in f.readlines()]
                for name in img_names:
                    ip = img_dir / name
                    mp = mask_dir / name
                    if ip.exists() and mp.exists():
                        self.images.append(str(ip))
                        self.masks.append(str(mp))
            else:
                for img_p in img_dir.glob("*.png"):
                    mask_p = mask_dir / img_p.name
                    if mask_p.exists():
                        self.images.append(str(img_p))
                        self.masks.append(str(mask_p))

    def __len__(self):
        return len(self.images)

    def _read_image(self, img_path: str):
        # 兼容中文路径：fromfile + imdecode
        arr = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, mask_path: str):
        arr = np.fromfile(mask_path, dtype=np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask

    def _map_labels(self, img_path: str, mask: np.ndarray) -> np.ndarray:
        # 统一标签映射（强制把所有 mask 映射到 0/1/2）
        # 兜底映射规则（论文里解释为"label normalization"）
        
        # 1) CoFly 数据集：所有非零都当 weed(2)
        if self.dataset_type == "cofly":
            mask = np.where(mask > 0, 2, 0).astype(np.uint8)
        
        # 2) combined 数据集里，文件名前缀 cofly_ 也按 cofly 规则映射
        elif self.dataset_type == "combined":
            name = Path(img_path).name
            if name.startswith("cofly_"):
                mask = np.where(mask > 0, 2, 0).astype(np.uint8)
        
        # 3) 兜底映射规则（处理各种编码）
        # {0,255}：把 255 当 weed(2)
        # {0,128,255}：128->crop(1), 255->weed(2)
        u = np.unique(mask)
        if u.max() > 2:
            if set(u.tolist()) <= {0, 255}:
                mask = np.where(mask == 255, 2, 0).astype(np.uint8)
            elif set(u.tolist()) <= {0, 128, 255}:
                mask2 = np.zeros_like(mask, dtype=np.uint8)
                mask2[mask == 128] = 1
                mask2[mask == 255] = 2
                mask = mask2
        
        # tobacco/combined(tobacco part) assumed 0/1/2 already
        return mask

    def _random_crop(self, image: np.ndarray, mask: np.ndarray, cx: int, cy: int):
        h, w = mask.shape[:2]
        s = self.crop_size
        x1 = int(np.clip(cx - s // 2, 0, max(0, w - s)))
        y1 = int(np.clip(cy - s // 2, 0, max(0, h - s)))
        x2 = x1 + s
        y2 = y1 + s
        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def _fac_crop(self, image: np.ndarray, mask: np.ndarray):
        """
        前景引导裁剪：在 crop/weed 像素附近裁剪
        若找不到足够前景则退化为随机裁剪
        """
        h, w = mask.shape[:2]
        s = self.crop_size

        # 如果原图本身比 crop_size 小，则直接 resize
        if h < s or w < s:
            fg_ratio = float((mask > 0).mean())
            self._fac_attempt_sum += 0.0
            self._fac_fg_ratio_sum += fg_ratio
            self._fac_sample_count += 1
            return image, mask

        min_fg = self._fac_min_fg_ratio()
        # foreground pixels (crop or weed)
        fg = np.where(mask > 0)
        if fg[0].size > 0 and np.random.rand() < self.fg_prob:
            # 采样一个前景点作为中心
            idx = np.random.randint(0, fg[0].size)
            cy, cx = int(fg[0][idx]), int(fg[1][idx])
            img_c, mask_c = self._random_crop(image, mask, cx, cy)
            # 前景比例不够时，限制尝试次数避免拖慢
            attempts = 1
            for _ in range(max(self.max_fac_tries - 1, 0)):
                fg_ratio = (mask_c > 0).mean()
                if fg_ratio >= min_fg:
                    self._fac_attempt_sum += attempts
                    self._fac_fg_ratio_sum += float(fg_ratio)
                    self._fac_sample_count += 1
                    return img_c, mask_c
                attempts += 1
                idx = np.random.randint(0, fg[0].size)
                cy, cx = int(fg[0][idx]), int(fg[1][idx])
                img_c, mask_c = self._random_crop(image, mask, cx, cy)
            fg_ratio = (mask_c > 0).mean()
            self._fac_attempt_sum += attempts
            self._fac_fg_ratio_sum += float(fg_ratio)
            self._fac_sample_count += 1
            return img_c, mask_c

        # fallback: random crop
        cx = np.random.randint(s // 2, w - s // 2)
        cy = np.random.randint(s // 2, h - s // 2)
        img_c, mask_c = self._random_crop(image, mask, cx, cy)
        fg_ratio = (mask_c > 0).mean()
        self._fac_attempt_sum += 0.0
        self._fac_fg_ratio_sum += float(fg_ratio)
        self._fac_sample_count += 1
        return img_c, mask_c

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = self._read_image(img_path)
        mask = self._read_mask(mask_path)
        mask = self._map_labels(img_path, mask)

        # label validity
        u = np.unique(mask)
        if u.size > 0 and u.max() > 2:
            raise RuntimeError(f"Invalid label {u.max()} in {mask_path}. Expected 0/1/2.")

        # FAC crop (train only)
        if self.enable_fac:
            image, mask = self._fac_crop(image, mask)

        # augment
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        # final resize + normalize + tensor
        out = self.final_resize(image=image, mask=mask)
        image, mask = out["image"], out["mask"]

        out = self.normalize(image=image, mask=mask)
        image, mask = out["image"], out["mask"]

        out = self.to_tensor(image=image, mask=mask)
        image = out["image"]
        mask = out["mask"].long()

        return image, mask

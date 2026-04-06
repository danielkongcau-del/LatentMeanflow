from bisect import bisect_right
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MaskImagePairDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        size=256,
        image_dir="image",
        mask_dir="mask",
        image_exts=(".jpg", ".jpeg", ".png", ".bmp"),
        mask_exts=(".png", ".jpg", ".jpeg", ".bmp"),
        strict_pairs=True,
        class_id=None,
        class_name=None,
    ):
        self.root = Path(root)
        self.split = split
        self.size = size
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_exts = image_exts
        self.mask_exts = mask_exts
        self.strict_pairs = strict_pairs
        self.class_id = class_id
        self.class_name = class_name

        self._pairs = self._collect_pairs()
        if not self._pairs:
            raise ValueError(f"No image/mask pairs found in {self.root}/{self.split}")

    def _collect_pairs(self):
        image_root = self.root / self.split / self.image_dir
        mask_root = self.root / self.split / self.mask_dir

        if not image_root.exists():
            raise FileNotFoundError(f"Image dir not found: {image_root}")
        if not mask_root.exists():
            raise FileNotFoundError(f"Mask dir not found: {mask_root}")

        masks = []
        for ext in self.mask_exts:
            masks.extend(mask_root.glob(f"*{ext}"))
        masks = sorted(masks)

        pairs = []
        for mask_path in masks:
            stem = mask_path.stem
            image_path = None
            for ext in self.image_exts:
                candidate = image_root / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                if self.strict_pairs:
                    raise FileNotFoundError(f"Missing image for mask: {mask_path}")
                continue
            pairs.append((image_path, mask_path))

        return pairs

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self._pairs[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.size is not None:
            image = image.resize((self.size, self.size), resample=Image.BICUBIC)
            mask = mask.resize((self.size, self.size), resample=Image.NEAREST)

        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = (mask >= 0.5).astype(np.float32)
        mask = mask * 2.0 - 1.0
        mask = mask[..., None]

        pair = np.concatenate([image, mask], axis=2).astype(np.float32)

        example = {
            "image": pair,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
        if self.class_id is not None:
            example["class_label"] = int(self.class_id)
        if self.class_name is not None:
            example["class_name"] = str(self.class_name)
        return example


class MultiMaskImagePairDataset(Dataset):
    def __init__(
        self,
        roots,
        split="train",
        size=256,
        image_dir="image",
        mask_dir="mask",
        image_exts=(".jpg", ".jpeg", ".png", ".bmp"),
        mask_exts=(".png", ".jpg", ".jpeg", ".bmp"),
        strict_pairs=True,
        class_names=None,
    ):
        if isinstance(roots, str):
            roots = [r.strip() for r in roots.split(",") if r.strip()]
        if not roots:
            raise ValueError("roots must contain at least one dataset path")

        if class_names is None:
            class_names = [Path(r).name for r in roots]
        if len(class_names) != len(roots):
            raise ValueError("class_names must match roots length")

        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.datasets = []
        self.cumulative_lengths = []

        total = 0
        for root, name in zip(roots, class_names):
            class_id = self.class_to_idx[name]
            dataset = MaskImagePairDataset(
                root=root,
                split=split,
                size=size,
                image_dir=image_dir,
                mask_dir=mask_dir,
                image_exts=image_exts,
                mask_exts=mask_exts,
                strict_pairs=strict_pairs,
                class_id=class_id,
                class_name=name,
            )
            self.datasets.append(dataset)
            total += len(dataset)
            self.cumulative_lengths.append(total)

        if total == 0:
            raise ValueError("No samples found across provided roots")

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_lengths, idx)
        prev_total = 0 if dataset_idx == 0 else self.cumulative_lengths[dataset_idx - 1]
        sample_idx = idx - prev_total
        return self.datasets[dataset_idx][sample_idx]

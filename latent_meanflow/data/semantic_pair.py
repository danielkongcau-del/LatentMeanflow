from bisect import bisect_right
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from latent_meanflow.utils.palette import (
    UNDEFINED_CLASS_ID,
    build_lookup_table,
    infer_num_classes,
    resolve_gray_to_class_id,
)


class SemanticImageMaskPairDataset(Dataset):
    def __init__(
        self,
        root,
        gray_to_class_id,
        split="train",
        size=256,
        image_dir="image",
        mask_dir="mask",
        image_exts=(".jpg", ".jpeg", ".png", ".bmp"),
        mask_exts=(".png", ".jpg", ".jpeg", ".bmp"),
        strict_pairs=True,
        ignore_index=None,
        class_label=None,
        class_name=None,
        extra_metadata=None,
    ):
        self.root = Path(root)
        self.split = split
        self.size = size
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_exts = image_exts
        self.mask_exts = mask_exts
        self.strict_pairs = strict_pairs
        self.class_label = class_label
        self.class_name = class_name
        self.extra_metadata = dict(extra_metadata or {})

        self.gray_to_class_id, self.ignore_index = resolve_gray_to_class_id(
            gray_to_class_id, ignore_index=ignore_index
        )
        self.num_classes = infer_num_classes(self.gray_to_class_id, ignore_index=self.ignore_index)
        self._undefined_class_id = UNDEFINED_CLASS_ID
        self._lookup = build_lookup_table(
            self.gray_to_class_id, undefined_value=self._undefined_class_id
        )
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
            image_path = None
            for ext in self.image_exts:
                candidate = image_root / f"{mask_path.stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                if self.strict_pairs:
                    raise FileNotFoundError(f"Missing image for mask: {mask_path}")
                continue
            pairs.append((image_path, mask_path))

        return pairs

    def _load_mask(self, mask_path):
        mask_image = Image.open(mask_path)
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        if self.size is not None:
            mask_image = mask_image.resize((self.size, self.size), resample=Image.NEAREST)

        mask_raw = np.array(mask_image, dtype=np.uint8)
        if int(mask_raw.min()) >= 0 and int(mask_raw.max()) <= int(self.num_classes) - 1:
            # Accept already-indexed masks produced by the teacher-data remap pipeline.
            mask_index = mask_raw.astype(np.int64, copy=False)
        else:
            mask_index = self._lookup[mask_raw]
            undefined_mask = mask_index == self._undefined_class_id
            if np.any(undefined_mask):
                unknown_gray_values = sorted(int(value) for value in np.unique(mask_raw[undefined_mask]).tolist())
                raise ValueError(
                    f"Mask {mask_path} contains gray values missing from gray_to_class_id: {unknown_gray_values}"
                )

        mask_index = mask_index.astype(np.int64, copy=False)
        onehot = np.zeros(mask_index.shape + (self.num_classes,), dtype=np.float32)
        if self.ignore_index is None:
            valid_mask = np.ones(mask_index.shape, dtype=bool)
        else:
            valid_mask = mask_index != self.ignore_index

        if np.any(valid_mask):
            flat_onehot = onehot.reshape(-1, self.num_classes)
            flat_mask = mask_index.reshape(-1)
            flat_valid = valid_mask.reshape(-1)
            valid_positions = np.nonzero(flat_valid)[0]
            flat_onehot[valid_positions, flat_mask[flat_valid]] = 1.0

        return mask_raw, mask_index, onehot

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self._pairs[idx]

        image = Image.open(image_path).convert("RGB")
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=Image.BICUBIC)
        image = np.array(image).astype(np.float32) / 127.5 - 1.0

        mask_raw, mask_index, mask_onehot = self._load_mask(mask_path)
        metadata = {
            "root": str(self.root),
            "split": self.split,
            "sample_stem": image_path.stem,
            "raw_gray_values": ",".join(str(int(value)) for value in np.unique(mask_raw).tolist()),
        }
        metadata.update(self.extra_metadata)
        if self.class_label is not None:
            metadata["class_label"] = int(self.class_label)
        if self.class_name is not None:
            metadata["class_name"] = str(self.class_name)

        example = {
            "image": image.astype(np.float32, copy=False),
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
            "num_classes": int(self.num_classes),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "metadata": metadata,
        }
        if self.class_label is not None:
            example["class_label"] = int(self.class_label)
        if self.class_name is not None:
            example["class_name"] = str(self.class_name)
        return example


class MultiSemanticImageMaskPairDataset(Dataset):
    def __init__(
        self,
        roots,
        gray_to_class_id,
        split="train",
        size=256,
        image_dir="image",
        mask_dir="mask",
        image_exts=(".jpg", ".jpeg", ".png", ".bmp"),
        mask_exts=(".png", ".jpg", ".jpeg", ".bmp"),
        strict_pairs=True,
        ignore_index=None,
        class_labels=None,
        class_names=None,
    ):
        if isinstance(roots, str):
            roots = [root.strip() for root in roots.split(",") if root.strip()]
        if not roots:
            raise ValueError("roots must contain at least one dataset path")

        if class_labels is not None and len(class_labels) != len(roots):
            raise ValueError("class_labels must match roots length")
        if class_names is not None and len(class_names) != len(roots):
            raise ValueError("class_names must match roots length")

        self.gray_to_class_id, self.ignore_index = resolve_gray_to_class_id(
            gray_to_class_id, ignore_index=ignore_index
        )
        self.num_classes = infer_num_classes(self.gray_to_class_id, ignore_index=self.ignore_index)
        self.datasets = []
        self.cumulative_lengths = []

        total = 0
        for dataset_idx, root in enumerate(roots):
            dataset = SemanticImageMaskPairDataset(
                root=root,
                gray_to_class_id=self.gray_to_class_id,
                split=split,
                size=size,
                image_dir=image_dir,
                mask_dir=mask_dir,
                image_exts=image_exts,
                mask_exts=mask_exts,
                strict_pairs=strict_pairs,
                ignore_index=self.ignore_index,
                class_label=None if class_labels is None else class_labels[dataset_idx],
                class_name=None if class_names is None else class_names[dataset_idx],
                extra_metadata={"dataset_index": int(dataset_idx), "source_root": str(Path(root))},
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
        sample = self.datasets[dataset_idx][sample_idx]
        sample["num_classes"] = int(self.num_classes)
        return sample

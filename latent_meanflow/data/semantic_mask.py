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


class SemanticMaskDataset(Dataset):
    def __init__(
        self,
        root,
        gray_to_class_id,
        split="train",
        size=256,
        mask_dir="masks",
        mask_exts=(".png", ".jpg", ".jpeg", ".bmp"),
        ignore_index=None,
        extra_metadata=None,
    ):
        self.root = Path(root)
        self.split = str(split)
        self.size = size
        self.mask_dir = str(mask_dir)
        self.mask_exts = tuple(mask_exts)
        self.extra_metadata = dict(extra_metadata or {})

        self.gray_to_class_id, self.ignore_index = resolve_gray_to_class_id(
            gray_to_class_id, ignore_index=ignore_index
        )
        self.num_classes = infer_num_classes(self.gray_to_class_id, ignore_index=self.ignore_index)
        self._undefined_class_id = UNDEFINED_CLASS_ID
        self._lookup = build_lookup_table(
            self.gray_to_class_id,
            undefined_value=self._undefined_class_id,
        )
        self._mask_paths = self._collect_mask_paths()
        if not self._mask_paths:
            raise ValueError(f"No semantic masks found in {self.root}/{self.split}/{self.mask_dir}")

    def _collect_mask_paths(self):
        mask_root = self.root / self.split / self.mask_dir
        if not mask_root.exists():
            raise FileNotFoundError(f"Mask dir not found: {mask_root}")

        mask_paths = []
        for ext in self.mask_exts:
            mask_paths.extend(mask_root.glob(f"*{ext}"))
        return sorted(mask_paths)

    def _load_mask(self, mask_path):
        mask_image = Image.open(mask_path)
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        if self.size is not None:
            mask_image = mask_image.resize((self.size, self.size), resample=Image.NEAREST)

        mask_raw = np.asarray(mask_image, dtype=np.uint8)
        mask_index = self._lookup[mask_raw]
        undefined_mask = mask_index == self._undefined_class_id
        if np.any(undefined_mask):
            unknown_values = sorted(int(value) for value in np.unique(mask_raw[undefined_mask]).tolist())
            raise ValueError(
                f"Mask {mask_path} contains gray values missing from gray_to_class_id: {unknown_values}"
            )

        mask_index = mask_index.astype(np.int64, copy=False)
        mask_onehot = np.zeros(mask_index.shape + (self.num_classes,), dtype=np.float32)
        if self.ignore_index is None:
            valid_mask = np.ones(mask_index.shape, dtype=bool)
        else:
            valid_mask = mask_index != self.ignore_index

        if np.any(valid_mask):
            flat_onehot = mask_onehot.reshape(-1, self.num_classes)
            flat_index = mask_index.reshape(-1)
            flat_valid = valid_mask.reshape(-1)
            valid_positions = np.nonzero(flat_valid)[0]
            flat_onehot[valid_positions, flat_index[flat_valid]] = 1.0

        return mask_raw, mask_index, mask_onehot

    def __len__(self):
        return len(self._mask_paths)

    def __getitem__(self, idx):
        mask_path = self._mask_paths[idx]
        mask_raw, mask_index, mask_onehot = self._load_mask(mask_path)
        metadata = {
            "root": str(self.root),
            "split": self.split,
            "sample_stem": mask_path.stem,
            "raw_gray_values": ",".join(str(int(value)) for value in np.unique(mask_raw).tolist()),
        }
        metadata.update(self.extra_metadata)
        return {
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
            "num_classes": int(self.num_classes),
            "mask_path": str(mask_path),
            "metadata": metadata,
        }


class MultiSemanticMaskDataset(Dataset):
    def __init__(
        self,
        roots,
        gray_to_class_id,
        split="train",
        size=256,
        mask_dir="masks",
        mask_exts=(".png", ".jpg", ".jpeg", ".bmp"),
        ignore_index=None,
    ):
        if isinstance(roots, str):
            roots = [root.strip() for root in roots.split(",") if root.strip()]
        if not roots:
            raise ValueError("roots must contain at least one dataset path")

        self.gray_to_class_id, self.ignore_index = resolve_gray_to_class_id(
            gray_to_class_id, ignore_index=ignore_index
        )
        self.num_classes = infer_num_classes(self.gray_to_class_id, ignore_index=self.ignore_index)
        self.datasets = []
        self.cumulative_lengths = []

        total = 0
        for dataset_idx, root in enumerate(roots):
            dataset = SemanticMaskDataset(
                root=root,
                gray_to_class_id=self.gray_to_class_id,
                split=split,
                size=size,
                mask_dir=mask_dir,
                mask_exts=mask_exts,
                ignore_index=self.ignore_index,
                extra_metadata={"dataset_index": int(dataset_idx), "source_root": str(Path(root))},
            )
            self.datasets.append(dataset)
            total += len(dataset)
            self.cumulative_lengths.append(total)

        if total == 0:
            raise ValueError("No semantic mask samples found across provided roots")

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_lengths, idx)
        prev_total = 0 if dataset_idx == 0 else self.cumulative_lengths[dataset_idx - 1]
        sample_idx = idx - prev_total
        sample = self.datasets[dataset_idx][sample_idx]
        sample["num_classes"] = int(self.num_classes)
        return sample

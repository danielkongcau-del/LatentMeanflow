from bisect import bisect_right
from pathlib import Path

import numpy as np

from latent_meanflow.data.semantic_mask import SemanticMaskDataset
from latent_meanflow.utils.palette import build_default_palette, colorize_mask_index


class SemanticPaletteMaskDataset(SemanticMaskDataset):
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
        palette_spec=None,
    ):
        super().__init__(
            root=root,
            gray_to_class_id=gray_to_class_id,
            split=split,
            size=size,
            mask_dir=mask_dir,
            mask_exts=mask_exts,
            ignore_index=ignore_index,
            extra_metadata=extra_metadata,
        )
        self.palette_spec = palette_spec or build_default_palette(
            self.num_classes,
            ignore_index=self.ignore_index,
        )

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        mask_index = np.asarray(sample["mask_index"], dtype=np.int64)
        palette_image = colorize_mask_index(
            mask_index,
            num_classes=self.num_classes,
            palette_spec=self.palette_spec,
            ignore_index=self.ignore_index,
        )
        sample["palette_image"] = np.transpose(palette_image.astype(np.float32) / 255.0, (2, 0, 1))
        sample["palette_rgb"] = palette_image
        return sample


class MultiSemanticPaletteMaskDataset:
    def __init__(
        self,
        roots,
        gray_to_class_id,
        split="train",
        size=256,
        mask_dir="masks",
        mask_exts=(".png", ".jpg", ".jpeg", ".bmp"),
        ignore_index=None,
        palette_spec=None,
    ):
        if isinstance(roots, str):
            roots = [root.strip() for root in roots.split(",") if root.strip()]
        if not roots:
            raise ValueError("roots must contain at least one dataset path")

        self.datasets = []
        self.cumulative_lengths = []
        self.palette_spec = palette_spec
        total = 0
        for dataset_idx, root in enumerate(roots):
            dataset = SemanticPaletteMaskDataset(
                root=root,
                gray_to_class_id=gray_to_class_id,
                split=split,
                size=size,
                mask_dir=mask_dir,
                mask_exts=mask_exts,
                ignore_index=ignore_index,
                extra_metadata={"dataset_index": int(dataset_idx), "source_root": str(Path(root))},
                palette_spec=self.palette_spec,
            )
            self.datasets.append(dataset)
            total += len(dataset)
            self.cumulative_lengths.append(total)

        if total == 0:
            raise ValueError("No semantic palette-mask samples found across provided roots")
        self.num_classes = int(self.datasets[0].num_classes)
        self.ignore_index = self.datasets[0].ignore_index

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_lengths, idx)
        prev_total = 0 if dataset_idx == 0 else self.cumulative_lengths[dataset_idx - 1]
        sample_idx = idx - prev_total
        sample = self.datasets[dataset_idx][sample_idx]
        sample["num_classes"] = int(self.num_classes)
        return sample

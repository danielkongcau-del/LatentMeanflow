import tempfile
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from latent_meanflow.data.semantic_pair import SemanticImageMaskPairDataset


def write_sample_dataset(root):
    image_dir = root / "train" / "images"
    mask_dir = root / "train" / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.array(
        [
            [[0, 64, 128], [255, 128, 0], [32, 64, 96]],
            [[16, 32, 48], [200, 210, 220], [255, 255, 255]],
            [[120, 60, 30], [80, 40, 20], [10, 20, 30]],
        ],
        dtype=np.uint8,
    )
    mask = np.array(
        [
            [0, 64, 128],
            [255, 200, 64],
            [128, 0, 255],
        ],
        dtype=np.uint8,
    )

    Image.fromarray(image, mode="RGB").save(image_dir / "sample.png")
    Image.fromarray(mask, mode="L").save(mask_dir / "sample.png")


def main():
    with tempfile.TemporaryDirectory(prefix="semantic_pair_smoke_") as tmpdir:
        root = Path(tmpdir)
        write_sample_dataset(root)

        dataset = SemanticImageMaskPairDataset(
            root=root,
            split="train",
            size=None,
            image_dir="images",
            mask_dir="masks",
            gray_to_class_id={0: 0, 64: 1, 128: 2, 255: 3, 200: -1},
            ignore_index=-1,
        )
        sample = dataset[0]

        assert sample["image"].shape == (3, 3, 3)
        assert sample["image"].dtype == np.float32
        assert float(sample["image"].min()) >= -1.0
        assert float(sample["image"].max()) <= 1.0

        assert sample["mask_index"].shape == (3, 3)
        assert sample["mask_index"].dtype == np.int64
        assert set(np.unique(sample["mask_index"]).tolist()) == {-1, 0, 1, 2, 3}

        assert sample["mask_onehot"].shape == (3, 3, 4)
        assert sample["mask_onehot"].dtype == np.float32
        sums = sample["mask_onehot"].sum(axis=-1)
        ignored = sample["mask_index"] == -1
        assert np.allclose(sums[~ignored], 1.0)
        assert np.allclose(sums[ignored], 0.0)

        assert sample["num_classes"] == 4
        assert sample["image_path"].endswith("sample.png")
        assert sample["mask_path"].endswith("sample.png")
        assert "metadata" in sample

        print("Semantic pair smoke test passed")
        print(f"image shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
        print(f"mask_index shape: {sample['mask_index'].shape}, dtype: {sample['mask_index'].dtype}")
        print(f"mask_onehot shape: {sample['mask_onehot'].shape}, dtype: {sample['mask_onehot'].dtype}")
        print(f"num_classes: {sample['num_classes']}")


if __name__ == "__main__":
    main()

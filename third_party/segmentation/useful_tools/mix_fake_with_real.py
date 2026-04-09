import argparse
import random
import shutil
from pathlib import Path


def resolve_image_mask_dirs(root: Path):
    for img_name, mask_name in (("images", "masks"), ("image", "mask")):
        img_dir = root / img_name
        mask_dir = root / mask_name
        if img_dir.is_dir() and mask_dir.is_dir():
            return img_dir, mask_dir
    return None, None


def list_pairs(image_dir: Path, mask_dir: Path):
    image_files = [p for p in image_dir.iterdir() if p.is_file()]
    mask_files = [p for p in mask_dir.iterdir() if p.is_file()]

    image_map = {}
    for img in image_files:
        image_map.setdefault(img.stem, img)

    pairs = []
    missing_images = []
    for mask in mask_files:
        img = image_map.get(mask.stem)
        if img is None:
            missing_images.append(mask.name)
            continue
        pairs.append((img, mask))

    pairs.sort(key=lambda x: x[0].name)
    return pairs, missing_images


def copy_pairs(pairs, dst_image_dir: Path, dst_mask_dir: Path, prefix: str = ""):
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    for img, mask in pairs:
        new_stem = f"{prefix}{img.stem}"
        dst_img = dst_image_dir / f"{new_stem}{img.suffix}"
        dst_mask = dst_mask_dir / f"{new_stem}{mask.suffix}"
        if dst_img.exists() or dst_mask.exists():
            raise FileExistsError(f"File exists: {dst_img} or {dst_mask}")
        shutil.copy2(img, dst_img)
        shutil.copy2(mask, dst_mask)


def parse_list(value: str):
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=str, default="Data/Original_Dataset")
    parser.add_argument("--fake-root", type=str, default="Data")
    parser.add_argument("--out-root", type=str, default="Data/Mixed")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--num-fake", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    real_root = Path(args.real_root)
    fake_root = Path(args.fake_root)
    out_root = Path(args.out_root)

    datasets = parse_list(args.datasets)
    if not datasets:
        datasets = [p.name for p in real_root.iterdir() if p.is_dir()]

    models = parse_list(args.models)
    if not models:
        ignore = {"Original_Dataset", "AblationExps", "aug", "Mixed"}
        models = [p.name for p in fake_root.iterdir()
                  if p.is_dir() and p.name not in ignore]

    if args.num_fake < 0:
        raise ValueError("--num-fake must be >= 0")

    rng = random.Random(args.seed)

    for dataset in datasets:
        real_dataset_dir = real_root / dataset
        if not real_dataset_dir.is_dir():
            print(f"[Skip] Real dataset not found: {real_dataset_dir}")
            continue

        real_splits = {}
        for split in ("train", "val", "test"):
            split_dir = real_dataset_dir / split
            img_dir, mask_dir = resolve_image_mask_dirs(split_dir)
            if img_dir is None:
                print(f"[Skip] Missing {split} image/mask dirs: {split_dir}")
                real_splits = {}
                break
            pairs, missing = list_pairs(img_dir, mask_dir)
            if missing:
                print(f"[Warn] {dataset}/{split}: {len(missing)} masks missing images")
            real_splits[split] = pairs

        if not real_splits:
            continue

        for model in models:
            fake_dataset_dir = fake_root / model / dataset
            img_dir, mask_dir = resolve_image_mask_dirs(fake_dataset_dir)
            if img_dir is None:
                print(f"[Skip] Fake dataset not found: {fake_dataset_dir}")
                continue

            fake_pairs, missing = list_pairs(img_dir, mask_dir)
            if missing:
                print(f"[Warn] {model}/{dataset}: {len(missing)} masks missing images")

            if args.num_fake > len(fake_pairs):
                raise ValueError(
                    f"Requested {args.num_fake} fake images, but only {len(fake_pairs)} available "
                    f"for {model}/{dataset}"
                )

            fake_pairs_shuffled = fake_pairs[:]
            rng.shuffle(fake_pairs_shuffled)
            fake_sample = fake_pairs_shuffled[: args.num_fake]

            out_base = out_root / dataset / f"{model}_fake{args.num_fake}"
            out_train = out_base / "train"
            out_val = out_base / "val"
            out_test = out_base / "test"

            if out_base.exists():
                raise FileExistsError(f"Output already exists: {out_base}")

            print(f"[Build] {dataset} + {model} (fake={args.num_fake}) -> {out_base}")

            if args.dry_run:
                continue

            copy_pairs(real_splits["train"], out_train / "image", out_train / "mask")
            fake_prefix = f"fake_{model}_"
            copy_pairs(fake_sample, out_train / "image", out_train / "mask", prefix=fake_prefix)

            copy_pairs(real_splits["val"], out_val / "image", out_val / "mask")
            copy_pairs(real_splits["test"], out_test / "image", out_test / "mask")

    print("Done.")


if __name__ == "__main__":
    main()

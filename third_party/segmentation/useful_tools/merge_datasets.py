import argparse
import shutil
from pathlib import Path


def parse_list(value: str):
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


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


def copy_pairs(pairs, dst_image_dir: Path, dst_mask_dir: Path, prefix: str):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, required=True,
                        help="Comma-separated dataset roots or split folders to merge")
    parser.add_argument("--out-root", type=str, default="Data/Merged")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma-separated splits to merge (default: train,val,test)")
    parser.add_argument("--single-split", type=str, default="train",
                        help="Split name to use when a source is a single split folder")
    parser.add_argument("--prefix", type=str, default="{source}_",
                        help="Filename prefix template. Use {source} and {split}.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = parse_list(args.sources)
    splits = parse_list(args.splits)
    out_root = Path(args.out_root)

    if not sources:
        raise ValueError("--sources is empty")
    if not splits:
        raise ValueError("--splits is empty")

    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            print(f"[Skip] Source not found: {src_path}")
            continue

        has_all_splits = all((src_path / split).is_dir() for split in splits)
        if has_all_splits:
            source_name = src_path.name
            for split in splits:
                split_dir = src_path / split
                img_dir, mask_dir = resolve_image_mask_dirs(split_dir)
                if img_dir is None:
                    print(f"[Skip] Missing image/mask dirs: {split_dir}")
                    continue
                pairs, missing = list_pairs(img_dir, mask_dir)
                if missing:
                    print(f"[Warn] {source_name}/{split}: {len(missing)} masks missing images")

                dst_img_dir = out_root / split / "image"
                dst_mask_dir = out_root / split / "mask"
                prefix = args.prefix.format(source=source_name, split=split)
                print(f"[Merge] {source_name}/{split} -> {out_root}/{split} ({len(pairs)} pairs)")
                if not args.dry_run:
                    copy_pairs(pairs, dst_img_dir, dst_mask_dir, prefix=prefix)
        else:
            img_dir, mask_dir = resolve_image_mask_dirs(src_path)
            if img_dir is None:
                print(f"[Skip] Not a dataset root or split folder: {src_path}")
                continue
            split = args.single_split
            source_name = src_path.name
            pairs, missing = list_pairs(img_dir, mask_dir)
            if missing:
                print(f"[Warn] {source_name}/{split}: {len(missing)} masks missing images")

            dst_img_dir = out_root / split / "image"
            dst_mask_dir = out_root / split / "mask"
            prefix = args.prefix.format(source=source_name, split=split)
            print(f"[Merge] {source_name} -> {out_root}/{split} ({len(pairs)} pairs)")
            if not args.dry_run:
                copy_pairs(pairs, dst_img_dir, dst_mask_dir, prefix=prefix)

    print("Done.")


if __name__ == "__main__":
    main()

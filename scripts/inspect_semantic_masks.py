import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from latent_meanflow.utils.palette import (
    build_gray_value_stats,
    build_mapping_template,
    render_mapping_template,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect grayscale semantic mask values and emit a mapping template.")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="One or more dataset roots, for example: data/remote",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to scan. Missing split directories are skipped.",
    )
    parser.add_argument("--mask-dir", type=str, default="masks")
    parser.add_argument("--mask-exts", nargs="+", default=[".png", ".jpg", ".jpeg", ".bmp"])
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON/YAML output path.")
    parser.add_argument(
        "--format",
        choices=["auto", "json", "yaml"],
        default="auto",
        help="Template output format. 'auto' uses the output suffix or YAML for stdout.",
    )
    return parser.parse_args()


def iter_mask_paths(roots, splits, mask_dir, mask_exts):
    for root in roots:
        root_path = Path(root)
        for split in splits:
            split_dir = root_path / split / mask_dir
            if not split_dir.exists():
                continue
            for ext in mask_exts:
                yield from sorted(split_dir.glob(f"*{ext}"))


def inspect_masks(mask_paths):
    gray_value_counts = Counter()
    file_count = 0
    for mask_path in mask_paths:
        mask_image = Image.open(mask_path)
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_array = np.array(mask_image, dtype=np.uint8)
        unique_values, counts = np.unique(mask_array, return_counts=True)
        for gray_value, count in zip(unique_values.tolist(), counts.tolist()):
            gray_value_counts[int(gray_value)] += int(count)
        file_count += 1
    return gray_value_counts, file_count


def resolve_output_format(output_path, output_format):
    if output_format != "auto":
        return output_format
    if output_path is None:
        return "yaml"
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        return "json"
    return "yaml"


def main():
    args = parse_args()

    mask_paths = list(iter_mask_paths(args.roots, args.splits, args.mask_dir, args.mask_exts))
    if not mask_paths:
        raise FileNotFoundError("No mask files found for the provided roots/splits/mask-dir")

    gray_value_counts, file_count = inspect_masks(mask_paths)
    stats, total_pixels = build_gray_value_stats(gray_value_counts)

    print(f"Scanned {file_count} mask files")
    print(f"Total pixels: {total_pixels}")
    print("Gray value statistics:")
    for stat in stats:
        ratio_pct = stat["pixel_ratio"] * 100.0
        print(
            f"  gray={stat['gray_value']:>3} pixels={stat['pixel_count']:>12} ratio={ratio_pct:>8.4f}%"
        )

    template = build_mapping_template(gray_value_counts.keys())
    output_payload = {
        "gray_to_class_id": template["gray_to_class_id"],
        "ignore_index": template["ignore_index"],
        "stats": {str(stat["gray_value"]): stat for stat in stats},
        "total_pixels": int(total_pixels),
        "num_masks": int(file_count),
    }

    output_format = resolve_output_format(args.output, args.format)
    rendered = render_mapping_template(output_payload, output_format=output_format)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote template to {args.output}")
    else:
        print()
        print(rendered)


if __name__ == "__main__":
    main()

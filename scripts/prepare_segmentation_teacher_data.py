import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from latent_meanflow.utils import prepare_segmentation_teacher_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a contiguous class-index dataset view for the vendored remote-sensing "
            "segmentation teacher harness. The source masks remain unchanged; this writes a "
            "project-layer indexed copy or hardlinked view for training teachers on real data."
        )
    )
    parser.add_argument("--src-root", type=Path, required=True, help="Source dataset root, for example data/remote.")
    parser.add_argument("--dst-root", type=Path, required=True, help="Prepared dataset root.")
    parser.add_argument(
        "--label-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "label_specs" / "remote_semantic.yaml",
        help="Gray-to-class label spec used to remap raw semantic masks to contiguous class ids.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--link-mode",
        choices=["auto", "copy", "hardlink"],
        default="auto",
        help="How to materialize image files into the prepared teacher dataset view.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = prepare_segmentation_teacher_dataset(
        src_root=args.src_root,
        dst_root=args.dst_root,
        label_spec_path=args.label_spec,
        splits=args.splits,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    print(f"Prepared segmentation teacher dataset at {args.dst_root.resolve()}")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

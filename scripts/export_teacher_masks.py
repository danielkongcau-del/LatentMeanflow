import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from latent_meanflow.utils import (
    load_teacher_model,
    predict_masks_for_paths,
    resolve_label_spec_metadata,
    write_teacher_mask_triplet,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen in-domain segmentation teacher on renderer outputs and export "
            "precomputed teacher masks in the nfe*/teacher_mask_raw format expected by "
            "eval_mask_layout_faithfulness.py."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Teacher training run directory containing train_args.json.")
    parser.add_argument("--ckpt", type=Path, default=None, help="Optional explicit teacher checkpoint path.")
    parser.add_argument("--generated-root", type=Path, required=True, help="Renderer output root containing nfe*/generated_image.")
    parser.add_argument(
        "--label-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "label_specs" / "remote_semantic.yaml",
    )
    parser.add_argument("--split", type=str, default="validation", help="Metadata-only split tag written into the summary.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=None)
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _prepare_outdir(path, overwrite):
    path = Path(path).resolve()
    if path.exists():
        existing = list(path.rglob("*"))
        if existing and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {path}. "
                "Use a fresh outdir or pass --overwrite."
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_nfe_dirs(generated_root, requested_nfe_values):
    generated_root = Path(generated_root).resolve()
    if requested_nfe_values:
        resolved = []
        for value in requested_nfe_values:
            path = generated_root / f"nfe{int(value)}"
            if path.exists():
                resolved.append((int(value), path))
        if resolved:
            return resolved
    auto_dirs = []
    for path in sorted(generated_root.glob("nfe*")):
        try:
            nfe = int(path.name.replace("nfe", ""))
        except ValueError:
            continue
        auto_dirs.append((nfe, path))
    return auto_dirs


def _collect_image_paths(nfe_dir):
    image_dir = nfe_dir / "generated_image"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing generated_image directory under {nfe_dir}")
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No generated images found under {image_dir}")
    return image_paths


def _load_rgb_uint8(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _write_summary_md(path, summary):
    lines = [
        "# Precomputed Teacher Mask Export",
        "",
        f"- teacher run: `{summary['teacher_run_dir']}`",
        f"- teacher checkpoint: `{summary['teacher_checkpoint']}`",
        f"- teacher net: `{summary['teacher_net_name']}`",
        f"- generated_root: `{summary['generated_root']}`",
        f"- split tag: `{summary['split']}`",
        f"- teacher_mask_root to pass into renderer eval: `{summary['teacher_mask_root']}`",
        "- contract: renderer eval expects `nfe*/teacher_mask_raw/*.png` under the exact path passed to `--teacher-mask-root`.",
        "",
        "| NFE | Exported teacher masks | Color masks | Overlays |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['nfe']} | {row['teacher_mask_count']} | {row['teacher_mask_color_count']} | {row['teacher_overlay_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    outdir = _prepare_outdir(args.outdir, overwrite=args.overwrite)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_metadata = resolve_label_spec_metadata(args.label_spec)
    model, teacher_metadata = load_teacher_model(
        run_dir=args.run_dir,
        device=device,
        checkpoint_path=args.ckpt,
    )
    if int(teacher_metadata["out_channels"]) != int(label_metadata["num_classes"]):
        raise ValueError(
            f"Teacher predicts {teacher_metadata['out_channels']} classes, "
            f"but the label spec expects {label_metadata['num_classes']}."
        )
    nfe_dirs = _resolve_nfe_dirs(args.generated_root, args.nfe_values)
    if not nfe_dirs:
        raise FileNotFoundError(f"No nfe* directories found under {Path(args.generated_root).resolve()}")

    rows = []
    input_size_hw = (int(teacher_metadata["height"]), int(teacher_metadata["width"]))
    for nfe, nfe_dir in nfe_dirs:
        image_paths = _collect_image_paths(nfe_dir)
        pred_masks = predict_masks_for_paths(
            image_paths=image_paths,
            model=model,
            input_size_hw=input_size_hw,
            device=device,
            batch_size=args.batch_size,
            output_size_mode="original",
        )
        export_dir = outdir / f"nfe{int(nfe)}"
        export_dir.mkdir(parents=True, exist_ok=True)
        for image_path, pred_mask in zip(image_paths, pred_masks):
            rgb_uint8 = _load_rgb_uint8(image_path)
            write_teacher_mask_triplet(
                mask_index=pred_mask.cpu().numpy().astype(np.int64, copy=False),
                rgb_uint8=rgb_uint8,
                outdir=export_dir,
                stem=image_path.stem,
                num_classes=label_metadata["num_classes"],
                overlay_alpha=args.overlay_alpha,
            )
        rows.append(
            {
                "nfe": int(nfe),
                "generated_dir": str((nfe_dir / "generated_image").resolve()),
                "teacher_mask_count": len(list((export_dir / "teacher_mask_raw").glob("*.png"))),
                "teacher_mask_color_count": len(list((export_dir / "teacher_mask_color").glob("*.png"))),
                "teacher_overlay_count": len(list((export_dir / "teacher_overlay").glob("*.png"))),
            }
        )

    summary = {
        "teacher_run_dir": teacher_metadata["run_dir"],
        "teacher_checkpoint": teacher_metadata["checkpoint_path"],
        "teacher_net_name": teacher_metadata["net_name"],
        "teacher_input_height": int(teacher_metadata["height"]),
        "teacher_input_width": int(teacher_metadata["width"]),
        "generated_root": str(Path(args.generated_root).resolve()),
        "teacher_mask_root": str(outdir.resolve()),
        "split": str(args.split),
        "label_spec": str(Path(args.label_spec).resolve()),
        "path_contract": {
            "teacher_mask_root_argument": str(outdir.resolve()),
            "required_subdirs": [
                "nfe*/teacher_mask_raw",
                "nfe*/teacher_mask_color",
                "nfe*/teacher_overlay",
            ],
            "recommended_layout": "outputs/precomputed_teacher_masks/<winner_alias>/<split>",
        },
        "results": rows,
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    summary_md_path = outdir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nfe",
                "generated_dir",
                "teacher_mask_count",
                "teacher_mask_color_count",
                "teacher_overlay_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    _write_summary_md(summary_md_path, summary)

    print(f"Saved precomputed teacher masks to {outdir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()

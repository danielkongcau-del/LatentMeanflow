import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from latent_meanflow.utils.image_tokenizer_audit import (
    compare_summaries,
    evaluate_image_tokenizer,
    write_eval_markdown,
    write_summary_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an image-only tokenizer checkpoint and optionally compare "
            "it against a reference tokenizer on the same split."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--reference-config", type=Path, default=None)
    parser.add_argument("--reference-ckpt", type=Path, default=None)
    parser.add_argument("--reference-name", type=str, default=None)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--collapse-std-threshold", type=float, default=0.05)
    parser.add_argument("--export-visuals", action="store_true")
    parser.add_argument("--visual-samples", type=int, default=6)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--error-heatmap-max", type=float, default=0.20)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "outputs" / "image_tokenizer_eval",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summaries = []
    primary_name = args.name or args.config.stem
    summaries.append(
        evaluate_image_tokenizer(
            name=primary_name,
            config_path=args.config,
            ckpt_path=args.ckpt,
            split=args.split,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            max_batches=args.max_batches,
            device=device,
            collapse_std_threshold=args.collapse_std_threshold,
            export_visuals=bool(args.export_visuals),
            visual_outdir=outdir / "visuals" / primary_name,
            visual_samples=args.visual_samples,
            crop_size=args.crop_size,
            error_heatmap_max=args.error_heatmap_max,
        )
    )

    comparison = None
    if args.reference_config is not None or args.reference_ckpt is not None:
        if args.reference_config is None or args.reference_ckpt is None:
            raise ValueError("Both --reference-config and --reference-ckpt are required for comparison mode.")
        reference_name = args.reference_name or args.reference_config.stem
        reference_summary = evaluate_image_tokenizer(
            name=reference_name,
            config_path=args.reference_config,
            ckpt_path=args.reference_ckpt,
            split=args.split,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            max_batches=args.max_batches,
            device=device,
            collapse_std_threshold=args.collapse_std_threshold,
            export_visuals=bool(args.export_visuals),
            visual_outdir=outdir / "visuals" / reference_name,
            visual_samples=args.visual_samples,
            crop_size=args.crop_size,
            error_heatmap_max=args.error_heatmap_max,
        )
        summaries.append(reference_summary)
        comparison = compare_summaries(candidate=summaries[0], reference=reference_summary)

    payload = {
        "seed": int(args.seed),
        "device": str(device),
        "split": args.split,
        "max_batches": None if args.max_batches is None else int(args.max_batches),
        "batch_size_override": None if args.batch_size is None else int(args.batch_size),
        "num_workers_override": None if args.num_workers is None else int(args.num_workers),
        "collapse_std_threshold": float(args.collapse_std_threshold),
        "summaries": summaries,
        "comparison": comparison,
    }

    summary_json_path = outdir / "summary.json"
    summary_md_path = outdir / "summary.md"
    summary_csv_path = outdir / "summary.csv"
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_eval_markdown(summary_md_path, summaries=summaries, comparison=comparison)
    write_summary_csv(summary_csv_path, summaries=summaries)

    print(f"Saved image tokenizer evaluation JSON to {summary_json_path}")
    print(f"Saved image tokenizer evaluation markdown to {summary_md_path}")
    print(f"Saved image tokenizer evaluation CSV to {summary_csv_path}")


if __name__ == "__main__":
    main()

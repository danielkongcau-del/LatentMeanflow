import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"
SCRIPT_ROOT = Path(__file__).resolve().parent

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT, SCRIPT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from find_checkpoint import find_best_checkpoint, find_last_checkpoint, resolve_run_dir
from latent_meanflow.utils.image_tokenizer_audit import (
    evaluate_image_tokenizer,
    flatten_summary,
    format_value,
    write_summary_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit one or more image-only tokenizer checkpoints, export visual "
            "diagnostics, and rank them for downstream mask-conditioned rendering."
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Repeatable spec: name|config_path|checkpoint_path",
    )
    parser.add_argument(
        "--candidate-run",
        action="append",
        default=[],
        help="Repeatable spec: name|config_path|run_dir. Uses monitor-aware checkpoint selection.",
    )
    parser.add_argument("--selection", choices=["best", "last"], default="best")
    parser.add_argument("--monitor", type=str, default=None)
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
        default=REPO_ROOT / "outputs" / "image_tokenizer_audit",
    )
    return parser.parse_args()


def _parse_spec(raw_spec, expected_parts):
    parts = [part.strip() for part in str(raw_spec).split("|")]
    if len(parts) != expected_parts or any(not part for part in parts):
        raise ValueError(
            f"Invalid spec '{raw_spec}'. Expected {'|'.join(['field'] * expected_parts)} with no empty parts."
        )
    return parts


def _default_monitor_from_config(config_path):
    config = OmegaConf.load(config_path)
    monitor = OmegaConf.select(config, "model.params.monitor", default=None)
    return "val/total_loss" if monitor is None else str(monitor)


def _resolve_run_candidate(spec, selection, monitor_override):
    name, config_path_raw, run_dir_raw = _parse_spec(spec, expected_parts=3)
    config_path = Path(config_path_raw).resolve()
    run_dir = resolve_run_dir(type("Args", (), {"run_dir": Path(run_dir_raw), "config": None, "run_tag": None, "logs_root": None})())

    if selection == "last":
        ckpt_path = find_last_checkpoint(run_dir, filename="last.ckpt")
        monitor = None
    else:
        monitor = monitor_override or _default_monitor_from_config(config_path)
        ckpt_path = find_best_checkpoint(run_dir, monitor=monitor)
    return {
        "name": name,
        "config_path": config_path,
        "checkpoint_path": Path(ckpt_path).resolve(),
        "run_dir": str(Path(run_dir).resolve()),
        "selection": selection,
        "monitor": monitor,
    }


def _resolve_direct_candidate(spec):
    name, config_path_raw, ckpt_path_raw = _parse_spec(spec, expected_parts=3)
    return {
        "name": name,
        "config_path": Path(config_path_raw).resolve(),
        "checkpoint_path": Path(ckpt_path_raw).resolve(),
        "run_dir": None,
        "selection": "explicit",
        "monitor": None,
    }


def _ranking_key(summary):
    rgb_lpips = 1.0 if summary["rgb_lpips"] is None else float(summary["rgb_lpips"])
    return (
        -float(summary["downstream_readiness"]["score"]),
        rgb_lpips,
        float(summary["rgb_l1"]),
    )


def _write_audit_markdown(path, payload):
    lines = [
        "# Image Tokenizer Audit",
        "",
        "## Ranking Heuristic",
        "",
        "- `downstream_readiness.score` is the primary ranking key.",
        "- The score combines 55% perceptual quality, 35% latent health, and 10% latent compactness.",
        "- Perceptual quality uses RGB LPIPS and RGB L1.",
        "- Latent health penalizes low per-channel std, uneven channel std spread, and collapsed channels.",
        "- Compactness only gives a small bonus to tighter latents, so `f=8` is not automatically preferred.",
        "",
        "## Ranked Candidates",
        "",
        "| Rank | Name | Readiness | LPIPS Wt | Adv | RGB LPIPS | RGB L1 | Collapsed Channels | Latent Shape |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for rank, summary in enumerate(payload["ranking"], start=1):
        collapse = summary["channel_collapse"]
        metadata = summary["config_metadata"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    summary["name"],
                    format_value(summary["downstream_readiness"]["score"]),
                    format_value(metadata["loss_weights"]["rgb_lpips_weight"]),
                    str(metadata["adversarial"]["enabled"]),
                    format_value(summary["rgb_lpips"]),
                    format_value(summary["rgb_l1"]),
                    f"{collapse['collapsed_channel_count']}/{len(summary['per_channel_stats'])}",
                    str(summary["latent_shape"]),
                ]
            )
            + " |"
        )

    for summary in payload["ranking"]:
        collapse = summary["channel_collapse"]
        readiness = summary["downstream_readiness"]
        lines.extend(
            [
                "",
                f"## {summary['name']}",
                "",
                f"- config: `{summary['config']}`",
                f"- checkpoint: `{summary['checkpoint']}`",
                f"- split: `{summary['split']}`",
                f"- latent shape: `{summary['latent_shape']}`",
                f"- downsample factor: `{summary['downsample_factor']}`",
                f"- adversarial enabled: `{summary['config_metadata']['adversarial']['enabled']}`",
                f"- generator adversarial weight: `{format_value(summary['config_metadata']['adversarial']['generator_adversarial_weight'])}`",
                f"- RGB LPIPS weight: `{format_value(summary['config_metadata']['loss_weights']['rgb_lpips_weight'])}`",
                f"- latent std floor weight: `{format_value(summary['config_metadata']['loss_weights']['latent_channel_std_floor_weight'])}`",
                f"- latent std floor: `{format_value(summary['config_metadata']['loss_weights']['latent_channel_std_floor'])}`",
                f"- RGB L1: `{format_value(summary['rgb_l1'])}`",
                f"- RGB LPIPS: `{format_value(summary['rgb_lpips'])}`",
                f"- latent mean/std: `{format_value(summary['latent_mean'])}` / `{format_value(summary['latent_std'])}`",
                f"- latent L2 norm mean/std: `{format_value(summary['latent_l2_norm_mean'])}` / `{format_value(summary['latent_l2_norm_std'])}`",
                f"- collapsed channels: `{collapse['collapsed_channel_count']}` / `{len(summary['per_channel_stats'])}`",
                f"- min/max per-channel std: `{format_value(collapse['min_channel_std'])}` / `{format_value(collapse['max_channel_std'])}`",
                f"- channel-std CV: `{format_value(collapse['channel_std_cv'])}`",
                f"- collapse severity: `{collapse['severity']}`",
                f"- readiness score: `{format_value(readiness['score'])}`",
                "",
                "| Channel | Mean | Std |",
                "| --- | ---: | ---: |",
            ]
        )
        for channel_name, stats in summary["per_channel_stats"].items():
            lines.append(f"| {channel_name} | {format_value(stats['mean'])} | {format_value(stats['std'])} |")
        visual_root = summary["visual_diagnostics"]["visual_root"]
        if visual_root is not None and summary["visual_diagnostics"]["exported_samples"] > 0:
            lines.extend(["", f"- visual diagnostics: `{visual_root}`"])

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if not args.candidate and not args.candidate_run:
        raise ValueError("Pass at least one --candidate or --candidate-run spec.")

    torch.manual_seed(int(args.seed))
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_specs = []
    for spec in args.candidate:
        resolved_specs.append(_resolve_direct_candidate(spec))
    for spec in args.candidate_run:
        resolved_specs.append(_resolve_run_candidate(spec, selection=args.selection, monitor_override=args.monitor))

    summaries = []
    for resolved in resolved_specs:
        name = resolved["name"]
        summaries.append(
            evaluate_image_tokenizer(
                name=name,
                config_path=resolved["config_path"],
                ckpt_path=resolved["checkpoint_path"],
                split=args.split,
                batch_size_override=args.batch_size,
                num_workers_override=args.num_workers,
                max_batches=args.max_batches,
                device=device,
                collapse_std_threshold=args.collapse_std_threshold,
                export_visuals=bool(args.export_visuals),
                visual_outdir=outdir / "visuals" / name,
                visual_samples=args.visual_samples,
                crop_size=args.crop_size,
                error_heatmap_max=args.error_heatmap_max,
            )
        )

    ranking = sorted(summaries, key=_ranking_key)
    ranking_rows = []
    for rank, summary in enumerate(ranking, start=1):
        row = flatten_summary(summary)
        row["rank"] = rank
        ranking_rows.append(row)

    payload = {
        "seed": int(args.seed),
        "device": str(device),
        "split": args.split,
        "max_batches": None if args.max_batches is None else int(args.max_batches),
        "batch_size_override": None if args.batch_size is None else int(args.batch_size),
        "num_workers_override": None if args.num_workers is None else int(args.num_workers),
        "collapse_std_threshold": float(args.collapse_std_threshold),
        "resolved_candidates": [
            {
                "name": item["name"],
                "config_path": str(item["config_path"]),
                "checkpoint_path": str(item["checkpoint_path"]),
                "run_dir": item["run_dir"],
                "selection": item["selection"],
                "monitor": item["monitor"],
            }
            for item in resolved_specs
        ],
        "ranking": ranking,
        "ranking_rows": ranking_rows,
    }

    summary_json_path = outdir / "summary.json"
    summary_md_path = outdir / "summary.md"
    summary_csv_path = outdir / "summary.csv"
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv_path, ranking)
    _write_audit_markdown(summary_md_path, payload)

    print(f"Saved tokenizer audit JSON to {summary_json_path}")
    print(f"Saved tokenizer audit markdown to {summary_md_path}")
    print(f"Saved tokenizer audit CSV to {summary_csv_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from latent_meanflow.utils import SUPPORTED_TEACHER_CANDIDATES, evaluate_teacher_on_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run an apples-to-apples held-out bakeoff for in-domain remote-sensing segmentation "
            "teacher candidates and rank them for renderer layout-faithfulness evaluation."
        )
    )
    parser.add_argument(
        "--candidate-run",
        action="append",
        required=True,
        help="Candidate spec in the form alias=run_dir. Repeat for every teacher run.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Prepared contiguous class-index teacher dataset root.",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "validation"], default="val")
    parser.add_argument(
        "--label-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "label_specs" / "remote_semantic.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--boundary-tolerance-px", type=int, default=2)
    parser.add_argument("--small-class-threshold-ratio", type=float, default=0.02)
    parser.add_argument(
        "--focus-class-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional class ids to treat as thin-structure / high-priority classes such as roads or channels.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
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


def _parse_candidate_run(spec):
    if "=" not in spec:
        raise ValueError(f"Invalid --candidate-run value: {spec}. Expected alias=run_dir.")
    alias, run_dir = spec.split("=", 1)
    alias = alias.strip()
    run_dir = run_dir.strip()
    if not alias or not run_dir:
        raise ValueError(f"Invalid --candidate-run value: {spec}. Expected alias=run_dir.")
    return alias, Path(run_dir).resolve()


def _mean_or_fallback(value, fallback):
    return float(fallback) if value is None else float(value)


def _selection_score(metrics):
    miou = _mean_or_fallback(metrics["miou"], 0.0)
    boundary_f1 = _mean_or_fallback(metrics["boundary_f1"], 0.0)
    small_class_miou = _mean_or_fallback(metrics["small_class_miou"], miou)
    worst_class_iou = _mean_or_fallback(metrics["worst_class_iou"], miou)
    focus_class_mean_iou = metrics.get("focus_class_mean_iou")
    if focus_class_mean_iou is None:
        return float(0.55 * miou + 0.20 * boundary_f1 + 0.15 * small_class_miou + 0.10 * worst_class_iou)
    return float(
        0.45 * miou
        + 0.20 * boundary_f1
        + 0.15 * small_class_miou
        + 0.10 * worst_class_iou
        + 0.10 * float(focus_class_mean_iou)
    )


def _lowest_iou_items(per_class_iou_by_name, top_k=3):
    pairs = [
        (str(class_name), float(iou))
        for class_name, iou in per_class_iou_by_name.items()
        if iou is not None
    ]
    pairs.sort(key=lambda item: item[1])
    return pairs[: int(top_k)]


def _write_summary_md(path, summary):
    winner = summary["winner_summary"]
    lines = [
        "# Segmentation Teacher Bakeoff",
        "",
        f"- dataset_root: `{summary['dataset_root']}`",
        f"- split: `{summary['split']}`",
        f"- label_spec: `{summary['label_spec']}`",
        f"- supported first-round candidates: `{', '.join(summary['supported_candidates'])}`",
        f"- winner: `{winner['candidate']}`",
        f"- winner checkpoint: `{winner['checkpoint_path']}`",
        f"- winner mIoU: `{_format_metric(winner['miou'])}`",
        f"- winner boundary_f1: `{_format_metric(winner['boundary_f1'])}`",
        f"- winner small_class_miou: `{_format_metric(winner['small_class_miou'])}`",
        "- selection rule:",
        "  - if no focus classes are set: `0.55*mIoU + 0.20*boundary_f1 + 0.15*small_class_miou + 0.10*worst_class_iou`",
        "  - if focus classes are set: `0.45*mIoU + 0.20*boundary_f1 + 0.15*small_class_miou + 0.10*worst_class_iou + 0.10*focus_class_mean_iou`",
        "- this score is only a tie-break helper; final selection still requires checking per-class IoU for thin/small classes.",
        "",
        "| Rank | Candidate | Net | Score | mIoU | Boundary F1 | Pixel Acc | Small-class mIoU | Focus-class mean IoU | Worst class |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["ranking"]:
        worst_class = "n/a"
        if row["worst_class_name"] is not None and row["worst_class_iou"] is not None:
            worst_class = f"{row['worst_class_name']} ({row['worst_class_iou']:.4f})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    row["candidate"],
                    row["net_name"],
                    f"{row['selection_score']:.4f}",
                    _format_metric(row["miou"]),
                    _format_metric(row["boundary_f1"]),
                    _format_metric(row["pixel_accuracy"]),
                    _format_metric(row["small_class_miou"]),
                    _format_metric(row["focus_class_mean_iou"]),
                    worst_class,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Candidate Notes")
    lines.append("")
    for row in summary["ranking"]:
        lines.append(f"### {row['rank']}. {row['candidate']}")
        lines.append("")
        lines.append(f"- checkpoint: `{row['checkpoint_path']}`")
        lines.append(f"- run_dir: `{row['run_dir']}`")
        lines.append(f"- net_name: `{row['net_name']}`")
        lines.append(f"- lowest IoU classes: `{json.dumps(row['lowest_iou_classes'], ensure_ascii=True)}`")
        lines.append(f"- small classes: `{json.dumps(row['small_class_names'], ensure_ascii=True)}`")
        lines.append(f"- focus classes: `{json.dumps(row['focus_class_names'], ensure_ascii=True)}`")
        if not row["focus_class_names"]:
            lines.append("- note: no focus classes were inferred from class names. If roads/channels/boundaries map to known ids, rerun with `--focus-class-ids`.")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def main():
    args = parse_args()
    split = "val" if args.split == "validation" else args.split
    outdir = _prepare_outdir(args.outdir, overwrite=args.overwrite)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for candidate_spec in args.candidate_run:
        alias, run_dir = _parse_candidate_run(candidate_spec)
        metrics = evaluate_teacher_on_dataset(
            run_dir=run_dir,
            dataset_root=args.dataset_root,
            split=split,
            label_spec_path=args.label_spec,
            device=device,
            batch_size=args.batch_size,
            boundary_tolerance_px=args.boundary_tolerance_px,
            small_class_threshold_ratio=args.small_class_threshold_ratio,
            focus_class_ids=args.focus_class_ids,
            max_samples=args.max_samples,
        )
        if int(metrics["teacher_out_channels"]) != int(metrics["num_classes"]):
            raise ValueError(
                f"Teacher '{alias}' predicts {metrics['teacher_out_channels']} classes, "
                f"but the label spec expects {metrics['num_classes']}."
            )
        results.append(
            {
                "candidate": alias,
                "selection_score": _selection_score(metrics),
                "lowest_iou_classes": _lowest_iou_items(metrics["per_class_iou_by_name"]),
                **metrics,
            }
        )

    results.sort(key=lambda item: item["selection_score"], reverse=True)
    ranking = []
    for rank, result in enumerate(results, start=1):
        ranking.append(
            {
                "rank": int(rank),
                "candidate": result["candidate"],
                "net_name": result["net_name"],
                "run_dir": result["run_dir"],
                "checkpoint_path": result["checkpoint_path"],
                "selection_score": float(result["selection_score"]),
                "miou": result["miou"],
                "boundary_f1": result["boundary_f1"],
                "pixel_accuracy": result["pixel_accuracy"],
                "small_class_miou": result["small_class_miou"],
                "focus_class_mean_iou": result["focus_class_mean_iou"],
                "small_class_names": result["small_class_names"],
                "focus_class_names": result["focus_class_names"],
                "worst_class_name": result["worst_class_name"],
                "worst_class_iou": result["worst_class_iou"],
                "lowest_iou_classes": result["lowest_iou_classes"],
                "epoch_budget": result["epoch_budget"],
                "train_batch_size": result["train_batch_size"],
                "input_height": result["input_height"],
                "input_width": result["input_width"],
                "sample_count": result["sample_count"],
                "per_class_iou_json": json.dumps(result["per_class_iou_by_name"], ensure_ascii=True, sort_keys=True),
                "per_class_pixel_ratio_json": json.dumps(
                    result["per_class_pixel_ratio_by_name"],
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "small_class_names_json": json.dumps(result["small_class_names"], ensure_ascii=True),
                "focus_class_names_json": json.dumps(result["focus_class_names"], ensure_ascii=True),
                "lowest_iou_classes_json": json.dumps(result["lowest_iou_classes"], ensure_ascii=True),
            }
        )

    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "split": split,
        "label_spec": str(args.label_spec.resolve()),
        "supported_candidates": list(SUPPORTED_TEACHER_CANDIDATES),
        "boundary_tolerance_px": int(args.boundary_tolerance_px),
        "small_class_threshold_ratio": float(args.small_class_threshold_ratio),
        "focus_class_ids": None if args.focus_class_ids is None else [int(value) for value in args.focus_class_ids],
        "ranking": ranking,
        "winner": ranking[0]["candidate"],
        "winner_checkpoint": ranking[0]["checkpoint_path"],
        "winner_summary": {
            "candidate": ranking[0]["candidate"],
            "net_name": ranking[0]["net_name"],
            "checkpoint_path": ranking[0]["checkpoint_path"],
            "selection_score": ranking[0]["selection_score"],
            "miou": ranking[0]["miou"],
            "boundary_f1": ranking[0]["boundary_f1"],
            "pixel_accuracy": ranking[0]["pixel_accuracy"],
            "small_class_miou": ranking[0]["small_class_miou"],
            "focus_class_mean_iou": ranking[0]["focus_class_mean_iou"],
            "worst_class_name": ranking[0]["worst_class_name"],
            "worst_class_iou": ranking[0]["worst_class_iou"],
        },
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    summary_md_path = outdir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rank",
            "candidate",
            "net_name",
            "run_dir",
            "checkpoint_path",
            "selection_score",
            "miou",
            "boundary_f1",
            "pixel_accuracy",
            "small_class_miou",
            "focus_class_mean_iou",
            "worst_class_name",
            "worst_class_iou",
            "epoch_budget",
            "train_batch_size",
            "input_height",
            "input_width",
            "sample_count",
            "per_class_iou_json",
            "per_class_pixel_ratio_json",
            "small_class_names_json",
            "focus_class_names_json",
            "lowest_iou_classes_json",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fieldnames} for row in ranking])
    _write_summary_md(summary_md_path, summary)

    print(f"Saved teacher bakeoff summary to {outdir}")
    print(f"Winner: {summary['winner']}")
    print(f"Winner checkpoint: {summary['winner_checkpoint']}")


if __name__ == "__main__":
    main()

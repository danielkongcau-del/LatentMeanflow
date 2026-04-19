import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.eval_mask_prior import (
    _compare_distribution,
    _load_generated_masks,
    _load_reference_masks_from_dataset,
    _load_reference_masks_from_dir,
    _mean_or_none,
    _nearest_real_mious,
    _pairwise_fake_mious,
    _prepare_outdir,
    _resolve_label_spec_metadata,
    _resolve_split_key,
    _std_or_none,
    _summarize_distribution,
    _to_json_ready,
)
from scripts.sample_latent_flow import load_config, load_model
from scripts.sample_token_mask_prior import (
    DEFAULT_CONFIG,
    DEFAULT_NFE_VALUES,
    DEFAULT_TOKENIZER_CONFIG,
    apply_tokenizer_overrides,
    extract_token_mask_prior_route_metadata,
    generate_token_mask_prior_sweep,
    resolve_configured_tokenizer_artifacts,
    validate_token_mask_prior_checkpoint_contract,
)
from scripts.sample_latent_flow import load_checkpoint_state


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the project-layer token-code p(mask) route. This reports the decoded semantic-mask "
            "distribution against the real split and includes token-usage diagnostics."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--tokenizer-config", type=Path, default=DEFAULT_TOKENIZER_CONFIG)
    parser.add_argument("--tokenizer-ckpt", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument(
        "--label-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "label_specs" / "remote_semantic.yaml",
    )
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=DEFAULT_NFE_VALUES)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--small-region-threshold-ratio", type=float, default=0.02)
    parser.add_argument(
        "--thin-structure-class-ids",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional semantic class ids to audit with thin-structure continuity statistics "
            "(skeleton length, endpoint count, fragment count)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-monitor", type=str, default="val/base_error_mean")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Extra OmegaConf dotlist override. Repeat as needed, without a leading --.",
    )
    return parser.parse_args()


def _check_monitor(config, expected_monitor):
    configured_monitor = config.model.params.get("monitor")
    if expected_monitor is not None and configured_monitor != expected_monitor:
        raise ValueError(
            f"Evaluation monitor mismatch: expected '{expected_monitor}', got '{configured_monitor}'. "
            "Use the best checkpoint selected by val/base_error_mean for the token-code mask baseline."
        )
    return configured_monitor


def _load_generated_codes(nfe_dir, n_samples):
    code_dir = Path(nfe_dir) / "code_raw"
    if not code_dir.exists():
        return None
    code_paths = sorted(code_dir.glob("*.png"))[: int(n_samples)]
    if not code_paths:
        return None
    return [np.asarray(Image.open(path), dtype=np.int64) for path in code_paths]


def _summarize_codes(code_grids, *, codebook_size):
    if not code_grids:
        return None
    flat = np.concatenate([grid.reshape(-1) for grid in code_grids], axis=0)
    counts = np.bincount(flat.astype(np.int64), minlength=int(codebook_size)).astype(np.float64)
    probs = counts / max(1.0, float(counts.sum()))
    unique_per_sample = np.asarray([np.unique(grid).size for grid in code_grids], dtype=np.float64)
    return {
        "active_code_count": int((counts > 0).sum()),
        "active_code_fraction": float((counts > 0).sum() / float(max(int(codebook_size), 1))),
        "code_perplexity": float(np.exp(-(probs * np.log(np.clip(probs, 1.0e-10, 1.0))).sum())),
        "unique_code_count_mean": float(unique_per_sample.mean()),
        "unique_code_count_std": float(unique_per_sample.std()),
    }


def _write_markdown_report(path, summary):
    route = summary.get("route_metadata", {})
    lines = [
        "# Token Mask Prior Evaluation Protocol",
        "",
        f"- task: `{summary['task']}`",
        f"- config: `{summary['config']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- tokenizer config: `{summary['tokenizer_config']}`",
        f"- tokenizer checkpoint: `{summary['tokenizer_checkpoint']}`",
        f"- monitor: `{summary['monitor']}`",
        f"- reference source: `{summary['reference_source']}`",
        f"- small-region threshold ratio: `{summary['small_region_threshold_ratio']}`",
        f"- thin-structure class ids: `{summary['thin_structure_class_ids']}`",
        f"- refinement mode: `{route.get('refinement_mode', 'unknown')}`",
        f"- corruption mode: `{route.get('corruption_mode', 'unknown')}`",
        f"- final full reveal: `{route.get('final_full_reveal', 'unknown')}`",
        f"- min keep fraction: `{route.get('min_keep_fraction', 'unknown')}`",
        f"- lock noise scale: `{route.get('lock_noise_scale', 'unknown')}`",
        f"- reveal noise scale: `{route.get('reveal_noise_scale', 'unknown')}`",
        f"- sample temperature: `{route.get('sample_temperature', 'unknown')}`",
        "",
        "Primary mask-quality readout stays distributional on the decoded semantic masks. Adjacency, largest-component share, hole statistics, and optional thin-structure continuity are included because remote-sensing layout quality depends on topology, not only class area. Token diagnostics remain secondary and answer whether the frozen-tokenizer code vocabulary is actually being used.",
        "",
        "| NFE | nearest-real mIoU | adjacency L1 | adjacency JSD | largest-CC gap | hole-count gap | boundary gap | thin-frag gap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(result["nfe"])),
                    f"{float(result['nearest_real_miou_mean']):.4f}"
                    if result["nearest_real_miou_mean"] is not None
                    else "n/a",
                    f"{float(result['adjacency_matrix_l1_mean']):.4f}",
                    f"{float(result['adjacency_matrix_jsd']):.4f}",
                    f"{float(result['largest_component_class_share_l1_mean']):.4f}",
                    f"{float(result['hole_count_l1_mean']):.4f}",
                    f"{float(result['boundary_length_ratio_gap']):.4f}",
                    "n/a"
                    if result["thin_structure_fragment_count_gap_mean"] is None
                    else f"{float(result['thin_structure_fragment_count_gap_mean']):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| NFE | pairwise fake mIoU | global class ratio L1 | area hist L1 | active codes | code perplexity | unique codes / sample |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in summary["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(result["nfe"])),
                    f"{float(result['pairwise_fake_miou_mean']):.4f}"
                    if result["pairwise_fake_miou_mean"] is not None
                    else "n/a",
                    f"{float(result['global_class_pixel_ratio_l1']):.4f}",
                    f"{float(result['class_area_histogram_l1_mean']):.4f}",
                    str(result["active_code_count"]) if result["active_code_count"] is not None else "n/a",
                    f"{float(result['code_perplexity']):.2f}" if result["code_perplexity"] is not None else "n/a",
                    f"{float(result['unique_code_count_mean']):.2f}" if result["unique_code_count_mean"] is not None else "n/a",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main():
    args = parse_args()
    if not args.tokenizer_ckpt.exists():
        raise FileNotFoundError(f"Frozen tokenizer checkpoint not found: {args.tokenizer_ckpt}")

    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)

    config = load_config(args.config, overrides=args.overrides)
    apply_tokenizer_overrides(
        config,
        tokenizer_config=args.tokenizer_config,
        tokenizer_ckpt=args.tokenizer_ckpt,
    )
    resolved_tokenizer_config_path, resolved_tokenizer_ckpt_path = resolve_configured_tokenizer_artifacts(
        config,
        route_name="Token-mask prior evaluation",
    )
    monitor = _check_monitor(config, args.expected_monitor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    generated_root = None if args.generated_root is None else args.generated_root.resolve()
    ckpt_path = None if args.ckpt is None else args.ckpt.resolve()
    if generated_root is None:
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError("Token-mask prior checkpoint not found. Pass --ckpt explicitly.")
        checkpoint_state = load_checkpoint_state(ckpt_path)
        validate_token_mask_prior_checkpoint_contract(
            config,
            ckpt_path,
            config_path=args.config,
            checkpoint_state=checkpoint_state,
        )
        model = load_model(config, ckpt_path, device=device, checkpoint_state=checkpoint_state)
        route_metadata = extract_token_mask_prior_route_metadata(config=config, model=model)
        generated_root = outdir / "generated"
        _prepare_outdir(generated_root, overwrite=args.overwrite)
        generate_token_mask_prior_sweep(
            model=model,
            outdir=generated_root,
            nfe_values=args.nfe_values,
            seed=args.seed,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
        )
        codebook_size = int(model.codebook_size)
    else:
        tokenizer_config = OmegaConf.load(args.tokenizer_config)
        codebook_size = int(OmegaConf.select(tokenizer_config, "model.params.codebook_size"))
        route_metadata = extract_token_mask_prior_route_metadata(config=config, model=None)

    split_key = _resolve_split_key(config, args.split)
    num_classes, ignore_index = _resolve_label_spec_metadata(args.label_spec)
    if args.mask_dir is not None:
        reference_masks, reference_source, _, ignore_index = _load_reference_masks_from_dir(
            args.mask_dir.resolve(),
            label_spec=args.label_spec.resolve(),
            size=None,
            n_samples=args.n_samples,
        )
    else:
        reference_masks, reference_source = _load_reference_masks_from_dataset(
            config,
            split=split_key,
            n_samples=args.n_samples,
        )

    reference_stats = _summarize_distribution(
        reference_masks,
        num_classes=num_classes,
        ignore_index=ignore_index,
        small_region_threshold_ratio=args.small_region_threshold_ratio,
        thin_structure_class_ids=args.thin_structure_class_ids,
    )

    results = []
    for nfe in args.nfe_values:
        nfe_dir = generated_root / f"nfe{int(nfe)}"
        if not nfe_dir.exists():
            continue
        generated_masks, stems = _load_generated_masks(nfe_dir, n_samples=args.n_samples)
        del stems
        generated_stats = _summarize_distribution(
            generated_masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
            small_region_threshold_ratio=args.small_region_threshold_ratio,
            thin_structure_class_ids=args.thin_structure_class_ids,
        )
        comparison = _compare_distribution(
            generated_stats,
            reference_stats,
            num_classes=num_classes,
            thin_structure_class_ids=args.thin_structure_class_ids,
        )
        nearest_real = _nearest_real_mious(
            generated_masks,
            reference_masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        pairwise_fake = _pairwise_fake_mious(
            generated_masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        code_summary = _summarize_codes(_load_generated_codes(nfe_dir, args.n_samples), codebook_size=codebook_size)
        results.append(
            {
                "nfe": int(nfe),
                "outdir": str(nfe_dir.resolve()),
                "nearest_real_miou_mean": _mean_or_none(nearest_real),
                "nearest_real_miou_std": _std_or_none(nearest_real),
                "pairwise_fake_miou_mean": _mean_or_none(pairwise_fake),
                "pairwise_fake_miou_std": _std_or_none(pairwise_fake),
                "global_class_pixel_ratio_l1": float(comparison["global_class_pixel_ratio_l1"]),
                "class_area_histogram_l1_mean": float(comparison["class_area_histogram_l1_mean"]),
                "class_area_histogram_l1_max": float(comparison["class_area_histogram_l1_max"]),
                "adjacency_matrix_l1_mean": float(comparison["adjacency_matrix_l1_mean"]),
                "adjacency_matrix_jsd": float(comparison["adjacency_matrix_jsd"]),
                "component_count_l1_mean": float(comparison["component_count_l1_mean"]),
                "component_area_ratio_l1_mean": float(comparison["component_area_ratio_l1_mean"]),
                "largest_component_class_share_l1_mean": float(comparison["largest_component_class_share_l1_mean"]),
                "hole_count_l1_mean": float(comparison["hole_count_l1_mean"]),
                "hole_area_ratio_l1_mean": float(comparison["hole_area_ratio_l1_mean"]),
                "boundary_length_ratio_gap": float(comparison["boundary_length_ratio_gap"]),
                "small_region_frequency_l1_mean": float(comparison["small_region_frequency_l1_mean"]),
                "thin_structure_skeleton_length_gap_mean": comparison["thin_structure_skeleton_length_gap_mean"],
                "thin_structure_endpoint_count_gap_mean": comparison["thin_structure_endpoint_count_gap_mean"],
                "thin_structure_fragment_count_gap_mean": comparison["thin_structure_fragment_count_gap_mean"],
                "unique_class_count_gap": float(comparison["unique_class_count_gap"]),
                "active_code_count": None if code_summary is None else int(code_summary["active_code_count"]),
                "active_code_fraction": None if code_summary is None else float(code_summary["active_code_fraction"]),
                "code_perplexity": None if code_summary is None else float(code_summary["code_perplexity"]),
                "unique_code_count_mean": None if code_summary is None else float(code_summary["unique_code_count_mean"]),
                "unique_code_count_std": None if code_summary is None else float(code_summary["unique_code_count_std"]),
                "generated_stats": _to_json_ready(generated_stats),
            }
        )

    summary = {
        "task": "p(token_codes) -> frozen tokenizer decode -> semantic_mask",
        "config": str(args.config.resolve()),
        "checkpoint": None if ckpt_path is None else str(ckpt_path.resolve()),
        "tokenizer_config": str(resolved_tokenizer_config_path),
        "tokenizer_checkpoint": str(resolved_tokenizer_ckpt_path),
        "generated_root": str(generated_root),
        "monitor": monitor,
        "reference_source": str(reference_source),
        "split": str(split_key),
        "label_spec": str(args.label_spec.resolve()),
        "small_region_threshold_ratio": float(args.small_region_threshold_ratio),
        "thin_structure_class_ids": sorted(
            {int(class_id) for class_id in (args.thin_structure_class_ids or []) if 0 <= int(class_id) < int(num_classes)}
        ),
        "codebook_size": int(codebook_size),
        "route_metadata": route_metadata,
        "reference_stats": _to_json_ready(reference_stats),
        "results": results,
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    summary_md_path = outdir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nfe",
                "outdir",
                "nearest_real_miou_mean",
                "nearest_real_miou_std",
                "pairwise_fake_miou_mean",
                "pairwise_fake_miou_std",
                "global_class_pixel_ratio_l1",
                "class_area_histogram_l1_mean",
                "class_area_histogram_l1_max",
                "adjacency_matrix_l1_mean",
                "adjacency_matrix_jsd",
                "component_count_l1_mean",
                "component_area_ratio_l1_mean",
                "largest_component_class_share_l1_mean",
                "hole_count_l1_mean",
                "hole_area_ratio_l1_mean",
                "boundary_length_ratio_gap",
                "small_region_frequency_l1_mean",
                "thin_structure_skeleton_length_gap_mean",
                "thin_structure_endpoint_count_gap_mean",
                "thin_structure_fragment_count_gap_mean",
                "unique_class_count_gap",
                "active_code_count",
                "active_code_fraction",
                "code_perplexity",
                "unique_code_count_mean",
                "unique_code_count_std",
            ],
        )
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != "generated_stats"} for row in results])
    _write_markdown_report(summary_md_path, summary)

    print(f"Saved token-mask evaluation JSON to {summary_json_path}")
    print(f"Saved token-mask evaluation CSV to {summary_csv_path}")
    print(f"Saved token-mask evaluation markdown to {summary_md_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ldm.util import instantiate_from_config
from scripts.sample_latent_flow import load_checkpoint_state, load_config, load_model
from scripts.sample_mask_prior import _prepare_outdir
from scripts.sample_token_mask_prior import (
    DEFAULT_CONFIG,
    DEFAULT_TOKENIZER_CONFIG,
    apply_tokenizer_overrides,
    resolve_configured_tokenizer_artifacts,
    validate_token_mask_prior_checkpoint_contract,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run cheap diagnostics for the token-code autoregressive prior on an existing checkpoint. "
            "This can inspect teacher-forced predictions and optionally run short prefix-conditioned rollout probes "
            "without paying for a full unconditional 4096-token sample."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, default=DEFAULT_TOKENIZER_CONFIG)
    parser.add_argument("--tokenizer-ckpt", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument(
        "--prefix-rollout-steps",
        dest="prefix_rollout_steps",
        type=int,
        action="append",
        default=[],
        help="Roll out only the last N tokens while conditioning on the true prefix. Repeat as needed.",
    )
    parser.add_argument("--expected-monitor", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Extra OmegaConf dotlist override. Repeat as needed, without a leading --.",
    )
    return parser.parse_args()


def _resolve_split_key(config, split_name):
    normalized = str(split_name).lower()
    if normalized == "val":
        normalized = "validation"
    if OmegaConf.select(config, f"data.params.{normalized}", default=None) is None:
        raise KeyError(f"Config does not define data.params.{normalized}")
    return normalized


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _collect_scalar_metrics(outputs, *, group_names=None):
    metrics = {}
    if group_names is None:
        group_names = (
            "loss_dict",
            "code_target_stats",
            "autoregressive_metrics",
            "teacher_forced_prediction_stats",
        )
    for group_name in group_names:
        group = outputs.get(group_name, {})
        for name, value in group.items():
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    continue
                metrics[name] = float(value.detach().cpu().item())
            elif isinstance(value, (int, float)):
                metrics[name] = float(value)
    return metrics


def _aggregate_diagnostics(model, dataloader, *, device, limit_batches=None, prefix_rollout_steps=None):
    metric_sums = {}
    total_examples = 0
    total_batches = 0
    rollout_steps = sorted({int(step) for step in (prefix_rollout_steps or [])})

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if limit_batches is not None and batch_idx >= int(limit_batches):
                break
            batch = _move_to_device(batch, device)
            outputs = model(batch)
            batch_size = int(model._get_log_batch_size(batch))
            metrics = _collect_scalar_metrics(outputs)
            for step in rollout_steps:
                rollout_metrics = model.prefix_conditioned_rollout_metrics(
                    code_grid=outputs["code_grid"],
                    target_mask_index=outputs["mask_index"],
                    rollout_steps=step,
                )
                metrics.update(
                    _collect_scalar_metrics(
                        {"prefix_rollout_metrics": rollout_metrics},
                        group_names=("prefix_rollout_metrics",),
                    )
                )
            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + (float(value) * float(batch_size))
            total_examples += batch_size
            total_batches += 1

    if total_batches <= 0:
        raise ValueError("No batches were processed for diagnostics.")

    denom = float(max(total_examples, 1))
    return {
        "num_batches": int(total_batches),
        "num_examples": int(total_examples),
        "metrics": {
            name: (value / denom)
            for name, value in sorted(metric_sums.items())
        },
    }


def _build_datamodule(config):
    datamodule = instantiate_from_config(config.data)
    prepare_data = getattr(datamodule, "prepare_data", None)
    if callable(prepare_data):
        prepare_data()
    setup = getattr(datamodule, "setup", None)
    if callable(setup):
        setup()
    return datamodule


def _resolve_dataloader(datamodule, split_key):
    if split_key == "train":
        return datamodule.train_dataloader()
    if split_key == "validation":
        return datamodule.val_dataloader()
    if split_key == "test":
        return datamodule.test_dataloader()
    if split_key == "predict":
        return datamodule.predict_dataloader()
    raise KeyError(f"Unsupported split '{split_key}'")


def _write_markdown_report(path, summary):
    lines = [
        "# Token-Code Prior Diagnostics",
        "",
        f"- config: `{summary['config']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- tokenizer config: `{summary['tokenizer_config']}`",
        f"- tokenizer checkpoint: `{summary['tokenizer_checkpoint']}`",
        f"- split: `{summary['split']}`",
        f"- monitor: `{summary['monitor']}`",
        f"- batches: `{summary['num_batches']}`",
        f"- examples: `{summary['num_examples']}`",
        f"- context length: `{summary['context_length']}`",
        f"- sequence length: `{summary['sequence_length']}`",
        f"- full-context teacher forcing: `{summary['context_length'] >= summary['sequence_length']}`",
        f"- prefix rollout steps: `{summary['prefix_rollout_steps']}`",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for name, value in summary["metrics"].items():
        lines.append(f"| `{name}` | `{float(value):.6f}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Token-code prior checkpoint not found: {args.ckpt}")
    if not args.tokenizer_ckpt.exists():
        raise FileNotFoundError(f"Frozen tokenizer checkpoint not found: {args.tokenizer_ckpt}")

    config = load_config(args.config, overrides=args.overrides)
    apply_tokenizer_overrides(
        config,
        tokenizer_config=args.tokenizer_config,
        tokenizer_ckpt=args.tokenizer_ckpt,
    )
    if args.batch_size is not None:
        OmegaConf.update(config, "data.params.batch_size", int(args.batch_size), merge=False)
    if args.num_workers is not None:
        OmegaConf.update(config, "data.params.num_workers", int(args.num_workers), merge=False)

    monitor = str(OmegaConf.select(config, "model.params.monitor", default=""))
    if args.expected_monitor is not None and monitor != str(args.expected_monitor):
        raise ValueError(
            f"Diagnostics monitor mismatch: expected '{args.expected_monitor}', got '{monitor}'."
        )

    resolved_tokenizer_config_path, resolved_tokenizer_ckpt_path = resolve_configured_tokenizer_artifacts(
        config,
        route_name="Token-code prior diagnostics",
    )
    checkpoint_state = load_checkpoint_state(args.ckpt.resolve())
    validate_token_mask_prior_checkpoint_contract(
        config,
        args.ckpt.resolve(),
        config_path=args.config,
        checkpoint_state=checkpoint_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.ckpt.resolve(), device=device, checkpoint_state=checkpoint_state)
    model.eval()

    datamodule = _build_datamodule(config)
    split_key = _resolve_split_key(config, args.split)
    dataloader = _resolve_dataloader(datamodule, split_key)
    summary = _aggregate_diagnostics(
        model,
        dataloader,
        device=device,
        limit_batches=args.limit_batches,
        prefix_rollout_steps=args.prefix_rollout_steps,
    )
    summary.update(
        {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.ckpt.resolve()),
            "tokenizer_config": str(resolved_tokenizer_config_path),
            "tokenizer_checkpoint": str(resolved_tokenizer_ckpt_path),
            "split": split_key,
            "monitor": monitor,
            "context_length": int(model.context_length),
            "sequence_length": int(model.code_sequence_length),
            "prefix_rollout_steps": sorted({int(step) for step in args.prefix_rollout_steps}),
            "config_overrides": list(args.overrides),
        }
    )

    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)
    summary_json_path = outdir / "summary.json"
    summary_md_path = outdir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_report(summary_md_path, summary)

    print(f"Saved token-code prior diagnostics to {outdir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary markdown: {summary_md_path}")


if __name__ == "__main__":
    main()

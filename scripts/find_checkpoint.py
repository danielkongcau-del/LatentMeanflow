import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_ROOT = REPO_ROOT / "logs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resolve an explicit checkpoint path from a run tag or a concrete run directory."
    )
    parser.add_argument("--config", type=Path, default=None, help="Config path used to derive the run tag from config.stem.")
    parser.add_argument("--run-tag", type=str, default=None, help="Explicit run tag substring to match in the checkpoint path.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory, checkpoints directory, or checkpoint file. Use this for reproducible benchmark selection.",
    )
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--filename", type=str, default="last.ckpt")
    parser.add_argument(
        "--selection",
        choices=["last", "best"],
        default="last",
        help="Checkpoint selection mode. 'best' reads the Lightning checkpoint callback state instead of relying on the latest file.",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        help="Optional exact monitor name expected in the checkpoint callback state, for example 'val/base_error_mean'.",
    )
    return parser.parse_args()


def resolve_run_tag(args):
    if args.run_dir is not None:
        return None
    if args.config is not None:
        return args.config.stem.lower()
    if args.run_tag is not None:
        return args.run_tag.strip().lower()
    raise ValueError("Pass --run-dir, or pass either --config or --run-tag.")


def ckpt_matches_run_tag(ckpt_path, run_tag):
    run_dir_name = ckpt_path.parent.parent.name.lower()
    return run_dir_name == run_tag or run_dir_name.endswith(f"_{run_tag}")


def resolve_run_dir(args):
    if args.run_dir is not None:
        candidate = args.run_dir.resolve()
        if candidate.is_file():
            return candidate.parent.parent
        if candidate.name.lower() == "checkpoints":
            return candidate.parent
        return candidate

    run_tag = resolve_run_tag(args)
    logs_root = args.logs_root
    if not logs_root.exists():
        raise FileNotFoundError(f"Logs root not found: {logs_root}")

    run_dirs = sorted(
        [path for path in logs_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        run_dir_name = run_dir.name.lower()
        if run_dir_name == run_tag or run_dir_name.endswith(f"_{run_tag}"):
            return run_dir
    raise FileNotFoundError(f"No run directory found under {logs_root} matching run tag '{run_tag}'.")


def _normalize_candidate_path(raw_path, run_dir):
    candidate = Path(str(raw_path))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    repo_relative = (REPO_ROOT / candidate).resolve()
    if repo_relative.exists():
        return repo_relative
    run_relative = (run_dir / candidate.name).resolve()
    if run_relative.exists():
        return run_relative
    checkpoints_relative = (run_dir / "checkpoints" / candidate.name).resolve()
    if checkpoints_relative.exists():
        return checkpoints_relative
    return repo_relative


def _extract_best_model_path(checkpoint_state, run_dir, monitor=None):
    callbacks = checkpoint_state.get("callbacks", {})
    if not isinstance(callbacks, dict):
        return None

    for callback_state in callbacks.values():
        if not isinstance(callback_state, dict):
            continue
        best_model_path = callback_state.get("best_model_path")
        callback_monitor = callback_state.get("monitor")
        if not best_model_path:
            continue
        if monitor is not None and callback_monitor != monitor:
            continue
        normalized = _normalize_candidate_path(best_model_path, run_dir=run_dir)
        if normalized.exists():
            return normalized
    return None


def find_last_checkpoint(run_dir, filename):
    checkpoint_path = run_dir / "checkpoints" / filename
    if checkpoint_path.exists():
        return checkpoint_path.resolve()
    raise FileNotFoundError(f"Checkpoint '{filename}' not found under {run_dir / 'checkpoints'}.")


def find_best_checkpoint(run_dir, monitor=None):
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    candidates = []
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        candidates.append(last_ckpt)
    candidates.extend(
        sorted(
            ckpt_dir.glob("*.ckpt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )

    seen = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        checkpoint_state = torch.load(candidate, map_location="cpu")
        best_path = _extract_best_model_path(checkpoint_state, run_dir=run_dir, monitor=monitor)
        if best_path is not None:
            return best_path.resolve()

    monitor_suffix = f" with monitor '{monitor}'" if monitor is not None else ""
    raise FileNotFoundError(
        f"Could not resolve a best checkpoint for run '{run_dir.name}'{monitor_suffix}. "
        "Pass an explicit --run-dir that contains a Lightning checkpoint with callback state."
    )


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if args.selection == "last":
        checkpoint_path = find_last_checkpoint(run_dir, filename=args.filename)
    else:
        checkpoint_path = find_best_checkpoint(run_dir, monitor=args.monitor)
    print(checkpoint_path.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise

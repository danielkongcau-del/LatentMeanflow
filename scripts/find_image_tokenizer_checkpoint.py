import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent

for path in (REPO_ROOT, SCRIPT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from find_checkpoint import find_best_checkpoint, find_last_checkpoint, resolve_run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resolve an image-tokenizer checkpoint path using an explicit, "
            "monitor-aware selection rule."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory, checkpoints directory, or checkpoint file.",
    )
    parser.add_argument("--selection", choices=["best", "last"], default="best")
    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        help="Exact monitor name. Defaults to config.model.params.monitor or val/total_loss.",
    )
    parser.add_argument("--filename", type=str, default="last.ckpt")
    return parser.parse_args()


def default_monitor(config_path):
    config = OmegaConf.load(config_path)
    monitor = OmegaConf.select(config, "model.params.monitor", default=None)
    return "val/total_loss" if monitor is None else str(monitor)


def main():
    args = parse_args()
    run_dir = resolve_run_dir(
        type(
            "Args",
            (),
            {
                "run_dir": args.run_dir,
                "config": None,
                "run_tag": None,
                "logs_root": None,
            },
        )()
    )

    if args.selection == "last":
        checkpoint_path = find_last_checkpoint(run_dir, filename=args.filename)
    else:
        checkpoint_path = find_best_checkpoint(run_dir, monitor=args.monitor or default_monitor(args.config))
    print(Path(checkpoint_path).resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise

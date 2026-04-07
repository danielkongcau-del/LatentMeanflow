import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_ROOT = REPO_ROOT / "logs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print the newest checkpoint path whose run directory matches a config stem or tag."
    )
    parser.add_argument("--config", type=Path, default=None, help="Config path used to derive the run tag from config.stem.")
    parser.add_argument("--run-tag", type=str, default=None, help="Explicit run tag substring to match in the checkpoint path.")
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--filename", type=str, default="last.ckpt")
    return parser.parse_args()


def resolve_run_tag(args):
    if args.config is not None:
        return args.config.stem.lower()
    if args.run_tag is not None:
        return args.run_tag.strip().lower()
    raise ValueError("Pass either --config or --run-tag.")


def ckpt_matches_run_tag(ckpt_path, run_tag):
    run_dir_name = ckpt_path.parent.parent.name.lower()
    return run_dir_name == run_tag or run_dir_name.endswith(f"_{run_tag}")


def main():
    args = parse_args()
    run_tag = resolve_run_tag(args)
    logs_root = args.logs_root
    if not logs_root.exists():
        raise FileNotFoundError(f"Logs root not found: {logs_root}")

    candidates = sorted(
        logs_root.rglob(args.filename),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if ckpt_matches_run_tag(candidate, run_tag):
            print(candidate.resolve())
            return

    raise FileNotFoundError(
        f"No checkpoint named '{args.filename}' found under {logs_root} matching run tag '{run_tag}'."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise

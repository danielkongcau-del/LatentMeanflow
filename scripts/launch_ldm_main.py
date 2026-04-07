import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_MAIN = REPO_ROOT / "third_party" / "latent-diffusion" / "main.py"


def configure_runtime_warnings():
    suppress_flag = os.environ.get("LMF_SUPPRESS_DDP_STREAM_WARNING", "1").strip().lower()
    if suppress_flag in {"0", "false", "no"}:
        return

    try:
        import torch
    except Exception:
        return

    graph_module = getattr(torch.autograd, "graph", None)
    if graph_module is None:
        return
    setter = getattr(graph_module, "set_warn_on_accumulate_grad_stream_mismatch", None)
    if setter is None:
        return
    setter(False)


def main():
    if not LDM_MAIN.exists():
        raise FileNotFoundError(f"latent-diffusion entrypoint not found: {LDM_MAIN}")

    configure_runtime_warnings()
    sys.argv[0] = str(LDM_MAIN)
    runpy.run_path(str(LDM_MAIN), run_name="__main__")


if __name__ == "__main__":
    main()

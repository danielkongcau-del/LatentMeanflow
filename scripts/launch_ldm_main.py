import os
import runpy
import signal
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


def configure_safe_signal_handlers():
    disable_flag = os.environ.get("LMF_DISABLE_LDM_SIGNAL_HANDLERS", "1").strip().lower()
    if disable_flag in {"0", "false", "no"}:
        return

    ignored_signals = []
    for name in ("SIGUSR1", "SIGUSR2"):
        signum = getattr(signal, name, None)
        if signum is not None:
            ignored_signals.append(signum)
    if not ignored_signals:
        return

    def _ignore_usr_signal(signum, frame):
        del frame
        try:
            signal_name = signal.Signals(signum).name
        except Exception:
            signal_name = str(signum)
        print(
            f"[launch_ldm_main] Ignoring {signal_name} to avoid vendored checkpoint/debug "
            "signal handlers deadlocking DDP startup.",
            flush=True,
        )

    original_signal = signal.signal

    def _patched_signal(signum, handler):
        if signum in ignored_signals:
            return original_signal(signum, _ignore_usr_signal)
        return original_signal(signum, handler)

    for signum in ignored_signals:
        original_signal(signum, _ignore_usr_signal)
    signal.signal = _patched_signal


def main():
    if not LDM_MAIN.exists():
        raise FileNotFoundError(f"latent-diffusion entrypoint not found: {LDM_MAIN}")

    configure_runtime_warnings()
    configure_safe_signal_handlers()
    sys.argv[0] = str(LDM_MAIN)
    runpy.run_path(str(LDM_MAIN), run_name="__main__")


if __name__ == "__main__":
    main()

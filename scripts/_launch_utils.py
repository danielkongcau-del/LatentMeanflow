import os
import signal
import subprocess


_GRACEFUL_TIMEOUT_SECONDS = 10
_FORCE_TIMEOUT_SECONDS = 5


def normalize_gpus_arg(gpus):
    if gpus is None:
        return None
    text = str(gpus).strip()
    if not text:
        return None
    if "," in text:
        return text
    if not text.isdigit():
        return text
    count = int(text)
    if count == 0:
        # The project wrappers historically documented "--gpus 0" as
        # "use device 0". Under Lightning 2 this must be passed as a device
        # list rather than devices=0, which is invalid.
        return "0,"
    if count == 1:
        return text
    return ",".join(str(i) for i in range(count))


def _terminate_process(process, signum):
    if process.poll() is not None:
        return
    try:
        process.send_signal(signum)
    except ProcessLookupError:
        return


def _terminate_process_group(process, signum):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return


def _wait_with_escalation(process, initial_signal):
    if process.poll() is not None:
        return

    if os.name == "posix":
        _terminate_process_group(process, initial_signal)
    else:
        _terminate_process(process, initial_signal)

    try:
        process.wait(timeout=_GRACEFUL_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix":
        _terminate_process_group(process, signal.SIGTERM)
    else:
        process.terminate()

    try:
        process.wait(timeout=_GRACEFUL_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    if hasattr(signal, "SIGKILL"):
        if os.name == "posix":
            _terminate_process_group(process, signal.SIGKILL)
        else:
            process.kill()
        try:
            process.wait(timeout=_FORCE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return


def run_managed_subprocess(cmd, cwd, env):
    cwd = os.fspath(cwd)

    if os.name != "posix":
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
        return

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )

    previous_handlers = {}

    def _signal_handler(signum, _frame):
        _wait_with_escalation(process, signal.SIGINT if signum == signal.SIGINT else signal.SIGTERM)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[handled_signal] = signal.getsignal(handled_signal)
        signal.signal(handled_signal, _signal_handler)

    try:
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
    except BaseException:
        _wait_with_escalation(process, signal.SIGTERM)
        raise
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)
        _wait_with_escalation(process, signal.SIGTERM)

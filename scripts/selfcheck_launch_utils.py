from unittest import mock

import _launch_utils


class _FakeProcess:
    def __init__(self):
        self.pid = 4242
        self._return_code = 0
        self.wait_calls = 0

    def poll(self):
        return self._return_code if self.wait_calls > 0 else None

    def wait(self, timeout=None):
        self.wait_calls += 1
        return self._return_code

    def send_signal(self, _signum):
        self._return_code = 0

    def terminate(self):
        self._return_code = 0

    def kill(self):
        self._return_code = 0


def main():
    fake_process = _FakeProcess()
    popen_kwargs = {}

    def fake_popen(cmd, cwd, env, start_new_session):
        popen_kwargs["cmd"] = cmd
        popen_kwargs["cwd"] = cwd
        popen_kwargs["env"] = env
        popen_kwargs["start_new_session"] = start_new_session
        return fake_process

    with mock.patch.object(_launch_utils.os, "name", "posix"), \
         mock.patch.object(_launch_utils.subprocess, "Popen", side_effect=fake_popen), \
         mock.patch.object(_launch_utils.os, "killpg", create=True) as killpg_mock, \
         mock.patch.object(_launch_utils.signal, "getsignal", side_effect=lambda signum: signum), \
         mock.patch.object(_launch_utils.signal, "signal") as signal_mock:
        _launch_utils.run_managed_subprocess(
            cmd=["python", "dummy.py"],
            cwd=".",
            env={"TEST_ENV": "1"},
        )

    assert popen_kwargs["start_new_session"] is True
    assert popen_kwargs["cmd"] == ["python", "dummy.py"]
    assert signal_mock.call_count >= 2
    assert killpg_mock.call_count == 0
    print("launch utils use start_new_session=True on posix")


if __name__ == "__main__":
    main()

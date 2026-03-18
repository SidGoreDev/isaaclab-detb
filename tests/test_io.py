import subprocess

from detb.io import capture_system_details


class _Result:
    def __init__(self, stdout: str):
        self.stdout = stdout


def test_capture_system_details_uses_nvidia_smi(monkeypatch):
    def fake_run(cmd, check, capture_output, text, cwd=None):
        if cmd[0] == "nvidia-smi":
            return _Result("NVIDIA GeForce RTX 5080 Laptop GPU, 581.29\n")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    operating_system, driver_version, gpu_model = capture_system_details(gpu_index=0)

    assert operating_system
    assert driver_version == "581.29"
    assert gpu_model == "NVIDIA GeForce RTX 5080 Laptop GPU"

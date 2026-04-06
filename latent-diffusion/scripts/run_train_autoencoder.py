import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAMING_DIR = ROOT / "taming-transformers"
if TAMING_DIR.exists():
    sys.path.insert(0, str(TAMING_DIR))
CONFIG = "configs/mask_image/autoencoder_kl_4ch_256.yaml"

# Set to None or "" to run on CPU.
GPUS = "0,1"

# Training epochs.
MAX_EPOCHS = 1000

# Batch size override (set to None to use config default).
BATCH_SIZE = 20

# Set to a logdir or checkpoint path to resume; leave empty to train from scratch.
RESUME_PATH = "/root/autodl-tmp/latent-diffusion/logs/autoencoder/checkpoints/last.ckpt"


def build_command():
    cmd = [sys.executable, "main.py", "-t", "--base", CONFIG]
    cmd.append("--lightning.callbacks.image_logger.params.disabled=False")
    cmd.append("--lightning.callbacks.image_logger.params.batch_frequency=430")

    if RESUME_PATH:
        cmd.extend(["--resume", RESUME_PATH])
    if GPUS:
        cmd.extend(["--gpus", GPUS])
    if MAX_EPOCHS is not None:
        cmd.extend(["--max_epochs", str(MAX_EPOCHS)])
    if BATCH_SIZE is not None:
        cmd.append(f"--data.params.batch_size={BATCH_SIZE}")
    return cmd


def main():
    cmd = build_command()
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    if TAMING_DIR.exists():
        env["PYTHONPATH"] = str(TAMING_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


if __name__ == "__main__":
    main()

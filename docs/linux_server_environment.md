# Linux Server Environment Setup

This document is the recommended environment setup for training LatentMeanflow on a remote Linux GPU server.

It is intentionally stricter than the lightweight local instructions in the main README because the goal here is:

- keep a single clean training environment
- avoid `torch` / CUDA wheel mismatches
- avoid `opencv-python` vs `opencv-python-headless` conflicts on headless servers
- avoid editable installs re-resolving dependencies and downgrading packages

## Target Stack

Validated project stack:

- Python `3.10`
- PyTorch `2.11.0+cu128`
- torchvision `0.26.0+cu128`
- torchaudio `2.11.0+cu128`
- NumPy `1.26.4`
- PyTorch Lightning `2.5.6`
- TensorBoard `2.19.0`

Project-specific lock file for Linux servers:

- [requirements/lmf-linux-server.txt](../requirements/lmf-linux-server.txt)

This lock file intentionally excludes:

- `torch`, `torchvision`, `torchaudio`
- editable local packages under `third_party/`
- the `clip` Git install

Those parts must be installed separately and in order.

## Why The Install Order Matters

Use this order and do not skip around:

1. Create a clean Python 3.10 environment.
2. Install the exact PyTorch CUDA wheel triplet first.
3. Install the locked Python dependencies for this repo.
4. Install vendored editable packages with `--no-deps`.
5. Install `CLIP` last with `--no-deps`.
6. Run import and CUDA verification before training.

The important failure modes this avoids are:

- installing editable packages too early and letting `pip` re-resolve `torch`/`numpy`
- mixing `opencv-python` with `opencv-python-headless`
- installing `clip` before its base environment is already stable
- mixing different CUDA wheel variants in one environment
- missing TensorBoard, which causes Lightning logger initialization to fail before training starts

## Step 0. System Packages

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y \
  git \
  build-essential \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1
```

Notes:

- `ffmpeg` is needed by `imageio-ffmpeg`.
- `libgl1` and `libglib2.0-0` avoid common OpenCV import failures on servers.
- This document assumes NVIDIA driver and CUDA runtime support are already present on the machine.

## Step 1. Create A Clean Conda Environment

```bash
conda create -n lmf python=3.10 -y
conda activate lmf
python -m pip install --upgrade pip setuptools wheel
```

Do not install any project packages before this step is complete.

## Step 2. Install The Exact PyTorch CUDA Triplet First

If your server is meant to match the current project environment, install:

```bash
python -m pip install \
  torch==2.11.0+cu128 \
  torchvision==0.26.0+cu128 \
  torchaudio==2.11.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

Important:

- Install all three in one command.
- Do not install `pytorch-cuda` from conda and then overwrite `torch` with pip in the same environment.
- Do not mix `cu121`, `cu124`, `cu128`, or CPU-only wheels in one environment.

If your server driver cannot run the `cu128` wheels, stop and align the PyTorch wheel variant first. Do not continue with a mixed environment.

## Step 3. Install The Locked Python Dependencies

Use the Linux server lock file:

```bash
python -m pip install --upgrade-strategy only-if-needed -r requirements/lmf-linux-server.txt
```

Why this file exists:

- it freezes the known-good package versions used by the current project
- it uses `opencv-python-headless` instead of `opencv-python`
- it includes `tensorboard`, which is required by the default Lightning TensorBoard logger
- it avoids the Windows-local drift that can happen with a looser requirements file

## Step 4. Install Vendored Editable Packages Without Dependency Resolution

Install vendored packages one by one and always use `--no-deps`:

```bash
python -m pip install --no-deps -e third_party/latent-diffusion/taming-transformers
python -m pip install --no-deps -e third_party/latent-diffusion
python -m pip install --no-deps -e third_party/flow_matching
```

Reason:

- the vendored `setup.py` files declare generic dependencies such as `torch` and `numpy`
- those are already installed and pinned
- `--no-deps` prevents `pip` from trying to upgrade, downgrade, or re-resolve your environment

## Step 5. Install CLIP Last

Install `CLIP` only after the environment is already stable:

```bash
python -m pip install --no-deps git+https://github.com/openai/CLIP.git
```

Its runtime dependencies are already covered by the locked base environment:

- `ftfy`
- `regex`
- `packaging`
- `torch`
- `torchvision`
- `tqdm`

## Step 6. Optional: Silence Albumentations Update Checks

On servers, this warning is just noise:

```bash
export NO_ALBUMENTATIONS_UPDATE=1
```

If you want it on every login, append it to your shell profile.

## Step 7. Verify The Environment Before Training

Run:

```bash
python - <<'PY'
import torch
import torchvision
import torchaudio
import numpy
import pytorch_lightning
import omegaconf
import transformers
import kornia
import torchmetrics
import imageio
import cv2
import clip
import ldm
import taming
import flow_matching

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("numpy", numpy.__version__)
print("pytorch_lightning", pytorch_lightning.__version__)
print("omegaconf", omegaconf.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_name", torch.cuda.get_device_name(0))
PY
```

Minimum expected result:

- all imports succeed
- `torch.cuda.is_available()` is `True`
- the reported `torch` / `torchvision` / `torchaudio` versions match the target stack
- TensorBoard-backed logging initializes successfully when training starts

## Step 8. Optional: Place VGG Weights

The current semantic tokenizer config keeps LPIPS disabled by default, so this is not required for the first semantic training runs.

If you later enable LPIPS or use legacy pipelines that need VGG weights:

```bash
python scripts/place_vgg16_weights.py --src /path/to/vgg16-397923af.pth
```

## Training Entry Points After Setup

Once the environment is ready, the first recommended training order is:

1. inspect semantic masks
2. tokenizer tiny overfit
3. tokenizer baseline
4. latent FM tiny pilot
5. latent MeanFlow tiny pilot
6. latent MeanFlow baseline

See [docs/first_semantic_pair_experiment_plan.md](./first_semantic_pair_experiment_plan.md) for the exact commands.

## Linux Launch Scripts

This repo now includes two Linux-friendly shell launchers:

- [scripts/train_tokenizer.sh](../scripts/train_tokenizer.sh)
- [scripts/train_meanflow.sh](../scripts/train_meanflow.sh)
- [scripts/sample_meanflow.sh](../scripts/sample_meanflow.sh)

They are thin wrappers around the existing Python launchers:

- `train_tokenizer.sh` wraps `scripts/train_semantic_autoencoder.py`
- `train_meanflow.sh` wraps `scripts/train_latent_meanflow.py`
- `sample_meanflow.sh` wraps `scripts/sample_latent_flow.py`

They are designed for remote servers:

- they resolve the repo root automatically
- they default `NO_ALBUMENTATIONS_UPDATE=1`
- they default latent-flow shell workflows to the objective-specific project baseline, with U-Net now backing the default MeanFlow and AlphaFlow routes
- they pass any extra CLI arguments straight through to the Python launcher

Examples:

```bash
./scripts/train_tokenizer.sh

CONFIG=configs/semantic_tokenizer_tiny_256.yaml MAX_EPOCHS=40 ./scripts/train_tokenizer.sh

./scripts/train_meanflow.sh

CONFIG=configs/latent_meanflow_semantic_256_tiny.yaml MAX_EPOCHS=3 ./scripts/train_meanflow.sh

OBJECTIVE=alphaflow CONFIG=configs/latent_alphaflow_semantic_256_unet.yaml ./scripts/train_meanflow.sh

RESUME=logs/2026-04-07T12-00-00_latent_meanflow_semantic_256_unet/checkpoints/last.ckpt ./scripts/train_meanflow.sh

./scripts/sample_meanflow.sh

NFE=8 OUTDIR=outputs/meanflow_nfe8 ./scripts/sample_meanflow.sh

CONFIG=configs/latent_meanflow_semantic_256_tiny.yaml OUTDIR=outputs/meanflow_tiny_samples NFE=2 ./scripts/sample_meanflow.sh

CONFIG=configs/latent_meanflow_semantic_256.yaml OUTDIR=outputs/meanflow_convnet_legacy NFE=4 ./scripts/sample_meanflow.sh
```

## Copy-Paste Install Block

If you want the whole sequence in one place:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

conda create -n lmf python=3.10 -y
conda activate lmf

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  torch==2.11.0+cu128 \
  torchvision==0.26.0+cu128 \
  torchaudio==2.11.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install --upgrade-strategy only-if-needed -r requirements/lmf-linux-server.txt

python -m pip install --no-deps -e third_party/latent-diffusion/taming-transformers
python -m pip install --no-deps -e third_party/latent-diffusion
python -m pip install --no-deps -e third_party/flow_matching
python -m pip install --no-deps git+https://github.com/openai/CLIP.git

export NO_ALBUMENTATIONS_UPDATE=1
```

## Troubleshooting

If `cv2` import fails with `libGL.so.1`:

- re-check Step 0 system packages

If `torch.cuda.is_available()` is `False`:

- check `nvidia-smi`
- check that the server driver supports the selected PyTorch CUDA wheel
- confirm you did not install CPU-only `torch`

If `pip` tries to change `torch` during editable installs:

- remove the environment and recreate it
- reinstall the vendored packages with `--no-deps`

If training fails with `Neither tensorboard nor tensorboardX is available`:

- re-run `python -m pip install --upgrade-strategy only-if-needed -r requirements/lmf-linux-server.txt`
- or install `python -m pip install tensorboard==2.19.0`

If you accidentally installed both `opencv-python` and `opencv-python-headless`:

- remove the environment and recreate it
- do not try to patch the OpenCV pair in-place on a training server

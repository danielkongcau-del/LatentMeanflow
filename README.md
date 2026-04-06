# LatentMeanflow

LatentMeanflow is a research workspace for class-conditional generation of paired RGB image and binary mask samples. The active implementation is built on top of a patched copy of CompVis `latent-diffusion`, while `flow_matching` is kept as a separate upstream reference for later experiments.

## Repository Layout

- `configs/`: project-owned training configs.
- `scripts/`: project-owned entry points for training, sampling, and utility setup.
- `latent_meanflow/`: project-specific Python code such as dataset loaders.
- `third_party/latent-diffusion/`: vendored upstream fork with local compatibility patches.
- `third_party/flow_matching/`: vendored upstream reference code, currently not on the main training path.
- `docs/third_party.md`: provenance and patch summary for vendored code.

## Data Layout

Training configs expect a class-separated dataset rooted at `data/`:

```text
data/
  borner/
    train/
      image/
      mask/
    val/
      image/
      mask/
  corn/
  tomato/
  worm/
```

Each `image/` file should have a matching file stem in `mask/`.

## Environment

The maintained environment target for this repository is:

- Python `3.10`
- PyTorch `2.11.0+cu128`
- torchvision `0.26.0+cu128`
- a single Conda environment such as `lmf`

After activating your environment, install PyTorch from the official CUDA 12.8 wheel index:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/lmf.txt
python -m pip install -e third_party/latent-diffusion/taming-transformers
python -m pip install -e third_party/latent-diffusion
python -m pip install -e third_party/flow_matching
python -m pip install git+https://github.com/openai/CLIP.git
```

If LPIPS asks for VGG weights, place them with:

```bash
python scripts/place_vgg16_weights.py --src /path/to/vgg16-397923af.pth
```

## Training

Train the 4-channel autoencoder:

```bash
python scripts/train_autoencoder.py --gpus 0
```

Train the latent diffusion model after the autoencoder checkpoint exists:

```bash
python scripts/train_ldm.py --gpus 0 --ae-ckpt logs/autoencoder/checkpoints/last.ckpt
```

Sample paired image and mask outputs:

```bash
python scripts/sample_mask_image_pairs.py --ckpt logs/ldm/checkpoints/last.ckpt --outdir outputs/worm
```

## Notes

- Training logs are written to `logs/autoencoder/` and `logs/ldm/`.
- The root-level configs and scripts are the project layer; avoid editing `third_party/` unless you are intentionally changing the vendored fork.
- `third_party/flow_matching/` is kept isolated so future flow-matching work does not pollute the current latent-diffusion pipeline.

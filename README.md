# LatentMeanflow

LatentMeanflow currently ships a legacy binary-mask / 4-channel latent diffusion baseline and a planned migration path toward a paired RGB image + multiclass semantic mask latent flow model. The active runtime remains a patched copy of CompVis `latent-diffusion`; the target research direction is `latent tokenizer -> latent flow matching -> MeanFlow / AlphaFlow`.

See [docs/latentmeanflow_migration_plan.md](docs/latentmeanflow_migration_plan.md) for the staged migration plan and [docs/semantic_label_spec.md](docs/semantic_label_spec.md) for the planned semantic label contract.

## Repository Layout

- `configs/`: project-owned training configs.
- `scripts/`: project-owned entry points for training, sampling, and utility setup.
- `latent_meanflow/`: project-specific Python code such as dataset loaders.
- `docs/`: project documentation, including migration planning and label contracts.
- `third_party/latent-diffusion/`: vendored upstream fork with local compatibility patches.
- `third_party/flow_matching/`: vendored upstream reference code. It is the future reference base for latent FM / MeanFlow / AlphaFlow, but it is not on the current training path.
- `docs/third_party.md`: provenance and patch summary for vendored code.

## Terminology

- `image-level class_label`: sample-level domain or category metadata. It is optional and applies to the whole image, not to individual pixels.
- `pixel-level semantic class`: the discrete label assigned to each mask pixel. This is the future primary paired-generation signal.
- `class_label != semantic_mask`: image-level metadata and pixel-level semantics are different contracts and should not be conflated.

## Data Layout

### Current legacy path

The checked-in legacy configs currently point to a paired dataset rooted at `data/remote`:

```text
data/
  remote/
    train/
      images/
      masks/
    val/
      images/
      masks/
    test/
      images/
      masks/
```

Each file in `images/` must have a same-stem partner in `masks/`.

The current project-layer loader treats each mask as a grayscale image, thresholds it to binary foreground/background, and packs `RGB + mask` into a single 4-channel tensor. That behavior is legacy and only describes the current baseline.

### Planned semantic path

The planned semantic route keeps `image_rgb` and `semantic_mask` as separate project-layer fields, uses an explicit label specification, and treats image-level `class_label` only as optional metadata. Multiclass semantic masks should not be reduced to binary foreground/background. See [docs/semantic_label_spec.md](docs/semantic_label_spec.md) for the planned contract.

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

### Legacy binary baseline

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
python scripts/sample_mask_image_pairs.py --ckpt logs/ldm/checkpoints/last.ckpt --outdir outputs/remote
```

These commands target the legacy binary / 4-channel baseline only. They do not implement the planned multiclass semantic latent flow path yet.

## Notes

- Training logs are written to `logs/autoencoder/` and `logs/ldm/`.
- The root-level configs, scripts, docs, and dataset code are the project layer; avoid editing `third_party/` unless you are intentionally changing the vendored fork.
- `third_party/flow_matching/` is kept isolated so future flow-matching work does not pollute the current latent-diffusion baseline.

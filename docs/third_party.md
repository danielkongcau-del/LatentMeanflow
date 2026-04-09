# Vendored Code Notes

## `third_party/latent-diffusion`

This directory is a vendored fork of CompVis `latent-diffusion`. It remains the runtime base for training and sampling, but project-owned configs, scripts, and dataset code have been moved out to the repository root.

Local patches still kept inside the vendored fork:

- `third_party/latent-diffusion/main.py`
  Lightning 2.x compatibility, fixed log directory names, extra logger fallbacks, pair-saving image logger support, and CUDA callback fixes.
- `third_party/latent-diffusion/ldm/models/autoencoder.py`
  Manual optimization for modern Lightning, 4-channel RGB+mask logging, and safer quantizer import fallback.
- `third_party/latent-diffusion/ldm/models/diffusion/ddpm.py`
  Lightning compatibility fixes plus 4-channel RGB+mask logging and class-label conditioning robustness.
- `third_party/latent-diffusion/ldm/modules/encoders/modules.py`
  Added `ClassLabeler` for class-conditional training from dataset labels.
- `third_party/latent-diffusion/taming-transformers/taming/modules/transformer/mingpt.py`
  Added a local fallback for `top_k_top_p_filtering` so newer `transformers` 4.x releases still work.
- `third_party/latent-diffusion/taming-transformers/taming/data/utils.py`
  Added a fallback for the removed `torch._six.string_classes`.
- `third_party/latent-diffusion/ldm/modules/image_degradation/utils_image.py`
  Replaced deprecated NumPy scalar aliases with modern built-in types.
- `third_party/latent-diffusion/taming-transformers/taming/data/coco.py`
  Replaced deprecated NumPy scalar aliases with modern built-in types.

The preserved `.git_local_backup/` directories contain the original nested Git metadata so upstream history and provenance are still recoverable locally.

`third_party/latent-diffusion/main.py` is the maintained training entrypoint for this repository. The older upstream script `third_party/latent-diffusion/taming-transformers/main.py` is still vendored, but is not part of the current LatentMeanflow training path and has not been fully ported to the modern Lightning CLI surface.

## `third_party/flow_matching`

This directory is a vendored copy of the upstream `flow_matching` repository. It is currently reference code only and is not on the active training path for the paired image-mask model.

Keeping it under `third_party/` avoids mixing exploratory upstream code with the project layer.

## `third_party/segmentation`

This directory is a vendored copy of a legacy remote-sensing / agricultural
segmentation codebase kept as a potential frozen segmentation-teacher source for
evaluation.

It is not wired into the main LatentMeanflow training path yet.

Local ignore rules live in:

- `third_party/segmentation/.gitignore`

Those rules keep the vendored code trackable while preventing routine teacher
training artifacts from polluting the main repository status:

- `Data/*` runtime contents, while keeping `Data/README.md`
- `logs/`, `outputs/`, `runs/`, `wandb/`, `checkpoints/`
- cached Python bytecode and editor metadata
- downloaded model weights such as `*.pth`, `*.pt`, `*.ckpt`

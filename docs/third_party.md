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
segmentation codebase now used as the in-domain teacher bakeoff harness for the
project-layer `p(image | semantic_mask)` layout-faithfulness evaluation route.

It is still not wired into the generator training path. Its current role is:

- train teacher candidates only on real remote-sensing image-mask data
- compare them on held-out `val` / `test`
- freeze the winning checkpoint
- export precomputed teacher masks for `--teacher-mask-root`

The project-layer runbook for this workflow lives in:

- `docs/segmentation_teacher_bakeoff.md`

Local ignore rules live in:

- `third_party/segmentation/.gitignore`

Those rules keep the vendored code trackable while preventing routine teacher
training artifacts from polluting the main repository status:

- `Data/*` runtime contents, while keeping `Data/README.md`
- `logs/`, `outputs/`, `runs/`, `wandb/`, `checkpoints/`
- cached Python bytecode and editor metadata
- downloaded model weights such as `*.pth`, `*.pt`, `*.ckpt`

## `third_party/SiT`

This directory is a vendored copy of the official SiT repository:

- upstream: `https://github.com/willisma/SiT`
- paper: `Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers`
- license: `MIT`

Its current role in this repository is reference-only:

- study DiT-style transformer backbones under an interpolant / transport objective
- borrow implementation ideas for future project-layer `mask prior` backbone experiments
- keep that reference code isolated from the active LatentMeanflow training path

It is not currently wired into any checked-in project-layer training, sampling, or
evaluation script.

Local ignore rules live in:

- `third_party/SiT/.gitignore`

Those rules keep the vendored source tree trackable while preventing routine
SiT-side runtime artifacts from polluting the main repository status:

- `wandb/`, `samples/`, `results/`
- downloaded weights under `pretrained_models/`
- local experiment state such as `logs/`, `outputs/`, `runs/`, `checkpoints/`
- cached Python bytecode and notebook checkpoint files

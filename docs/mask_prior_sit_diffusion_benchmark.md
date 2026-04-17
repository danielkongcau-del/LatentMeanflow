# SiT Diffusion Mask Prior Benchmark

This document defines the next apples-to-apples benchmark for the project-layer
upstream route:

- task: `p(semantic_mask)`
- task type: unconditional semantic layout generation
- representation: direct `K x H x W` semantic-mask field
- no mask tokenizer
- no image branch
- no change to the frozen `p(image | semantic_mask)` renderer

The purpose of this step is narrow and explicit:

- test whether a stronger SiT-style transformer backbone plus a standard
  diffusion baseline is a better fit for unconditional semantic-mask
  generation than the current U-Net + AlphaFlow baseline
- keep the evaluation protocol fixed so the comparison stays
  apples-to-apples

This step does **not** claim that flow is dead. It only asks whether the
current direct-mask U-Net flow baseline is being limited by backbone capacity
and objective choice. If the SiT-style diffusion baseline wins under the same
protocol, the next comparison should be the same SiT-style backbone under flow
or AlphaFlow.

## Compared Routes

Legacy baseline:

- backbone: project-layer `LatentIntervalUNet`
- objective: AlphaFlow
- config: `configs/latent_alphaflow_mask_prior_unet.yaml`
- train entrypoint: `scripts/train_mask_prior.py`
- sample entrypoint: `scripts/sample_mask_prior.py`

New benchmark baseline:

- backbone: project-layer `LatentIntervalSiT`
- objective: standard Gaussian diffusion training
- sampler: DDIM-style deterministic few-step sampler
- config: `configs/latent_diffusion_mask_prior_sit.yaml`
- train entrypoint: `scripts/train_mask_prior_diffusion.py`
- sample entrypoint: `scripts/sample_mask_prior_diffusion.py`

Both routes still model `p(semantic_mask)` only.

## Fixed Comparison Matrix

Keep the following fixed across the benchmark:

- same dataset root: `outputs/segmentation_teacher_data/remote_semantic_indexed`
- same label spec: `configs/label_specs/remote_semantic.yaml`
- same task: unconditional semantic-mask generation
- same decode contract: sampled `K`-channel field -> `argmax` -> discrete mask
- same frozen downstream renderer
- same frozen in-domain teacher
- same mask-only metrics
- same compose-to-image metrics
- same `NFE=8/4/2/1` sweep
- same seed

Do **not** change the renderer config, renderer checkpoint rule, or teacher
workflow while running this comparison.

## Why Diffusion First

Use the SiT-style diffusion baseline before SiT + AlphaFlow for three reasons:

1. The current blocker is quality, not one-step efficiency.
2. Standard diffusion is the more stable control when testing whether a
   stronger transformer backbone can better model global semantic layout,
   geometry, and class-region relations.
3. Holding the new backbone fixed while first switching to a conventional
   diffusion objective reduces confounding. If this baseline already improves
   the frozen-protocol metrics, the next flow comparison becomes much clearer.

## Tiny Pilot

Use the tiny config for shape checks and deliberate overfit:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/latent_diffusion_mask_prior_sit_tiny.yaml \
  --scale-lr true \
  --gpus 0
```

Success criteria for the tiny pilot:

- clearly memorize the dominant large-region layout
- main class boundaries remain visible
- thin or narrow structures do not collapse completely

## Base Training

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/latent_diffusion_mask_prior_sit.yaml \
  --scale-lr true \
  --gpus 0
```

Sample a fixed few-step sweep:

```bash
python scripts/sample_mask_prior_diffusion.py \
  --config configs/latent_diffusion_mask_prior_sit.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_samples/sit_diffusion \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

The sampling and evaluation scripts in this benchmark require explicit
checkpoint paths. They do not fall back to `last.ckpt`.

## Shared Evaluation Protocol

The benchmark intentionally reuses the checked-in evaluation scripts without
changing their metric definitions.

Mask-only evaluation:

```bash
python scripts/eval_mask_prior.py \
  --config configs/latent_diffusion_mask_prior_sit.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_eval/sit_diffusion \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Compose-to-image evaluation:

```bash
python scripts/eval_mask_prior_composed_renderer.py \
  --mask-config configs/latent_diffusion_mask_prior_sit.yaml \
  --mask-ckpt <best-mask-prior-ckpt> \
  --renderer-config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml \
  --renderer-ckpt <best-renderer-ckpt> \
  --renderer-tokenizer-config configs/autoencoder_image_lpips_adv_256.yaml \
  --renderer-tokenizer-ckpt <best-image-tokenizer-ckpt> \
  --teacher-run-dir logs/segmentation_teacher/<winner_run> \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --outdir outputs/mask_prior_compose_eval/sit_diffusion \
  --n-samples 32 \
  --mask-nfe-values 8 4 2 1 \
  --renderer-nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

## Minimum Readout Standard

For the new baseline to count as a useful upgrade, the expectation is:

- `NFE=8/4` mask samples are visibly more coherent than the U-Net + AlphaFlow
  baseline
- mask-only distribution statistics do not collapse
- compose-to-image teacher metrics are clearly better than the legacy U-Net
  baseline under the same frozen renderer and teacher

If this SiT-style diffusion baseline still fails, then the next variables to
consider are:

- a semantic-mask tokenizer
- stronger flow variants such as OT-CFM
- a same-backbone SiT + AlphaFlow comparison

Do not introduce those extra variables in this benchmark step.

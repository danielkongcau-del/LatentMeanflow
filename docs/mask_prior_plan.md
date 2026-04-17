# Mask Prior Plan

## Route Definition

This document defines a new project-layer baseline for unconditional semantic-mask generation:

- route: `p(semantic_mask)`
- task type: unconditional multi-class semantic layout generation
- representation: direct `mask_onehot` field during training, `argmax` back to `mask_index` at sampling

This route is explicitly separate from the current renderer route:

- `p(mask)` models semantic layout by itself
- `p(image | mask)` renders an image conditioned on a semantic mask

The current project decomposition is therefore:

`p(mask) + p(image | mask)`

This document does **not** redefine the current renderer mathematics. It adds a new upstream route in parallel.

## Why This First Baseline Is Conservative

The first `p(mask)` baseline intentionally does **not** use a mask tokenizer.

Reasons:

1. Remote-sensing semantic masks are layout-heavy and topology-heavy, so the first question is whether a few-step flow prior can capture large-scale layout directly.
2. A tokenizer would add another compression bottleneck and another potential failure mode before the unconditional prior itself has been validated.
3. Direct mask-space training keeps failure analysis simple:
   - global layout wrong
   - boundaries unstable
   - thin structures missing
   - collapse or fragmentation under low NFE

The first `p(mask)` baseline also intentionally does **not** model `p(image, mask)` jointly.

Reasons:

1. Joint modeling would re-couple mask generation with image appearance, making it harder to locate the actual failure source.
2. The project already has a validated `p(image | mask)` route, so the more natural next step is to solve the missing upstream factor by itself.
3. The project goal is now operationally decomposed, not fused.

## Legacy Baseline Design

The first main baseline is:

- trainer: `latent_meanflow.trainers.mask_prior_trainer.MaskPriorTrainer`
- backbone: project-layer `LatentIntervalUNet`
- objective: AlphaFlow
- sampler: interval sampler with fixed `NFE=8/4/2/1` evaluation sweep
- decoding: sampled `K`-channel field -> `argmax` -> discrete class-id mask

Interpretation:

- the model learns a direct flow prior over `K x H x W` semantic-mask fields
- there is no image branch
- there is no mask tokenizer
- there is no latent downsampling beyond the native mask resolution configured for training

## Stronger Benchmark Follow-Up

The next benchmark route keeps the same `p(semantic_mask)` task and the same
evaluation protocol, but changes backbone and objective:

- backbone: project-layer `LatentIntervalSiT`
- objective: standard diffusion
- sampler: DDIM-style few-step sampler
- configs:
  - `configs/latent_diffusion_mask_prior_sit_tiny.yaml`
  - `configs/latent_diffusion_mask_prior_sit.yaml`

This route still does **not** use:

- a mask tokenizer
- an image branch
- the upstream `third_party/SiT/train.py` or `sample.py` entrypoints

Use [docs/mask_prior_sit_diffusion_benchmark.md](docs/mask_prior_sit_diffusion_benchmark.md)
for the fixed U-Net AlphaFlow vs SiT benchmark comparisons.

## Direct Discrete Follow-Up

The continuous one-hot SiT diffusion route remains checked in as a control, but
the current parallel upgrade path changes the modeled state itself:

- trainer: `latent_meanflow.trainers.discrete_mask_prior_trainer.DiscreteMaskPriorTrainer`
- backbone: project-layer `LatentIntervalSiT`
- objective: absorbing-mask discrete diffusion over `mask_index`
- sampler: seeded stochastic few-step progressive reveal
- configs:
  - `configs/discrete_mask_prior_sit_tiny.yaml`
  - `configs/discrete_mask_prior_sit.yaml`

Phase A / MVP rules for this route:

- task is still `p(semantic_mask)` only
- state object is discrete `mask_index`, not a continuous one-hot field
- an absorbing `MASK` token is added during training and sampling
- there is still no mask tokenizer
- there is still no image branch
- there are still no boundary / area / adjacency / topology auxiliary losses
- frozen renderer and frozen teacher evaluation stay unchanged

This route exists in parallel. It does **not** delete or redefine:

- the legacy U-Net + AlphaFlow mask-prior baseline
- the continuous one-hot SiT diffusion control baseline
- the frozen `p(image | semantic_mask)` renderer workflow

## Deliverables

Project-layer files for this route:

- `latent_meanflow/trainers/mask_prior_trainer.py`
- `latent_meanflow/data/semantic_mask.py`
- `scripts/train_mask_prior.py`
- `scripts/sample_mask_prior.py`
- `scripts/eval_mask_prior.py`
- `scripts/eval_mask_prior_composed_renderer.py`
- `configs/latent_alphaflow_mask_prior_unet_tiny.yaml`
- `configs/latent_alphaflow_mask_prior_unet.yaml`

Project-layer files for the stronger benchmark route:

- `latent_meanflow/models/backbones/latent_interval_sit.py`
- `latent_meanflow/objectives/diffusion.py`
- `latent_meanflow/samplers/diffusion.py`
- `scripts/train_mask_prior_diffusion.py`
- `scripts/sample_mask_prior_diffusion.py`
- `configs/latent_diffusion_mask_prior_sit_tiny.yaml`
- `configs/latent_diffusion_mask_prior_sit.yaml`
- `docs/mask_prior_sit_diffusion_benchmark.md`

Project-layer files for the direct discrete semantic-variable follow-up:

- `latent_meanflow/trainers/discrete_mask_prior_trainer.py`
- `latent_meanflow/objectives/discrete_mask_diffusion.py`
- `latent_meanflow/samplers/discrete_mask_diffusion.py`
- `configs/discrete_mask_prior_sit_tiny.yaml`
- `configs/discrete_mask_prior_sit.yaml`
- `tests/test_discrete_mask_prior_smoke.py`

## Success Criteria

Tiny overfit target:

- remember large contiguous class regions
- recover main boundaries
- let narrow roads / thin channels at least partially emerge

Base target:

- `NFE=8/4` should not collapse into random local fragments or pure noise-like blocks
- if `NFE=8` still lacks global layout coherence, the next upgrade should be backbone capacity, not a premature one-step blame cycle

Discrete-route Phase A target:

- keep the same `NFE=8/4/2/1` evaluation sweep
- confirm that moving from continuous one-hot fields to discrete semantic
  variables improves large-region layout coherence and reduces speckle
  fragmentation
- if this still fails, the next variables should be structure-aware losses or a
  tokenizer, not a silent return to the old continuous objective

## Recommended Workflow

1. Overfit with the tiny config.
2. Verify that `mask_raw/`, `mask_color/`, and `panel/` all look structurally coherent.
3. Train the base AlphaFlow route.
4. Run the fixed `NFE=8/4/2/1` sweep.
5. Evaluate generated masks against the real-mask bank before composing with the renderer.
6. Use `docs/mask_prior_eval_protocol.md` for the fixed mask-only and compose evaluation protocol.

## Commands

Tiny pilot:

```bash
python scripts/train_mask_prior.py \
  --config configs/latent_alphaflow_mask_prior_unet_tiny.yaml \
  --scale-lr true \
  --gpus 0
```

Base AlphaFlow training:

```bash
python scripts/train_mask_prior.py \
  --config configs/latent_alphaflow_mask_prior_unet.yaml \
  --scale-lr true \
  --gpus 0
```

Sampling sweep:

```bash
python scripts/sample_mask_prior.py \
  --config configs/latent_alphaflow_mask_prior_unet.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_samples/alphaflow_unet \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --overwrite
```

Evaluation:

```bash
python scripts/eval_mask_prior.py \
  --config configs/latent_alphaflow_mask_prior_unet.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_eval/alphaflow_unet \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --overwrite
```

Notes:

- `scripts/sample_mask_prior.py` and `scripts/eval_mask_prior.py` now require an explicit `--ckpt`; they no longer fall back to `last.ckpt`.
- `scripts/train_mask_prior.py` now exposes `--scale-lr true|false` explicitly and keeps `resume` in safe mode by default.

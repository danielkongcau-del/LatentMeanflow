# Mask Prior Plan

## Stage Update

The direct pixel-space mask-prior routes remain checked in, but they are no
longer the promoted upstream mainline for multimodal `p(semantic_mask)` work.

Current conclusion:

- the checked-in direct discrete route is still a useful negative but
  informative baseline
- `memorize_1` showed that a direct pixel-space route can memorize one fixed
  layout
- `memorize_4` collapsed toward a small number of low-quality prototypes
- therefore the direct pixel-space route is not being promoted as the mainline
  for unconditional multimodal `p(semantic_mask)`

Next mainline:

- `mask-only semantic tokenizer -> latent/token prior`
- the tokenizer has passed reconstruction review and is now treated as fixed
- the current patch implements a frozen-tokenizer continuous latent diffusion
  prior v1
- codebook / VQ token priors are still `not implemented yet`

## Frozen-Tokenizer Latent Prior v1

The current upstream control baseline is now:

- frozen `SemanticMaskAutoencoder`
- deterministic tokenizer latent target via
  `encode_batch(..., sample_posterior=False)`
- continuous latent diffusion prior on `z`
- decode path:
  `sample latent z -> tokenizer.decode_latents(z) -> semantic_mask`

This route exists to answer a narrow question with minimal extra variables:

- after removing the direct pixel-space bottleneck, does unconditional
  multimodal `p(semantic_mask)` become easier immediately?

This patch does **not** implement:

- codebook / VQ tokenizer
- token autoregressive prior
- latent discrete diffusion
- latent flow / AlphaFlow prior
- image branch coupling
- boundary / area / adjacency / topology auxiliary losses

## Frozen-Tokenizer Latent Prior Deliverables

Project-layer files for the first latent-prior control baseline:

- `latent_meanflow/trainers/semantic_mask_latent_prior_trainer.py`
- `configs/latent_semantic_mask_prior_diffusion_sit_tiny.yaml`
- `configs/latent_semantic_mask_prior_diffusion_sit.yaml`
- `configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_1.yaml`
- `configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_4.yaml`
- `tests/test_semantic_mask_latent_prior_smoke.py`

This route deliberately reuses existing project-layer wrappers:

- `scripts/train_mask_prior_diffusion.py`
- `scripts/sample_mask_prior_diffusion.py`
- `scripts/eval_mask_prior.py`

`tokenizer_ckpt_path` stays `null` in checked-in configs on purpose. Pass it
explicitly at runtime via `--set`.

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
- sampler:
  - control configs: seeded stochastic few-step progressive reveal
  - high-mask ablation configs: remask-low-confidence iterative refinement
- configs:
  - `configs/discrete_mask_prior_sit_tiny.yaml`
  - `configs/discrete_mask_prior_sit.yaml`
  - `configs/ablations/discrete_mask_prior_sit_highmask_refine_tiny.yaml`
  - `configs/ablations/discrete_mask_prior_sit_highmask_refine.yaml`

Phase A / MVP rules for this route:

- task is still `p(semantic_mask)` only
- state object is discrete `mask_index`, not a continuous one-hot field
- an absorbing `MASK` token is added during training and sampling
- there is still no mask tokenizer
- there is still no image branch
- there are still no boundary / area / adjacency / topology auxiliary losses
- frozen renderer and frozen teacher evaluation stay unchanged

Current parallel split inside this route:

- `configs/discrete_mask_prior_sit*.yaml`
  - control discrete baseline
  - Bernoulli masking
  - seeded progressive reveal sampler
- `configs/ablations/discrete_mask_prior_sit_highmask_refine*.yaml`
  - exact-count corruption
  - full-mask / high-mask sample mixing inside each minibatch
  - class-balanced masked cross entropy
  - remask-low-confidence iterative refinement sampler
- `configs/ablations/discrete_mask_prior_sit_proposal_visible_refine*.yaml`
  - same improved discrete objective as the high-mask / refine ablation
  - proposal-visible iterative refinement sampler
  - next-step state is the previous full proposal, not sparse locked-only context

## Decisive Memorization Diagnostic

The current improved discrete route also has two deliberate overfit diagnostics:

- `configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_1.yaml`
- `configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_4.yaml`

Purpose:

- keep the current improved discrete high-mask / class-balanced / refinement route unchanged
- shrink both train and validation to the same fixed bank from the training split
- answer whether this route can memorize coherent layout geometry on 1 mask or 4 masks

Interpretation:

- if the route still collapses into majority-class speckle on these fixed banks,
  then the current objective / sampler semantics are still insufficient even for
  basic layout memory
- if the route does memorize coherent connected regions on these fixed banks,
  then the full-data failure is more likely a scaling / generalization / target-alignment problem

These configs are deliberate overfit diagnostics, not normal generalization benchmarks.

## Proposal-Visible Refinement Ablation

This parallel ablation keeps the current improved discrete objective fixed and
changes only the sampler state semantics:

- `configs/ablations/discrete_mask_prior_sit_proposal_visible_refine_tiny.yaml`
- `configs/ablations/discrete_mask_prior_sit_proposal_visible_refine.yaml`
- `configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_1.yaml`
- `configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_4.yaml`

Purpose:

- test whether the current `memorize_4` failure is primarily caused by sparse
  locked-only context in the sampler
- let the next step see a full editable coarse proposal instead of only a small
  set of frozen high-confidence tokens

Interpretation:

- if proposal-visible refinement makes `memorize_4` cover multiple coherent
  layouts instead of collapsing to one low-quality prototype, then the previous
  failure was primarily a sampler-context problem
- if `memorize_4` still collapses badly, then the direct pixel-space discrete
  route is likely hitting a more basic multimodal layout bottleneck

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
- `configs/ablations/discrete_mask_prior_sit_highmask_refine_tiny.yaml`
- `configs/ablations/discrete_mask_prior_sit_highmask_refine.yaml`
- `configs/ablations/discrete_mask_prior_sit_proposal_visible_refine_tiny.yaml`
- `configs/ablations/discrete_mask_prior_sit_proposal_visible_refine.yaml`
- `configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_1.yaml`
- `configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_4.yaml`
- `configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_1.yaml`
- `configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_4.yaml`
- `latent_meanflow/data/subset.py`
- `tests/test_discrete_mask_prior_smoke.py`
- `tests/test_fixed_subset_dataset.py`

Project-layer files for the new mask-only semantic tokenizer mainline:

- `latent_meanflow/models/semantic_mask_autoencoder.py`
- `scripts/train_semantic_mask_tokenizer.py`
- `scripts/eval_semantic_mask_tokenizer.py`
- `configs/semantic_mask_tokenizer_tiny_256.yaml`
- `configs/semantic_mask_tokenizer_mid_256.yaml`
- `configs/semantic_mask_tokenizer_mid_plus_256.yaml`
- `configs/diagnostics/semantic_mask_tokenizer_memorize_1_256.yaml`
- `configs/diagnostics/semantic_mask_tokenizer_memorize_4_256.yaml`
- `tests/test_semantic_mask_tokenizer_smoke.py`

Project-layer files for the frozen-tokenizer latent prior control baseline:

- `latent_meanflow/trainers/semantic_mask_latent_prior_trainer.py`
- `configs/latent_semantic_mask_prior_diffusion_sit_tiny.yaml`
- `configs/latent_semantic_mask_prior_diffusion_sit.yaml`
- `configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_1.yaml`
- `configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_4.yaml`
- `tests/test_semantic_mask_latent_prior_smoke.py`

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

Frozen-tokenizer latent-prior v1 target:

- keep the tokenizer fixed and deterministic
- move the training target from pixel-space `semantic_mask` to continuous latent
  `z`
- reuse the same `NFE=8/4/2/1` few-step evaluation sweep
- use `memorize_1` and `memorize_4` as the decisive first diagnostic before
  long full-data training
- if latent `memorize_4` still collapses badly, the next step should be a true
  token / codebook prior rather than more pixel-space patching

## Mask-Only Semantic Tokenizer Route

The new mainline starts by compressing `semantic_mask` into a more tractable
latent space before reopening the upstream `p(semantic_mask)` prior question.

Route definition:

- task: reconstruct `semantic_mask` only
- model: `latent_meanflow.models.semantic_mask_autoencoder.SemanticMaskAutoencoder`
- dataset: `latent_meanflow.data.semantic_mask.MultiSemanticMaskDataset`
- input fields:
  - `mask_index`
  - `mask_onehot`
- output fields:
  - `mask_logits`
  - `mask_index`
  - `mask_onehot`
- no image branch
- no direct unconditional prior in this patch

Purpose:

- validate that `semantic_mask` can be reconstructed cleanly in a mask-only
  latent space before implementing any latent/token prior
- stop spending mainline time on direct pixel-space multimodal collapse

Checked-in configs:

- `configs/semantic_mask_tokenizer_tiny_256.yaml`
- `configs/semantic_mask_tokenizer_mid_256.yaml`
- `configs/semantic_mask_tokenizer_mid_plus_256.yaml`
- `configs/diagnostics/semantic_mask_tokenizer_memorize_1_256.yaml`
- `configs/diagnostics/semantic_mask_tokenizer_memorize_4_256.yaml`

Checked-in entry points:

- `scripts/train_semantic_mask_tokenizer.py`
- `scripts/eval_semantic_mask_tokenizer.py`

Commands:

Tiny tokenizer:

```bash
python scripts/train_semantic_mask_tokenizer.py \
  --config configs/semantic_mask_tokenizer_tiny_256.yaml \
  --scale-lr true \
  --gpus 0
```

Main tokenizer candidate:

```bash
python scripts/train_semantic_mask_tokenizer.py \
  --config configs/semantic_mask_tokenizer_mid_plus_256.yaml \
  --scale-lr true \
  --gpus 0
```

Memorize-4 diagnostic:

```bash
python scripts/train_semantic_mask_tokenizer.py \
  --config configs/diagnostics/semantic_mask_tokenizer_memorize_4_256.yaml \
  --scale-lr false \
  --gpus 0
```

Reconstruction evaluation:

```bash
python scripts/eval_semantic_mask_tokenizer.py \
  --config configs/semantic_mask_tokenizer_mid_plus_256.yaml \
  --ckpt <best-semantic-mask-tokenizer-ckpt> \
  --outdir outputs/semantic_mask_tokenizer_eval/mid_plus \
  --split validation \
  --n-samples 32 \
  --batch-size 4 \
  --seed 23 \
  --overwrite
```

## Frozen-Tokenizer Latent Prior Route

This is the current second-step upstream mainline:

- task: unconditional `p(semantic_mask)`
- representation during prior training: continuous tokenizer latent `z`
- tokenizer:
  - mask-only
  - frozen
  - deterministic posterior mode only
- prior:
  - backbone: `LatentIntervalSiT`
  - objective: `GaussianDiffusionObjective`
  - sampler: `DDIMDiffusionSampler`
- no image branch
- no codebook / VQ tokenizer
- no token autoregressive prior
- no latent discrete diffusion
- no latent flow / AlphaFlow in this patch

Checked-in configs:

- `configs/latent_semantic_mask_prior_diffusion_sit_tiny.yaml`
- `configs/latent_semantic_mask_prior_diffusion_sit.yaml`
- `configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_1.yaml`
- `configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_4.yaml`

Interpretation of the decisive diagnostics:

- if frozen-tokenizer latent prior passes `memorize_4`, then the direct
  pixel-space bottleneck was the main problem
- if it still collapses badly, the next mainline should move toward a true
  token / codebook prior rather than more direct pixel-space patching

Commands:

Tiny latent prior:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/latent_semantic_mask_prior_diffusion_sit_tiny.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --scale-lr true \
  --gpus 0
```

Base latent prior:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/latent_semantic_mask_prior_diffusion_sit.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --scale-lr true \
  --gpus 0
```

Memorize-1 diagnostic:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_1.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --scale-lr false \
  --gpus 0
```

Memorize-4 diagnostic:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_4.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --scale-lr false \
  --gpus 0
```

Sampling sweep:

```bash
python scripts/sample_mask_prior_diffusion.py \
  --config configs/latent_semantic_mask_prior_diffusion_sit.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --ckpt <best-latent-prior-ckpt> \
  --outdir outputs/latent_semantic_mask_prior_samples/base \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Distributional evaluation:

```bash
python scripts/eval_mask_prior.py \
  --config configs/latent_semantic_mask_prior_diffusion_sit.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --ckpt <best-latent-prior-ckpt> \
  --outdir outputs/latent_semantic_mask_prior_eval/base \
  --split validation \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Memorize diagnostics are deliberate overfit controls, not generalization
benchmarks.

Memorize-4 reconstruction evaluation:

```bash
python scripts/eval_semantic_mask_tokenizer.py \
  --config configs/diagnostics/semantic_mask_tokenizer_memorize_4_256.yaml \
  --ckpt <best-semantic-mask-tokenizer-memorize-4-ckpt> \
  --outdir outputs/semantic_mask_tokenizer_eval/memorize_4 \
  --split validation \
  --n-samples 4 \
  --batch-size 4 \
  --seed 23 \
  --overwrite
```

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

Decisive memorization diagnostic:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_1.yaml \
  --scale-lr false \
  --gpus 0

python scripts/train_mask_prior_diffusion.py \
  --config configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_4.yaml \
  --scale-lr false \
  --gpus 0
```

Proposal-visible memorization diagnostic:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_1.yaml \
  --scale-lr false \
  --gpus 0

python scripts/train_mask_prior_diffusion.py \
  --config configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_4.yaml \
  --scale-lr false \
  --gpus 0
```

Diagnostic sampling sweep:

```bash
python scripts/sample_mask_prior_diffusion.py \
  --config configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_1.yaml \
  --ckpt <best-memorize-1-ckpt> \
  --outdir outputs/mask_prior_diagnostics/memorize_1_samples \
  --n-samples 16 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite

python scripts/sample_mask_prior_diffusion.py \
  --config configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_4.yaml \
  --ckpt <best-memorize-4-ckpt> \
  --outdir outputs/mask_prior_diagnostics/memorize_4_samples \
  --n-samples 16 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Proposal-visible diagnostic sampling sweep:

```bash
python scripts/sample_mask_prior_diffusion.py \
  --config configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_1.yaml \
  --ckpt <best-proposal-visible-memorize-1-ckpt> \
  --outdir outputs/mask_prior_diagnostics/proposal_visible_memorize_1_samples \
  --n-samples 16 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite

python scripts/sample_mask_prior_diffusion.py \
  --config configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_4.yaml \
  --ckpt <best-proposal-visible-memorize-4-ckpt> \
  --outdir outputs/mask_prior_diagnostics/proposal_visible_memorize_4_samples \
  --n-samples 16 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Diagnostic evaluation:

```bash
python scripts/eval_mask_prior.py \
  --config configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_1.yaml \
  --ckpt <best-memorize-1-ckpt> \
  --outdir outputs/mask_prior_diagnostics/memorize_1_eval \
  --n-samples 16 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite

python scripts/eval_mask_prior.py \
  --config configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_4.yaml \
  --ckpt <best-memorize-4-ckpt> \
  --outdir outputs/mask_prior_diagnostics/memorize_4_eval \
  --n-samples 16 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Proposal-visible diagnostic evaluation:

```bash
python scripts/eval_mask_prior.py \
  --config configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_1.yaml \
  --ckpt <best-proposal-visible-memorize-1-ckpt> \
  --outdir outputs/mask_prior_diagnostics/proposal_visible_memorize_1_eval \
  --n-samples 16 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite

python scripts/eval_mask_prior.py \
  --config configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_4.yaml \
  --ckpt <best-proposal-visible-memorize-4-ckpt> \
  --outdir outputs/mask_prior_diagnostics/proposal_visible_memorize_4_eval \
  --n-samples 16 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

These evaluation runs are intentionally using the same fixed tiny bank for
train and validation. In this diagnostic setting, `nearest_real_miou_mean` is
useful as an overfit readout rather than a generalization benchmark.

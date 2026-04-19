# SiT Mask Prior Benchmark

## Stage Conclusion

The checked-in direct pixel-space routes remain useful benchmark artifacts, but
they are not the promoted mainline anymore.

Current conclusion:

- direct pixel-space discrete routing passed `memorize_1`
- the same route collapsed on `memorize_4` toward a small number of prototypes
- frozen tokenizer + continuous latent diffusion prior v1 also failed on the
  same tiny-bank style diagnostics
- that makes both continuous routes negative but informative baselines rather
  than the preferred mainline for unconditional multimodal `p(semantic_mask)`

Next mainline:

- `frozen balanced VQ tokenizer -> token-code mask generator`
- the tokenizer remains frozen infrastructure
- the first conservative token-code diffusion baseline is now implemented in the
  project layer

## Mask-Only Discrete Tokenizer

The new upstream mainline after the continuous-prior failures is:

- mask-only VQ / codebook tokenizer
- encode path:
  `semantic_mask -> encoder -> discrete code grid`
- decode path:
  `discrete code grid -> quantized embeddings -> decoder -> semantic_mask`

Why this route exists:

- the continuous latent prior repeated the old continuous-field failure mode
- the next variable to change is state semantics, not another continuous
  objective
- tokenizer reconstruction was strong enough to freeze and hand off to the next
  upstream generator stage

The current mainline still does **not** implement:

- token autoregressive prior
- class-conditional token prior
- any new continuous latent prior
- joint image+mask generation
- structural auxiliary losses

The discrete-tokenizer half now has checked-in project-layer files:

- `latent_meanflow/models/semantic_mask_vq_autoencoder.py`
- `scripts/train_semantic_mask_vq_tokenizer.py`
- `scripts/eval_semantic_mask_vq_tokenizer.py`
- `configs/semantic_mask_vq_tokenizer_tiny_256.yaml`
- `configs/semantic_mask_vq_tokenizer_main_256.yaml`
- `configs/semantic_mask_vq_tokenizer_main_stable_256.yaml`
- `configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml`
- `configs/diagnostics/semantic_mask_vq_tokenizer_memorize_1_256.yaml`
- `configs/diagnostics/semantic_mask_vq_tokenizer_memorize_4_256.yaml`
- `configs/diagnostics/semantic_mask_vq_tokenizer_memorize_1_hifi_256.yaml`
- `configs/diagnostics/semantic_mask_vq_tokenizer_memorize_4_hifi_256.yaml`

Current tokenizer geometry:

- tiny token grid: `64x64`
- main token grid: `64x64`
- token sequence length: `4096`
- tiny codebook size: `64`
- main codebook size: `512`
- high-fidelity overfit token grid: `128x128`
- high-fidelity overfit sequence length: `16384`
- high-fidelity overfit codebook size: `1024`

Current promoted main config:

- `configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml`
- balanced quantizer path:
  cosine matching + EMA codebook update + dead-code refresh
- balanced loss path:
  train-set scanned class-balanced CE
- checkpoint monitor:
  `val/mask_ce_unweighted`

Stable fallback config:

- `configs/semantic_mask_vq_tokenizer_main_stable_256.yaml`
- stable quantizer path:
  cosine matching + EMA codebook update + dead-code refresh
- checkpoint monitor:
  `val/mask_ce`

Tail-aware main variant:

- `configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml`
- same `64x64` token grid and stabilized quantizer
- train-set scanned class-balanced CE for rare semantic classes
- checkpoint monitor:
  `val/mask_ce_unweighted`

The direct pixel-space prior benchmarks below remain checked in, but they are
now comparison controls rather than the promoted upstream mainline.

This document therefore serves two purposes:

- record the stage conclusion that the continuous prior mainline failed
- preserve the direct-mask benchmark matrix as negative but informative controls

- task: `p(semantic_mask)`
- task type: unconditional semantic layout generation
- representation: direct mask-space modeling under fixed project-layer scripts
- no mask tokenizer
- no image branch
- no change to the frozen `p(image | semantic_mask)` renderer

The purpose of the retained benchmark section is narrow and explicit:

- test whether a stronger SiT-style transformer backbone plus either a
  conventional diffusion control or a direct discrete baseline is a better fit
  for unconditional semantic-mask generation than the current U-Net +
  AlphaFlow baseline
- keep the evaluation protocol fixed so the comparison stays
  apples-to-apples

This benchmark does **not** claim that flow is dead. It only asks whether the
current direct-mask U-Net flow baseline is being limited by backbone capacity
and objective choice. If a SiT-style baseline wins under the same protocol, the
next comparison should be the same SiT-style backbone under flow or AlphaFlow.

## Discrete Tokenizer Commands

Tokenizer-only training:

```bash
python scripts/train_semantic_mask_vq_tokenizer.py \
  --config configs/semantic_mask_vq_tokenizer_tiny_256.yaml \
  --scale-lr true \
  --gpus 0

python scripts/train_semantic_mask_vq_tokenizer.py \
  --config configs/semantic_mask_vq_tokenizer_main_stable_256.yaml \
  --scale-lr true \
  --gpus 0

python scripts/train_semantic_mask_vq_tokenizer.py \
  --config configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml \
  --scale-lr true \
  --gpus 0

python scripts/train_semantic_mask_vq_tokenizer.py \
  --config configs/diagnostics/semantic_mask_vq_tokenizer_memorize_1_256.yaml \
  --scale-lr false \
  --gpus 0

python scripts/train_semantic_mask_vq_tokenizer.py \
  --config configs/diagnostics/semantic_mask_vq_tokenizer_memorize_4_256.yaml \
  --scale-lr false \
  --gpus 0

python scripts/train_semantic_mask_vq_tokenizer.py \
  --config configs/diagnostics/semantic_mask_vq_tokenizer_memorize_1_hifi_256.yaml \
  --scale-lr false \
  --gpus 0

python scripts/train_semantic_mask_vq_tokenizer.py \
  --config configs/diagnostics/semantic_mask_vq_tokenizer_memorize_4_hifi_256.yaml \
  --scale-lr false \
  --gpus 0
```

Tokenizer-only reconstruction evaluation:

```bash
python scripts/eval_semantic_mask_vq_tokenizer.py \
  --config configs/semantic_mask_vq_tokenizer_main_stable_256.yaml \
  --ckpt /path/to/semantic_mask_vq_tokenizer.ckpt \
  --outdir outputs/semantic_mask_vq_tokenizer_eval/main \
  --split validation \
  --n-samples 32 \
  --batch-size 4 \
  --seed 23 \
  --overwrite

python scripts/eval_semantic_mask_vq_tokenizer.py \
  --config configs/diagnostics/semantic_mask_vq_tokenizer_memorize_4_256.yaml \
  --ckpt /path/to/semantic_mask_vq_tokenizer_memorize_4.ckpt \
  --outdir outputs/semantic_mask_vq_tokenizer_eval/memorize_4 \
  --split validation \
  --n-samples 4 \
  --batch-size 4 \
  --seed 23 \
  --overwrite

python scripts/eval_semantic_mask_vq_tokenizer.py \
  --config configs/diagnostics/semantic_mask_vq_tokenizer_memorize_4_hifi_256.yaml \
  --ckpt /path/to/semantic_mask_vq_tokenizer_memorize_4_hifi.ckpt \
  --outdir outputs/semantic_mask_vq_tokenizer_eval/memorize_4_hifi \
  --split validation \
  --n-samples 4 \
  --batch-size 4 \
  --seed 23 \
  --overwrite
```

The VQ reconstruction eval also exports class diagnostics:

- `analysis/confusion_matrix.json`
- `analysis/worst_miou.json`
- `analysis/worst_per_class.json`
- `analysis/worst_miou_panel/`
- `analysis/worst_per_class_panel/`

These commands evaluate reconstruction only. Unconditional generation is now
evaluated separately through the token-code mask-generator scripts:

- `scripts/train_token_mask_prior.py`
- `scripts/sample_token_mask_prior.py`
- `scripts/eval_token_mask_prior.py`

## Compared Routes

Legacy baseline:

- backbone: project-layer `LatentIntervalUNet`
- objective: AlphaFlow
- config: `configs/latent_alphaflow_mask_prior_unet.yaml`
- train entrypoint: `scripts/train_mask_prior.py`
- sample entrypoint: `scripts/sample_mask_prior.py`

Continuous SiT control baseline:

- backbone: project-layer `LatentIntervalSiT`
- objective: standard Gaussian diffusion training
- sampler: DDIM-style deterministic few-step sampler
- config: `configs/latent_diffusion_mask_prior_sit.yaml`
- train entrypoint: `scripts/train_mask_prior_diffusion.py`
- sample entrypoint: `scripts/sample_mask_prior_diffusion.py`

Current benchmark candidate:

- backbone: project-layer `LatentIntervalSiT`
- objective: absorbing-mask discrete diffusion over `mask_index`
- sampler: seeded stochastic few-step progressive reveal
- configs:
  - `configs/discrete_mask_prior_sit_tiny.yaml`
  - `configs/discrete_mask_prior_sit.yaml`
- train entrypoint: `scripts/train_mask_prior_diffusion.py`
- sample entrypoint: `scripts/sample_mask_prior_diffusion.py`

Current collapse-mitigation ablation:

- backbone: project-layer `LatentIntervalSiT`
- objective:
  - exact-count corruption
  - full-mask / high-mask sample mixing inside each minibatch
  - class-balanced masked cross entropy
- sampler: remask-low-confidence iterative refinement
- configs:
  - `configs/ablations/discrete_mask_prior_sit_highmask_refine_tiny.yaml`
  - `configs/ablations/discrete_mask_prior_sit_highmask_refine.yaml`
- train entrypoint: `scripts/train_mask_prior_diffusion.py`
- sample entrypoint: `scripts/sample_mask_prior_diffusion.py`

Proposal-visible refinement ablation:

- backbone: project-layer `LatentIntervalSiT`
- objective:
  - exact-count corruption
  - full-mask / high-mask sample mixing inside each minibatch
  - class-balanced masked cross entropy
- sampler: proposal-visible iterative refinement
- configs:
  - `configs/ablations/discrete_mask_prior_sit_proposal_visible_refine_tiny.yaml`
  - `configs/ablations/discrete_mask_prior_sit_proposal_visible_refine.yaml`
- train entrypoint: `scripts/train_mask_prior_diffusion.py`
- sample entrypoint: `scripts/sample_mask_prior_diffusion.py`

Decisive memorization diagnostic:

- route: the same improved discrete high-mask / refine ablation
- configs:
  - `configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_1.yaml`
  - `configs/diagnostics/discrete_mask_prior_sit_highmask_refine_memorize_4.yaml`
- data rule:
  - both train and validation point to the same fixed 1-mask or 4-mask bank
  - the bank is drawn from the training split on purpose
- purpose:
  - determine whether the current route can memorize coherent layout geometry at all

Interpretation:

- if these diagnostics still produce majority-class collapse or speckle on 1-4 masks,
  the current objective / sampler semantics are still insufficient
- if these diagnostics memorize coherent connected regions, then the full-data
  failure is more likely a scaling / generalization problem

Proposal-visible memorization diagnostic:

- route: the same improved discrete objective with proposal-visible refinement
- configs:
  - `configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_1.yaml`
  - `configs/diagnostics/discrete_mask_prior_sit_proposal_visible_refine_memorize_4.yaml`
- data rule:
  - both train and validation point to the same fixed 1-mask or 4-mask bank
  - the bank is drawn from the training split on purpose
- purpose:
  - test whether `memorize_4` failure is primarily caused by sparse locked-only
    sampler context instead of a complete route-level failure

Interpretation:

- if proposal-visible refinement makes `memorize_4` cover multiple coherent
  layouts instead of collapsing to one low-quality prototype, then the previous
  failure was primarily a sampler-context problem
- if it still collapses badly, then the direct pixel-space discrete route is
  likely missing a more basic multimodal layout mechanism

All three routes still model `p(semantic_mask)` only.

Frozen-tokenizer latent prior control baseline:

- trainer: `latent_meanflow.trainers.semantic_mask_latent_prior_trainer.SemanticMaskLatentPriorTrainer`
- tokenizer:
  - `configs/semantic_mask_tokenizer_mid_plus_256.yaml`
  - checkpoint passed explicitly at runtime via `--set`
- backbone: project-layer `LatentIntervalSiT`
- objective: standard Gaussian diffusion training
- sampler: DDIM-style deterministic few-step sampler
- configs:
  - `configs/latent_semantic_mask_prior_diffusion_sit_tiny.yaml`
  - `configs/latent_semantic_mask_prior_diffusion_sit.yaml`
- decisive diagnostics:
  - `configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_1.yaml`
  - `configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_4.yaml`

Interpretation:

- if frozen-tokenizer latent prior passes `memorize_4`, then the direct
  pixel-space bottleneck was the main failure source
- if it still collapses badly, the next mainline should move toward a true
  token / codebook prior rather than more direct pixel-space patching

## Fixed Comparison Matrix

Keep the following fixed across the benchmark:

- same dataset root: `outputs/segmentation_teacher_data/remote_semantic_indexed`
- same label spec: `configs/label_specs/remote_semantic.yaml`
- same task: unconditional semantic-mask generation
- same downstream interface: generated outputs must end as discrete
  `mask_index` maps
- same frozen downstream renderer
- same frozen in-domain teacher
- same mask-only metrics
- same compose-to-image metrics
- same `NFE=8/4/2/1` sweep
- same seed

Do **not** change the renderer config, renderer checkpoint rule, or teacher
workflow while running this comparison.

## Why The Discrete Phase A Baseline Exists

The continuous SiT diffusion route stays checked in as a control, but the
current Phase A upgrade changes the state object itself for three reasons:

1. The failure mode in the continuous route is not just low capacity. It is
   also a mismatch between the true target and the modeled state: final masks
   are discrete semantic variables, not continuous one-hot fields.
2. This Phase A patch changes only one major variable at the project layer:
   state semantics. It still avoids a tokenizer, an image branch, and any
   structure auxiliary losses.
3. If this direct discrete route improves mask-only and compose metrics under
   the same frozen protocol, the next comparison against SiT + AlphaFlow
   becomes much clearer.

## Tiny Pilot

Use the tiny config for shape checks and deliberate overfit.

Continuous one-hot control:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/latent_diffusion_mask_prior_sit_tiny.yaml \
  --scale-lr true \
  --gpus 0
```

Direct discrete Phase A:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/discrete_mask_prior_sit_tiny.yaml \
  --scale-lr true \
  --gpus 0
```

Direct discrete high-mask / refine ablation:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/ablations/discrete_mask_prior_sit_highmask_refine_tiny.yaml \
  --scale-lr true \
  --gpus 0
```

Proposal-visible refinement ablation:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/ablations/discrete_mask_prior_sit_proposal_visible_refine_tiny.yaml \
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
  --config configs/discrete_mask_prior_sit.yaml \
  --scale-lr true \
  --gpus 0
```

Improved ablation:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/ablations/discrete_mask_prior_sit_highmask_refine.yaml \
  --scale-lr true \
  --gpus 0
```

Proposal-visible base ablation:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/ablations/discrete_mask_prior_sit_proposal_visible_refine.yaml \
  --scale-lr true \
  --gpus 0
```

Decisive memorization diagnostics:

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

Proposal-visible memorization diagnostics:

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

For `memorize_4`, the checked-in config uses `batch_size=4` so each optimizer
step sees the whole 4-mask bank. That makes the diagnostic sharper: failure is
less likely to be explained away by minibatch subsampling noise.

Sample a fixed few-step sweep:

```bash
python scripts/sample_mask_prior_diffusion.py \
  --config configs/discrete_mask_prior_sit.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_samples/discrete_sit \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

The sampling and evaluation scripts in this benchmark require explicit
checkpoint paths. They do not fall back to `last.ckpt`.

Phase A notes:

- the sampler is seeded stochastic, not pure deterministic MAP fill-in
- `sample_latents(..., noise=...)` remains meaningful because the existing
  project-layer scripts reuse it for reproducible few-step sweeps
- this patch does not add boundary / area / adjacency / topology auxiliary
  losses yet

Frozen-tokenizer latent prior control commands:

```bash
python scripts/eval_semantic_mask_tokenizer.py \
  --config configs/semantic_mask_tokenizer_mid_plus_256.yaml \
  --ckpt /path/to/semantic_mask_tokenizer.ckpt \
  --outdir outputs/semantic_mask_tokenizer_eval/mid_plus_validation \
  --split validation \
  --n-samples 256 \
  --batch-size 8 \
  --overwrite

python scripts/train_mask_prior_diffusion.py \
  --config configs/latent_semantic_mask_prior_diffusion_sit_tiny.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --set model.params.latent_normalization_config.mode=per_channel_affine \
  --set model.params.latent_normalization_config.stats_path=outputs/semantic_mask_tokenizer_eval/mid_plus_validation/latent_stats.json \
  --scale-lr true \
  --gpus 0

python scripts/train_mask_prior_diffusion.py \
  --config configs/latent_semantic_mask_prior_diffusion_sit.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --set model.params.latent_normalization_config.mode=per_channel_affine \
  --set model.params.latent_normalization_config.stats_path=outputs/semantic_mask_tokenizer_eval/mid_plus_validation/latent_stats.json \
  --scale-lr true \
  --gpus 0

python scripts/train_mask_prior_diffusion.py \
  --config configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_1.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --set model.params.latent_normalization_config.mode=per_channel_affine \
  --set model.params.latent_normalization_config.stats_path=outputs/semantic_mask_tokenizer_eval/mid_plus_validation/latent_stats.json \
  --scale-lr false \
  --gpus 0

python scripts/train_mask_prior_diffusion.py \
  --config configs/diagnostics/latent_semantic_mask_prior_diffusion_memorize_4.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --set model.params.latent_normalization_config.mode=per_channel_affine \
  --set model.params.latent_normalization_config.stats_path=outputs/semantic_mask_tokenizer_eval/mid_plus_validation/latent_stats.json \
  --scale-lr false \
  --gpus 0
```

Frozen-tokenizer latent prior sampling:

```bash
python scripts/sample_mask_prior_diffusion.py \
  --config configs/latent_semantic_mask_prior_diffusion_sit.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --set model.params.latent_normalization_config.mode=per_channel_affine \
  --set model.params.latent_normalization_config.stats_path=outputs/semantic_mask_tokenizer_eval/mid_plus_validation/latent_stats.json \
  --ckpt <best-latent-prior-ckpt> \
  --outdir outputs/mask_prior_samples/latent_semantic_sit \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

## Shared Evaluation Protocol

The benchmark intentionally reuses the checked-in evaluation scripts without
changing their metric definitions.

Mask-only evaluation:

```bash
python scripts/eval_mask_prior.py \
  --config configs/discrete_mask_prior_sit.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_eval/discrete_sit \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Frozen-tokenizer latent prior evaluation:

```bash
python scripts/eval_mask_prior.py \
  --config configs/latent_semantic_mask_prior_diffusion_sit.yaml \
  --set model.params.tokenizer_ckpt_path=/path/to/semantic_mask_tokenizer.ckpt \
  --set model.params.latent_normalization_config.mode=per_channel_affine \
  --set model.params.latent_normalization_config.stats_path=outputs/semantic_mask_tokenizer_eval/mid_plus_validation/latent_stats.json \
  --ckpt <best-latent-prior-ckpt> \
  --outdir outputs/mask_prior_eval/latent_semantic_sit \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

`memorize_1` and `memorize_4` remain deliberate overfit diagnostics, not
generalization benchmarks.

Diagnostic sampling and evaluation reuse the same checked-in scripts:

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

Proposal-visible diagnostic sampling and evaluation reuse the same checked-in scripts:

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

These diagnostic evaluations are deliberate overfit checks, not normal
generalization benchmarks. Because the reference bank is the same fixed tiny
bank used for training, `nearest_real_miou_mean` is intentionally being used as
an overfit readout.

Compose-to-image evaluation:

```bash
python scripts/eval_mask_prior_composed_renderer.py \
  --mask-config configs/discrete_mask_prior_sit.yaml \
  --mask-ckpt <best-mask-prior-ckpt> \
  --renderer-config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml \
  --renderer-ckpt <best-renderer-ckpt> \
  --renderer-tokenizer-config configs/autoencoder_image_lpips_adv_256.yaml \
  --renderer-tokenizer-ckpt <best-image-tokenizer-ckpt> \
  --teacher-run-dir logs/segmentation_teacher/<winner_run> \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --outdir outputs/mask_prior_compose_eval/discrete_sit \
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

If this Phase A discrete baseline still fails, then the next variables to
consider are:

- structure-aware auxiliary losses
- a semantic-mask tokenizer
- stronger flow variants such as OT-CFM
- a same-backbone SiT + AlphaFlow comparison

Do not introduce those extra variables in this benchmark step.

The tokenizer follow-up now exists as a checked-in parallel route:

- train entrypoint: `scripts/train_semantic_mask_tokenizer.py`
- eval entrypoint: `scripts/eval_semantic_mask_tokenizer.py`
- configs:
  - `configs/semantic_mask_tokenizer_tiny_256.yaml`
  - `configs/semantic_mask_tokenizer_mid_256.yaml`
  - `configs/semantic_mask_tokenizer_mid_plus_256.yaml`
  - `configs/diagnostics/semantic_mask_tokenizer_memorize_1_256.yaml`
  - `configs/diagnostics/semantic_mask_tokenizer_memorize_4_256.yaml`

Use it to answer the prerequisite question:

- can a mask-only tokenizer reconstruct 4 fixed semantic masks cleanly enough
  to justify a later latent/token prior?

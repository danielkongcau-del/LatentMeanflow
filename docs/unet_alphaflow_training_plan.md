# U-Net AlphaFlow Training Plan

This runbook defines the current project-layer U-Net AlphaFlow main line for
high-quality few-step paired `RGB image + semantic mask` generation.

It does not change the AlphaFlow math. It only fixes the training recipe,
capacity tiers, checkpoint rule, and evaluation protocol around the existing
implementation.

## Scope

- Keep the semantic tokenizer fixed.
- Keep the paired image-mask task fixed.
- Keep the AlphaFlow objective math fixed.
- Keep the sampling protocol fixed.
- Keep the best-checkpoint rule fixed.

This is a project-layer recipe family, not a paper-equivalent claim.
In particular, the current implementation still uses the detached online target
branch already checked into the project instead of a separate EMA teacher.

## Config Roles

- `configs/latent_alphaflow_semantic_256_unet_tiny.yaml`
  - Tiny/debug pilot.
  - Use it to verify that the promoted U-Net AlphaFlow route trains, logs, and
    samples without divergence.
- `configs/latent_alphaflow_semantic_256_unet.yaml`
  - Default project baseline.
  - This is the recommended first real long run today.
- `configs/latent_alphaflow_semantic_256_unet_large.yaml`
  - Same U-Net family, wider backbone, same effective batch target as the base
    route.
  - Use it only after the base route is stable.
- `configs/latent_alphaflow_semantic_256_unet_paper_attempt.yaml`
  - Same base U-Net family as the project baseline, but with a smoother
    curriculum transition and much smaller clamping.
  - This is paper-aligned in schedule semantics only, not paper-equivalent.
- `configs/latent_alphaflow_semantic_256.yaml`
  - Legacy ConvNet rollback baseline.

## Budget-Aware Schedule

The promoted U-Net AlphaFlow configs use:

- `latent_meanflow.objectives.alphaflow.BudgetSigmoidAlphaScheduler`

Instead of hard-coding an absolute `start_step` and `end_step`, the schedule now
tracks fractions of the total optimizer-step budget:

- `transition_start_fraction`
- `transition_end_fraction`
- `gamma`
- `clamp_eta`

The trainer injects the actual fit budget at `on_fit_start()` using the current
Lightning estimate of optimizer steps. The config still carries fallback budget
numbers so the route can be instantiated for self-checks and offline tooling.

## Effective Batch Size

Use this formula:

`effective_batch = per_device_batch * num_gpus * accumulate_grad_batches`

Checked-in 2-GPU recipes:

- Tiny U-Net AlphaFlow:
  - per-device batch = `2`
  - accumulate = `1`
  - effective batch = `4`
- Base U-Net AlphaFlow:
  - per-device batch = `8`
  - accumulate = `4`
  - effective batch = `64`
- Large U-Net AlphaFlow:
  - per-device batch = `4`
  - accumulate = `8`
  - effective batch = `64`
- Paper-attempt U-Net AlphaFlow:
  - per-device batch = `8`
  - accumulate = `4`
  - effective batch = `64`

The checked-in `base_learning_rate` values are chosen so that the legacy
latent-diffusion learning-rate scaling still lands near the same effective LR
target on the 2-GPU recipes above.

## Checkpoint Rule

Always select the best checkpoint by:

- `val/base_error_mean`

Do not use:

- `val/alphaflow_loss` as the main checkpoint-selection rule
- `last.ckpt` for the primary quality comparison
- ad-hoc "most recent run under logs" logic

Resolve the checkpoint explicitly from the intended run directory:

```bash
python scripts/find_checkpoint.py --run-dir logs/<your_run> --selection best --monitor val/base_error_mean
```

## Recommended Run Order

1. `configs/latent_alphaflow_semantic_256_unet_tiny.yaml`
2. `configs/latent_alphaflow_semantic_256_unet.yaml`
3. `configs/latent_alphaflow_semantic_256_unet_large.yaml`
4. `configs/latent_alphaflow_semantic_256_unet_paper_attempt.yaml`

Use the paper-attempt route only after the project baseline is stable and you
want to test whether a smoother curriculum plus smaller clamping improves the
few-step quality curve.

## Training Commands

These examples assume:

- `lmf` environment is active
- tokenizer config: `configs/autoencoder_semantic_pair_24gb_lpips_256.yaml`
- tokenizer checkpoint: `logs/autoencoder/checkpoints/last.ckpt`
- two GPUs

### Tiny / Debug

```bash
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 \
OBJECTIVE=alphaflow \
CONFIG=configs/latent_alphaflow_semantic_256_unet_tiny.yaml \
TOKENIZER_CONFIG=configs/autoencoder_semantic_pair_24gb_lpips_256.yaml \
TOKENIZER_CKPT=logs/autoencoder/checkpoints/last.ckpt \
GPUS=2 \
MAX_EPOCHS=5 \
bash scripts/train_meanflow.sh
```

### Base / Project Baseline

```bash
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 \
OBJECTIVE=alphaflow \
CONFIG=configs/latent_alphaflow_semantic_256_unet.yaml \
TOKENIZER_CONFIG=configs/autoencoder_semantic_pair_24gb_lpips_256.yaml \
TOKENIZER_CKPT=logs/autoencoder/checkpoints/last.ckpt \
GPUS=2 \
MAX_EPOCHS=3000 \
bash scripts/train_meanflow.sh
```

### Large / Capacity Bump

```bash
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 \
OBJECTIVE=alphaflow \
CONFIG=configs/latent_alphaflow_semantic_256_unet_large.yaml \
TOKENIZER_CONFIG=configs/autoencoder_semantic_pair_24gb_lpips_256.yaml \
TOKENIZER_CKPT=logs/autoencoder/checkpoints/last.ckpt \
GPUS=2 \
MAX_EPOCHS=3000 \
bash scripts/train_meanflow.sh
```

### Paper-Aligned Attempt

```bash
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 \
OBJECTIVE=alphaflow \
CONFIG=configs/latent_alphaflow_semantic_256_unet_paper_attempt.yaml \
TOKENIZER_CONFIG=configs/autoencoder_semantic_pair_24gb_lpips_256.yaml \
TOKENIZER_CKPT=logs/autoencoder/checkpoints/last.ckpt \
GPUS=2 \
MAX_EPOCHS=3000 \
bash scripts/train_meanflow.sh
```

## Few-Step Evaluation Protocol

Use the same seed and the same `NFE=8/4/2/1` sweep for every AlphaFlow U-Net
comparison.

For the full paired RGB + semantic-mask reporting protocol, including
teacher-aligned `mIoU`, `Boundary F1`, and pair-consistency metrics, use
[docs/semantic_pair_eval_protocol.md](semantic_pair_eval_protocol.md). The
backbone sweep below is still the correct first pass for image/mask export and
checkpoint fixation.

The benchmark script always writes:

- `image/`
- `mask_raw/`
- `mask_color/`
- `overlay/`

and also saves:

- `summary.json`
- `summary.csv`

### Base Sweep

```bash
python scripts/find_checkpoint.py --run-dir logs/<timestamp>_latent_alphaflow_semantic_256_unet --selection best --monitor val/base_error_mean

python scripts/eval_backbone_nfe_sweep.py \
  --config configs/latent_alphaflow_semantic_256_unet.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/benchmarks/alphaflow_unet_base \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1
```

### Large Sweep

```bash
python scripts/find_checkpoint.py --run-dir logs/<timestamp>_latent_alphaflow_semantic_256_unet_large --selection best --monitor val/base_error_mean

python scripts/eval_backbone_nfe_sweep.py \
  --config configs/latent_alphaflow_semantic_256_unet_large.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/benchmarks/alphaflow_unet_large \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1
```

### Paper-Attempt Sweep

```bash
python scripts/find_checkpoint.py --run-dir logs/<timestamp>_latent_alphaflow_semantic_256_unet_paper_attempt --selection best --monitor val/base_error_mean

python scripts/eval_backbone_nfe_sweep.py \
  --config configs/latent_alphaflow_semantic_256_unet_paper_attempt.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/benchmarks/alphaflow_unet_paper_attempt \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1
```

## Pass Criteria

- If the U-Net AlphaFlow large route is clearly better than the base route at
  `NFE=4/2` without regressing at `NFE=8`, the extra capacity is worth keeping.
- If the paper-attempt route beats the project baseline on `NFE=4/2` while
  staying at least neutral on `NFE=8`, the smoother schedule is worth
  continuing.
- If `NFE=8` is still poor, prioritize investigating:
  - tokenizer latent quality
  - objective choice
  - training budget

Do not blame one-step difficulty first if `NFE=8` never becomes competitive.

## Recommendation

The next formal long run should be:

- `configs/latent_alphaflow_semantic_256_unet.yaml`

It is the safest route for today because it is already the promoted default
U-Net AlphaFlow project baseline, it uses the budget-aware schedule, and it
hits the effective batch target of `64` without requiring the larger model's
VRAM footprint.

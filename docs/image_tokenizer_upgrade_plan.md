# Image Tokenizer Upgrade Plan

This note defines the next project-layer image-only tokenizer upgrade without
changing downstream latent-prior math and without claiming SD-VAE equivalence.

## Design Summary

The stronger tokenizer route keeps the interface fixed:

- RGB in
- RGB out
- semantic mask stays outside latent `z`
- downstream latent interface remains `4` channels
- primary base route remains `256 -> 64x64x4`

The upgrade adds three knobs on top of the current base tokenizer:

1. `RGB L1`
   Keeps coarse color and structure anchored.
2. `optional LPIPS`
   Pushes reconstructions toward better perceptual sharpness.
3. `optional patch adversarial loss`
   Pushes local texture and boundary realism harder than L1 alone.

It also adds one explicit anti-collapse mechanism:

- `latent_channel_std_floor_penalty`
  The model computes batch-wise per-channel latent std from `posterior.mode()`
  and penalizes channels whose std falls below a configured floor.

This does not hide collapse. It makes collapse measurable and directly costly.

## New Configs

Base-geometry route:

- [configs/autoencoder_image_adv_256.yaml](/e:/CodeSpace/LarentMeanflow/configs/autoencoder_image_adv_256.yaml)
- [configs/autoencoder_image_lpips_adv_256.yaml](/e:/CodeSpace/LarentMeanflow/configs/autoencoder_image_lpips_adv_256.yaml)

Follow-up `f=8` branch:

- [configs/autoencoder_image_f8_adv_256.yaml](/e:/CodeSpace/LarentMeanflow/configs/autoencoder_image_f8_adv_256.yaml)
- [configs/autoencoder_image_f8_lpips_adv_256.yaml](/e:/CodeSpace/LarentMeanflow/configs/autoencoder_image_f8_lpips_adv_256.yaml)

## Training Recipes

### Baseline image-only tokenizer

Use the existing baseline if you need the old route unchanged:

```bash
python scripts/train_image_autoencoder.py \
  --config configs/autoencoder_image_256.yaml \
  --gpus 0
```

### LPIPS-only tokenizer

Use this when you want a direct comparison against the currently measured
baseline-to-beat:

```bash
python scripts/train_image_autoencoder.py \
  --config configs/autoencoder_image_24gb_lpips_256.yaml \
  --gpus 0
```

### LPIPS + adversarial tokenizer

This is the recommended first long run:

```bash
python scripts/train_image_autoencoder.py \
  --config configs/autoencoder_image_lpips_adv_256.yaml \
  --gpus 0
```

## What Counts As A Better Tokenizer

The new checkpoint should beat the current baseline on most of the following:

- lower or comparable `rgb_lpips`
- lower or comparable `rgb_l1`
- sharper reconstruction panels on roads, narrow channels, and field edges
- `collapsed_channel_count == 0` under the default audit threshold
- materially better `min_channel_std`
- higher downstream-readiness score in the audit report

It is not enough to look slightly prettier on a few images if the latent health
gets worse.

## When To Keep Base Geometry

Stay on `64x64x4` first when:

- the current bottleneck is reconstruction sharpness, not latent size
- field edges, roads, and narrow structures still blur under audit crops
- the `f=8` branch has not yet shown healthy latent channels
- you need the least disruptive drop-in tokenizer for current mask-conditioned
  rendering experiments

## When `f=8` Is Worth Trying

Try `f=8` after the base LPIPS+adversarial route is healthy and audited.

Signals that justify the `f=8` follow-up:

- base `64x64x4` route no longer shows collapse
- blur on boundary/error crop panels is materially reduced
- the tighter latent is needed for memory or sampling efficiency
- `f=8` no longer loses badly on audit LPIPS or boundary-sensitive visuals

## Recommended Next Runs

1. First long run:
   [configs/autoencoder_image_lpips_adv_256.yaml](/e:/CodeSpace/LarentMeanflow/configs/autoencoder_image_lpips_adv_256.yaml)
2. Stronger follow-up branch:
   [configs/autoencoder_image_f8_lpips_adv_256.yaml](/e:/CodeSpace/LarentMeanflow/configs/autoencoder_image_f8_lpips_adv_256.yaml)

## How Anti-Collapse Is Measured

Training logs expose:

- `latent_std_floor_penalty`
- `latent_batch_channel_std_mean`
- `latent_batch_channel_std_min`
- `latent_batch_channel_std_max`
- `latent_batch_channel_std_cv`
- `latent_utilized_channel_count`
- `latent_utilized_channel_fraction`

Audit-time reports expose:

- per-channel std
- collapsed-channel count
- min/max channel std
- channel-std CV

This makes the anti-collapse mechanism explicit instead of hidden.

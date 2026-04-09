# Image Tokenizer Audit

This note defines how to audit the current project-layer image-only tokenizers
before changing their architecture or the downstream mask-conditioned renderer.

## What To Audit

Use [scripts/eval_image_tokenizer.py](/e:/CodeSpace/LarentMeanflow/scripts/eval_image_tokenizer.py)
for one checkpoint or [scripts/audit_image_tokenizers.py](/e:/CodeSpace/LarentMeanflow/scripts/audit_image_tokenizers.py)
for a ranked multi-checkpoint sweep.

The audit treats three failure modes separately:

- `blur severity`
  The reconstruction looks smooth at first glance but edges, narrow roads,
  water channels, and field boundaries are softened or washed out.
- `channel collapse severity`
  One or more latent channels carry almost no useful variance even when the
  global latent std still looks healthy.
- `downstream readiness`
  Whether a tokenizer is currently usable as the reference bottleneck for
  `p(image | semantic_mask)`.

## Signs Of Blur

Numerical signs:

- higher `rgb_lpips`
- higher `rgb_l1`
- high error concentration around semantic boundaries in crop panels

Visual signs:

- roads expand into soft ribbons instead of staying crisp
- narrow water channels disappear or merge into nearby regions
- field edges lose their straight or stepped local geometry
- absolute-error heatmaps light up around object boundaries even when the
  interior region looks acceptable

The audit scripts export:

- input image
- reconstruction
- absolute error heatmap
- boundary-focused crops
- edge-error-focused crops
- pure error-focused crops

Those crops are chosen from mask boundaries and local reconstruction error so
that remote-sensing failures are easier to see on roads, channels, and field
edges.

## Signs Of Channel Collapse

The audit uses an explicit collapse heuristic:

- `collapsed_channel_count`: number of latent channels with std below a chosen
  threshold, default `0.05`
- `min_channel_std` and `max_channel_std`
- `channel_std_cv`: coefficient of variation across per-channel std values
- `channel_std_min_to_max_ratio`

Interpretation:

- `collapsed_channel_count == 0` means no direct collapse signal
- any channel std below `0.05` is a real warning, not a stylistic difference
- very low `min_channel_std` with normal `latent_std` usually means the global
  latent statistics are hiding a dead or near-dead channel
- high `channel_std_cv` means variance is distributed unevenly across channels

## Current Baseline Decision

As of April 9, 2026, the only image-only tokenizer checkpoint with explicit
measured evidence in hand is:

- config: [configs/autoencoder_image_24gb_lpips_256.yaml](/e:/CodeSpace/LarentMeanflow/configs/autoencoder_image_24gb_lpips_256.yaml)
- checkpoint: `logs/autoencoder/checkpoints/last-v1.ckpt`
- reported metrics:
  - `rgb_l1 = 0.0471466805`
  - `rgb_lpips = 0.171143384`
  - `latent_mean = -0.001948`
  - `latent_std = 0.836498`
  - `latent_l2_norm_mean = 106.941`
  - per-channel stds:
    - `channel_0 = 1.000769`
    - `channel_1 = 0.961164`
    - `channel_2 = 0.005034`
    - `channel_3 = 0.934528`

This is enough to make two hard calls:

- channel collapse is real
- the current baseline to beat is still the LPIPS-enabled base tokenizer above

That baseline is not being called "best" because it is healthy. It is the
baseline because it is the tokenizer that already has real measured numbers and
is already wired into downstream mask-conditioned image generation work.

## Why `f=8` Is Not Automatically Better

The `f=8` tokenizers:

- use a tighter `32x32x4` latent instead of `64x64x4`
- are more compact on paper
- may be better eventually

But tighter latent geometry is not enough by itself. A stronger compressor only
helps downstream if:

- reconstruction blur does not get worse in the crop diagnostics
- channel health does not collapse
- the downstream renderer is not starved of spatial detail

For that reason, the audit gives compactness only a small weight in the
downstream-readiness score. `f=8` gets a small credit for being tighter, but it
can still lose the ranking if LPIPS, L1, or channel health are worse.

## Recommended Current Default

Until an `f=8` checkpoint has been audited with the same script and the same
split:

- keep downstream work on `64x64x4` first
- treat `f=8` as a candidate branch, not the default baseline

This avoids replacing a known bottleneck with an unknown one.

## Commands

Pick the monitored best checkpoint from a run directory:

```bash
python scripts/find_image_tokenizer_checkpoint.py \
  --config configs/autoencoder_image_24gb_lpips_256.yaml \
  --run-dir logs/autoencoder_image
```

Audit two direct checkpoints:

```bash
python scripts/audit_image_tokenizers.py \
  --candidate base_lpips|configs/autoencoder_image_24gb_lpips_256.yaml|/path/to/base_lpips.ckpt \
  --candidate f8_lpips|configs/autoencoder_image_f8_24gb_lpips_256.yaml|/path/to/f8_lpips.ckpt \
  --export-visuals \
  --outdir outputs/image_tokenizer_audit/current
```

Audit a run directory with monitor-aware selection:

```bash
python scripts/audit_image_tokenizers.py \
  --candidate-run base_lpips|configs/autoencoder_image_24gb_lpips_256.yaml|logs/autoencoder_image \
  --selection best \
  --export-visuals \
  --outdir outputs/image_tokenizer_audit/base_lpips_run
```

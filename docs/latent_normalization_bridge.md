# Latent Normalization Bridge

This note defines the explicit project-layer bridge between the image-only
tokenizer and downstream latent flow training for `p(image | semantic_mask)`.

## What The Bridge Does

The bridge sits between tokenizer encode/decode and the latent flow trainer:

- `encode`: tokenizer latent `z_raw -> z_train`
- `decode`: sampled or reconstructed `z_train -> z_raw -> RGB decode`

It is deliberately small and explicit:

- it does not change FM / MeanFlow / AlphaFlow math
- it does not touch `third_party/`
- it does not silently normalize every route
- it is enabled only when the config asks for it

Supported modes:

- `none`
- `global_affine`
- `per_channel_affine`

`per_channel_affine` is the recommended first bridge. It is a diagonal
whitening-style transform, not a full covariance whitening claim.

## When It Is Likely To Help

Use the bridge when tokenizer audit shows one or more of the following:

- large per-channel std imbalance
- partially collapsed latent channels
- latent std far from the Gaussian-like scale assumed by standard noise mixing

If tokenizer channels have very different scales, the flow trainer sees a
badly conditioned target space even if the objective itself is unchanged.

## Required Stats Artifact

Generate tokenizer latent stats with the same tokenizer config and checkpoint
that downstream training will actually use:

```bash
python scripts/eval_image_tokenizer.py \
  --config configs/autoencoder_image_256.yaml \
  --ckpt logs/autoencoder_image/checkpoints/last.ckpt \
  --outdir outputs/image_tokenizer_eval/autoencoder_image_256
```

This writes:

- `outputs/image_tokenizer_eval/autoencoder_image_256/summary.json`
- `outputs/image_tokenizer_eval/autoencoder_image_256/latent_stats.json`
- `outputs/image_tokenizer_eval/autoencoder_image_256/latent_stats/<name>.json`

The normalized downstream configs point at `latent_stats.json`.

The bridge loader can also read an existing tokenizer `summary.json` or audit
`summary.json`. If the file contains multiple candidates, set
`summary_name` explicitly in `latent_normalization_config`.

## First Comparison Plan

Keep everything fixed except the bridge:

- same tokenizer config
- same tokenizer checkpoint
- same downstream config family
- same seed
- same training budget

Recommended first pair:

- non-normalized: [configs/latent_alphaflow_mask2image_unet.yaml](/e:/CodeSpace/LarentMeanflow/configs/latent_alphaflow_mask2image_unet.yaml)
- normalized: [configs/latent_alphaflow_mask2image_unet_norm.yaml](/e:/CodeSpace/LarentMeanflow/configs/latent_alphaflow_mask2image_unet_norm.yaml)

Run the stats export first, then the normalized pilot:

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/latent_alphaflow_mask2image_unet_norm.yaml \
  --gpus 0
```

## How To Read The Result

The bridge is worth keeping if it improves one or more of:

- training stability
- `val/base_error_mean`
- few-step sampling at `NFE=4/2`
- decoded sharpness at fixed tokenizer

The bridge is not the right fix if:

- the tokenizer itself is still too blurry
- collapse is so severe that one channel needs extreme amplification even after
  the configured `std_floor`

In that case, fix the tokenizer first and rerun the stats export.

## Decoding Correctness

Decoding stays correct because the trainer now applies:

- normalize before flow training
- inverse affine denormalize before tokenizer decode

So the tokenizer itself still decodes the latent geometry it was trained on.

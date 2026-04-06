# LatentMeanflow Migration Plan

## Summary

This repository currently implements a legacy binary-mask / 4-channel latent diffusion baseline. The next research path is a paired RGB image + multiclass semantic mask latent flow model with the staged direction:

`multiclass data pipeline -> joint semantic tokenizer -> latent flow matching -> MeanFlow / AlphaFlow`

This document audits the current project-layer assumptions and records the migration plan without changing training logic or touching vendored runtime code.

## Current-State Audit

Audited files:

- `README.md`
- `configs/autoencoder_kl_4ch_256.yaml`
- `configs/ldm_4ch_256.yaml`
- `configs/paired_ddpm_256.yaml`
- `latent_meanflow/data/mask_image_pair.py`
- `scripts/sample_mask_image_pairs.py`
- `scripts/train_autoencoder.py`
- `scripts/train_ldm.py`

Observed dataset reality:

- A spot-check of `data/remote/train/masks` shows multiple discrete grayscale values: `36, 73, 109, 146, 182, 219, 255`.
- The current data is therefore already multiclass semantic in practice, while the legacy loader still collapses masks to binary.

| Current assumption | Where it appears | Why it mismatches the target | Planned migration direction |
| --- | --- | --- | --- |
| The task is described as binary-mask, class-conditional generation. | `README.md` | The target model should generate paired RGB image + multiclass semantic mask samples, not binary foreground/background masks. | Reframe the repo as a legacy baseline plus a future semantic latent-flow path. |
| The dataset loader converts every mask to grayscale, thresholds it to binary, and returns a packed `RGB + mask` tensor as `image`. | `latent_meanflow/data/mask_image_pair.py` | Thresholding destroys multiclass semantic information, and packing mask into channel 4 erases the distinction between image content and semantic labels. | Add a parallel semantic dataset that preserves a discrete `semantic_mask` and keeps it separate from `image_rgb`. |
| Optional `class_label` and `class_name` are derived from dataset roots and attached at sample level. | `latent_meanflow/data/mask_image_pair.py` | Image-level metadata is not the same thing as pixel-level semantic classes. | Keep `class_label` and `class_name` only as optional metadata in the future semantic path. |
| Autoencoder input and output are fixed to 4 channels. | `configs/autoencoder_kl_4ch_256.yaml` | This assumes a single packed tensor with `RGB + binary mask`. It does not represent a joint semantic pair contract. | Replace this with a future tokenizer design that treats `image_rgb` and `semantic_mask` as separate modalities. |
| The discriminator is configured for 4-channel inputs. | `configs/autoencoder_kl_4ch_256.yaml` | This is another packed-representation assumption tied to binary mask channel 4. | Redesign discriminator and reconstruction targets in the future tokenizer stage rather than extending the packed 4-channel baseline. |
| The latent diffusion model is defined over 4-channel latents. | `configs/ldm_4ch_256.yaml`, `configs/paired_ddpm_256.yaml` | The future model should operate on paired semantic latents, not on a packed binary-mask latent. | Introduce a future latent tokenizer contract first, then build latent FM on top of that contract. |
| The latent diffusion model uses `class_label` through `cond_stage_key` and `ClassLabeler`. | `configs/ldm_4ch_256.yaml` | This is image-level conditioning. It does not model pixel-level semantic structure. | Make semantic mask the primary paired signal and keep image-level labels optional. |
| Sampling assumes output channel layout `RGB(3) + mask(1)` and thresholds the mask again at save time. | `scripts/sample_mask_image_pairs.py` | This hardcodes binary-mask decoding and discards multiclass semantics after generation. | Build a future semantic sampler around explicit semantic mask tensors, not thresholded channel-4 output. |
| Training wrappers default to the legacy 4-channel configs. | `scripts/train_autoencoder.py`, `scripts/train_ldm.py` | The current CLI surface only exposes the legacy baseline. | Preserve these wrappers for the legacy path and add future semantic entrypoints in parallel later. |
| `paired_ddpm_256.yaml` remains a 4-channel DDPM baseline. | `configs/paired_ddpm_256.yaml` | It still encodes the same packed binary-mask assumption and is not a semantic pair model. | Keep it as a legacy baseline reference only. |

## Terminology and Contracts

### Core distinction

| Concept | Meaning | Scope | Status |
| --- | --- | --- | --- |
| `class_label` | Sample-level domain or category metadata for the whole example. | Image-level | Optional metadata only |
| `semantic_mask` | Per-pixel discrete semantic label map paired with the RGB image. | Pixel-level | Future primary signal |

`class_label != semantic_mask`

### Legacy contract

The current project-layer contract is:

- `image`: float tensor containing packed `RGB + binary mask`
- `mask`: binary mask derived by grayscale conversion and thresholding
- `class_label`: optional image-level metadata
- `class_name`: optional image-level metadata

This contract is useful for the checked-in baseline only.

### Future semantic contract

The planned project-layer semantic contract is:

- `image_rgb`: normalized RGB image tensor
- `semantic_mask`: single-channel discrete label map with integer semantic ids
- `class_label`: optional image-level metadata
- `class_name`: optional image-level metadata
- `label_spec`: explicit mapping from raw label values to semantic ids and names

The future semantic contract must preserve semantic classes without thresholding and without folding them into a binary channel.

## Phase Roadmap

### Phase A: Multiclass semantic data pipeline

- Add a parallel semantic dataset implementation without modifying the legacy dataset.
- Use an explicit label specification instead of inferring class definitions during training.
- Keep `image_rgb` and `semantic_mask` separate at the project-layer interface.
- Preserve `class_label` only as optional image-level metadata.

### Phase B: Joint semantic pair autoencoder / tokenizer

- Design a joint tokenizer for matched `image_rgb + semantic_mask` pairs.
- Stop treating semantic masks as a binary fourth channel.
- Keep packed representations, if any, as internal implementation details rather than project-layer contracts.

### Phase C: Latent flow matching baseline

- Build a latent flow matching baseline on top of the joint paired latent representation.
- Move the main modeling focus from image-level conditioning to paired semantic latent state.
- Re-evaluate vendored-runtime requirements only if the project layer cannot support the baseline cleanly.

### Phase D: MeanFlow / AlphaFlow extension

- Extend the latent FM baseline toward MeanFlow / AlphaFlow.
- Keep the paired latent contract stable.
- Keep the semantic label specification stable across the extension stages.

## Non-Goals of This Phase

- No new trainer implementation
- No changes to vendored runtime code
- No deletion or rewrite of the legacy binary baseline
- No semantic training config or sampling implementation yet

## Recommended First Code Patch

The first executable migration patch should add a parallel semantic dataset instead of editing the legacy binary loader:

- Add `latent_meanflow/data/semantic_mask_image_pair.py`
- Read same-stem image/mask pairs without thresholding the mask
- Return `image_rgb` and `semantic_mask` as separate fields
- Keep `class_label` and `class_name` optional
- Allow validation against an explicit label specification
- Leave `latent_meanflow/data/mask_image_pair.py` unchanged
- Leave the existing training scripts unchanged for now

## Risks and Compatibility Notes

- The existing configs and scripts should remain runnable as a legacy baseline, but they are not semantically correct for multiclass mask training.
- Any checkpoint trained on the current 4-channel packed representation should be treated as a legacy artifact, not as a drop-in initializer for a future semantic tokenizer.
- Future semantic work should default to project-layer additions first; vendored-runtime edits should happen only after project-layer options are exhausted.

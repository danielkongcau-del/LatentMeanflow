# Semantic Label Spec

## Status

This document defines the planned project-layer semantic mask contract. It is a specification only and is not implemented by the current legacy binary pipeline.

Current legacy behavior still thresholds masks to binary. Future semantic data code must follow this document instead of the legacy thresholding path.

## Semantic Contract

- `image_rgb`: normalized RGB image tensor
- `semantic_mask`: single-channel discrete label map with integer semantic ids
- `class_label`: optional image-level metadata
- `class_name`: optional image-level metadata
- `label_spec`: explicit mapping from raw grayscale values to semantic ids and semantic names

Rules:

- `semantic_mask` is not a binary alpha matte.
- `semantic_mask` must remain discrete and must not be thresholded.
- Project-layer data interfaces should not normalize `semantic_mask` into `[-1, 1]`.
- Project-layer data interfaces should not require one-hot encoding as the public contract.
- Image-level `class_label` cannot be used as a substitute for pixel-level semantic labels.

## Dataset Layout Assumption

The planned semantic path still assumes paired same-stem image/mask files:

```text
data/
  remote/
    train/
      images/
      masks/
    val/
      images/
      masks/
    test/
      images/
      masks/
```

The semantic interpretation of the mask is defined by `label_spec`, not by binary foreground/background thresholding.

## Label Table

The raw values below were observed in the current `data/remote` masks. Semantic names are placeholders for now and should be replaced with domain names before Phase A training starts.

Global defaults:

- Background class: none defined by default
- `ignore_index`: none

| Raw grayscale value | Semantic index | Semantic name | Is background | Notes |
| --- | --- | --- | --- | --- |
| 36 | 0 | `semantic_036` | `false` | Placeholder canonical name |
| 73 | 1 | `semantic_073` | `false` | Placeholder canonical name |
| 109 | 2 | `semantic_109` | `false` | Placeholder canonical name |
| 146 | 3 | `semantic_146` | `false` | Placeholder canonical name |
| 182 | 4 | `semantic_182` | `false` | Placeholder canonical name |
| 219 | 5 | `semantic_219` | `false` | Placeholder canonical name |
| 255 | 6 | `semantic_255` | `false` | Placeholder canonical name |

If a background class or `ignore_index` is introduced later, update this document before semantic training code is added.

## Planned Loader Expectations

Planned project-layer semantic loaders should:

- preserve raw label values until they are mapped through `label_spec`
- emit a discrete `semantic_mask`
- validate unknown or out-of-spec raw values
- keep `class_label` and `class_name` optional and independent from semantic labels

Planned, not implemented yet:

- a project-layer semantic dataset module
- semantic training configs
- semantic sampling scripts

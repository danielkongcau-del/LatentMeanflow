from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from latent_meanflow.conditioning import LatentConditioning
from latent_meanflow.trainers.latent_flow_trainer import LatentFlowTrainer
from latent_meanflow.trainers.latent_fm_trainer import LatentFMTrainer
from latent_meanflow.utils.palette import infer_num_classes, resolve_gray_to_class_id


def resolve_mask_condition_channels(*, semantic_mask_label_spec_path=None, mask_num_classes=None):
    if mask_num_classes is not None:
        return int(mask_num_classes)
    if semantic_mask_label_spec_path is None:
        return None
    label_spec_path = Path(semantic_mask_label_spec_path)
    gray_to_class_id, ignore_index = resolve_gray_to_class_id(label_spec_path)
    return int(infer_num_classes(gray_to_class_id, ignore_index=ignore_index))


def _patch_backbone_condition_channels(
    backbone_config,
    *,
    semantic_mask_label_spec_path=None,
    mask_num_classes=None,
):
    backbone_cfg = deepcopy(backbone_config)
    backbone_cfg.setdefault("params", {})
    resolved_channels = resolve_mask_condition_channels(
        semantic_mask_label_spec_path=semantic_mask_label_spec_path,
        mask_num_classes=mask_num_classes,
    )
    if resolved_channels is not None:
        backbone_cfg["params"]["spatial_condition_channels"] = int(resolved_channels)
    return backbone_cfg


class _MaskConditionedImageMixin:
    def _init_mask_conditioning(
        self,
        *,
        mask_condition_key="mask_onehot",
        mask_index_key="mask_index",
        condition_source="latent_resized_mask",
    ):
        self.mask_condition_key = str(mask_condition_key)
        self.mask_index_key = str(mask_index_key)
        self.condition_source = str(condition_source)
        if self.condition_source not in {"latent_resized_mask", "fullres_mask"}:
            raise ValueError(
                f"Unsupported condition_source={self.condition_source!r}. "
                "Expected 'latent_resized_mask' or 'fullres_mask'."
            )
        spatial_condition_channels = int(getattr(self.backbone, "spatial_condition_channels", 0))
        if spatial_condition_channels <= 0:
            raise ValueError(
                "Mask-conditioned image trainers require a backbone with "
                "spatial_condition_channels > 0."
            )
        self.mask_condition_channels = spatial_condition_channels
        # For image|mask logging, num_classes refers to semantic mask classes,
        # not tokenizer classes. The image-only tokenizer keeps num_classes=0.
        self.num_classes = self.mask_condition_channels

    def _get_log_batch_size(self, batch):
        return int(batch["image"].shape[0])

    def _prepare_mask_onehot(self, mask_onehot):
        if not isinstance(mask_onehot, torch.Tensor):
            mask_onehot = torch.as_tensor(mask_onehot)
        if mask_onehot.ndim != 4:
            raise ValueError(
                f"Expected mask_onehot with rank 4, got shape {tuple(mask_onehot.shape)}"
            )

        if mask_onehot.shape[1] == self.mask_condition_channels:
            spatial = mask_onehot
        elif mask_onehot.shape[-1] == self.mask_condition_channels:
            spatial = mask_onehot.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "mask_onehot channel mismatch: expected either BCHW or BHWC with "
                f"{self.mask_condition_channels} channels, got shape {tuple(mask_onehot.shape)}"
            )
        return spatial.float().contiguous()

    def build_condition_from_mask_onehot(self, mask_onehot, *, device=None, dtype=torch.float32):
        fullres_spatial = self._prepare_mask_onehot(mask_onehot)
        spatial = fullres_spatial
        if spatial.shape[-2:] != self.latent_spatial_shape:
            spatial = F.interpolate(spatial, size=self.latent_spatial_shape, mode="nearest")
        if device is None:
            device = spatial.device
        spatial = spatial.to(device=device, dtype=dtype)
        if self.condition_source == "latent_resized_mask":
            return LatentConditioning(spatial=spatial)
        fullres_spatial = fullres_spatial.to(device=device, dtype=dtype)
        return LatentConditioning(spatial=spatial, spatial_fullres=fullres_spatial)

    def _get_condition(self, batch):
        if self.mask_condition_key not in batch:
            raise KeyError(
                "Mask-conditioned image generation requires "
                f"'{self.mask_condition_key}' in the batch."
            )
        return self.build_condition_from_mask_onehot(batch[self.mask_condition_key])

    def _prepare_condition(self, condition, *, device, dtype):
        if condition is None:
            return None
        if isinstance(condition, LatentConditioning):
            return condition.to(device=device, dtype=dtype)
        if isinstance(condition, dict):
            return LatentConditioning(
                class_label=condition.get("class_label"),
                spatial=condition.get("spatial"),
                spatial_fullres=condition.get("spatial_fullres"),
            ).to(device=device, dtype=dtype)
        if isinstance(condition, torch.Tensor):
            return LatentConditioning(spatial=condition).to(device=device, dtype=dtype)
        raise TypeError(f"Unsupported condition type for mask-conditioned route: {type(condition)}")

    def predict_field(self, z_t, t=None, condition=None, r=None, delta_t=None):
        condition = self._prepare_condition(condition, device=z_t.device, dtype=z_t.dtype)
        r = self._prepare_time(r, device=z_t.device, dtype=z_t.dtype)
        t = self._prepare_time(t, device=z_t.device, dtype=z_t.dtype)
        delta_t = self._prepare_time(delta_t, device=z_t.device, dtype=z_t.dtype)
        return self.backbone(z_t, t=t, condition=condition, r=r, delta_t=delta_t)

    @torch.no_grad()
    def log_images(self, batch, split="train", **kwargs):
        encoded = self.encode_batch(batch)
        decoded_inputs = self.decode_latents(encoded["z"])
        batch_size = min(encoded["z"].shape[0], kwargs.get("max_images", 4))
        condition = self._get_condition(batch)
        if condition is not None:
            condition = condition[:batch_size]
        sampled_latents = self.sample_latents(
            batch_size=batch_size,
            nfe=getattr(self, "log_sample_nfe", None),
            num_steps=getattr(self, "log_sample_steps", None),
            device=self.device,
            condition=condition,
        )
        sampled = self.decode_latents(sampled_latents)
        return {
            "inputs_image": encoded["image"][:batch_size].to(self.device),
            "inputs_mask_index": batch[self.mask_index_key][:batch_size].unsqueeze(1).float().to(self.device),
            "reconstructions_image": decoded_inputs["rgb_recon"][:batch_size],
            "samples_image": sampled["rgb_recon"][:batch_size],
        }


class MaskConditionedLatentFlowTrainer(_MaskConditionedImageMixin, LatentFlowTrainer):
    def __init__(
        self,
        tokenizer_config_path,
        tokenizer_ckpt_path,
        backbone_config,
        objective_config,
        sampler_config,
        objective_name=None,
        sample_posterior=False,
        freeze_tokenizer=True,
        log_sample_nfe=2,
        latent_normalization_config=None,
        monitor="val/loss",
        mask_condition_key="mask_onehot",
        mask_index_key="mask_index",
        condition_source="latent_resized_mask",
        semantic_mask_label_spec_path=None,
        mask_num_classes=None,
    ):
        backbone_config = _patch_backbone_condition_channels(
            backbone_config,
            semantic_mask_label_spec_path=semantic_mask_label_spec_path,
            mask_num_classes=mask_num_classes,
        )
        super().__init__(
            tokenizer_config_path=tokenizer_config_path,
            tokenizer_ckpt_path=tokenizer_ckpt_path,
            backbone_config=backbone_config,
            objective_config=objective_config,
            sampler_config=sampler_config,
            objective_name=objective_name,
            sample_posterior=sample_posterior,
            freeze_tokenizer=freeze_tokenizer,
            use_class_condition=False,
            class_label_key="class_label",
            log_sample_nfe=log_sample_nfe,
            latent_normalization_config=latent_normalization_config,
            monitor=monitor,
        )
        self._init_mask_conditioning(
            mask_condition_key=mask_condition_key,
            mask_index_key=mask_index_key,
            condition_source=condition_source,
        )
        backbone_condition_source = str(getattr(self.backbone, "condition_source", self.condition_source))
        if backbone_condition_source != self.condition_source:
            raise ValueError(
                "Mask-conditioned trainer/backbone condition_source mismatch: "
                f"trainer={self.condition_source!r}, backbone={backbone_condition_source!r}"
            )
        self.semantic_mask_label_spec_path = semantic_mask_label_spec_path
        if getattr(self.tokenizer, "num_classes", 0) != 0:
            raise ValueError(
                "Mask-conditioned image generation expects an image-only tokenizer with num_classes=0."
            )

    def sample_latents(self, batch_size, nfe=None, device=None, condition=None, noise=None, num_steps=None):
        if num_steps is not None and nfe is None:
            nfe = num_steps
        if device is None:
            device = self.device
        condition = self._prepare_condition(condition, device=device, dtype=torch.float32)
        latent_shape = (self.latent_channels, *self.latent_spatial_shape)
        return self.sampler.sample(
            model_fn=self.predict_field,
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=device,
            condition=condition,
            noise=noise,
            nfe=nfe,
        )


class MaskConditionedLatentFMTrainer(_MaskConditionedImageMixin, LatentFMTrainer):
    def __init__(
        self,
        tokenizer_config_path,
        tokenizer_ckpt_path,
        backbone_config,
        objective_config,
        sample_posterior=False,
        freeze_tokenizer=True,
        log_sample_steps=16,
        monitor="val/fm_loss",
        mask_condition_key="mask_onehot",
        mask_index_key="mask_index",
        condition_source="latent_resized_mask",
        semantic_mask_label_spec_path=None,
        mask_num_classes=None,
    ):
        backbone_config = _patch_backbone_condition_channels(
            backbone_config,
            semantic_mask_label_spec_path=semantic_mask_label_spec_path,
            mask_num_classes=mask_num_classes,
        )
        super().__init__(
            tokenizer_config_path=tokenizer_config_path,
            tokenizer_ckpt_path=tokenizer_ckpt_path,
            backbone_config=backbone_config,
            objective_config=objective_config,
            sample_posterior=sample_posterior,
            freeze_tokenizer=freeze_tokenizer,
            use_class_condition=False,
            class_label_key="class_label",
            log_sample_steps=log_sample_steps,
            monitor=monitor,
        )
        self._init_mask_conditioning(
            mask_condition_key=mask_condition_key,
            mask_index_key=mask_index_key,
            condition_source=condition_source,
        )
        backbone_condition_source = str(getattr(self.backbone, "condition_source", self.condition_source))
        if backbone_condition_source != self.condition_source:
            raise ValueError(
                "Mask-conditioned trainer/backbone condition_source mismatch: "
                f"trainer={self.condition_source!r}, backbone={backbone_condition_source!r}"
            )
        self.semantic_mask_label_spec_path = semantic_mask_label_spec_path
        if getattr(self.tokenizer, "num_classes", 0) != 0:
            raise ValueError(
                "Mask-conditioned image generation expects an image-only tokenizer with num_classes=0."
            )

    def sample_latents(self, batch_size, num_steps=32, device=None, condition=None, noise=None, nfe=None):
        if device is None:
            device = self.device
        if nfe is not None:
            num_steps = int(nfe)
        condition = self._prepare_condition(condition, device=device, dtype=torch.float32)
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.latent_channels,
                *self.latent_spatial_shape,
                device=device,
            )
        z = noise
        time_grid = torch.linspace(1.0, 0.0, int(num_steps) + 1, device=device)
        for step_idx in range(len(time_grid) - 1):
            t_curr = time_grid[step_idx].expand(batch_size)
            delta_t = time_grid[step_idx + 1] - time_grid[step_idx]
            velocity = self.predict_field(z, t=t_curr, condition=condition)
            z = z + delta_t * velocity
        return z

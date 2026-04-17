from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config

from latent_meanflow.utils.palette import infer_num_classes, resolve_gray_to_class_id


def resolve_discrete_mask_prior_metadata(*, semantic_mask_label_spec_path=None, mask_num_classes=None):
    if semantic_mask_label_spec_path is not None:
        gray_to_class_id, ignore_index = resolve_gray_to_class_id(Path(semantic_mask_label_spec_path))
        return {
            "num_classes": int(infer_num_classes(gray_to_class_id, ignore_index=ignore_index)),
            "ignore_index": None if ignore_index is None else int(ignore_index),
        }
    if mask_num_classes is None:
        raise ValueError(
            "DiscreteMaskPriorTrainer requires either semantic_mask_label_spec_path or mask_num_classes."
        )
    return {
        "num_classes": int(mask_num_classes),
        "ignore_index": None,
    }


def _normalize_spatial_shape(mask_spatial_shape):
    if isinstance(mask_spatial_shape, int):
        edge = int(mask_spatial_shape)
        return (edge, edge)
    if isinstance(mask_spatial_shape, (str, bytes)) or not isinstance(mask_spatial_shape, Sequence):
        raise ValueError(
            "mask_spatial_shape must be an int or a 2-item sequence, "
            f"got {mask_spatial_shape!r}"
        )
    values = tuple(mask_spatial_shape)
    if len(values) != 2:
        raise ValueError(
            "mask_spatial_shape must be an int or a 2-item sequence, "
            f"got {mask_spatial_shape!r}"
        )
    height = int(values[0])
    width = int(values[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"mask_spatial_shape must be positive, got {(height, width)}")
    return (height, width)


class DiscreteMaskPriorTrainer(pl.LightningModule):
    def __init__(
        self,
        backbone_config,
        objective_config,
        sampler_config,
        objective_name=None,
        mask_key="mask_onehot",
        mask_index_key="mask_index",
        semantic_mask_label_spec_path=None,
        mask_num_classes=None,
        mask_spatial_shape=(256, 256),
        log_sample_nfe=4,
        monitor="val/base_error_mean",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_config", "objective_config", "sampler_config"])

        self.mask_key = str(mask_key)
        self.mask_index_key = str(mask_index_key)
        self.log_sample_nfe = int(log_sample_nfe)
        self.monitor = str(monitor)
        self.learning_rate = 1.0e-4
        self.semantic_mask_label_spec_path = semantic_mask_label_spec_path
        self.mask_spatial_shape = _normalize_spatial_shape(mask_spatial_shape)

        metadata = resolve_discrete_mask_prior_metadata(
            semantic_mask_label_spec_path=semantic_mask_label_spec_path,
            mask_num_classes=mask_num_classes,
        )
        self.num_classes = int(metadata["num_classes"])
        self.ignore_index = metadata["ignore_index"]
        self.mask_token_id = int(self.num_classes)
        self.model_input_channels = int(self.num_classes + 1)
        self.model_output_channels = int(self.num_classes)
        # Keep the legacy latent_* names only for script compatibility. This route does not use compressed latents.
        # The modeled state is discrete mask_index. mask_onehot is only an auxiliary view.
        self.latent_channels = 1
        self.latent_spatial_shape = tuple(self.mask_spatial_shape)

        backbone_cfg = deepcopy(backbone_config)
        backbone_cfg.setdefault("params", {})
        if int(backbone_cfg["params"].get("spatial_condition_channels", 0)) > 0:
            raise ValueError("DiscreteMaskPriorTrainer is unconditional and must not use spatial conditioning.")
        if backbone_cfg["params"].get("condition_num_classes") is not None:
            raise ValueError("DiscreteMaskPriorTrainer baseline does not use image-level class conditioning.")
        backbone_cfg["params"]["in_channels"] = int(self.model_input_channels)
        backbone_cfg["params"]["out_channels"] = int(self.model_output_channels)

        self.backbone = instantiate_from_config(backbone_cfg)
        self.objective = instantiate_from_config(objective_config)
        self.sampler = instantiate_from_config(sampler_config)
        self.objective_name = (
            str(objective_name)
            if objective_name is not None
            else getattr(self.objective, "name", self.objective.__class__.__name__.lower())
        )

        for module in (self.objective, self.sampler):
            configure_state = getattr(module, "configure_discrete_state", None)
            if callable(configure_state):
                configure_state(
                    num_classes=self.num_classes,
                    mask_token_id=self.mask_token_id,
                    ignore_index=self.ignore_index,
                )

    def _get_log_batch_size(self, batch):
        if self.mask_index_key not in batch:
            raise KeyError(
                f"DiscreteMaskPriorTrainer discrete route requires '{self.mask_index_key}' in the batch. "
                f"'{self.mask_key}' is auxiliary only and cannot recover the discrete state."
            )
        return int(batch[self.mask_index_key].shape[0])

    def _should_sync_dist(self):
        trainer = getattr(self, "trainer", None)
        return trainer is not None and int(getattr(trainer, "world_size", 1)) > 1

    def on_fit_start(self):
        super().on_fit_start()
        self.print(
            "[DiscreteMaskPriorTrainer] "
            f"num_classes={self.num_classes}, mask_token_id={self.mask_token_id}, "
            f"mask_spatial_shape={self.mask_spatial_shape}, objective={self.objective_name}"
        )

    def _prepare_mask_onehot(self, mask_onehot):
        if not isinstance(mask_onehot, torch.Tensor):
            mask_onehot = torch.as_tensor(mask_onehot)
        if mask_onehot.ndim != 4:
            raise ValueError(
                f"Expected mask_onehot with rank 4, got shape {tuple(mask_onehot.shape)}"
            )

        if mask_onehot.shape[1] == self.num_classes:
            spatial = mask_onehot
        elif mask_onehot.shape[-1] == self.num_classes:
            spatial = mask_onehot.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "mask_onehot channel mismatch: expected either BCHW or BHWC with "
                f"{self.num_classes} channels, got shape {tuple(mask_onehot.shape)}"
            )

        if tuple(spatial.shape[-2:]) != self.mask_spatial_shape:
            spatial = F.interpolate(
                spatial.float(),
                size=self.mask_spatial_shape,
                mode="nearest",
            )
        return spatial.float().contiguous()

    def _prepare_mask_index(self, mask_index):
        if not isinstance(mask_index, torch.Tensor):
            mask_index = torch.as_tensor(mask_index)
        if mask_index.ndim == 4 and mask_index.shape[1] == 1:
            mask_index = mask_index[:, 0]
        if mask_index.ndim != 3:
            raise ValueError(
                f"Expected mask_index with shape [B, H, W] or [B, 1, H, W], got {tuple(mask_index.shape)}"
            )
        if tuple(mask_index.shape[-2:]) != self.mask_spatial_shape:
            mask_index = F.interpolate(
                mask_index.unsqueeze(1).float(),
                size=self.mask_spatial_shape,
                mode="nearest",
            )[:, 0]
        return mask_index.long().contiguous()

    def _build_valid_mask(self, mask_index):
        valid_mask = torch.ones_like(mask_index, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask &= mask_index != int(self.ignore_index)
        return valid_mask

    def _validate_auxiliary_mask_onehot(self, mask_index, mask_onehot):
        if mask_index.ndim != 3:
            raise ValueError(
                f"_validate_auxiliary_mask_onehot expects mask_index [B, H, W], got {tuple(mask_index.shape)}"
            )
        if mask_onehot.ndim != 4 or mask_onehot.shape[1] != self.num_classes:
            raise ValueError(
                "_validate_auxiliary_mask_onehot expects mask_onehot [B, K, H, W] with "
                f"K={self.num_classes}, got {tuple(mask_onehot.shape)}"
            )

        valid_mask = self._build_valid_mask(mask_index)
        if not torch.any(valid_mask):
            return

        derived_mask_index = torch.argmax(mask_onehot, dim=1)
        channel_sum = mask_onehot.sum(dim=1)
        expected_sum = torch.ones_like(channel_sum, dtype=mask_onehot.dtype)
        valid_channel_mass = torch.isclose(
            channel_sum[valid_mask],
            expected_sum[valid_mask],
            atol=1.0e-4,
            rtol=0.0,
        )
        valid_class_match = derived_mask_index[valid_mask] == mask_index[valid_mask]

        if torch.all(valid_channel_mass) and torch.all(valid_class_match):
            return

        invalid_mass_count = int((~valid_channel_mass).sum().item())
        disagree_count = int((~valid_class_match).sum().item())
        raise ValueError(
            "DiscreteMaskPriorTrainer batch simultaneously provided mask_index and mask_onehot, "
            "but they disagree on valid pixels. Discrete route requires them to be consistent. "
            f"invalid_onehot_mass_pixels={invalid_mass_count}, disagree_pixels={disagree_count}"
        )

    def encode_batch(self, batch):
        if self.mask_index_key not in batch:
            raise KeyError(
                f"DiscreteMaskPriorTrainer discrete route requires '{self.mask_index_key}' in the batch. "
                f"'{self.mask_key}' is auxiliary only and must not be used to recover mask_index."
            )
        mask_index = self._prepare_mask_index(batch[self.mask_index_key])

        if self.mask_key in batch:
            mask_onehot = self._prepare_mask_onehot(batch[self.mask_key])
            self._validate_auxiliary_mask_onehot(mask_index, mask_onehot)
        else:
            safe_mask_index = mask_index.clone()
            safe_mask_index = safe_mask_index.masked_fill(safe_mask_index < 0, 0)
            if self.ignore_index is not None:
                safe_mask_index = safe_mask_index.masked_fill(
                    mask_index == int(self.ignore_index),
                    0,
                )
            mask_onehot = F.one_hot(safe_mask_index, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            if self.ignore_index is not None:
                valid_mask = (mask_index != int(self.ignore_index)).unsqueeze(1)
                mask_onehot = mask_onehot * valid_mask.to(dtype=mask_onehot.dtype)

        return {
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
        }

    def _prepare_time(self, value, device):
        if value is None:
            return None
        return value.to(device=device, dtype=torch.float32)

    def _discrete_to_onehot(self, z_t):
        if z_t.ndim == 4 and z_t.shape[1] == self.model_input_channels:
            return z_t.to(dtype=torch.float32)
        if z_t.ndim == 4 and z_t.shape[1] == 1:
            z_t = z_t[:, 0]
        if z_t.ndim != 3:
            raise ValueError(
                "DiscreteMaskPriorTrainer.predict_field expects mask indices [B, H, W] or "
                f"one-hot [B, {self.model_input_channels}, H, W], got {tuple(z_t.shape)}"
            )
        if torch.any(z_t < 0):
            raise ValueError("DiscreteMaskPriorTrainer.predict_field received negative mask indices.")
        if torch.any(z_t > self.mask_token_id):
            raise ValueError(
                f"DiscreteMaskPriorTrainer.predict_field received indices above MASK token id {self.mask_token_id}."
            )
        onehot = F.one_hot(z_t.to(dtype=torch.long), num_classes=self.model_input_channels)
        return onehot.permute(0, 3, 1, 2).to(dtype=torch.float32)

    def predict_field(self, z_t, t=None, condition=None, r=None, delta_t=None):
        if condition is not None:
            raise ValueError("DiscreteMaskPriorTrainer baseline is unconditional and does not accept condition.")
        z_t_onehot = self._discrete_to_onehot(z_t).to(device=self.device)
        t = self._prepare_time(t, device=z_t_onehot.device)
        r = self._prepare_time(r, device=z_t_onehot.device)
        delta_t = self._prepare_time(delta_t, device=z_t_onehot.device)
        return self.backbone(z_t_onehot, t=t, condition=None, r=r, delta_t=delta_t)

    def decode_latents(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z)
        if z.ndim == 4 and z.shape[1] == 1:
            z = z[:, 0]
        if z.ndim != 3:
            raise ValueError(
                f"DiscreteMaskPriorTrainer.decode_latents expects [B, H, W], got {tuple(z.shape)}"
            )
        mask_index = z.to(dtype=torch.long)
        if torch.any(mask_index < 0) or torch.any(mask_index >= self.num_classes):
            raise ValueError(
                f"DiscreteMaskPriorTrainer.decode_latents expects class ids in [0, {self.num_classes - 1}], "
                f"got range [{int(mask_index.min().item())}, {int(mask_index.max().item())}]"
            )
        mask_onehot = F.one_hot(mask_index, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        return {
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
        }

    def sample_latents(self, batch_size, nfe=None, device=None, condition=None, noise=None):
        if condition is not None:
            raise ValueError("DiscreteMaskPriorTrainer baseline is unconditional and does not accept condition.")
        if device is None:
            device = self.device
        compatibility_shape = (self.latent_channels, *self.latent_spatial_shape)
        return self.sampler.sample(
            model_fn=self.predict_field,
            batch_size=batch_size,
            latent_shape=compatibility_shape,
            device=device,
            condition=None,
            noise=noise,
            nfe=nfe,
        )

    def forward(self, batch, objective_step=None):
        encoded = self.encode_batch(batch)
        mask_index = encoded["mask_index"]
        if objective_step is None:
            objective_step = int(self.global_step)
        objective_outputs = self.objective(
            self.predict_field,
            mask_index,
            condition=None,
            global_step=objective_step,
            target_model_fn=self.predict_field,
        )
        class_mass = encoded["mask_onehot"].detach().mean(dim=(0, 2, 3))
        mask_bridge_metrics = {
            "mask_class_mass_min": class_mass.min(),
            "mask_class_mass_max": class_mass.max(),
        }
        return {
            "mask_index": mask_index,
            "mask_onehot": encoded["mask_onehot"],
            "mask_bridge_metrics": mask_bridge_metrics,
            **objective_outputs,
        }

    def _collect_log_scalars(self, split, loss_dict):
        metrics = {}
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    continue
                metrics[f"{split}/{name}"] = value.detach()
            elif isinstance(value, (int, float)):
                metrics[f"{split}/{name}"] = torch.tensor(float(value), device=self.device)
        return metrics

    def shared_step(self, batch, split):
        outputs = self(batch, objective_step=int(self.global_step))
        loss_dict = outputs.get("loss_dict", {"loss": outputs["loss"], "total_loss": outputs["loss"]})
        metrics = self._collect_log_scalars(split, loss_dict)
        metrics.update(self._collect_log_scalars(split, outputs.get("mask_bridge_metrics", {})))
        prog_bar_metrics = {}
        base_error_key = f"{split}/base_error_mean"
        if base_error_key in metrics:
            prog_bar_metrics[base_error_key] = metrics.pop(base_error_key)
        batch_size = self._get_log_batch_size(batch)
        sync_dist = self._should_sync_dist()
        self.log(
            f"{split}/loss",
            outputs["loss"].detach(),
            prog_bar=True,
            logger=True,
            on_step=(split == "train"),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        if prog_bar_metrics:
            self.log_dict(
                prog_bar_metrics,
                prog_bar=True,
                logger=True,
                on_step=(split == "train"),
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )
        if metrics:
            self.log_dict(
                metrics,
                prog_bar=False,
                logger=True,
                on_step=(split == "train"),
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )
        return outputs["loss"], metrics.get(f"{split}/loss", outputs["loss"].detach())

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, split="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, detached_loss = self.shared_step(batch, split="val")
        return detached_loss

    def configure_optimizers(self):
        lr = float(getattr(self, "learning_rate", 1.0e-4))
        return torch.optim.Adam(self.backbone.parameters(), lr=lr, betas=(0.9, 0.999))

    @torch.no_grad()
    def log_images(self, batch, split="train", **kwargs):
        encoded = self.encode_batch(batch)
        batch_size = min(encoded["mask_index"].shape[0], kwargs.get("max_images", 4))
        sampled_latents = self.sample_latents(
            batch_size=batch_size,
            nfe=self.log_sample_nfe,
            device=self.device,
        )
        sampled = self.decode_latents(sampled_latents)
        return {
            "inputs_mask_index": encoded["mask_index"][:batch_size].unsqueeze(1).float().to(self.device),
            "samples_mask_index": sampled["mask_index"][:batch_size].unsqueeze(1).float(),
        }

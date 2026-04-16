import math
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config

from latent_meanflow.utils.palette import infer_num_classes, resolve_gray_to_class_id


def resolve_mask_prior_num_classes(*, semantic_mask_label_spec_path=None, mask_num_classes=None):
    if mask_num_classes is not None:
        return int(mask_num_classes)
    if semantic_mask_label_spec_path is None:
        return None
    gray_to_class_id, ignore_index = resolve_gray_to_class_id(Path(semantic_mask_label_spec_path))
    return int(infer_num_classes(gray_to_class_id, ignore_index=ignore_index))


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


class MaskPriorTrainer(pl.LightningModule):
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
        monitor="val/loss",
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

        resolved_num_classes = resolve_mask_prior_num_classes(
            semantic_mask_label_spec_path=semantic_mask_label_spec_path,
            mask_num_classes=mask_num_classes,
        )
        if resolved_num_classes is None:
            raise ValueError(
                "MaskPriorTrainer requires either semantic_mask_label_spec_path or mask_num_classes."
            )
        self.num_classes = int(resolved_num_classes)
        self.latent_channels = int(self.num_classes)
        self.latent_spatial_shape = tuple(self.mask_spatial_shape)

        backbone_cfg = deepcopy(backbone_config)
        backbone_cfg.setdefault("params", {})
        if int(backbone_cfg["params"].get("spatial_condition_channels", 0)) > 0:
            raise ValueError("MaskPriorTrainer is unconditional and must not use spatial conditioning.")
        if backbone_cfg["params"].get("condition_num_classes") is not None:
            raise ValueError("MaskPriorTrainer baseline does not use image-level class conditioning.")
        backbone_cfg["params"]["in_channels"] = int(self.num_classes)

        self.backbone = instantiate_from_config(backbone_cfg)
        self.objective = instantiate_from_config(objective_config)
        self.sampler = instantiate_from_config(sampler_config)
        self.objective_name = (
            str(objective_name)
            if objective_name is not None
            else getattr(self.objective, "name", self.objective.__class__.__name__.lower())
        )

    def _get_log_batch_size(self, batch):
        return int(batch[self.mask_key].shape[0])

    def _should_sync_dist(self):
        trainer = getattr(self, "trainer", None)
        return trainer is not None and int(getattr(trainer, "world_size", 1)) > 1

    def _prepare_time(self, value, device, dtype):
        if value is None:
            return None
        return value.to(device=device, dtype=dtype)

    def _configure_objective_training_budget(self):
        objective = getattr(self, "objective", None)
        set_budget = getattr(objective, "set_training_budget", None)
        if set_budget is None:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return

        total_steps = getattr(trainer, "estimated_stepping_batches", None)
        try:
            total_steps = None if total_steps is None else int(total_steps)
        except (TypeError, ValueError):
            total_steps = None
        if total_steps is not None and total_steps <= 0:
            total_steps = None

        total_epochs = getattr(trainer, "max_epochs", None)
        try:
            total_epochs = None if total_epochs is None else int(total_epochs)
        except (TypeError, ValueError):
            total_epochs = None
        if total_epochs is not None and total_epochs <= 0:
            total_epochs = None

        optimizer_steps_per_epoch = None
        if total_steps is not None and total_epochs is not None:
            optimizer_steps_per_epoch = max(1, int(math.ceil(total_steps / float(total_epochs))))

        set_budget(
            total_steps=total_steps,
            total_epochs=total_epochs,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        )

    def on_fit_start(self):
        super().on_fit_start()
        self._configure_objective_training_budget()
        self.print(
            "[MaskPriorTrainer] "
            f"num_classes={self.num_classes}, mask_spatial_shape={self.mask_spatial_shape}, "
            f"objective={self.objective_name}"
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

    def encode_batch(self, batch):
        if self.mask_key not in batch:
            raise KeyError(f"MaskPriorTrainer requires '{self.mask_key}' in the batch.")
        x_mask = self._prepare_mask_onehot(batch[self.mask_key])
        if self.mask_index_key in batch:
            mask_index = self._prepare_mask_index(batch[self.mask_index_key])
        else:
            mask_index = torch.argmax(x_mask, dim=1)
        return {
            "z": x_mask,
            "mask_index": mask_index,
        }

    def predict_field(self, z_t, t=None, condition=None, r=None, delta_t=None):
        if condition is not None:
            raise ValueError("MaskPriorTrainer baseline is unconditional and does not accept condition.")
        r = self._prepare_time(r, device=z_t.device, dtype=z_t.dtype)
        t = self._prepare_time(t, device=z_t.device, dtype=z_t.dtype)
        delta_t = self._prepare_time(delta_t, device=z_t.device, dtype=z_t.dtype)
        return self.backbone(z_t, t=t, condition=None, r=r, delta_t=delta_t)

    def decode_latents(self, z):
        mask_index = torch.argmax(z, dim=1)
        mask_onehot = F.one_hot(mask_index, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        return {
            "mask_logits": z,
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
        }

    def sample_latents(self, batch_size, nfe=None, device=None, condition=None, noise=None):
        if condition is not None:
            raise ValueError("MaskPriorTrainer baseline is unconditional and does not accept condition.")
        if device is None:
            device = self.device
        latent_shape = (self.latent_channels, *self.latent_spatial_shape)
        return self.sampler.sample(
            model_fn=self.predict_field,
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=device,
            condition=None,
            noise=noise,
            nfe=nfe,
        )

    def forward(self, batch, objective_step=None):
        encoded = self.encode_batch(batch)
        x_mask = encoded["z"]
        if objective_step is None:
            objective_step = int(self.global_step)
        objective_outputs = self.objective(
            self.predict_field,
            x_mask,
            condition=None,
            global_step=objective_step,
            target_model_fn=self.predict_field,
        )
        class_mass = x_mask.detach().mean(dim=(0, 2, 3))
        mask_bridge_metrics = {
            "mask_input_mean": x_mask.detach().mean(),
            "mask_input_std": x_mask.detach().std(unbiased=False),
            "mask_channel_mass_min": class_mass.min(),
            "mask_channel_mass_max": class_mass.max(),
        }
        return {
            "x_mask": x_mask,
            "mask_index": encoded["mask_index"],
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
        batch_size = min(encoded["z"].shape[0], kwargs.get("max_images", 4))
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

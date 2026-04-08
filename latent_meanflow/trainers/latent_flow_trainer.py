import math
from copy import deepcopy

import pytorch_lightning as pl
import torch
from ldm.util import instantiate_from_config

from latent_meanflow.models.tokenizer import SemanticTokenizerAdapter


class LatentFlowTrainer(pl.LightningModule):
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
        use_class_condition=False,
        class_label_key="class_label",
        log_sample_nfe=2,
        monitor="val/loss",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_config", "objective_config", "sampler_config"])

        self.sample_posterior = bool(sample_posterior)
        self.freeze_tokenizer = bool(freeze_tokenizer)
        self.use_class_condition = bool(use_class_condition)
        self.class_label_key = str(class_label_key)
        self.log_sample_nfe = int(log_sample_nfe)
        self.monitor = str(monitor)
        self.learning_rate = 1.0e-4

        self.tokenizer = SemanticTokenizerAdapter.from_pretrained(
            config_path=tokenizer_config_path,
            ckpt_path=tokenizer_ckpt_path,
            eval_mode=True,
            freeze=self.freeze_tokenizer,
        )
        self.latent_channels = self.tokenizer.latent_channels
        self.latent_spatial_shape = self.tokenizer.latent_spatial_shape
        self.num_classes = self.tokenizer.num_classes

        backbone_cfg = deepcopy(backbone_config)
        backbone_cfg.setdefault("params", {})
        backbone_cfg["params"]["in_channels"] = self.latent_channels
        if self.use_class_condition:
            backbone_cfg["params"]["condition_num_classes"] = self.num_classes
        self.backbone = instantiate_from_config(backbone_cfg)
        self.objective = instantiate_from_config(objective_config)
        self.sampler = instantiate_from_config(sampler_config)
        self.objective_name = (
            str(objective_name)
            if objective_name is not None
            else getattr(self.objective, "name", self.objective.__class__.__name__.lower())
        )

    def _get_condition(self, batch):
        if not self.use_class_condition:
            return None
        if self.class_label_key not in batch:
            raise KeyError(
                f"use_class_condition=True but batch does not contain '{self.class_label_key}'"
            )
        return batch[self.class_label_key].long()

    def _get_log_batch_size(self, batch):
        return int(batch["image"].shape[0])

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

    def predict_field(self, z_t, t=None, condition=None, r=None, delta_t=None):
        if condition is not None:
            condition = condition.to(device=z_t.device, dtype=torch.long)
        r = self._prepare_time(r, device=z_t.device, dtype=z_t.dtype)
        t = self._prepare_time(t, device=z_t.device, dtype=z_t.dtype)
        delta_t = self._prepare_time(delta_t, device=z_t.device, dtype=z_t.dtype)
        return self.backbone(z_t, t=t, condition=condition, r=r, delta_t=delta_t)

    def encode_batch(self, batch):
        return self.tokenizer.encode_batch(batch, sample_posterior=self.sample_posterior)

    def decode_latents(self, z):
        return self.tokenizer.decode_latents(z)

    def sample_latents(self, batch_size, nfe=None, device=None, condition=None, noise=None):
        if device is None:
            device = self.device
        if condition is not None:
            condition = condition.to(device=device, dtype=torch.long)
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

    def forward(self, batch, objective_step=None):
        encoded = self.encode_batch(batch)
        x_lat = encoded["z"]
        condition = self._get_condition(batch)
        if objective_step is None:
            objective_step = int(self.global_step)
        objective_outputs = self.objective(
            self.predict_field,
            x_lat,
            condition=condition,
            global_step=objective_step,
            target_model_fn=self.predict_field,
        )
        return {
            "x_lat": x_lat,
            "posterior": encoded["posterior"],
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
        parameters = list(self.backbone.parameters())
        if self.tokenizer.trainable:
            parameters.extend(self.tokenizer.parameters())
        return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999))

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
            nfe=self.log_sample_nfe,
            device=self.device,
            condition=condition,
        )
        sampled = self.decode_latents(sampled_latents)
        return {
            "inputs_image": encoded["image"][:batch_size].to(self.device),
            "inputs_mask_index": encoded["mask_index"][:batch_size].unsqueeze(1).float().to(self.device),
            "reconstructions_image": decoded_inputs["rgb_recon"][:batch_size],
            "reconstructions_mask_index": decoded_inputs["mask_index"][:batch_size].unsqueeze(1).float(),
            "samples_image": sampled["rgb_recon"][:batch_size],
            "samples_mask_index": sampled["mask_index"][:batch_size].unsqueeze(1).float(),
        }

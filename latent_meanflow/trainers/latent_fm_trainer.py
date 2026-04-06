from copy import deepcopy

import pytorch_lightning as pl
import torch
from ldm.util import instantiate_from_config

from latent_meanflow.models.tokenizer import SemanticTokenizerAdapter


class LatentFMTrainer(pl.LightningModule):
    def __init__(
        self,
        tokenizer_config_path,
        tokenizer_ckpt_path,
        backbone_config,
        objective_config,
        sample_posterior=False,
        freeze_tokenizer=True,
        use_class_condition=False,
        class_label_key="class_label",
        log_sample_steps=16,
        monitor="val/fm_loss",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_config", "objective_config"])

        self.sample_posterior = bool(sample_posterior)
        self.freeze_tokenizer = bool(freeze_tokenizer)
        self.use_class_condition = bool(use_class_condition)
        self.class_label_key = str(class_label_key)
        self.log_sample_steps = int(log_sample_steps)
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

    def _get_condition(self, batch):
        if not self.use_class_condition:
            return None
        if self.class_label_key not in batch:
            raise KeyError(
                f"use_class_condition=True but batch does not contain '{self.class_label_key}'"
            )
        return batch[self.class_label_key].long()

    def encode_batch(self, batch):
        return self.tokenizer.encode_batch(batch, sample_posterior=self.sample_posterior)

    def decode_latents(self, z):
        return self.tokenizer.decode_latents(z)

    def sample_latents(self, batch_size, num_steps=32, device=None, condition=None, noise=None):
        if device is None:
            device = self.device
        if condition is not None:
            condition = condition.to(device=device, dtype=torch.long)
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
            velocity = self.backbone(z, t_curr, condition=condition)
            z = z + delta_t * velocity
        return z

    def forward(self, batch):
        encoded = self.encode_batch(batch)
        x_lat = encoded["z"]
        condition = self._get_condition(batch)
        objective_outputs = self.objective(self.backbone, x_lat, condition=condition)
        return {
            "x_lat": x_lat,
            "posterior": encoded["posterior"],
            **objective_outputs,
        }

    def shared_step(self, batch, split):
        outputs = self(batch)
        fm_loss = outputs["fm_loss"]
        self.log(
            f"{split}/fm_loss",
            fm_loss,
            prog_bar=True,
            logger=True,
            on_step=(split == "train"),
            on_epoch=True,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, split="train")
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, split="val")
        return outputs["fm_loss"]

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
            num_steps=self.log_sample_steps,
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

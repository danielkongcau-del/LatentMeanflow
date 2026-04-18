import math
from copy import deepcopy

import pytorch_lightning as pl
import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

from latent_meanflow.models.tokenizer import SemanticTokenizerAdapter
from latent_meanflow.utils.latent_normalization import build_latent_normalizer


class SemanticMaskLatentPriorTrainer(pl.LightningModule):
    def __init__(
        self,
        tokenizer_config_path,
        tokenizer_ckpt_path=None,
        backbone_config=None,
        objective_config=None,
        sampler_config=None,
        objective_name=None,
        tokenizer_sample_posterior=False,
        freeze_tokenizer=True,
        log_sample_nfe=8,
        latent_normalization_config=None,
        monitor="val/base_error_mean",
        mask_key="mask_onehot",
        mask_index_key="mask_index",
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["backbone_config", "objective_config", "sampler_config", "latent_normalization_config"]
        )

        if backbone_config is None or objective_config is None or sampler_config is None:
            raise ValueError(
                "SemanticMaskLatentPriorTrainer requires backbone_config, objective_config, and sampler_config."
            )
        if bool(tokenizer_sample_posterior):
            raise ValueError(
                "SemanticMaskLatentPriorTrainer requires deterministic tokenizer latents. "
                "Set tokenizer_sample_posterior=False."
            )
        if not bool(freeze_tokenizer):
            raise ValueError(
                "SemanticMaskLatentPriorTrainer requires a fully frozen tokenizer. "
                "Set freeze_tokenizer=True."
            )

        self.tokenizer_sample_posterior = False
        self.freeze_tokenizer = True
        self.log_sample_nfe = int(log_sample_nfe)
        self.monitor = str(monitor)
        self.mask_key = str(mask_key)
        self.mask_index_key = str(mask_index_key)
        self.learning_rate = 1.0e-4
        if latent_normalization_config is None:
            self.latent_normalization_config = None
        elif OmegaConf.is_config(latent_normalization_config):
            self.latent_normalization_config = OmegaConf.to_container(
                latent_normalization_config,
                resolve=True,
            )
        else:
            self.latent_normalization_config = deepcopy(latent_normalization_config)
        self.tokenizer_config_path = str(tokenizer_config_path)
        self.tokenizer_ckpt_path = None if tokenizer_ckpt_path in {None, ""} else str(tokenizer_ckpt_path)

        self.tokenizer, self.tokenizer_has_ckpt = self._load_tokenizer(
            config_path=self.tokenizer_config_path,
            ckpt_path=self.tokenizer_ckpt_path,
        )
        self.latent_channels = self.tokenizer.latent_channels
        self.latent_spatial_shape = self.tokenizer.latent_spatial_shape
        self.num_classes = self.tokenizer.num_classes
        self.mask_spatial_shape = self._infer_mask_spatial_shape()
        self.latent_normalizer = build_latent_normalizer(
            self.latent_normalization_config,
            latent_channels=self.latent_channels,
        )

        backbone_cfg = deepcopy(backbone_config)
        backbone_cfg.setdefault("params", {})
        if int(backbone_cfg["params"].get("spatial_condition_channels", 0)) > 0:
            raise ValueError("SemanticMaskLatentPriorTrainer is unconditional and must not use spatial conditioning.")
        if backbone_cfg["params"].get("condition_num_classes") is not None:
            raise ValueError("SemanticMaskLatentPriorTrainer does not use image-level class conditioning.")
        backbone_cfg["params"]["in_channels"] = int(self.latent_channels)
        backbone_cfg["params"].setdefault("out_channels", int(self.latent_channels))

        self.backbone = instantiate_from_config(backbone_cfg)
        self.objective = instantiate_from_config(objective_config)
        self.sampler = instantiate_from_config(sampler_config)
        self.objective_name = (
            str(objective_name)
            if objective_name is not None
            else getattr(self.objective, "name", self.objective.__class__.__name__.lower())
        )

    def _load_tokenizer(self, *, config_path, ckpt_path):
        if ckpt_path is None:
            tokenizer = SemanticTokenizerAdapter.from_config(
                config_path=config_path,
                eval_mode=True,
                freeze=True,
            )
            return tokenizer, False
        tokenizer = SemanticTokenizerAdapter.from_pretrained(
            config_path=config_path,
            ckpt_path=ckpt_path,
            eval_mode=True,
            freeze=True,
        )
        return tokenizer, True

    def _infer_mask_spatial_shape(self):
        probe_latents = torch.zeros((1, self.latent_channels, *self.latent_spatial_shape), dtype=torch.float32)
        with torch.no_grad():
            decoded = self.tokenizer.decode_latents(probe_latents)
        return tuple(int(v) for v in decoded["mask_index"].shape[-2:])

    def _require_tokenizer_ckpt(self):
        if self.tokenizer_has_ckpt:
            return
        raise RuntimeError(
            "SemanticMaskLatentPriorTrainer requires model.params.tokenizer_ckpt_path to be set before training, "
            "sampling, or evaluation. The checked-in config keeps it null on purpose; pass it via --set."
        )

    def _get_log_batch_size(self, batch):
        if self.mask_index_key in batch:
            return int(torch.as_tensor(batch[self.mask_index_key]).shape[0])
        if self.mask_key in batch:
            return int(torch.as_tensor(batch[self.mask_key]).shape[0])
        raise KeyError(
            f"SemanticMaskLatentPriorTrainer requires '{self.mask_index_key}' and/or '{self.mask_key}' in the batch."
        )

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
        normalization_info = self.latent_normalizer.describe()
        self.print(
            "[SemanticMaskLatentPriorTrainer] "
            f"objective={self.objective_name}, tokenizer_has_ckpt={self.tokenizer_has_ckpt}, "
            f"latent_spatial_shape={self.latent_spatial_shape}, mask_spatial_shape={self.mask_spatial_shape}, "
            "latent_normalization="
            f"mode={normalization_info['mode']}, enabled={normalization_info['enabled']}, "
            f"stats_path={normalization_info['stats_path']}, std_floor={normalization_info['std_floor']}, "
            f"clamped_channel_count={normalization_info['clamped_channel_count']}"
        )

    def normalize_latents(self, z):
        return self.latent_normalizer.normalize(z)

    def denormalize_latents(self, z):
        return self.latent_normalizer.denormalize(z)

    def encode_batch(self, batch):
        self._require_tokenizer_ckpt()
        encoded = self.tokenizer.encode_batch(batch, sample_posterior=False)
        raw_z = encoded["z"]
        encoded["z_tokenizer"] = raw_z
        encoded["z"] = self.normalize_latents(raw_z)
        return encoded

    def decode_latents(self, z):
        self._require_tokenizer_ckpt()
        return self.tokenizer.decode_latents(self.denormalize_latents(z))

    def predict_field(self, z_t, t=None, condition=None, r=None, delta_t=None):
        if condition is not None:
            raise ValueError("SemanticMaskLatentPriorTrainer is unconditional and does not accept condition.")
        r = self._prepare_time(r, device=z_t.device, dtype=z_t.dtype)
        t = self._prepare_time(t, device=z_t.device, dtype=z_t.dtype)
        delta_t = self._prepare_time(delta_t, device=z_t.device, dtype=z_t.dtype)
        return self.backbone(z_t, t=t, condition=None, r=r, delta_t=delta_t)

    def sample_latents(self, batch_size, nfe=None, device=None, condition=None, noise=None):
        self._require_tokenizer_ckpt()
        if condition is not None:
            raise ValueError("SemanticMaskLatentPriorTrainer is unconditional and does not accept condition.")
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
        x_lat = encoded["z"]
        if objective_step is None:
            objective_step = int(self.global_step)
        objective_outputs = self.objective(
            self.predict_field,
            x_lat,
            condition=None,
            global_step=objective_step,
            target_model_fn=self.predict_field,
        )
        channel_std = x_lat.detach().permute(1, 0, 2, 3).reshape(self.latent_channels, -1).std(
            dim=1,
            unbiased=False,
        )
        latent_target_stats = {
            "latent_input_mean": x_lat.detach().mean(),
            "latent_input_std": x_lat.detach().std(unbiased=False),
            "latent_input_abs_mean": x_lat.detach().abs().mean(),
            "latent_input_channel_std_min": channel_std.min(),
            "latent_input_channel_std_max": channel_std.max(),
            "latent_normalization_enabled": torch.tensor(
                1.0 if self.latent_normalizer.enabled else 0.0,
                device=x_lat.device,
                dtype=x_lat.dtype,
            ),
        }
        return {
            "x_lat": x_lat,
            "z_tokenizer": encoded["z_tokenizer"],
            "posterior": encoded["posterior"],
            "mask_index": encoded["mask_index"],
            "mask_onehot": encoded["mask_onehot"],
            "latent_target_stats": latent_target_stats,
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
        metrics.update(self._collect_log_scalars(split, outputs.get("latent_target_stats", {})))
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
        return torch.optim.Adam(list(self.backbone.parameters()), lr=lr, betas=(0.9, 0.999))

    @torch.no_grad()
    def log_images(self, batch, split="train", **kwargs):
        del split
        encoded = self.encode_batch(batch)
        decoded_inputs = self.decode_latents(encoded["z"])
        batch_size = min(encoded["z"].shape[0], kwargs.get("max_images", 4))
        sampled_latents = self.sample_latents(
            batch_size=batch_size,
            nfe=self.log_sample_nfe,
            device=self.device,
        )
        sampled = self.decode_latents(sampled_latents)
        return {
            "inputs_mask_index": encoded["mask_index"][:batch_size].unsqueeze(1).float().to(self.device),
            "tokenizer_reconstructions_mask_index": decoded_inputs["mask_index"][:batch_size].unsqueeze(1).float(),
            "samples_mask_index": sampled["mask_index"][:batch_size].unsqueeze(1).float(),
        }

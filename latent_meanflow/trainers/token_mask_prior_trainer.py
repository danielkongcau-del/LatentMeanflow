import math
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

from latent_meanflow.losses.semantic_structure import (
    adjacency_l1_loss,
    area_ratio_l1_loss,
    boundary_bce_loss,
    build_valid_mask,
    mask_index_to_boundary_target,
    semantic_probs_to_soft_boundary,
)
from latent_meanflow.models.tokenizer import SemanticTokenizerAdapter


class TokenMaskPriorTrainer(pl.LightningModule):
    def __init__(
        self,
        tokenizer_config_path,
        tokenizer_ckpt_path,
        backbone_config,
        objective_config,
        sampler_config,
        objective_name=None,
        freeze_tokenizer=True,
        tokenizer_sample_posterior=False,
        semantic_ce_weight=0.0,
        semantic_dice_weight=0.0,
        boundary_loss_weight=0.0,
        area_ratio_loss_weight=0.0,
        adjacency_loss_weight=0.0,
        log_sample_nfe=4,
        monitor="val/base_error_mean",
        mask_key="mask_onehot",
        mask_index_key="mask_index",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_config", "objective_config", "sampler_config"])

        if tokenizer_ckpt_path in {None, ""}:
            raise ValueError(
                "TokenMaskPriorTrainer requires tokenizer_ckpt_path to be set explicitly. "
                "Freeze the balanced VQ tokenizer checkpoint and pass it via config or CLI override."
            )
        if bool(tokenizer_sample_posterior):
            raise ValueError(
                "TokenMaskPriorTrainer requires tokenizer_sample_posterior=False. "
                "The frozen VQ tokenizer exposes deterministic code indices."
            )
        if not bool(freeze_tokenizer):
            raise ValueError(
                "TokenMaskPriorTrainer requires freeze_tokenizer=True. "
                "This route must keep the balanced tokenizer frozen."
            )

        self.log_sample_nfe = int(log_sample_nfe)
        self.monitor = str(monitor)
        self.learning_rate = 1.0e-4
        self.mask_key = str(mask_key)
        self.mask_index_key = str(mask_index_key)
        self.tokenizer_config_path = str(tokenizer_config_path)
        self.tokenizer_ckpt_path = str(tokenizer_ckpt_path)
        self.semantic_ce_weight = float(semantic_ce_weight)
        self.semantic_dice_weight = float(semantic_dice_weight)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.area_ratio_loss_weight = float(area_ratio_loss_weight)
        self.adjacency_loss_weight = float(adjacency_loss_weight)

        self.tokenizer = SemanticTokenizerAdapter.from_pretrained(
            config_path=self.tokenizer_config_path,
            ckpt_path=self.tokenizer_ckpt_path,
            eval_mode=True,
            freeze=True,
        )
        if any(param.requires_grad for param in self.tokenizer.parameters()):
            raise ValueError("Frozen token-mask prior requires tokenizer parameters to stay non-trainable.")
        self.ignore_index = self.tokenizer.ignore_index

        self.num_classes = int(self.tokenizer.num_classes)
        self.codebook_size = int(self.tokenizer.codebook_size)
        self.mask_token_id = int(self.codebook_size)
        self.token_vocabulary_size = int(self.codebook_size + 1)
        self.token_spatial_shape = tuple(int(v) for v in self.tokenizer.latent_spatial_shape)
        self.mask_spatial_shape = self._infer_mask_spatial_shape()
        # Keep latent_* names for compatibility with the existing sampling/eval utilities.
        self.latent_channels = 1
        self.latent_spatial_shape = tuple(self.token_spatial_shape)

        backbone_cfg = deepcopy(backbone_config)
        backbone_cfg.setdefault("params", {})
        if int(backbone_cfg["params"].get("spatial_condition_channels", 0)) > 0:
            raise ValueError("TokenMaskPriorTrainer is unconditional and must not use spatial conditioning.")
        if backbone_cfg["params"].get("condition_num_classes") is not None:
            raise ValueError("TokenMaskPriorTrainer does not use image-level class conditioning.")
        backbone_cfg["params"]["in_channels"] = int(self.token_vocabulary_size)
        backbone_cfg["params"]["out_channels"] = int(self.codebook_size)
        if "input_size" in backbone_cfg["params"]:
            backbone_cfg["params"]["input_size"] = list(self.token_spatial_shape)

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
                    num_classes=self.codebook_size,
                    mask_token_id=self.mask_token_id,
                    ignore_index=None,
                )

    def _infer_mask_spatial_shape(self):
        probe_codes = torch.zeros((1, *self.token_spatial_shape), dtype=torch.long)
        with torch.no_grad():
            decoded = self.tokenizer.decode_codes(probe_codes)
        return tuple(int(v) for v in decoded["mask_index"].shape[-2:])

    def _prepare_codes(self, codes):
        if not isinstance(codes, torch.Tensor):
            codes = torch.as_tensor(codes)
        if codes.ndim == 4 and int(codes.shape[1]) == 1:
            codes = codes[:, 0]
        if codes.ndim != 3:
            raise ValueError(
                f"TokenMaskPriorTrainer expects code indices with shape [B, Ht, Wt], got {tuple(codes.shape)}"
            )
        if tuple(int(v) for v in codes.shape[-2:]) != self.token_spatial_shape:
            raise ValueError(
                "TokenMaskPriorTrainer code-grid shape mismatch: "
                f"expected {self.token_spatial_shape}, got {tuple(int(v) for v in codes.shape[-2:])}"
            )
        return codes.long().contiguous()

    def _validate_sampled_codes(self, codes):
        min_code = int(codes.min().item())
        max_code = int(codes.max().item())
        if min_code < 0 or max_code >= self.codebook_size:
            raise ValueError(
                "Sampled token grid must contain only valid tokenizer code ids in "
                f"[0, {self.codebook_size - 1}] before decode; got range [{min_code}, {max_code}]"
            )

    def _discrete_to_onehot(self, z_t):
        if z_t.ndim == 4 and int(z_t.shape[1]) == self.token_vocabulary_size:
            return z_t.to(dtype=torch.float32)
        codes = self._prepare_codes(z_t)
        if torch.any(codes < 0):
            raise ValueError("TokenMaskPriorTrainer received negative token ids.")
        if torch.any(codes > self.mask_token_id):
            raise ValueError(
                f"TokenMaskPriorTrainer received token ids above MASK token id {self.mask_token_id}."
            )
        onehot = F.one_hot(codes.to(dtype=torch.long), num_classes=self.token_vocabulary_size)
        return onehot.permute(0, 3, 1, 2).to(dtype=torch.float32)

    def _prepare_time(self, value, device):
        if value is None:
            return None
        return value.to(device=device, dtype=torch.float32)

    def _get_log_batch_size(self, batch):
        if self.mask_index_key in batch:
            return int(torch.as_tensor(batch[self.mask_index_key]).shape[0])
        if self.mask_key in batch:
            return int(torch.as_tensor(batch[self.mask_key]).shape[0])
        raise KeyError(
            f"TokenMaskPriorTrainer requires '{self.mask_index_key}' and/or '{self.mask_key}' in the batch."
        )

    def _should_sync_dist(self):
        trainer = getattr(self, "trainer", None)
        return trainer is not None and int(getattr(trainer, "world_size", 1)) > 1

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
            "[TokenMaskPriorTrainer] "
            f"tokenizer_config={self.tokenizer_config_path}, tokenizer_ckpt={self.tokenizer_ckpt_path}, "
            f"codebook_size={self.codebook_size}, token_spatial_shape={self.token_spatial_shape}, "
            f"mask_spatial_shape={self.mask_spatial_shape}, objective={self.objective_name}, "
            f"semantic_ce_weight={self.semantic_ce_weight:.3f}, "
            f"semantic_dice_weight={self.semantic_dice_weight:.3f}, "
            f"boundary_loss_weight={self.boundary_loss_weight:.3f}, "
            f"area_ratio_loss_weight={self.area_ratio_loss_weight:.3f}, "
            f"adjacency_loss_weight={self.adjacency_loss_weight:.3f}"
        )

    def _token_usage_metrics(self, codes):
        flat_codes = codes.reshape(int(codes.shape[0]), -1)
        code_counts = torch.bincount(flat_codes.reshape(-1), minlength=self.codebook_size).to(dtype=torch.float32)
        total = code_counts.sum().clamp_min(1.0)
        probs = code_counts / total
        unique_per_sample = (F.one_hot(flat_codes, num_classes=self.codebook_size).sum(dim=1) > 0).sum(dim=1)
        return {
            "target_unique_code_count_mean": unique_per_sample.to(dtype=torch.float32).mean().detach(),
            "target_unique_code_fraction_mean": (
                unique_per_sample.to(dtype=torch.float32) / float(max(self.codebook_size, 1))
            ).mean().detach(),
            "target_active_code_count": (code_counts > 0).sum().to(dtype=torch.float32).detach(),
            "target_active_code_fraction": ((code_counts > 0).sum().to(dtype=torch.float32) / float(self.codebook_size)).detach(),
            "target_code_perplexity": torch.exp(-(probs * torch.log(probs.clamp_min(1.0e-10))).sum()).detach(),
        }

    def encode_batch(self, batch):
        encoded = self.tokenizer.encode_batch(batch, sample_posterior=False)
        if "codes" not in encoded:
            raise KeyError("Frozen tokenizer did not return 'codes'; token-code mask generation requires discrete codes.")
        codes = self._prepare_codes(encoded["codes"])
        if torch.any(codes < 0) or torch.any(codes >= self.codebook_size):
            raise ValueError(
                f"Tokenizer encode produced code ids outside [0, {self.codebook_size - 1}]."
            )
        return {
            "codes": codes,
            "z": codes,
            "mask_index": encoded["mask_index"],
            "mask_onehot": encoded["mask_onehot"],
            "quantizer_stats": encoded.get("quantizer_stats", None),
        }

    def decode_latents(self, z):
        codes = self._prepare_codes(z)
        self._validate_sampled_codes(codes)
        decoded = self.tokenizer.decode_codes(codes)
        decoded["codes"] = codes
        return decoded

    def _semantic_decode_bridge_enabled(self):
        return any(
            weight > 0.0
            for weight in (
                self.semantic_ce_weight,
                self.semantic_dice_weight,
                self.boundary_loss_weight,
                self.area_ratio_loss_weight,
                self.adjacency_loss_weight,
            )
        )

    def _decode_semantic_from_code_logits(self, code_logits):
        if tuple(int(v) for v in code_logits.shape[1:]) != (self.codebook_size, *self.token_spatial_shape):
            raise ValueError(
                "TokenMaskPriorTrainer semantic bridge expects code logits with shape "
                f"[B, {self.codebook_size}, {self.token_spatial_shape[0]}, {self.token_spatial_shape[1]}], "
                f"got {tuple(code_logits.shape)}"
            )
        return self.tokenizer.decode_code_distribution(code_logits=code_logits)

    def _semantic_auxiliary_terms(self, *, code_logits, mask_index):
        decoded = self._decode_semantic_from_code_logits(code_logits)
        semantic_losses = self.tokenizer.semantic_auxiliary_losses(
            mask_logits=decoded["mask_logits"],
            mask_index=mask_index,
        )
        semantic_aux_total = (
            self.semantic_ce_weight * semantic_losses["semantic_ce"]
            + self.semantic_dice_weight * semantic_losses["semantic_dice"]
        )
        with torch.no_grad():
            semantic_metrics = self.tokenizer.compute_mask_metrics(
                mask_index=mask_index,
                mask_logits=decoded["mask_logits"],
            )
        return {
            "decoded": decoded,
            "semantic_ce": semantic_losses["semantic_ce"],
            "semantic_dice": semantic_losses["semantic_dice"],
            "semantic_aux_total": semantic_aux_total,
            "semantic_metrics": {
                "semantic_pixel_accuracy": semantic_metrics["pixel_accuracy"].detach(),
                "semantic_miou": semantic_metrics["miou"].detach(),
            },
        }

    def _semantic_structure_terms(self, *, mask_probs, mask_index, mask_onehot):
        valid_mask = build_valid_mask(mask_index, ignore_index=self.ignore_index)
        boundary_target = mask_index_to_boundary_target(mask_index, ignore_index=self.ignore_index)
        boundary_pred = semantic_probs_to_soft_boundary(mask_probs, valid_mask=valid_mask)
        boundary_loss = boundary_bce_loss(
            boundary_pred,
            boundary_target,
            valid_mask=valid_mask,
        )
        area_ratio_loss, pred_area_ratio, target_area_ratio = area_ratio_l1_loss(
            mask_probs,
            mask_onehot,
            valid_mask=valid_mask,
        )
        adjacency_loss, pred_adjacency, target_adjacency = adjacency_l1_loss(
            mask_probs,
            mask_onehot,
            valid_mask=valid_mask,
        )
        structure_total = (
            self.boundary_loss_weight * boundary_loss
            + self.area_ratio_loss_weight * area_ratio_loss
            + self.adjacency_loss_weight * adjacency_loss
        )
        return {
            "boundary_loss": boundary_loss,
            "area_ratio_loss": area_ratio_loss,
            "adjacency_loss": adjacency_loss,
            "structure_aux_total": structure_total,
            "boundary_target": boundary_target.detach(),
            "boundary_pred": boundary_pred.detach(),
            "pred_area_ratio": pred_area_ratio.detach(),
            "target_area_ratio": target_area_ratio.detach(),
            "pred_adjacency": pred_adjacency.detach(),
            "target_adjacency": target_adjacency.detach(),
        }

    def predict_field(self, z_t, t=None, condition=None, r=None, delta_t=None):
        if condition is not None:
            raise ValueError("TokenMaskPriorTrainer is unconditional and does not accept condition.")
        z_t_onehot = self._discrete_to_onehot(z_t).to(device=self.device)
        t = self._prepare_time(t, device=z_t_onehot.device)
        r = self._prepare_time(r, device=z_t_onehot.device)
        delta_t = self._prepare_time(delta_t, device=z_t_onehot.device)
        return self.backbone(z_t_onehot, t=t, condition=None, r=r, delta_t=delta_t)

    def sample_latents(self, batch_size, nfe=None, device=None, condition=None, noise=None):
        if condition is not None:
            raise ValueError("TokenMaskPriorTrainer is unconditional and does not accept condition.")
        if device is None:
            device = self.device
        compatibility_shape = (self.latent_channels, *self.latent_spatial_shape)
        sampled_codes = self.sampler.sample(
            model_fn=self.predict_field,
            batch_size=batch_size,
            latent_shape=compatibility_shape,
            device=device,
            condition=None,
            noise=noise,
            nfe=nfe,
        )
        sampled_codes = self._prepare_codes(sampled_codes)
        self._validate_sampled_codes(sampled_codes)
        return sampled_codes

    def forward(self, batch, objective_step=None):
        encoded = self.encode_batch(batch)
        code_grid = encoded["codes"]
        if objective_step is None:
            objective_step = int(self.global_step)
        objective_outputs = self.objective(
            self.predict_field,
            code_grid,
            condition=None,
            global_step=objective_step,
            target_model_fn=self.predict_field,
        )
        total_loss = objective_outputs["loss"]
        loss_dict = dict(objective_outputs.get("loss_dict", {}))
        semantic_bridge_metrics = {}
        semantic_outputs = {}
        structure_outputs = {}
        if self._semantic_decode_bridge_enabled():
            semantic_outputs = self._semantic_auxiliary_terms(
                code_logits=objective_outputs["pred_field"],
                mask_index=encoded["mask_index"],
            )
            semantic_bridge_metrics = semantic_outputs["semantic_metrics"]
            total_loss = total_loss + semantic_outputs["semantic_aux_total"]
            loss_dict.update(
                {
                    "semantic_ce": semantic_outputs["semantic_ce"].detach(),
                    "semantic_dice": semantic_outputs["semantic_dice"].detach(),
                    "semantic_aux_total": semantic_outputs["semantic_aux_total"].detach(),
                }
            )
            structure_outputs = self._semantic_structure_terms(
                mask_probs=semantic_outputs["decoded"]["mask_probs"],
                mask_index=encoded["mask_index"],
                mask_onehot=encoded["mask_onehot"],
            )
            total_loss = total_loss + structure_outputs["structure_aux_total"]
            loss_dict.update(
                {
                    "boundary_loss": structure_outputs["boundary_loss"].detach(),
                    "area_ratio_loss": structure_outputs["area_ratio_loss"].detach(),
                    "adjacency_loss": structure_outputs["adjacency_loss"].detach(),
                    "structure_aux_total": structure_outputs["structure_aux_total"].detach(),
                }
            )
        loss_dict["total_loss"] = total_loss.detach()
        loss_dict["base_error_mean"] = total_loss.detach()
        return {
            **objective_outputs,
            "code_grid": code_grid,
            "mask_index": encoded["mask_index"],
            "mask_onehot": encoded["mask_onehot"],
            "code_target_stats": self._token_usage_metrics(code_grid),
            "semantic_mask_logits": semantic_outputs.get("decoded", {}).get("mask_logits"),
            "semantic_mask_probs": semantic_outputs.get("decoded", {}).get("mask_probs"),
            "semantic_mask_index": semantic_outputs.get("decoded", {}).get("mask_index"),
            "boundary_target": structure_outputs.get("boundary_target"),
            "boundary_pred": structure_outputs.get("boundary_pred"),
            "pred_area_ratio": structure_outputs.get("pred_area_ratio"),
            "target_area_ratio": structure_outputs.get("target_area_ratio"),
            "pred_adjacency": structure_outputs.get("pred_adjacency"),
            "target_adjacency": structure_outputs.get("target_adjacency"),
            "semantic_bridge_metrics": semantic_bridge_metrics,
            "loss": total_loss,
            "loss_dict": loss_dict,
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
        metrics.update(self._collect_log_scalars(split, outputs.get("code_target_stats", {})))
        metrics.update(self._collect_log_scalars(split, outputs.get("semantic_bridge_metrics", {})))
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
        del split
        encoded = self.encode_batch(batch)
        decoded_inputs = self.decode_latents(encoded["codes"])
        batch_size = min(int(encoded["codes"].shape[0]), int(kwargs.get("max_images", 4)))
        sampled_codes = self.sample_latents(
            batch_size=batch_size,
            nfe=self.log_sample_nfe,
            device=self.device,
        )
        sampled = self.decode_latents(sampled_codes)
        return {
            "inputs_mask_index": encoded["mask_index"][:batch_size].unsqueeze(1).float().to(self.device),
            "tokenizer_reconstructions_mask_index": decoded_inputs["mask_index"][:batch_size].unsqueeze(1).float(),
            "samples_mask_index": sampled["mask_index"][:batch_size].unsqueeze(1).float(),
        }

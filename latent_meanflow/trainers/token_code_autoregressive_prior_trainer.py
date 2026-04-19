import hashlib
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config

from latent_meanflow.losses.semantic_structure import build_valid_mask, mask_index_to_boundary_target
from latent_meanflow.models.backbones.token_code_mingpt import ensure_taming_transformers_on_path
from latent_meanflow.models.tokenizer import SemanticTokenizerAdapter


class TokenCodeAutoregressivePriorTrainer(pl.LightningModule):
    def __init__(
        self,
        tokenizer_config_path,
        tokenizer_ckpt_path,
        backbone_config,
        objective_name="token_code_autoregressive",
        permuter_config=None,
        freeze_tokenizer=True,
        tokenizer_sample_posterior=False,
        log_sample_nfe=1,
        enable_validation_sample_metrics=True,
        validation_sample_batch_size=4,
        validation_sample_metric_batches=4,
        validation_sample_nfe=None,
        validation_sample_seed=1234,
        monitor="val/sampled_monitor_error",
        sample_temperature=1.0,
        sample_top_k=None,
        sample_greedy=False,
        weight_decay=0.01,
        optimizer_beta1=0.9,
        optimizer_beta2=0.95,
        mask_key="mask_onehot",
        mask_index_key="mask_index",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_config", "permuter_config"])

        if tokenizer_ckpt_path in {None, ""}:
            raise ValueError(
                "TokenCodeAutoregressivePriorTrainer requires tokenizer_ckpt_path to be set explicitly. "
                "Freeze the balanced VQ tokenizer checkpoint and pass it via config or CLI override."
            )
        if bool(tokenizer_sample_posterior):
            raise ValueError(
                "TokenCodeAutoregressivePriorTrainer requires tokenizer_sample_posterior=False. "
                "The frozen VQ tokenizer exposes deterministic code indices."
            )
        if not bool(freeze_tokenizer):
            raise ValueError(
                "TokenCodeAutoregressivePriorTrainer requires freeze_tokenizer=True. "
                "This route must keep the balanced tokenizer frozen."
            )

        self.log_sample_nfe = int(log_sample_nfe)
        self.monitor = str(monitor)
        self.learning_rate = 1.0e-4
        self.mask_key = str(mask_key)
        self.mask_index_key = str(mask_index_key)
        self.tokenizer_config_path = str(tokenizer_config_path)
        self.tokenizer_ckpt_path = str(tokenizer_ckpt_path)
        self.objective_name = str(objective_name)
        self.enable_validation_sample_metrics = bool(enable_validation_sample_metrics)
        self.validation_sample_batch_size = max(0, int(validation_sample_batch_size))
        self.validation_sample_metric_batches = max(0, int(validation_sample_metric_batches))
        self.validation_sample_nfe = (
            int(log_sample_nfe) if validation_sample_nfe is None else max(1, int(validation_sample_nfe))
        )
        self.validation_sample_seed = int(validation_sample_seed)
        self.sample_temperature = float(sample_temperature)
        self.sample_top_k = None if sample_top_k in {None, ""} else max(1, int(sample_top_k))
        self.sample_greedy = bool(sample_greedy)
        self.weight_decay = float(weight_decay)
        self.optimizer_betas = (float(optimizer_beta1), float(optimizer_beta2))
        self.per_class_metric_logging_limit = 32
        self.supports_nfe_sweep = False
        self.route_family = "autoregressive"
        self._reset_validation_sample_metric_state()
        if self.monitor.startswith("val/sampled_") and (
            not self.enable_validation_sample_metrics
            or self.validation_sample_batch_size <= 0
            or self.validation_sample_metric_batches <= 0
        ):
            raise ValueError(
                f"TokenCodeAutoregressivePriorTrainer monitor={self.monitor} requires "
                "enable_validation_sample_metrics=True, validation_sample_batch_size > 0, "
                "and validation_sample_metric_batches > 0."
            )

        self.tokenizer = SemanticTokenizerAdapter.from_pretrained(
            config_path=self.tokenizer_config_path,
            ckpt_path=self.tokenizer_ckpt_path,
            eval_mode=True,
            freeze=True,
        )
        if any(param.requires_grad for param in self.tokenizer.parameters()):
            raise ValueError(
                "Frozen token-code autoregressive prior requires tokenizer parameters to stay non-trainable."
            )

        self.ignore_index = self.tokenizer.ignore_index
        self.num_classes = int(self.tokenizer.num_classes)
        self.codebook_size = int(self.tokenizer.codebook_size)
        self.bos_token_id = int(self.codebook_size)
        self.token_vocabulary_size = int(self.codebook_size + 1)
        self.token_spatial_shape = tuple(int(v) for v in self.tokenizer.latent_spatial_shape)
        self.code_sequence_length = int(self.token_spatial_shape[0] * self.token_spatial_shape[1])
        self.mask_spatial_shape = self._infer_mask_spatial_shape()
        self.latent_channels = 1
        self.latent_spatial_shape = tuple(self.token_spatial_shape)

        ensure_taming_transformers_on_path()
        permuter_cfg = deepcopy(permuter_config) if permuter_config is not None else {
            "target": "taming.modules.transformer.permuter.Identity"
        }
        permuter_cfg.setdefault("params", {})
        permuter_target = str(permuter_cfg.get("target", ""))
        if not permuter_target.endswith(".Identity"):
            permuter_cfg["params"].setdefault("H", int(self.token_spatial_shape[0]))
            permuter_cfg["params"].setdefault("W", int(self.token_spatial_shape[1]))
        self.permuter = instantiate_from_config(permuter_cfg)
        self.permuter_name = permuter_target.rsplit(".", 1)[-1] if permuter_target else "Identity"

        backbone_cfg = deepcopy(backbone_config)
        backbone_cfg.setdefault("params", {})
        backbone_cfg["params"]["vocab_size"] = int(self.token_vocabulary_size)
        backbone_cfg["params"].setdefault("block_size", int(self.code_sequence_length))
        self.backbone = instantiate_from_config(backbone_cfg)
        self.context_length = int(self.backbone.get_block_size())
        if self.context_length <= 0:
            raise ValueError("TokenCodeAutoregressivePriorTrainer requires backbone block_size > 0.")

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
                "TokenCodeAutoregressivePriorTrainer expects code indices with shape [B, Ht, Wt], "
                f"got {tuple(codes.shape)}"
            )
        if tuple(int(v) for v in codes.shape[-2:]) != self.token_spatial_shape:
            raise ValueError(
                "TokenCodeAutoregressivePriorTrainer code-grid shape mismatch: "
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

    def _prepare_metric_mask_index(self, mask_index):
        if not isinstance(mask_index, torch.Tensor):
            mask_index = torch.as_tensor(mask_index)
        if mask_index.ndim == 2:
            mask_index = mask_index.unsqueeze(0)
        if mask_index.ndim == 4 and int(mask_index.shape[1]) == 1:
            mask_index = mask_index[:, 0]
        if mask_index.ndim != 3:
            raise ValueError(
                "TokenCodeAutoregressivePriorTrainer semantic diagnostics expect mask_index with shape "
                f"[B, H, W] or [B, 1, H, W], got {tuple(mask_index.shape)}"
            )
        return mask_index.long().contiguous()

    def _mask_distribution_summary(self, mask_index):
        mask_index = self._prepare_metric_mask_index(mask_index)
        valid_mask = build_valid_mask(mask_index, ignore_index=self.ignore_index)
        safe_targets = mask_index.clone()
        safe_targets[~valid_mask] = 0
        onehot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        onehot = onehot * valid_mask.unsqueeze(1).float()
        class_counts = onehot.sum(dim=(2, 3))
        valid_counts = valid_mask.view(int(mask_index.shape[0]), -1).sum(dim=1).clamp_min(1).float()
        class_ratios = class_counts / valid_counts.unsqueeze(1)
        mean_class_ratios = class_ratios.mean(dim=0)
        boundary_ratio = mask_index_to_boundary_target(mask_index, ignore_index=self.ignore_index).mean(dim=(1, 2, 3))
        return {
            "mean_class_ratios": mean_class_ratios.detach(),
            "majority_class_ratio_mean": class_ratios.max(dim=1).values.mean().detach(),
            "unique_class_count_mean": (class_counts > 0).sum(dim=1).float().mean().detach(),
            "boundary_ratio_mean": boundary_ratio.mean().detach(),
            "class_entropy": (-(mean_class_ratios * torch.log(mean_class_ratios.clamp_min(1.0e-10))).sum()).detach(),
        }

    def _distribution_monitor_error(
        self,
        *,
        class_hist_l1,
        boundary_ratio_gap,
        majority_class_ratio_gap,
        class_entropy_gap,
        unique_class_count_gap,
    ):
        return (
            class_hist_l1
            + boundary_ratio_gap
            + majority_class_ratio_gap
            + class_entropy_gap
            + (unique_class_count_gap / float(max(self.num_classes, 1)))
        ).detach()

    def _reset_validation_sample_metric_state(self):
        self._validation_sample_metric_sums = {}
        self._validation_sample_metric_batches_seen = 0

    def _accumulate_validation_sample_metrics(self, metrics):
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    continue
                scalar = value.detach().to(device=torch.device("cpu"), dtype=torch.float32)
            elif isinstance(value, (int, float)):
                scalar = torch.tensor(float(value), dtype=torch.float32)
            else:
                continue
            if name in self._validation_sample_metric_sums:
                self._validation_sample_metric_sums[name] += scalar
            else:
                self._validation_sample_metric_sums[name] = scalar.clone()
        self._validation_sample_metric_batches_seen += 1

    def _finalize_validation_sample_metrics(self):
        if self._validation_sample_metric_batches_seen <= 0:
            return {}
        denom = float(self._validation_sample_metric_batches_seen)
        return {
            f"val/{name}": (value / denom).to(device=self.device)
            for name, value in self._validation_sample_metric_sums.items()
        }

    def _mask_distribution_gap_metrics(self, *, pred_mask_index, target_mask_index, prefix):
        pred_summary = self._mask_distribution_summary(pred_mask_index)
        target_summary = self._mask_distribution_summary(target_mask_index)
        class_ratio_gap = (
            pred_summary["mean_class_ratios"] - target_summary["mean_class_ratios"]
        ).abs().detach()
        class_hist_l1 = class_ratio_gap.sum().detach()
        majority_class_ratio_gap = (
            pred_summary["majority_class_ratio_mean"] - target_summary["majority_class_ratio_mean"]
        ).abs().detach()
        unique_class_count_gap = (
            pred_summary["unique_class_count_mean"] - target_summary["unique_class_count_mean"]
        ).abs().detach()
        boundary_ratio_gap = (
            pred_summary["boundary_ratio_mean"] - target_summary["boundary_ratio_mean"]
        ).abs().detach()
        class_entropy_gap = (
            pred_summary["class_entropy"] - target_summary["class_entropy"]
        ).abs().detach()
        metrics = {
            f"{prefix}_class_hist_l1": class_hist_l1,
            f"{prefix}_pred_majority_class_ratio": pred_summary["majority_class_ratio_mean"],
            f"{prefix}_target_majority_class_ratio": target_summary["majority_class_ratio_mean"],
            f"{prefix}_majority_class_ratio_gap": majority_class_ratio_gap,
            f"{prefix}_pred_unique_class_count": pred_summary["unique_class_count_mean"],
            f"{prefix}_target_unique_class_count": target_summary["unique_class_count_mean"],
            f"{prefix}_unique_class_count_gap": unique_class_count_gap,
            f"{prefix}_boundary_ratio_gap": boundary_ratio_gap,
            f"{prefix}_pred_class_entropy": pred_summary["class_entropy"],
            f"{prefix}_target_class_entropy": target_summary["class_entropy"],
            f"{prefix}_class_entropy_gap": class_entropy_gap,
            f"{prefix}_monitor_error": self._distribution_monitor_error(
                class_hist_l1=class_hist_l1,
                boundary_ratio_gap=boundary_ratio_gap,
                majority_class_ratio_gap=majority_class_ratio_gap,
                class_entropy_gap=class_entropy_gap,
                unique_class_count_gap=unique_class_count_gap,
            ),
        }
        if int(self.num_classes) <= int(self.per_class_metric_logging_limit):
            for class_idx in range(int(self.num_classes)):
                metrics[f"{prefix}_pred_class_ratio_{class_idx}"] = pred_summary["mean_class_ratios"][class_idx]
                metrics[f"{prefix}_target_class_ratio_{class_idx}"] = target_summary["mean_class_ratios"][class_idx]
                metrics[f"{prefix}_class_ratio_gap_{class_idx}"] = class_ratio_gap[class_idx]
        return metrics

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
            "target_active_code_fraction": (
                (code_counts > 0).sum().to(dtype=torch.float32) / float(self.codebook_size)
            ).detach(),
            "target_code_perplexity": torch.exp(-(probs * torch.log(probs.clamp_min(1.0e-10))).sum()).detach(),
        }

    def _get_log_batch_size(self, batch):
        if self.mask_index_key in batch:
            return int(torch.as_tensor(batch[self.mask_index_key]).shape[0])
        if self.mask_key in batch:
            return int(torch.as_tensor(batch[self.mask_key]).shape[0])
        raise KeyError(
            "TokenCodeAutoregressivePriorTrainer requires "
            f"'{self.mask_index_key}' and/or '{self.mask_key}' in the batch."
        )

    def _should_sync_dist(self):
        trainer = getattr(self, "trainer", None)
        return trainer is not None and int(getattr(trainer, "world_size", 1)) > 1

    def _is_global_zero(self):
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "is_global_zero"):
            return bool(trainer.is_global_zero)
        return int(getattr(self, "global_rank", 0)) == 0

    def _log_info(self, message):
        if self._is_global_zero():
            print(message)

    def _codes_to_sequence(self, codes):
        codes = self._prepare_codes(codes)
        sequence = codes.view(int(codes.shape[0]), -1)
        sequence = self.permuter(sequence)
        return sequence.long().contiguous()

    def _sequence_to_codes(self, sequence):
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence)
        if sequence.ndim != 2:
            raise ValueError(
                "TokenCodeAutoregressivePriorTrainer expects sequence tokens with shape [B, T], "
                f"got {tuple(sequence.shape)}"
            )
        if int(sequence.shape[1]) != self.code_sequence_length:
            raise ValueError(
                "TokenCodeAutoregressivePriorTrainer sequence length mismatch: "
                f"expected {self.code_sequence_length}, got {int(sequence.shape[1])}"
            )
        sequence = self.permuter(sequence.long(), reverse=True)
        return sequence.view(int(sequence.shape[0]), *self.token_spatial_shape).contiguous()

    def _build_teacher_forcing_window(self, code_sequence):
        batch_size, sequence_length = code_sequence.shape
        window_length = min(int(self.context_length), int(sequence_length))
        if sequence_length <= window_length:
            starts = torch.zeros(batch_size, device=code_sequence.device, dtype=torch.long)
        elif self.training:
            starts = torch.randint(
                low=0,
                high=int(sequence_length - window_length + 1),
                size=(batch_size,),
                device=code_sequence.device,
            )
        else:
            starts = torch.zeros(batch_size, device=code_sequence.device, dtype=torch.long)

        prefix = torch.cat(
            [
                torch.full(
                    (batch_size, 1),
                    fill_value=self.bos_token_id,
                    device=code_sequence.device,
                    dtype=torch.long,
                ),
                code_sequence,
            ],
            dim=1,
        )
        offsets = torch.arange(window_length, device=code_sequence.device, dtype=torch.long).unsqueeze(0)
        input_indices = starts.unsqueeze(1) + offsets
        target_indices = input_indices + 1
        input_tokens = prefix.gather(1, input_indices)
        target_tokens = prefix.gather(1, target_indices)
        return input_tokens, target_tokens, starts

    def _sequence_hash_seed(self, noise):
        if noise is None:
            return None
        payload = noise.detach().to(device=torch.device("cpu"), dtype=torch.float32).contiguous().numpy().tobytes()
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)

    def _sample_generator(self, *, device, noise):
        seed = self._sequence_hash_seed(noise)
        if seed is None:
            return None
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed)
        return generator

    def _sample_next_token(self, logits, *, generator):
        logits = logits[:, : self.codebook_size]
        if self.sample_top_k is not None and self.sample_top_k < int(logits.shape[-1]):
            threshold = torch.topk(logits, k=self.sample_top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if self.sample_greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    def _supports_full_context_kv_cache(self):
        return int(self.context_length) >= int(self.code_sequence_length)

    def encode_batch(self, batch):
        encoded = self.tokenizer.encode_batch(batch, sample_posterior=False)
        if "codes" not in encoded:
            raise KeyError(
                "Frozen tokenizer did not return 'codes'; token-code autoregressive generation requires discrete codes."
            )
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

    def forward(self, batch, objective_step=None):
        del objective_step
        encoded = self.encode_batch(batch)
        code_grid = encoded["codes"]
        code_sequence = self._codes_to_sequence(code_grid)
        input_tokens, target_tokens, window_starts = self._build_teacher_forcing_window(code_sequence)
        logits, loss = self.backbone(input_tokens, targets=target_tokens)
        if loss is None:
            raise RuntimeError("mingpt backbone did not return a teacher-forced loss.")
        predicted_tokens = torch.argmax(logits, dim=-1)
        token_accuracy = (predicted_tokens == target_tokens).float().mean().detach()
        loss_dict = {
            "autoregressive_ce": loss.detach(),
            "base_error_mean": loss.detach(),
            "total_loss": loss.detach(),
        }
        autoregressive_metrics = {
            "teacher_forced_token_accuracy": token_accuracy,
            "teacher_forced_window_start_mean": window_starts.float().mean().detach(),
            "teacher_forced_window_coverage": (
                torch.tensor(float(input_tokens.shape[1]), device=loss.device) / float(max(self.code_sequence_length, 1))
            ).detach(),
        }
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "code_grid": code_grid,
            "code_sequence": code_sequence,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "next_token_logits": logits,
            "mask_index": encoded["mask_index"],
            "mask_onehot": encoded["mask_onehot"],
            "code_target_stats": self._token_usage_metrics(code_grid),
            "autoregressive_metrics": autoregressive_metrics,
        }

    def _collect_log_scalars(self, split, scalars):
        metrics = {}
        for name, value in scalars.items():
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    continue
                metrics[f"{split}/{name}"] = value.detach()
            elif isinstance(value, (int, float)):
                metrics[f"{split}/{name}"] = torch.tensor(float(value), device=self.device)
        return metrics

    def shared_step(self, batch, split):
        outputs = self(batch, objective_step=int(self.global_step))
        metrics = self._collect_log_scalars(split, outputs.get("loss_dict", {}))
        metrics.update(self._collect_log_scalars(split, outputs.get("code_target_stats", {})))
        metrics.update(self._collect_log_scalars(split, outputs.get("autoregressive_metrics", {})))
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
        return outputs["loss"], metrics.get(f"{split}/loss", outputs["loss"].detach()), outputs

    def training_step(self, batch, batch_idx):
        del batch_idx
        loss, _, _ = self.shared_step(batch, split="train")
        return loss

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._reset_validation_sample_metric_state()

    @torch.no_grad()
    def _validation_sample_metrics(self, target_mask_index):
        if not self.enable_validation_sample_metrics or self.validation_sample_batch_size <= 0:
            return {}

        batch_size = min(int(target_mask_index.shape[0]), self.validation_sample_batch_size)
        if batch_size <= 0:
            return {}

        device = self.device if isinstance(self.device, torch.device) else torch.device("cpu")
        generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
        generator.manual_seed(self.validation_sample_seed)
        noise = torch.randn(
            (batch_size, self.latent_channels, *self.latent_spatial_shape),
            generator=generator,
            device=device,
        )
        sampled_codes = self.sample_latents(
            batch_size=batch_size,
            nfe=self.validation_sample_nfe,
            device=device,
            noise=noise,
        )
        sampled = self.decode_latents(sampled_codes)
        return self._mask_distribution_gap_metrics(
            pred_mask_index=sampled["mask_index"],
            target_mask_index=target_mask_index[:batch_size],
            prefix="sampled",
        )

    def validation_step(self, batch, batch_idx):
        _, detached_loss, outputs = self.shared_step(batch, split="val")
        if batch_idx < self.validation_sample_metric_batches and self._is_global_zero():
            sample_metrics = self._validation_sample_metrics(outputs["mask_index"].detach())
            if sample_metrics:
                self._accumulate_validation_sample_metrics(sample_metrics)
        return detached_loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self._is_global_zero():
            sample_metrics = self._finalize_validation_sample_metrics()
            if sample_metrics:
                effective_batch_size = max(
                    1,
                    int(self._validation_sample_metric_batches_seen) * max(1, self.validation_sample_batch_size),
                )
                self.log_dict(
                    sample_metrics,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=effective_batch_size,
                    sync_dist=False,
                )
        self._reset_validation_sample_metric_state()

    def on_fit_start(self):
        super().on_fit_start()
        self._log_info(
            "[TokenCodeAutoregressivePriorTrainer] "
            f"tokenizer_config={self.tokenizer_config_path}, tokenizer_ckpt={self.tokenizer_ckpt_path}, "
            f"codebook_size={self.codebook_size}, token_spatial_shape={self.token_spatial_shape}, "
            f"mask_spatial_shape={self.mask_spatial_shape}, context_length={self.context_length}, "
            f"sequence_length={self.code_sequence_length}, permuter={self.permuter_name}, "
            f"objective={self.objective_name}, monitor={self.monitor}, "
            f"sample_temperature={self.sample_temperature:.3f}, sample_top_k={self.sample_top_k}, "
            f"sample_greedy={self.sample_greedy}, validation_sample_metrics={self.enable_validation_sample_metrics}, "
            f"validation_sample_batch_size={self.validation_sample_batch_size}, "
            f"validation_sample_metric_batches={self.validation_sample_metric_batches}"
        )

    @torch.no_grad()
    def sample_latents(self, batch_size, nfe=None, device=None, condition=None, noise=None):
        del nfe
        if condition is not None:
            raise ValueError("TokenCodeAutoregressivePriorTrainer is unconditional and does not accept condition.")
        if device is None:
            device = self.device
        batch_size = int(batch_size)
        generated_tokens = torch.empty(
            (batch_size, self.code_sequence_length),
            device=device,
            dtype=torch.long,
        )
        generator = self._sample_generator(device=device, noise=noise)
        if self._supports_full_context_kv_cache():
            current_token = torch.full(
                (batch_size, 1),
                fill_value=self.bos_token_id,
                device=device,
                dtype=torch.long,
            )
            past = None
            for step_idx in range(self.code_sequence_length):
                logits, _, present = self.backbone.forward_with_past(
                    current_token,
                    past=past,
                    past_length=(step_idx if past is not None else None),
                )
                next_logits = logits[:, -1, :] / float(max(self.sample_temperature, 1.0e-6))
                next_token = self._sample_next_token(next_logits, generator=generator)
                generated_tokens[:, step_idx] = next_token[:, 0]
                if past is None:
                    past = [present]
                else:
                    past.append(present)
                current_token = next_token
            sampled_codes = self._sequence_to_codes(generated_tokens)
            self._validate_sampled_codes(sampled_codes)
            return sampled_codes

        context_tokens = torch.full(
            (batch_size, 1),
            fill_value=self.bos_token_id,
            device=device,
            dtype=torch.long,
        )
        for step_idx in range(self.code_sequence_length):
            logits, _ = self.backbone(context_tokens, targets=None)
            next_logits = logits[:, -1, :] / float(max(self.sample_temperature, 1.0e-6))
            next_token = self._sample_next_token(next_logits, generator=generator)
            generated_tokens[:, step_idx] = next_token[:, 0]
            # Keep the sampling context capped at block_size while preserving the
            # existing sliding-window semantics that reindex positions inside each crop.
            if int(context_tokens.shape[1]) < self.context_length:
                context_tokens = torch.cat([context_tokens, next_token], dim=1)
            else:
                context_tokens = torch.cat([context_tokens[:, 1:], next_token], dim=1)
        sampled_codes = self._sequence_to_codes(generated_tokens)
        self._validate_sampled_codes(sampled_codes)
        return sampled_codes

    def configure_optimizers(self):
        lr = float(getattr(self, "learning_rate", 1.0e-4))
        optimizer_groups = self.backbone.optimizer_groups(weight_decay=self.weight_decay)
        return torch.optim.AdamW(optimizer_groups, lr=lr, betas=self.optimizer_betas)

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

import math
import traceback
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.distributed as dist
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
        semantic_ce_use_class_weights=False,
        semantic_dice_weight=0.0,
        boundary_loss_weight=0.0,
        area_ratio_loss_weight=0.0,
        adjacency_loss_weight=0.0,
        log_sample_nfe=4,
        enable_validation_sample_metrics=True,
        validation_sample_batch_size=4,
        validation_sample_nfe=None,
        validation_sample_seed=1234,
        monitor="val/sampled_monitor_error",
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
        self.semantic_ce_use_class_weights = bool(semantic_ce_use_class_weights)
        self.semantic_dice_weight = float(semantic_dice_weight)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.area_ratio_loss_weight = float(area_ratio_loss_weight)
        self.adjacency_loss_weight = float(adjacency_loss_weight)
        self.enable_validation_sample_metrics = bool(enable_validation_sample_metrics)
        self.validation_sample_batch_size = max(0, int(validation_sample_batch_size))
        self.validation_sample_nfe = (
            int(log_sample_nfe) if validation_sample_nfe is None else max(1, int(validation_sample_nfe))
        )
        self.validation_sample_seed = int(validation_sample_seed)
        self.per_class_metric_logging_limit = 32
        if self.monitor == "val/sampled_monitor_error" and (
            not self.enable_validation_sample_metrics or self.validation_sample_batch_size <= 0
        ):
            raise ValueError(
                "TokenMaskPriorTrainer monitor=val/sampled_monitor_error requires "
                "enable_validation_sample_metrics=True and validation_sample_batch_size > 0."
            )

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

    def _is_global_zero(self):
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "is_global_zero"):
            return bool(trainer.is_global_zero)
        return int(getattr(self, "global_rank", 0)) == 0

    def _log_info(self, message):
        if self._is_global_zero():
            print(message)

    def _get_train_dataset(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            raise RuntimeError("TokenMaskPriorTrainer requires an attached trainer to inspect the train dataset.")

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is not None:
            datasets = getattr(datamodule, "datasets", None)
            if isinstance(datasets, dict) and "train" in datasets:
                dataset = datasets["train"]
                if hasattr(dataset, "data") and hasattr(dataset.data, "__len__") and hasattr(dataset.data, "__getitem__"):
                    return dataset.data
                return dataset

        train_dataloader = getattr(trainer, "train_dataloader", None)
        if train_dataloader is not None:
            if isinstance(train_dataloader, (list, tuple)):
                train_dataloader = train_dataloader[0]
            dataset = getattr(train_dataloader, "dataset", None)
            if dataset is not None:
                if hasattr(dataset, "data") and hasattr(dataset.data, "__len__") and hasattr(dataset.data, "__getitem__"):
                    return dataset.data
                return dataset

        raise RuntimeError("TokenMaskPriorTrainer could not resolve the train dataset for code-count scanning.")

    def _batchify_dataset_sample(self, sample):
        if not isinstance(sample, dict):
            raise TypeError(
                "TokenMaskPriorTrainer class-balance scan expects dataset samples to be dictionaries, "
                f"got {type(sample).__name__}."
            )

        batch = {}
        for key in (self.mask_index_key, self.mask_key, "num_classes"):
            if key not in sample:
                continue
            value = sample[key]
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                try:
                    tensor = torch.as_tensor(value)
                except Exception:
                    batch[key] = value
                    continue
            batch[key] = tensor.unsqueeze(0).to(device=self.device)

        if self.mask_index_key not in batch and self.mask_key not in batch:
            raise KeyError(
                "TokenMaskPriorTrainer class-balance scan requires train dataset samples to include "
                f"'{self.mask_index_key}' and/or '{self.mask_key}'."
            )
        return batch

    def _scan_train_semantic_class_counts(self):
        dataset = self._get_train_dataset()
        dataset_length = int(len(dataset))
        counts = torch.zeros(self.num_classes, dtype=torch.float64)
        self._log_info(
            "[TokenMaskPriorTrainer] "
            f"scanning train semantic class counts for tokenizer CE over {dataset_length} samples"
        )
        progress_stride = max(min(dataset_length // 4, 500), 100)
        for sample_idx in range(dataset_length):
            sample = dataset[sample_idx]
            if self.mask_index_key not in sample:
                raise KeyError(
                    "TokenMaskPriorTrainer semantic class-count scan requires train dataset samples to include "
                    f"'{self.mask_index_key}'."
                )
            mask_index = self._prepare_metric_mask_index(sample[self.mask_index_key]).view(-1)
            if self.ignore_index is not None:
                mask_index = mask_index[mask_index != int(self.ignore_index)]
            if int(mask_index.numel()) <= 0:
                continue
            bincount = torch.bincount(mask_index.clamp(min=0), minlength=self.num_classes)
            counts += bincount[: self.num_classes].to(dtype=counts.dtype)
            if (sample_idx + 1) % progress_stride == 0 or (sample_idx + 1) == dataset_length:
                self._log_info(
                    "[TokenMaskPriorTrainer] "
                    f"semantic class-count scan progress {sample_idx + 1}/{dataset_length}"
                )
        return counts

    def _scan_train_code_counts(self):
        dataset = self._get_train_dataset()
        dataset_length = int(len(dataset))
        counts = torch.zeros(self.codebook_size, dtype=torch.float64)
        self._log_info(
            "[TokenMaskPriorTrainer] "
            f"scanning train tokenizer code counts for class-balanced objective over {dataset_length} samples"
        )
        progress_stride = max(min(dataset_length // 4, 500), 100)
        for sample_idx in range(dataset_length):
            sample = dataset[sample_idx]
            encoded = self.tokenizer.encode_batch(self._batchify_dataset_sample(sample), sample_posterior=False)
            if "codes" not in encoded:
                raise KeyError(
                    "Frozen tokenizer did not return 'codes' during class-balance scan; "
                    "token-code prior balancing requires discrete tokenizer codes."
                )
            codes = self._prepare_codes(encoded["codes"]).view(-1).detach().cpu()
            bincount = torch.bincount(codes, minlength=self.codebook_size)
            counts += bincount[: self.codebook_size].to(dtype=counts.dtype)
            if (sample_idx + 1) % progress_stride == 0 or (sample_idx + 1) == dataset_length:
                self._log_info(
                    "[TokenMaskPriorTrainer] "
                    f"code-count scan progress {sample_idx + 1}/{dataset_length}"
                )
        return counts

    def _broadcast_count_vector(self, counts, *, vector_length, scan_failed=False, failure_message):
        device = self.device if isinstance(self.device, torch.device) else torch.device("cpu")
        if device.type not in {"cpu", "cuda"}:
            device = torch.device("cpu")
        status_tensor = torch.ones(1, dtype=torch.int64, device=device)
        counts_tensor = torch.zeros(vector_length, dtype=torch.float64, device=device)
        if scan_failed:
            status_tensor.zero_()
        elif counts is not None:
            counts_tensor.copy_(counts.to(device=device, dtype=torch.float64))
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(status_tensor, src=0)
            if int(status_tensor.item()) == 0:
                raise RuntimeError(failure_message)
            dist.broadcast(counts_tensor, src=0)
        elif int(status_tensor.item()) == 0:
            raise RuntimeError(failure_message)
        return counts_tensor.cpu().to(dtype=torch.float32)

    def _get_tokenizer_loss_module(self):
        return getattr(self.tokenizer.tokenizer, "loss", None)

    def _maybe_configure_tokenizer_loss_class_balance(self):
        if not self.semantic_ce_use_class_weights:
            return

        loss_module = self._get_tokenizer_loss_module()
        configure_class_balance = getattr(loss_module, "configure_class_balance", None)
        needs_class_count_scan = getattr(loss_module, "needs_class_count_scan", None)
        if not callable(configure_class_balance) or not callable(needs_class_count_scan):
            return
        if not bool(needs_class_count_scan()):
            return

        class_counts = None
        scan_failed = False
        if self._is_global_zero():
            try:
                class_counts = self._scan_train_semantic_class_counts()
            except Exception:
                scan_failed = True
                print("[TokenMaskPriorTrainer] tokenizer semantic class-count scan failed on global rank 0.", flush=True)
                traceback.print_exc()
        class_counts = self._broadcast_count_vector(
            class_counts,
            vector_length=self.num_classes,
            scan_failed=scan_failed,
            failure_message=(
                "TokenMaskPriorTrainer tokenizer semantic class-count scan failed on global rank 0. "
                "See the rank-0 traceback above."
            ),
        )
        configure_class_balance(class_counts)
        class_weight_summary = getattr(loss_module, "class_weight_summary", None)
        weight_summary = class_weight_summary() if callable(class_weight_summary) else None
        summary_text = ""
        if isinstance(weight_summary, dict):
            summary_text = (
                ", "
                f"class_weight_min={float(weight_summary['class_weight_min']):.4f}, "
                f"class_weight_max={float(weight_summary['class_weight_max']):.4f}, "
                f"class_weight_mean={float(weight_summary['class_weight_mean']):.4f}"
            )
        self._log_info(
            "[TokenMaskPriorTrainer] "
            f"tokenizer_semantic_class_balance_mode={getattr(loss_module, 'class_balance_mode', 'none')}, "
            f"class_counts={class_counts.tolist()}{summary_text}"
        )

    def _maybe_configure_objective_class_balance(self):
        configure_class_balance = getattr(self.objective, "configure_class_balance", None)
        needs_class_count_scan = getattr(self.objective, "needs_class_count_scan", None)
        if not callable(configure_class_balance) or not callable(needs_class_count_scan):
            return
        if not bool(needs_class_count_scan()):
            return

        class_counts = None
        scan_failed = False
        if self._is_global_zero():
            try:
                class_counts = self._scan_train_code_counts()
            except Exception:
                scan_failed = True
                print("[TokenMaskPriorTrainer] code-count scan failed on global rank 0.", flush=True)
                traceback.print_exc()
        class_counts = self._broadcast_count_vector(
            class_counts,
            vector_length=self.codebook_size,
            scan_failed=scan_failed,
            failure_message=(
                "TokenMaskPriorTrainer code-count scan failed on global rank 0. "
                "See the rank-0 traceback above."
            ),
        )
        configure_class_balance(class_counts)
        self._log_info(
            "[TokenMaskPriorTrainer] "
            f"objective_code_count_scan active_codes={int((class_counts > 0).sum().item())}/{self.codebook_size}"
        )

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
        self._maybe_configure_tokenizer_loss_class_balance()
        self._maybe_configure_objective_class_balance()
        self._log_info(
            "[TokenMaskPriorTrainer] "
            f"tokenizer_config={self.tokenizer_config_path}, tokenizer_ckpt={self.tokenizer_ckpt_path}, "
            f"codebook_size={self.codebook_size}, token_spatial_shape={self.token_spatial_shape}, "
            f"mask_spatial_shape={self.mask_spatial_shape}, objective={self.objective_name}, "
            f"monitor={self.monitor}, "
            f"semantic_ce_weight={self.semantic_ce_weight:.3f}, "
            f"semantic_ce_use_class_weights={self.semantic_ce_use_class_weights}, "
            f"semantic_dice_weight={self.semantic_dice_weight:.3f}, "
            f"boundary_loss_weight={self.boundary_loss_weight:.3f}, "
            f"area_ratio_loss_weight={self.area_ratio_loss_weight:.3f}, "
            f"adjacency_loss_weight={self.adjacency_loss_weight:.3f}, "
            f"validation_sample_metrics={self.enable_validation_sample_metrics}, "
            f"validation_sample_batch_size={self.validation_sample_batch_size}, "
            f"validation_sample_nfe={self.validation_sample_nfe}"
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

    def _prepare_metric_mask_index(self, mask_index):
        if not isinstance(mask_index, torch.Tensor):
            mask_index = torch.as_tensor(mask_index)
        if mask_index.ndim == 2:
            mask_index = mask_index.unsqueeze(0)
        if mask_index.ndim == 4 and int(mask_index.shape[1]) == 1:
            mask_index = mask_index[:, 0]
        if mask_index.ndim != 3:
            raise ValueError(
                "TokenMaskPriorTrainer semantic diagnostics expect mask_index with shape [B, H, W] or [B, 1, H, W], "
                f"got {tuple(mask_index.shape)}"
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

    def _distribution_monitor_error(self, *, class_hist_l1, boundary_ratio_gap, unique_class_count_gap):
        return (
            class_hist_l1
            + boundary_ratio_gap
            + (unique_class_count_gap / float(max(self.num_classes, 1)))
        ).detach()

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
                unique_class_count_gap=unique_class_count_gap,
            ),
        }
        if int(self.num_classes) <= int(self.per_class_metric_logging_limit):
            for class_idx in range(int(self.num_classes)):
                metrics[f"{prefix}_pred_class_ratio_{class_idx}"] = pred_summary["mean_class_ratios"][class_idx]
                metrics[f"{prefix}_target_class_ratio_{class_idx}"] = target_summary["mean_class_ratios"][class_idx]
                metrics[f"{prefix}_class_ratio_gap_{class_idx}"] = class_ratio_gap[class_idx]
        return metrics

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
            use_class_weights=self.semantic_ce_use_class_weights,
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
            semantic_metrics.update(
                self._mask_distribution_gap_metrics(
                    pred_mask_index=decoded["mask_index"],
                    target_mask_index=mask_index,
                    prefix="teacher_forced",
                )
            )
        return {
            "decoded": decoded,
            "semantic_ce": semantic_losses["semantic_ce"],
            "semantic_dice": semantic_losses["semantic_dice"],
            "semantic_aux_total": semantic_aux_total,
            "semantic_metrics": {
                "semantic_pixel_accuracy": semantic_metrics["pixel_accuracy"].detach(),
                "semantic_miou": semantic_metrics["miou"].detach(),
                **{
                    name: value.detach()
                    for name, value in semantic_metrics.items()
                    if name not in {"pixel_accuracy", "miou"}
                },
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
        return outputs["loss"], metrics.get(f"{split}/loss", outputs["loss"].detach()), outputs

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, split="train")
        return loss

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
        if batch_idx == 0 and self._is_global_zero():
            sample_metrics = self._validation_sample_metrics(outputs["mask_index"].detach())
            if sample_metrics:
                self.log_dict(
                    {f"val/{name}": value for name, value in sample_metrics.items()},
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=min(self._get_log_batch_size(batch), self.validation_sample_batch_size),
                    sync_dist=False,
                )
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

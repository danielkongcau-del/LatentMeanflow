import hashlib
import math
import traceback
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ldm.util import instantiate_from_config

from latent_meanflow.losses.semantic_structure import (
    adjacency_l1_loss,
    area_ratio_l1_loss,
    boundary_bce_loss,
    build_valid_mask,
    mask_index_to_boundary_target,
    semantic_probs_to_soft_boundary,
)
from latent_meanflow.models.backbones.token_code_mingpt import ensure_taming_transformers_on_path
from latent_meanflow.models.tokenizer import SemanticTokenizerAdapter
from latent_meanflow.samplers.token_code_maskgit_sampler import TokenCodeMaskGitSampler


class TokenCodeMaskGitPriorTrainer(pl.LightningModule):
    def __init__(
        self,
        tokenizer_config_path,
        tokenizer_ckpt_path,
        backbone_config,
        objective_name="token_code_maskgit",
        permuter_config=None,
        freeze_tokenizer=True,
        tokenizer_sample_posterior=False,
        log_sample_nfe=8,
        enable_validation_sample_metrics=True,
        validation_sample_batch_size=4,
        validation_sample_metric_batches=4,
        validation_sample_nfe=None,
        validation_sample_seed=1234,
        monitor="val/sampled_monitor_error",
        label_smoothing=0.1,
        sample_temperature=1.0,
        sample_top_k=None,
        sample_base_gumbel_temp=4.5,
        corruption_mode="exact_count",
        full_mask_batch_fraction=0.25,
        high_mask_batch_fraction=0.50,
        high_mask_min_ratio=0.85,
        semantic_ce_weight=1.0,
        semantic_ce_use_class_weights=True,
        semantic_dice_weight=0.5,
        boundary_loss_weight=0.25,
        area_ratio_loss_weight=0.10,
        adjacency_loss_weight=0.10,
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
                "TokenCodeMaskGitPriorTrainer requires tokenizer_ckpt_path to be set explicitly. "
                "Freeze the balanced VQ tokenizer checkpoint and pass it via config or CLI override."
            )
        if bool(tokenizer_sample_posterior):
            raise ValueError(
                "TokenCodeMaskGitPriorTrainer requires tokenizer_sample_posterior=False. "
                "The frozen VQ tokenizer exposes deterministic code indices."
            )
        if not bool(freeze_tokenizer):
            raise ValueError(
                "TokenCodeMaskGitPriorTrainer requires freeze_tokenizer=True. "
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
        self.label_smoothing = float(label_smoothing)
        self.sample_temperature = float(sample_temperature)
        self.sample_top_k = None if sample_top_k in {None, ""} else max(1, int(sample_top_k))
        self.sample_base_gumbel_temp = float(sample_base_gumbel_temp)
        self.corruption_mode = str(corruption_mode).lower()
        self.full_mask_batch_fraction = float(full_mask_batch_fraction)
        self.high_mask_batch_fraction = float(high_mask_batch_fraction)
        self.high_mask_min_ratio = float(high_mask_min_ratio)
        self.semantic_ce_weight = float(semantic_ce_weight)
        self.semantic_ce_use_class_weights = bool(semantic_ce_use_class_weights)
        self.semantic_dice_weight = float(semantic_dice_weight)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.area_ratio_loss_weight = float(area_ratio_loss_weight)
        self.adjacency_loss_weight = float(adjacency_loss_weight)
        self.weight_decay = float(weight_decay)
        self.optimizer_betas = (float(optimizer_beta1), float(optimizer_beta2))
        self.per_class_metric_logging_limit = 32
        self.supports_nfe_sweep = True
        self.route_family = "maskgit"
        self._reset_validation_sample_metric_state()

        self.tokenizer = SemanticTokenizerAdapter.from_pretrained(
            config_path=self.tokenizer_config_path,
            ckpt_path=self.tokenizer_ckpt_path,
            eval_mode=True,
            freeze=True,
        )
        if any(param.requires_grad for param in self.tokenizer.parameters()):
            raise ValueError("Frozen token-code MaskGIT prior requires tokenizer parameters to stay non-trainable.")

        self.ignore_index = self.tokenizer.ignore_index
        self.num_classes = int(self.tokenizer.num_classes)
        self.codebook_size = int(self.tokenizer.codebook_size)
        self.mask_token_id = int(self.codebook_size)
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
        backbone_cfg["params"]["vocab_size"] = int(self.codebook_size)
        backbone_cfg["params"]["n_tokens"] = int(self.code_sequence_length)
        self.backbone = instantiate_from_config(backbone_cfg)
        self.sampler = TokenCodeMaskGitSampler(
            codebook_size=self.codebook_size,
            mask_token_id=self.mask_token_id,
            default_nfe=self.log_sample_nfe,
            mask_schedule_type=getattr(self.backbone, "mask_schedule_type", "cosine"),
            sample_temperature=self.sample_temperature,
            top_k=self.sample_top_k,
            base_gumbel_temp=self.sample_base_gumbel_temp,
        )

        if self.monitor.startswith("val/sampled_") and (
            not self.enable_validation_sample_metrics
            or self.validation_sample_batch_size <= 0
            or self.validation_sample_metric_batches <= 0
        ):
            raise ValueError(
                f"TokenCodeMaskGitPriorTrainer monitor={self.monitor} requires "
                "enable_validation_sample_metrics=True, validation_sample_batch_size > 0, "
                "and validation_sample_metric_batches > 0."
            )
        if self.corruption_mode not in {"exact_count", "bernoulli"}:
            raise ValueError(
                "TokenCodeMaskGitPriorTrainer corruption_mode must be one of "
                f"{{'exact_count', 'bernoulli'}}, got {self.corruption_mode!r}"
            )
        if not 0.0 <= self.full_mask_batch_fraction <= 1.0:
            raise ValueError(
                "full_mask_batch_fraction must be in [0, 1], got "
                f"{self.full_mask_batch_fraction}"
            )
        if not 0.0 <= self.high_mask_batch_fraction <= 1.0:
            raise ValueError(
                "high_mask_batch_fraction must be in [0, 1], got "
                f"{self.high_mask_batch_fraction}"
            )
        if (self.full_mask_batch_fraction + self.high_mask_batch_fraction) > 1.0 + 1.0e-8:
            raise ValueError(
                "full_mask_batch_fraction + high_mask_batch_fraction must be <= 1.0, got "
                f"{self.full_mask_batch_fraction + self.high_mask_batch_fraction}"
            )
        if not 0.0 <= self.high_mask_min_ratio <= 1.0:
            raise ValueError(f"high_mask_min_ratio must be in [0, 1], got {self.high_mask_min_ratio}")

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
                "TokenCodeMaskGitPriorTrainer expects code indices with shape [B, Ht, Wt], "
                f"got {tuple(codes.shape)}"
            )
        if tuple(int(v) for v in codes.shape[-2:]) != self.token_spatial_shape:
            raise ValueError(
                "TokenCodeMaskGitPriorTrainer code-grid shape mismatch: "
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
                "TokenCodeMaskGitPriorTrainer semantic diagnostics expect mask_index with shape "
                f"[B, H, W] or [B, 1, H, W], got {tuple(mask_index.shape)}"
            )
        return mask_index.long().contiguous()

    def _prepare_mask_onehot(self, mask_onehot):
        if not isinstance(mask_onehot, torch.Tensor):
            mask_onehot = torch.as_tensor(mask_onehot)
        if mask_onehot.ndim != 4:
            raise ValueError(
                "TokenCodeMaskGitPriorTrainer semantic structure losses expect mask_onehot with shape "
                f"[B, H, W, K] or [B, K, H, W], got {tuple(mask_onehot.shape)}"
            )
        if int(mask_onehot.shape[-1]) == self.num_classes:
            channel_first = mask_onehot.permute(0, 3, 1, 2)
        elif int(mask_onehot.shape[1]) == self.num_classes:
            channel_first = mask_onehot
        else:
            raise ValueError(
                f"mask_onehot class dimension mismatch: expected num_classes={self.num_classes}, got {tuple(mask_onehot.shape)}"
            )
        return channel_first.to(dtype=torch.float32).contiguous()

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

    def _get_log_batch_size(self, batch):
        if self.mask_index_key in batch:
            return int(torch.as_tensor(batch[self.mask_index_key]).shape[0])
        if self.mask_key in batch:
            return int(torch.as_tensor(batch[self.mask_key]).shape[0])
        raise KeyError(
            "TokenCodeMaskGitPriorTrainer requires "
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

    def _broadcast_from_rank_zero(self, value):
        trainer = getattr(self, "trainer", None)
        strategy = getattr(trainer, "strategy", None) if trainer is not None else None
        broadcast = getattr(strategy, "broadcast", None) if strategy is not None else None
        if self._should_sync_dist() and callable(broadcast):
            return broadcast(value, src=0)
        return value

    def _broadcast_metric_dict(self, metrics):
        payload = {}
        for name, value in dict(metrics or {}).items():
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    continue
                payload[str(name)] = float(value.detach().to(device=torch.device("cpu")).item())
            elif isinstance(value, (int, float)):
                payload[str(name)] = float(value)
        payload = self._broadcast_from_rank_zero(payload)
        device = self.device if isinstance(self.device, torch.device) else torch.device("cpu")
        if device.type not in {"cpu", "cuda"}:
            device = torch.device("cpu")
        return {
            str(name): torch.tensor(float(value), device=device, dtype=torch.float32)
            for name, value in dict(payload or {}).items()
        }

    def _broadcast_metric_batch_count(self, batches_seen):
        return int(self._broadcast_from_rank_zero(int(batches_seen)))

    def _log_info(self, message):
        if self._is_global_zero():
            print(message)

    def _get_train_dataset(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            raise RuntimeError("TokenCodeMaskGitPriorTrainer requires an attached trainer to inspect the train dataset.")

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

        raise RuntimeError("TokenCodeMaskGitPriorTrainer could not resolve the train dataset for class-count scanning.")

    def _batchify_dataset_sample(self, sample):
        if not isinstance(sample, dict):
            raise TypeError(
                "TokenCodeMaskGitPriorTrainer class-balance scan expects dataset samples to be dictionaries, "
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
                "TokenCodeMaskGitPriorTrainer class-balance scan requires train dataset samples to include "
                f"'{self.mask_index_key}' and/or '{self.mask_key}'."
            )
        return batch

    def _scan_train_semantic_class_counts(self):
        dataset = self._get_train_dataset()
        dataset_length = int(len(dataset))
        counts = torch.zeros(self.num_classes, dtype=torch.float64)
        self._log_info(
            "[TokenCodeMaskGitPriorTrainer] "
            f"scanning train semantic class counts for semantic CE over {dataset_length} samples"
        )
        progress_stride = max(min(dataset_length // 4, 500), 100)
        for sample_idx in range(dataset_length):
            sample = dataset[sample_idx]
            if self.mask_index_key not in sample:
                raise KeyError(
                    "TokenCodeMaskGitPriorTrainer semantic class-count scan requires train dataset samples to include "
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
                    "[TokenCodeMaskGitPriorTrainer] "
                    f"semantic class-count scan progress {sample_idx + 1}/{dataset_length}"
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
                print(
                    "[TokenCodeMaskGitPriorTrainer] tokenizer semantic class-count scan failed on global rank 0.",
                    flush=True,
                )
                traceback.print_exc()
        class_counts = self._broadcast_count_vector(
            class_counts,
            vector_length=self.num_classes,
            scan_failed=scan_failed,
            failure_message=(
                "TokenCodeMaskGitPriorTrainer tokenizer semantic class-count scan failed on global rank 0. "
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
            "[TokenCodeMaskGitPriorTrainer] "
            f"tokenizer_semantic_class_balance_mode={getattr(loss_module, 'class_balance_mode', 'none')}, "
            f"class_counts={class_counts.tolist()}{summary_text}"
        )

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
                "TokenCodeMaskGitPriorTrainer expects sequence tokens with shape [B, T], "
                f"got {tuple(sequence.shape)}"
            )
        if int(sequence.shape[1]) != self.code_sequence_length:
            raise ValueError(
                "TokenCodeMaskGitPriorTrainer sequence length mismatch: "
                f"expected {self.code_sequence_length}, got {int(sequence.shape[1])}"
            )
        sequence = self.permuter(sequence.long(), reverse=True)
        return sequence.view(int(sequence.shape[0]), *self.token_spatial_shape).contiguous()

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

    def _token_usage_metrics(self, code_grid):
        codes = self._prepare_codes(code_grid)
        flat_codes = codes.view(int(codes.shape[0]), -1)
        code_counts = torch.bincount(flat_codes.reshape(-1), minlength=self.codebook_size).to(dtype=torch.float32)
        probs = code_counts / code_counts.sum().clamp_min(1.0)
        unique_per_sample = (F.one_hot(flat_codes, num_classes=self.codebook_size).sum(dim=1) > 0).sum(dim=1)
        return {
            "target_unique_code_count_mean": unique_per_sample.float().mean().detach(),
            "target_unique_code_fraction_mean": (
                unique_per_sample.to(dtype=torch.float32) / float(max(self.codebook_size, 1))
            ).mean().detach(),
            "target_active_code_count": (code_counts > 0).sum().to(dtype=torch.float32).detach(),
            "target_active_code_fraction": (
                (code_counts > 0).sum().to(dtype=torch.float32) / float(max(self.codebook_size, 1))
            ).detach(),
            "target_code_perplexity": torch.exp(
                -(probs * torch.log(probs.clamp_min(1.0e-10))).sum()
            ).detach(),
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

    def encode_batch(self, batch):
        encoded = self.tokenizer.encode_batch(batch, sample_posterior=False)
        if "codes" not in encoded:
            raise KeyError(
                "Frozen tokenizer did not return 'codes'; token-code MaskGIT generation requires discrete codes."
            )
        codes = self._prepare_codes(encoded["codes"])
        if torch.any(codes < 0) or torch.any(codes >= self.codebook_size):
            raise ValueError(f"Tokenizer encode produced code ids outside [0, {self.codebook_size - 1}].")
        return {
            "codes": codes,
            "z": codes,
            "mask_index": encoded["mask_index"],
            "mask_onehot": self._prepare_mask_onehot(encoded["mask_onehot"]),
            "quantizer_stats": encoded.get("quantizer_stats", None),
        }

    def decode_latents(self, z):
        codes = self._prepare_codes(z)
        self._validate_sampled_codes(codes)
        decoded = self.tokenizer.decode_codes(codes)
        decoded["codes"] = codes
        return decoded

    def _sample_effective_mask_ratio(self, *, batch_size, device):
        base_ratio = torch.rand(batch_size, device=device, dtype=torch.float32)
        effective_mask_ratio = base_ratio.clone()

        mode_selector = torch.rand(batch_size, device=device, dtype=torch.float32)
        full_mask_rows = mode_selector < self.full_mask_batch_fraction
        high_mask_threshold = self.full_mask_batch_fraction + self.high_mask_batch_fraction
        high_mask_rows = (~full_mask_rows) & (mode_selector < high_mask_threshold)
        if torch.any(full_mask_rows):
            effective_mask_ratio[full_mask_rows] = 1.0
        if torch.any(high_mask_rows):
            high_uniform = torch.rand(batch_size, device=device, dtype=torch.float32)
            high_ratio = self.high_mask_min_ratio + (1.0 - self.high_mask_min_ratio) * high_uniform
            effective_mask_ratio[high_mask_rows] = high_ratio[high_mask_rows]
        return effective_mask_ratio, full_mask_rows, high_mask_rows

    def _sample_exact_count_mask(self, *, batch_size, sequence_length, effective_mask_ratio, device):
        target_masked_counts = torch.round(
            effective_mask_ratio * torch.full((batch_size,), float(sequence_length), device=device)
        ).to(dtype=torch.long)
        target_masked_counts = target_masked_counts.clamp(min=1, max=int(sequence_length))
        random_scores = torch.rand((batch_size, sequence_length), device=device, dtype=torch.float32)
        mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
        for batch_idx in range(batch_size):
            target_count = int(target_masked_counts[batch_idx].item())
            selected = torch.topk(random_scores[batch_idx], k=target_count, largest=True).indices
            mask[batch_idx, selected] = True
        return mask, target_masked_counts

    def _sample_bernoulli_mask(self, *, batch_size, sequence_length, effective_mask_ratio, device):
        mask = (
            torch.rand((batch_size, sequence_length), device=device, dtype=torch.float32)
            < effective_mask_ratio.view(batch_size, 1)
        )
        masked_counts = mask.sum(dim=1)
        force_rows = masked_counts == 0
        if torch.any(force_rows):
            force_indices = torch.randint(
                low=0,
                high=int(sequence_length),
                size=(int(force_rows.sum().item()),),
                device=device,
            )
            mask[force_rows, force_indices] = True
            masked_counts = mask.sum(dim=1)
        return mask, masked_counts.to(dtype=torch.long)

    def _build_masked_modeling_inputs(self, code_sequence):
        batch_size, sequence_length = code_sequence.shape
        effective_mask_ratio, full_mask_rows, high_mask_rows = self._sample_effective_mask_ratio(
            batch_size=batch_size,
            device=code_sequence.device,
        )
        if self.corruption_mode == "exact_count":
            mask, target_masked_counts = self._sample_exact_count_mask(
                batch_size=batch_size,
                sequence_length=sequence_length,
                effective_mask_ratio=effective_mask_ratio,
                device=code_sequence.device,
            )
        elif self.corruption_mode == "bernoulli":
            mask, target_masked_counts = self._sample_bernoulli_mask(
                batch_size=batch_size,
                sequence_length=sequence_length,
                effective_mask_ratio=effective_mask_ratio,
                device=code_sequence.device,
            )
        else:
            raise ValueError(f"Unsupported corruption_mode: {self.corruption_mode}")

        target_tokens = code_sequence.clone()
        masked_input_tokens = torch.where(mask, torch.full_like(code_sequence, self.mask_token_id), code_sequence)
        loss_targets = target_tokens.masked_fill(~mask, -100)
        return {
            "masked_input_tokens": masked_input_tokens,
            "target_tokens": target_tokens,
            "loss_targets": loss_targets,
            "mask_positions": mask,
            "effective_mask_ratio": effective_mask_ratio,
            "target_masked_counts": target_masked_counts,
            "full_mask_rows": full_mask_rows,
            "high_mask_rows": high_mask_rows,
        }

    def _sequence_logits_to_code_logits(self, token_logits):
        if token_logits.ndim != 3:
            raise ValueError(
                "TokenCodeMaskGitPriorTrainer expects token_logits with shape [B, T, K], "
                f"got {tuple(token_logits.shape)}"
            )
        if int(token_logits.shape[1]) != self.code_sequence_length or int(token_logits.shape[2]) != self.codebook_size:
            raise ValueError(
                "TokenCodeMaskGitPriorTrainer token_logits shape mismatch: expected "
                f"[B, {self.code_sequence_length}, {self.codebook_size}], got {tuple(token_logits.shape)}"
            )
        permuted = self.permuter(token_logits, reverse=True)
        return permuted.view(
            int(token_logits.shape[0]),
            int(self.token_spatial_shape[0]),
            int(self.token_spatial_shape[1]),
            int(self.codebook_size),
        ).permute(0, 3, 1, 2).contiguous()

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
                "TokenCodeMaskGitPriorTrainer semantic bridge expects code logits with shape "
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
        structure_aux_total = (
            self.boundary_loss_weight * boundary_loss
            + self.area_ratio_loss_weight * area_ratio_loss
            + self.adjacency_loss_weight * adjacency_loss
        )
        return {
            "boundary_loss": boundary_loss,
            "area_ratio_loss": area_ratio_loss,
            "adjacency_loss": adjacency_loss,
            "structure_aux_total": structure_aux_total,
            "boundary_target": boundary_target.detach(),
            "boundary_pred": boundary_pred.detach(),
            "pred_area_ratio": pred_area_ratio.detach(),
            "target_area_ratio": target_area_ratio.detach(),
            "pred_adjacency": pred_adjacency.detach(),
            "target_adjacency": target_adjacency.detach(),
        }

    def _sample_sequence_logits(self, token_sequence):
        return self.backbone(token_sequence)

    def forward(self, batch, objective_step=None):
        del objective_step
        encoded = self.encode_batch(batch)
        code_grid = encoded["codes"]
        code_sequence = self._codes_to_sequence(code_grid)
        masking_outputs = self._build_masked_modeling_inputs(code_sequence)
        masked_input_tokens = masking_outputs["masked_input_tokens"]
        target_tokens = masking_outputs["target_tokens"]
        loss_targets = masking_outputs["loss_targets"]
        mask = masking_outputs["mask_positions"]
        logits = self.backbone(masked_input_tokens)
        code_loss = F.cross_entropy(
            logits.reshape(-1, self.codebook_size),
            loss_targets.reshape(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )
        total_loss = code_loss
        loss_dict = {
            "maskgit_ce": code_loss.detach(),
            "base_error_mean": code_loss.detach(),
        }
        semantic_bridge_metrics = {}
        semantic_outputs = {}
        structure_outputs = {}
        if self._semantic_decode_bridge_enabled():
            code_logits = self._sequence_logits_to_code_logits(logits)
            semantic_outputs = self._semantic_auxiliary_terms(
                code_logits=code_logits,
                mask_index=encoded["mask_index"],
            )
            structure_outputs = self._semantic_structure_terms(
                mask_probs=semantic_outputs["decoded"]["mask_probs"],
                mask_index=encoded["mask_index"],
                mask_onehot=encoded["mask_onehot"],
            )
            total_loss = total_loss + semantic_outputs["semantic_aux_total"] + structure_outputs["structure_aux_total"]
            loss_dict.update(
                {
                    "semantic_ce": semantic_outputs["semantic_ce"].detach(),
                    "semantic_dice": semantic_outputs["semantic_dice"].detach(),
                    "semantic_aux_total": semantic_outputs["semantic_aux_total"].detach(),
                    "boundary_loss": structure_outputs["boundary_loss"].detach(),
                    "area_ratio_loss": structure_outputs["area_ratio_loss"].detach(),
                    "adjacency_loss": structure_outputs["adjacency_loss"].detach(),
                    "structure_aux_total": structure_outputs["structure_aux_total"].detach(),
                }
            )
            semantic_bridge_metrics = semantic_outputs["semantic_metrics"]
        loss_dict["total_loss"] = total_loss.detach()

        predicted_tokens = torch.argmax(logits, dim=-1)
        masked_token_count = mask.sum().clamp_min(1)
        masked_token_accuracy = (
            (predicted_tokens[mask] == target_tokens[mask]).float().mean().detach()
            if bool(torch.any(mask))
            else torch.tensor(1.0, device=logits.device)
        )
        maskgit_metrics = {
            "masked_token_accuracy": masked_token_accuracy,
            "masked_token_fraction": (mask.sum().to(dtype=torch.float32) / float(max(mask.numel(), 1))).detach(),
            "masked_token_count": masked_token_count.to(dtype=torch.float32).detach(),
            "full_mask_row_fraction": masking_outputs["full_mask_rows"].to(dtype=torch.float32).mean().detach(),
            "high_mask_row_fraction": masking_outputs["high_mask_rows"].to(dtype=torch.float32).mean().detach(),
            "effective_mask_ratio_mean": masking_outputs["effective_mask_ratio"].mean().detach(),
        }
        return {
            "loss": total_loss,
            "loss_dict": loss_dict,
            "code_grid": code_grid,
            "code_sequence": code_sequence,
            "masked_input_tokens": masked_input_tokens,
            "target_tokens": target_tokens,
            "mask_positions": mask,
            "full_mask_rows": masking_outputs["full_mask_rows"],
            "high_mask_rows": masking_outputs["high_mask_rows"],
            "effective_mask_ratio": masking_outputs["effective_mask_ratio"],
            "target_masked_counts": masking_outputs["target_masked_counts"],
            "token_logits": logits,
            "mask_index": encoded["mask_index"],
            "mask_onehot": encoded["mask_onehot"],
            "code_target_stats": self._token_usage_metrics(code_grid),
            "maskgit_metrics": maskgit_metrics,
            "semantic_bridge_metrics": semantic_bridge_metrics,
            **(
                {
                    "semantic_mask_logits": semantic_outputs["decoded"]["mask_logits"],
                    "semantic_mask_probs": semantic_outputs["decoded"]["mask_probs"],
                    "semantic_mask_index": semantic_outputs["decoded"]["mask_index"],
                }
                if semantic_outputs
                else {}
            ),
            **structure_outputs,
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
        metrics.update(self._collect_log_scalars(split, outputs.get("maskgit_metrics", {})))
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
        sample_metrics = self._broadcast_metric_dict(
            self._finalize_validation_sample_metrics() if self._is_global_zero() else {}
        )
        sample_batches_seen = self._broadcast_metric_batch_count(
            self._validation_sample_metric_batches_seen if self._is_global_zero() else 0
        )
        if sample_metrics:
            effective_batch_size = max(1, int(sample_batches_seen) * max(1, self.validation_sample_batch_size))
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
        self._maybe_configure_tokenizer_loss_class_balance()
        self._log_info(
            "[TokenCodeMaskGitPriorTrainer] "
            f"tokenizer_config={self.tokenizer_config_path}, tokenizer_ckpt={self.tokenizer_ckpt_path}, "
            f"codebook_size={self.codebook_size}, token_spatial_shape={self.token_spatial_shape}, "
            f"mask_spatial_shape={self.mask_spatial_shape}, sequence_length={self.code_sequence_length}, "
            f"permuter={self.permuter_name}, objective={self.objective_name}, monitor={self.monitor}, "
            f"corruption_mode={self.corruption_mode}, "
            f"full_mask_batch_fraction={self.full_mask_batch_fraction:.2f}, "
            f"high_mask_batch_fraction={self.high_mask_batch_fraction:.2f}, "
            f"high_mask_min_ratio={self.high_mask_min_ratio:.2f}, "
            f"label_smoothing={self.label_smoothing:.3f}, sample_temperature={self.sample_temperature:.3f}, "
            f"sample_top_k={self.sample_top_k}, sample_base_gumbel_temp={self.sample_base_gumbel_temp:.3f}, "
            f"semantic_ce_weight={self.semantic_ce_weight:.3f}, "
            f"semantic_ce_use_class_weights={self.semantic_ce_use_class_weights}, "
            f"semantic_dice_weight={self.semantic_dice_weight:.3f}, "
            f"boundary_loss_weight={self.boundary_loss_weight:.3f}, "
            f"area_ratio_loss_weight={self.area_ratio_loss_weight:.3f}, "
            f"adjacency_loss_weight={self.adjacency_loss_weight:.3f}, "
            f"validation_sample_metrics={self.enable_validation_sample_metrics}, "
            f"validation_sample_batch_size={self.validation_sample_batch_size}, "
            f"validation_sample_metric_batches={self.validation_sample_metric_batches}"
        )

    @torch.no_grad()
    def sample_latents(self, batch_size, nfe=None, device=None, condition=None, noise=None):
        if condition is not None:
            raise ValueError("TokenCodeMaskGitPriorTrainer is unconditional and does not accept condition.")
        if device is None:
            device = self.device
        generator = self._sample_generator(device=device, noise=noise)
        generated_tokens = self.sampler.sample(
            model_fn=self._sample_sequence_logits,
            batch_size=int(batch_size),
            sequence_length=self.code_sequence_length,
            device=device,
            generator=generator,
            nfe=nfe,
        )
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

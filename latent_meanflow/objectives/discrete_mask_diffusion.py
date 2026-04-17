import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_full_range_absorbing_schedule(*, min_mask_ratio, max_mask_ratio, owner_name):
    min_mask_ratio = float(min_mask_ratio)
    max_mask_ratio = float(max_mask_ratio)
    if min_mask_ratio != 0.0 or max_mask_ratio != 1.0:
        raise ValueError(
            f"{owner_name} currently supports only full-range absorbing schedule "
            f"(min_mask_ratio=0.0, max_mask_ratio=1.0). "
            "This MVP discrete route assumes all-MASK start and final full reveal."
        )


def discrete_mask_ratio_schedule(
    t,
    *,
    mask_schedule="linear",
    min_mask_ratio=0.0,
    max_mask_ratio=1.0,
):
    mask_schedule = str(mask_schedule).lower()
    t = t.to(dtype=torch.float32).clamp_(0.0, 1.0)
    min_mask_ratio = float(min_mask_ratio)
    max_mask_ratio = float(max_mask_ratio)
    if max_mask_ratio < min_mask_ratio:
        raise ValueError(
            f"max_mask_ratio must be >= min_mask_ratio, got {max_mask_ratio} < {min_mask_ratio}"
        )

    if mask_schedule == "linear":
        ratio = min_mask_ratio + (max_mask_ratio - min_mask_ratio) * t
    else:
        raise ValueError(f"Unsupported mask_schedule: {mask_schedule}")
    return ratio.clamp(min=0.0, max=1.0)


def _as_float_tensor(values, *, name):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 tensor/list, got shape {tuple(tensor.shape)}")
    return tensor


class DiscreteMaskDiffusionObjective(nn.Module):
    name = "discrete_mask_diffusion"
    prediction_type = "categorical_logits"

    def __init__(
        self,
        time_eps=1.0e-3,
        loss_type="cross_entropy",
        mask_schedule="linear",
        min_mask_ratio=0.0,
        max_mask_ratio=1.0,
        ignore_index=None,
        corruption_mode="bernoulli",
        full_mask_batch_fraction=0.0,
        high_mask_batch_fraction=0.0,
        high_mask_min_ratio=0.85,
        class_balance_mode="none",
        effective_num_beta=0.9999,
        class_counts=None,
    ):
        super().__init__()
        self.time_eps = float(time_eps)
        self.loss_type = str(loss_type).lower()
        self.mask_schedule = str(mask_schedule)
        self.min_mask_ratio = float(min_mask_ratio)
        self.max_mask_ratio = float(max_mask_ratio)
        self.ignore_index = None if ignore_index is None else int(ignore_index)
        self.corruption_mode = str(corruption_mode).lower()
        self.full_mask_batch_fraction = float(full_mask_batch_fraction)
        self.high_mask_batch_fraction = float(high_mask_batch_fraction)
        self.high_mask_min_ratio = float(high_mask_min_ratio)
        self.class_balance_mode = str(class_balance_mode).lower()
        self.effective_num_beta = float(effective_num_beta)
        _validate_full_range_absorbing_schedule(
            min_mask_ratio=self.min_mask_ratio,
            max_mask_ratio=self.max_mask_ratio,
            owner_name="DiscreteMaskDiffusionObjective",
        )

        if not 0.0 <= self.time_eps < 1.0:
            raise ValueError(f"time_eps must be in [0, 1), got {self.time_eps}")
        if self.loss_type != "cross_entropy":
            raise ValueError(
                f"DiscreteMaskDiffusionObjective currently supports only cross_entropy, got {self.loss_type!r}"
            )
        if self.corruption_mode not in {"bernoulli", "exact_count"}:
            raise ValueError(
                "DiscreteMaskDiffusionObjective corruption_mode must be one of "
                f"{{'bernoulli', 'exact_count'}}, got {self.corruption_mode!r}"
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
            raise ValueError(
                f"high_mask_min_ratio must be in [0, 1], got {self.high_mask_min_ratio}"
            )
        if self.class_balance_mode not in {"none", "inverse_sqrt_frequency", "effective_num"}:
            raise ValueError(
                "class_balance_mode must be one of "
                f"{{'none', 'inverse_sqrt_frequency', 'effective_num'}}, got {self.class_balance_mode!r}"
            )
        if self.class_balance_mode == "effective_num" and not 0.0 < self.effective_num_beta < 1.0:
            raise ValueError(
                f"effective_num_beta must be in (0, 1) for effective_num, got {self.effective_num_beta}"
            )

        self.num_classes = None
        self.mask_token_id = None
        self._pending_class_counts = None if class_counts is None else _as_float_tensor(
            class_counts,
            name="class_counts",
        )

        self.register_buffer("class_counts", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("class_weights", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("active_class_mask", torch.empty(0, dtype=torch.bool), persistent=False)

    def _buffer_device(self):
        for name in ("class_weights", "class_counts", "active_class_mask"):
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        return torch.device("cpu")

    def configure_discrete_state(self, *, num_classes, mask_token_id, ignore_index=None):
        self.num_classes = int(num_classes)
        self.mask_token_id = int(mask_token_id)
        self.ignore_index = self.ignore_index if ignore_index is None else int(ignore_index)
        if self.mask_token_id != self.num_classes:
            raise ValueError(
                f"Expected absorbing MASK token id to equal num_classes, got {self.mask_token_id} vs {self.num_classes}"
            )

        buffer_device = self._buffer_device()
        default_weights = torch.ones(self.num_classes, dtype=torch.float32, device=buffer_device)
        active_mask = torch.ones(self.num_classes, dtype=torch.bool, device=buffer_device)
        self.class_weights = default_weights
        self.active_class_mask = active_mask
        if self.class_balance_mode == "none":
            self.class_counts = torch.zeros(self.num_classes, dtype=torch.float32, device=buffer_device)
        elif self._pending_class_counts is not None:
            self.configure_class_balance(self._pending_class_counts)

    def configure_class_balance(self, class_counts):
        self._require_state()
        buffer_device = self._buffer_device()
        class_counts = _as_float_tensor(class_counts, name="class_counts").to(device=buffer_device)
        if class_counts.numel() != self.num_classes:
            raise ValueError(
                f"class_counts length must equal num_classes={self.num_classes}, got {class_counts.numel()}"
            )
        if torch.any(class_counts < 0):
            raise ValueError("class_counts must be non-negative")

        self.class_counts = class_counts.to(dtype=torch.float32, device=buffer_device)
        active_mask = self.class_counts > 0
        self.active_class_mask = active_mask

        weights = torch.ones(self.num_classes, dtype=torch.float32, device=buffer_device)
        if self.class_balance_mode == "none":
            self.class_weights = weights
            return

        if torch.any(active_mask):
            active_counts = self.class_counts[active_mask].clamp_min(1.0)
            if self.class_balance_mode == "inverse_sqrt_frequency":
                active_weights = active_counts.rsqrt()
            elif self.class_balance_mode == "effective_num":
                beta = self.effective_num_beta
                active_weights = (1.0 - beta) / (1.0 - torch.pow(beta, active_counts))
            else:
                raise ValueError(f"Unsupported class_balance_mode: {self.class_balance_mode}")
            active_weights = active_weights / active_weights.mean().clamp_min(1.0e-8)
            weights[active_mask] = active_weights

        self.class_weights = weights

    def needs_class_count_scan(self):
        return self.class_balance_mode != "none" and self.class_counts.numel() == 0

    def _class_weight_summary(self, *, device):
        if self.class_weights.numel() == 0:
            weights = torch.ones(self.num_classes, device=device, dtype=torch.float32)
            active_mask = torch.ones(self.num_classes, device=device, dtype=torch.bool)
        else:
            weights = self.class_weights.to(device=device, dtype=torch.float32)
            if self.active_class_mask.numel() == 0:
                active_mask = torch.ones_like(weights, dtype=torch.bool)
            else:
                active_mask = self.active_class_mask.to(device=device, dtype=torch.bool)

        if torch.any(active_mask):
            active_weights = weights[active_mask]
        else:
            active_weights = weights
        return {
            "class_weight_min": active_weights.min().detach(),
            "class_weight_max": active_weights.max().detach(),
            "class_weight_mean": active_weights.mean().detach(),
        }

    def _require_state(self):
        if self.num_classes is None or self.mask_token_id is None:
            raise RuntimeError(
                "DiscreteMaskDiffusionObjective must be configured with num_classes and mask_token_id "
                "before forward()."
            )

    def _mask_ratio_from_t(self, t):
        return discrete_mask_ratio_schedule(
            t,
            mask_schedule=self.mask_schedule,
            min_mask_ratio=self.min_mask_ratio,
            max_mask_ratio=self.max_mask_ratio,
        )

    def _build_valid_mask(self, mask_index):
        valid_mask = torch.ones_like(mask_index, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask &= mask_index != int(self.ignore_index)
        return valid_mask

    def _sample_effective_mask_ratio(self, *, batch_size, device, valid_mask):
        base_t = torch.rand(batch_size, device=device, dtype=torch.float32)
        base_t = base_t * (1.0 - self.time_eps) + self.time_eps
        base_mask_ratio = self._mask_ratio_from_t(base_t)

        effective_mask_ratio = base_mask_ratio.clone()
        valid_counts = valid_mask.view(batch_size, -1).sum(dim=1)
        has_valid = valid_counts > 0

        if torch.any(has_valid):
            mode_selector = torch.rand(batch_size, device=device, dtype=torch.float32)
            full_mask_rows = has_valid & (mode_selector < self.full_mask_batch_fraction)
            high_mask_threshold = self.full_mask_batch_fraction + self.high_mask_batch_fraction
            high_mask_rows = (
                has_valid
                & (~full_mask_rows)
                & (mode_selector < high_mask_threshold)
            )
            if torch.any(full_mask_rows):
                effective_mask_ratio[full_mask_rows] = 1.0
            if torch.any(high_mask_rows):
                high_uniform = torch.rand(batch_size, device=device, dtype=torch.float32)
                high_ratio = self.high_mask_min_ratio + (1.0 - self.high_mask_min_ratio) * high_uniform
                effective_mask_ratio[high_mask_rows] = high_ratio[high_mask_rows]
        else:
            full_mask_rows = torch.zeros(batch_size, device=device, dtype=torch.bool)
            high_mask_rows = torch.zeros(batch_size, device=device, dtype=torch.bool)

        effective_t = effective_mask_ratio.clamp(min=self.time_eps, max=1.0)
        return effective_t, effective_mask_ratio, full_mask_rows, high_mask_rows

    def _ensure_masked_positions(self, masked_positions, valid_mask):
        batch_size = masked_positions.shape[0]
        flat_masked = masked_positions.view(batch_size, -1)
        flat_valid = valid_mask.view(batch_size, -1)
        valid_counts = flat_valid.sum(dim=1)
        masked_counts = flat_masked.sum(dim=1)
        force_rows = torch.nonzero((valid_counts > 0) & (masked_counts == 0), as_tuple=False).flatten()
        if force_rows.numel() <= 0:
            return masked_positions

        valid_float = flat_valid.to(dtype=torch.float32)
        forced_indices = torch.multinomial(valid_float[force_rows], num_samples=1).squeeze(1)
        flat_masked[force_rows, forced_indices] = True
        return flat_masked.view_as(masked_positions)

    def _sample_exact_count_mask(self, *, valid_mask, effective_mask_ratio):
        batch_size = valid_mask.shape[0]
        flat_valid = valid_mask.view(batch_size, -1)
        flat_masked = torch.zeros_like(flat_valid)
        valid_counts = flat_valid.sum(dim=1)
        target_masked_counts = torch.round(
            effective_mask_ratio * valid_counts.to(dtype=torch.float32)
        ).to(dtype=torch.long)
        positive_rows = valid_counts > 0
        target_masked_counts = torch.where(
            positive_rows,
            target_masked_counts.clamp(min=1),
            target_masked_counts,
        )
        target_masked_counts = torch.minimum(target_masked_counts, valid_counts)

        if int(target_masked_counts.max().item()) > 0:
            random_scores = torch.rand(flat_valid.shape, device=valid_mask.device, dtype=torch.float32)
            random_scores = random_scores.masked_fill(~flat_valid, -1.0)
            for batch_idx in range(batch_size):
                target_count = int(target_masked_counts[batch_idx].item())
                if target_count <= 0:
                    continue
                selected = torch.topk(
                    random_scores[batch_idx],
                    k=target_count,
                    largest=True,
                ).indices
                flat_masked[batch_idx, selected] = True

        masked_positions = flat_masked.view_as(valid_mask)
        masked_counts = flat_masked.sum(dim=1)
        return masked_positions, masked_counts, target_masked_counts

    def _sample_masked_positions(self, *, valid_mask, effective_mask_ratio):
        batch_size = valid_mask.shape[0]
        valid_counts = valid_mask.view(batch_size, -1).sum(dim=1)
        if self.corruption_mode == "bernoulli":
            masked_positions = (
                torch.rand(valid_mask.shape, device=valid_mask.device, dtype=torch.float32)
                < effective_mask_ratio.view(batch_size, 1, 1)
            ) & valid_mask
            masked_positions = self._ensure_masked_positions(masked_positions, valid_mask)
            masked_counts = masked_positions.view(batch_size, -1).sum(dim=1)
            target_masked_counts = masked_counts.clone()
            return masked_positions, masked_counts, target_masked_counts
        if self.corruption_mode == "exact_count":
            return self._sample_exact_count_mask(
                valid_mask=valid_mask,
                effective_mask_ratio=effective_mask_ratio,
            )
        raise ValueError(f"Unsupported corruption_mode: {self.corruption_mode}")

    def forward(self, model_fn, x_lat, condition=None, global_step=None, **kwargs):
        _ = global_step, kwargs
        self._require_state()

        if not isinstance(x_lat, torch.Tensor):
            x_lat = torch.as_tensor(x_lat)
        if x_lat.ndim != 3:
            raise ValueError(
                f"DiscreteMaskDiffusionObjective expects mask_index with shape [B, H, W], got {tuple(x_lat.shape)}"
            )

        mask_index = x_lat.to(dtype=torch.long)
        batch_size = int(mask_index.shape[0])
        device = mask_index.device

        valid_mask = self._build_valid_mask(mask_index)
        t, effective_mask_ratio, full_mask_rows, high_mask_rows = self._sample_effective_mask_ratio(
            batch_size=batch_size,
            device=device,
            valid_mask=valid_mask,
        )
        masked_positions, masked_counts, target_masked_counts = self._sample_masked_positions(
            valid_mask=valid_mask,
            effective_mask_ratio=effective_mask_ratio,
        )

        corrupted = mask_index.clone()
        corrupted[~valid_mask] = self.mask_token_id
        corrupted[masked_positions] = self.mask_token_id

        pred_logits = model_fn(corrupted, t=t, condition=condition)
        expected_shape = (batch_size, self.num_classes, *mask_index.shape[-2:])
        if tuple(pred_logits.shape) != expected_shape:
            raise ValueError(
                f"DiscreteMaskDiffusionObjective expected logits shape {expected_shape}, got {tuple(pred_logits.shape)}"
            )

        safe_target = mask_index.masked_fill(~valid_mask, 0)
        ce = F.cross_entropy(pred_logits, safe_target, reduction="none")
        pixel_class_weights = self.class_weights.to(
            device=device,
            dtype=pred_logits.dtype,
        )[safe_target]
        masked_loss_weight = masked_positions.to(dtype=pred_logits.dtype)
        weighted_loss = ce * pixel_class_weights * masked_loss_weight
        loss = weighted_loss.sum() / masked_loss_weight.sum().clamp_min(1.0)

        pred_mask = torch.argmax(pred_logits, dim=1)
        masked_accuracy = (
            ((pred_mask == safe_target) & masked_positions).to(dtype=pred_logits.dtype).sum()
            / masked_loss_weight.sum().clamp_min(1.0)
        )

        valid_counts = valid_mask.view(batch_size, -1).sum(dim=1).clamp_min(1)
        masked_fraction = masked_counts.to(dtype=pred_logits.dtype) / valid_counts.to(dtype=pred_logits.dtype)

        loss_dict = {
            "discrete_diffusion_loss": loss.detach(),
            "total_loss": loss.detach(),
            "base_error_mean": loss.detach(),
            "masked_fraction_mean": masked_fraction.mean().detach(),
            "masked_token_accuracy": masked_accuracy.detach(),
            "full_mask_fraction_mean": full_mask_rows.to(dtype=pred_logits.dtype).mean().detach(),
            "high_mask_fraction_mean": high_mask_rows.to(dtype=pred_logits.dtype).mean().detach(),
            "effective_mask_ratio_mean": effective_mask_ratio.mean().detach(),
            "t_mean": t.mean().detach(),
            "mask_ratio_mean": effective_mask_ratio.mean().detach(),
        }
        loss_dict.update(self._class_weight_summary(device=device))

        return {
            "loss": loss,
            "discrete_diffusion_loss": loss,
            "loss_dict": loss_dict,
            "t": t,
            "z_t": corrupted,
            "pred_field": pred_logits,
            "target_field": safe_target,
            "masked_positions": masked_positions,
            "masked_counts": masked_counts,
            "target_masked_counts": target_masked_counts,
            "valid_mask": valid_mask,
            "effective_mask_ratio": effective_mask_ratio,
            "full_mask_rows": full_mask_rows,
            "high_mask_rows": high_mask_rows,
        }

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
    ):
        super().__init__()
        self.time_eps = float(time_eps)
        self.loss_type = str(loss_type).lower()
        self.mask_schedule = str(mask_schedule)
        self.min_mask_ratio = float(min_mask_ratio)
        self.max_mask_ratio = float(max_mask_ratio)
        self.ignore_index = None if ignore_index is None else int(ignore_index)
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

        self.num_classes = None
        self.mask_token_id = None

    def configure_discrete_state(self, *, num_classes, mask_token_id, ignore_index=None):
        self.num_classes = int(num_classes)
        self.mask_token_id = int(mask_token_id)
        self.ignore_index = self.ignore_index if ignore_index is None else int(ignore_index)
        if self.mask_token_id != self.num_classes:
            raise ValueError(
                f"Expected absorbing MASK token id to equal num_classes, got {self.mask_token_id} vs {self.num_classes}"
            )

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

        t = torch.rand(batch_size, device=device, dtype=torch.float32)
        t = t * (1.0 - self.time_eps) + self.time_eps
        mask_ratio = self._mask_ratio_from_t(t)

        valid_mask = self._build_valid_mask(mask_index)
        masked_positions = (
            torch.rand(mask_index.shape, device=device, dtype=torch.float32)
            < mask_ratio.view(batch_size, 1, 1)
        ) & valid_mask
        masked_positions = self._ensure_masked_positions(masked_positions, valid_mask)

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
        loss_weight = masked_positions.to(dtype=pred_logits.dtype)
        loss = (ce * loss_weight).sum() / loss_weight.sum().clamp_min(1.0)

        pred_mask = torch.argmax(pred_logits, dim=1)
        masked_accuracy = (
            ((pred_mask == safe_target) & masked_positions).to(dtype=pred_logits.dtype).sum()
            / loss_weight.sum().clamp_min(1.0)
        )

        valid_counts = valid_mask.view(batch_size, -1).sum(dim=1).clamp_min(1)
        masked_counts = masked_positions.view(batch_size, -1).sum(dim=1)
        masked_fraction = masked_counts.to(dtype=pred_logits.dtype) / valid_counts.to(dtype=pred_logits.dtype)

        loss_dict = {
            "discrete_diffusion_loss": loss.detach(),
            "total_loss": loss.detach(),
            "base_error_mean": loss.detach(),
            "masked_fraction_mean": masked_fraction.mean().detach(),
            "masked_token_accuracy": masked_accuracy.detach(),
            "t_mean": t.mean().detach(),
            "mask_ratio_mean": mask_ratio.mean().detach(),
        }

        return {
            "loss": loss,
            "discrete_diffusion_loss": loss,
            "loss_dict": loss_dict,
            "t": t,
            "z_t": corrupted,
            "pred_field": pred_logits,
            "target_field": safe_target,
            "masked_positions": masked_positions,
            "valid_mask": valid_mask,
        }

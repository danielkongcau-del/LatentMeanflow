import math

import torch
import torch.nn as nn

from latent_meanflow.objectives.discrete_mask_diffusion import discrete_mask_ratio_schedule


def _gaussian_to_uniform(noise):
    noise = noise.to(dtype=torch.float32)
    uniform = 0.5 * (1.0 + torch.erf(noise / math.sqrt(2.0)))
    return uniform.clamp_(1.0e-6, 1.0 - 1.0e-6)


class SeededDiscreteMaskDiffusionSampler(nn.Module):
    def __init__(
        self,
        default_nfe=8,
        mask_schedule="linear",
        min_mask_ratio=0.0,
        max_mask_ratio=1.0,
        reveal_noise_scale=0.15,
        sample_temperature=1.0,
    ):
        super().__init__()
        self.default_nfe = int(default_nfe)
        self.mask_schedule = str(mask_schedule)
        self.min_mask_ratio = float(min_mask_ratio)
        self.max_mask_ratio = float(max_mask_ratio)
        self.reveal_noise_scale = float(reveal_noise_scale)
        self.sample_temperature = float(sample_temperature)

        self.num_classes = None
        self.mask_token_id = None

    def configure_discrete_state(self, *, num_classes, mask_token_id, ignore_index=None):
        _ = ignore_index
        self.num_classes = int(num_classes)
        self.mask_token_id = int(mask_token_id)
        if self.mask_token_id != self.num_classes:
            raise ValueError(
                f"Expected absorbing MASK token id to equal num_classes, got {self.mask_token_id} vs {self.num_classes}"
            )

    def _require_state(self):
        if self.num_classes is None or self.mask_token_id is None:
            raise RuntimeError(
                "SeededDiscreteMaskDiffusionSampler must be configured with num_classes and mask_token_id "
                "before sample()."
            )

    def _normalize_mask_ratio(self, t):
        raw_ratio = discrete_mask_ratio_schedule(
            t,
            mask_schedule=self.mask_schedule,
            min_mask_ratio=self.min_mask_ratio,
            max_mask_ratio=self.max_mask_ratio,
        )
        denom = max(self.max_mask_ratio - self.min_mask_ratio, 1.0e-8)
        return ((raw_ratio - self.min_mask_ratio) / denom).clamp(0.0, 1.0)

    def _build_ratio_schedule(self, nfe, device):
        if nfe <= 0:
            raise ValueError(f"nfe must be positive, got {nfe}")
        times = torch.linspace(1.0, 0.0, steps=nfe + 1, device=device, dtype=torch.float32)
        ratios = self._normalize_mask_ratio(times)
        ratios[0] = 1.0
        ratios[-1] = 0.0
        return ratios

    def _parse_latent_shape(self, latent_shape):
        shape = tuple(int(dim) for dim in latent_shape)
        if len(shape) == 2:
            return shape
        if len(shape) == 3:
            return (shape[-2], shape[-1])
        raise ValueError(
            "Discrete sampler expects latent_shape to be [H, W] or [C, H, W] for compatibility, "
            f"got {latent_shape!r}"
        )

    def _resolve_noise(self, noise, *, batch_size, latent_shape, device):
        spatial_shape = self._parse_latent_shape(latent_shape)
        expected_hw = tuple(int(value) for value in spatial_shape)

        if noise is None:
            if len(tuple(latent_shape)) == 3:
                noise = torch.randn(batch_size, *latent_shape, device=device)
            else:
                noise = torch.randn(batch_size, *expected_hw, device=device)
        else:
            noise = noise.to(device=device)

        if noise.ndim == 4:
            if tuple(noise.shape[0:1]) != (batch_size,) or tuple(noise.shape[-2:]) != expected_hw:
                raise ValueError(
                    f"Noise shape mismatch: expected [B, C, H, W] with batch={batch_size}, hw={expected_hw}; "
                    f"got {tuple(noise.shape)}"
                )
            reduced_noise = noise.to(dtype=torch.float32).mean(dim=1)
        elif noise.ndim == 3:
            if tuple(noise.shape) != (batch_size, *expected_hw):
                raise ValueError(
                    f"Noise shape mismatch: expected {(batch_size, *expected_hw)}, got {tuple(noise.shape)}"
                )
            reduced_noise = noise.to(dtype=torch.float32)
        else:
            raise ValueError(
                "Discrete sampler noise must be None, [B, H, W], or [B, C, H, W], "
                f"got {tuple(noise.shape)}"
            )

        return reduced_noise.contiguous()

    def _step_uniform(self, base_uniform, *, step_idx, offset):
        step = float(step_idx + 1)
        mixed = torch.frac(
            base_uniform * (1.0 + 0.7548776662466927 * float(offset + 1))
            + 0.6180339887498948 * step
            + 0.4142135623730950 * float(offset + 1)
        )
        return mixed.clamp_(1.0e-6, 1.0 - 1.0e-6)

    def _class_uniform(self, pixel_uniform, *, step_idx, device):
        class_offsets = torch.arange(self.num_classes, device=device, dtype=pixel_uniform.dtype)
        uniform = torch.frac(
            pixel_uniform.unsqueeze(1) * (1.3247179572447458 + 0.137 * float(step_idx + 1))
            + class_offsets.view(1, -1) * 0.6180339887498948
            + 0.5
        )
        return uniform.clamp_(1.0e-6, 1.0 - 1.0e-6)

    def _sample_impl(
        self,
        *,
        model_fn,
        batch_size,
        latent_shape,
        device,
        condition=None,
        noise=None,
        nfe=None,
        return_history=False,
    ):
        self._require_state()
        nfe = self.default_nfe if nfe is None else int(nfe)
        height, width = self._parse_latent_shape(latent_shape)
        total_positions = int(height * width)
        ratio_schedule = self._build_ratio_schedule(nfe=nfe, device=device)

        base_noise = self._resolve_noise(
            noise,
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=device,
        )
        base_uniform = _gaussian_to_uniform(base_noise)

        state = torch.full(
            (batch_size, height, width),
            fill_value=self.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        history = [state.clone()] if return_history else None

        for step_idx in range(nfe):
            remaining_mask = state == self.mask_token_id
            remaining_counts = remaining_mask.view(batch_size, -1).sum(dim=1)
            if int(remaining_counts.max().item()) <= 0:
                if history is not None:
                    history.append(state.clone())
                continue

            current_ratio = remaining_counts.to(dtype=torch.float32) / float(max(total_positions, 1))
            logits = model_fn(state, t=current_ratio, condition=condition)
            expected_shape = (batch_size, self.num_classes, height, width)
            if tuple(logits.shape) != expected_shape:
                raise ValueError(
                    f"Discrete sampler expected logits shape {expected_shape}, got {tuple(logits.shape)}"
                )
            probs = torch.softmax(logits.to(dtype=torch.float32), dim=1)
            confidence = torch.max(probs, dim=1).values

            target_ratio = float(ratio_schedule[step_idx + 1].item())
            flat_state = state.view(batch_size, -1)
            flat_remaining = remaining_mask.view(batch_size, -1)
            flat_confidence = confidence.view(batch_size, -1)
            flat_noise = base_uniform.view(batch_size, -1)

            for batch_idx in range(batch_size):
                remaining_positions = torch.nonzero(flat_remaining[batch_idx], as_tuple=False).flatten()
                remaining_total = int(remaining_positions.numel())
                if remaining_total <= 0:
                    continue

                if step_idx == (nfe - 1):
                    target_remaining = 0
                else:
                    target_remaining = int(round(float(total_positions) * target_ratio))
                    if remaining_total > 1:
                        target_remaining = min(target_remaining, remaining_total - 1)
                    else:
                        target_remaining = 0
                target_remaining = max(0, target_remaining)
                reveal_count = max(0, remaining_total - target_remaining)
                if reveal_count <= 0:
                    reveal_count = 1
                reveal_count = min(reveal_count, remaining_total)

                reveal_uniform = self._step_uniform(
                    flat_noise[batch_idx, remaining_positions],
                    step_idx=step_idx,
                    offset=0,
                )
                reveal_score = flat_confidence[batch_idx, remaining_positions] + self.reveal_noise_scale * (
                    reveal_uniform - 0.5
                )
                if reveal_count < remaining_total:
                    reveal_order = torch.topk(reveal_score, k=reveal_count, largest=True).indices
                    reveal_positions = remaining_positions[reveal_order]
                else:
                    reveal_positions = remaining_positions

                pixel_logits = logits[batch_idx].permute(1, 2, 0).reshape(-1, self.num_classes)[reveal_positions]
                class_uniform = self._class_uniform(
                    self._step_uniform(
                        flat_noise[batch_idx, reveal_positions],
                        step_idx=step_idx,
                        offset=1,
                    ),
                    step_idx=step_idx,
                    device=device,
                )
                if self.sample_temperature > 0.0:
                    gumbel = -torch.log(-torch.log(class_uniform))
                    sampled_class = torch.argmax(
                        pixel_logits / self.sample_temperature + gumbel,
                        dim=1,
                    )
                else:
                    sampled_class = torch.argmax(pixel_logits, dim=1)

                flat_state[batch_idx, reveal_positions] = sampled_class.to(dtype=flat_state.dtype)

            state = flat_state.view(batch_size, height, width)
            if history is not None:
                history.append(state.clone())

        return history if return_history else state

    def sample(self, model_fn, batch_size, latent_shape, device, condition=None, noise=None, nfe=None):
        return self._sample_impl(
            model_fn=model_fn,
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=device,
            condition=condition,
            noise=noise,
            nfe=nfe,
            return_history=False,
        )

    def sample_with_history(
        self,
        model_fn,
        batch_size,
        latent_shape,
        device,
        condition=None,
        noise=None,
        nfe=None,
    ):
        return self._sample_impl(
            model_fn=model_fn,
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=device,
            condition=condition,
            noise=noise,
            nfe=nfe,
            return_history=True,
        )

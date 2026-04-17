import math

import torch
import torch.nn as nn

from latent_meanflow.objectives.discrete_mask_diffusion import (
    _validate_full_range_absorbing_schedule,
    discrete_mask_ratio_schedule,
)


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
        refinement_mode="progressive_reveal",
        lock_schedule="linear",
        final_full_reveal=True,
        min_keep_fraction=0.0,
        lock_noise_scale=None,
    ):
        super().__init__()
        self.default_nfe = int(default_nfe)
        self.mask_schedule = str(mask_schedule)
        self.min_mask_ratio = float(min_mask_ratio)
        self.max_mask_ratio = float(max_mask_ratio)
        self.reveal_noise_scale = float(reveal_noise_scale)
        self.sample_temperature = float(sample_temperature)
        self.refinement_mode = str(refinement_mode).lower()
        self.lock_schedule = str(lock_schedule).lower()
        self.final_full_reveal = bool(final_full_reveal)
        self.min_keep_fraction = float(min_keep_fraction)
        self.lock_noise_scale = (
            self.reveal_noise_scale if lock_noise_scale is None else float(lock_noise_scale)
        )
        _validate_full_range_absorbing_schedule(
            min_mask_ratio=self.min_mask_ratio,
            max_mask_ratio=self.max_mask_ratio,
            owner_name="SeededDiscreteMaskDiffusionSampler",
        )

        if self.refinement_mode not in {
            "progressive_reveal",
            "remask_low_confidence",
            "proposal_visible_refine",
        }:
            raise ValueError(
                "SeededDiscreteMaskDiffusionSampler refinement_mode must be one of "
                "{'progressive_reveal', 'remask_low_confidence', 'proposal_visible_refine'}, "
                f"got {self.refinement_mode!r}"
            )
        if self.lock_schedule != "linear":
            raise ValueError(
                f"SeededDiscreteMaskDiffusionSampler currently supports only lock_schedule='linear', got {self.lock_schedule!r}"
            )
        if not 0.0 <= self.min_keep_fraction <= 1.0:
            raise ValueError(f"min_keep_fraction must be in [0, 1], got {self.min_keep_fraction}")

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

    def _call_model(self, *, model_fn, state, t, condition, height, width):
        batch_size = int(state.shape[0])
        logits = model_fn(state, t=t, condition=condition)
        expected_shape = (batch_size, self.num_classes, height, width)
        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                f"Discrete sampler expected logits shape {expected_shape}, got {tuple(logits.shape)}"
            )
        return logits

    def _predict_logits(self, *, model_fn, state, condition, height, width):
        batch_size = int(state.shape[0])
        remaining_mask = state == self.mask_token_id
        remaining_counts = remaining_mask.view(batch_size, -1).sum(dim=1)
        current_ratio = remaining_counts.to(dtype=torch.float32) / float(max(height * width, 1))
        logits = self._call_model(
            model_fn=model_fn,
            state=state,
            t=current_ratio,
            condition=condition,
            height=height,
            width=width,
        )
        return logits, remaining_mask

    def _predict_logits_from_editable_fraction(
        self,
        *,
        model_fn,
        state,
        locked_mask,
        condition,
        height,
        width,
    ):
        batch_size = int(state.shape[0])
        editable_mask = ~locked_mask
        editable_counts = editable_mask.view(batch_size, -1).sum(dim=1)
        # Proposal-visible refinement uses editable fraction as time. After step 0 the
        # state is usually fully proposed, so literal MASK count is no longer a valid
        # proxy for how much of the layout is still mutable.
        editable_fraction = editable_counts.to(dtype=torch.float32) / float(max(height * width, 1))
        logits = self._call_model(
            model_fn=model_fn,
            state=state,
            t=editable_fraction,
            condition=condition,
            height=height,
            width=width,
        )
        return logits, editable_mask, editable_fraction

    def _sample_classes(self, pixel_logits, *, pixel_uniform, step_idx, device):
        probs = torch.softmax(pixel_logits.to(dtype=torch.float32), dim=1)
        if self.sample_temperature > 0.0:
            class_uniform = self._class_uniform(pixel_uniform, step_idx=step_idx, device=device)
            gumbel = -torch.log(-torch.log(class_uniform))
            sampled_class = torch.argmax(pixel_logits / self.sample_temperature + gumbel, dim=1)
        else:
            sampled_class = torch.argmax(pixel_logits, dim=1)
        sampled_confidence = probs.gather(1, sampled_class.unsqueeze(1)).squeeze(1)
        max_confidence = probs.max(dim=1).values
        return sampled_class, sampled_confidence, max_confidence

    def _select_lock_positions(
        self,
        *,
        unlocked_positions,
        sampled_confidence,
        flat_locked_row,
        flat_noise_row,
        ratio_schedule,
        step_idx,
        nfe,
        total_positions,
    ):
        unlocked_total = int(unlocked_positions.numel())
        if unlocked_total <= 0:
            return unlocked_positions
        if step_idx == (nfe - 1):
            return unlocked_positions

        target_remaining = int(round(float(total_positions) * float(ratio_schedule[step_idx + 1].item())))
        target_remaining = max(0, min(target_remaining, total_positions))
        current_locked = int(flat_locked_row.sum().item())
        target_locked = max(0, total_positions - target_remaining)
        additional_lock = max(0, target_locked - current_locked)
        min_lock = max(1, int(math.ceil(unlocked_total * self.min_keep_fraction)))
        additional_lock = max(additional_lock, min_lock)
        additional_lock = min(additional_lock, unlocked_total)

        lock_uniform = self._step_uniform(
            flat_noise_row[unlocked_positions],
            step_idx=step_idx,
            offset=2,
        )
        lock_score = sampled_confidence + self.lock_noise_scale * (lock_uniform - 0.5)
        if additional_lock < unlocked_total:
            chosen = torch.topk(lock_score, k=additional_lock, largest=True).indices
            return unlocked_positions[chosen]
        return unlocked_positions

    def _progressive_reveal(
        self,
        *,
        model_fn,
        batch_size,
        height,
        width,
        device,
        condition,
        base_uniform,
        ratio_schedule,
        nfe,
        return_history,
    ):
        total_positions = int(height * width)
        state = torch.full(
            (batch_size, height, width),
            fill_value=self.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        history = None
        if return_history:
            history = {
                "state_history": [state.clone()],
                "proposal_history": [],
                "locked_mask_history": [state != self.mask_token_id],
            }

        for step_idx in range(nfe):
            logits, remaining_mask = self._predict_logits(
                model_fn=model_fn,
                state=state,
                condition=condition,
                height=height,
                width=width,
            )
            remaining_counts = remaining_mask.view(batch_size, -1).sum(dim=1)
            if int(remaining_counts.max().item()) <= 0:
                if history is not None:
                    history["proposal_history"].append(state.clone())
                    history["state_history"].append(state.clone())
                    history["locked_mask_history"].append(state != self.mask_token_id)
                continue

            probs = torch.softmax(logits.to(dtype=torch.float32), dim=1)
            confidence = torch.max(probs, dim=1).values

            target_ratio = float(ratio_schedule[step_idx + 1].item())
            flat_state = state.view(batch_size, -1)
            flat_remaining = remaining_mask.view(batch_size, -1)
            flat_confidence = confidence.view(batch_size, -1)
            flat_noise = base_uniform.view(batch_size, -1)
            flat_logits = logits.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)

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
                reveal_count = max(1, remaining_total - target_remaining)
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

                pixel_logits = flat_logits[batch_idx, reveal_positions]
                pixel_uniform = self._step_uniform(
                    flat_noise[batch_idx, reveal_positions],
                    step_idx=step_idx,
                    offset=1,
                )
                sampled_class, _, _ = self._sample_classes(
                    pixel_logits,
                    pixel_uniform=pixel_uniform,
                    step_idx=step_idx,
                    device=device,
                )
                flat_state[batch_idx, reveal_positions] = sampled_class.to(dtype=flat_state.dtype)

            state = flat_state.view(batch_size, height, width)
            if history is not None:
                history["proposal_history"].append(state.clone())
                history["state_history"].append(state.clone())
                history["locked_mask_history"].append(state != self.mask_token_id)

        return history if return_history else state

    def _remask_low_confidence(
        self,
        *,
        model_fn,
        batch_size,
        height,
        width,
        device,
        condition,
        base_uniform,
        ratio_schedule,
        nfe,
        return_history,
    ):
        total_positions = int(height * width)
        state = torch.full(
            (batch_size, height, width),
            fill_value=self.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        locked_mask = torch.zeros_like(state, dtype=torch.bool)
        history = None
        if return_history:
            history = {
                "state_history": [state.clone()],
                "proposal_history": [],
                "locked_mask_history": [locked_mask.clone()],
            }

        flat_noise = base_uniform.view(batch_size, -1)

        for step_idx in range(nfe):
            logits, _ = self._predict_logits(
                model_fn=model_fn,
                state=state,
                condition=condition,
                height=height,
                width=width,
            )
            flat_state = state.view(batch_size, -1)
            flat_locked = locked_mask.view(batch_size, -1)
            flat_logits = logits.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            proposal = flat_state.clone()

            for batch_idx in range(batch_size):
                unlocked_positions = torch.nonzero(~flat_locked[batch_idx], as_tuple=False).flatten()
                unlocked_total = int(unlocked_positions.numel())
                if unlocked_total <= 0:
                    continue

                pixel_logits = flat_logits[batch_idx, unlocked_positions]
                pixel_uniform = self._step_uniform(
                    flat_noise[batch_idx, unlocked_positions],
                    step_idx=step_idx,
                    offset=1,
                )
                sampled_class, sampled_confidence, _ = self._sample_classes(
                    pixel_logits,
                    pixel_uniform=pixel_uniform,
                    step_idx=step_idx,
                    device=device,
                )
                proposal[batch_idx, unlocked_positions] = sampled_class.to(dtype=proposal.dtype)

                lock_positions = self._select_lock_positions(
                    unlocked_positions=unlocked_positions,
                    sampled_confidence=sampled_confidence,
                    flat_locked_row=flat_locked[batch_idx],
                    flat_noise_row=flat_noise[batch_idx],
                    ratio_schedule=ratio_schedule,
                    step_idx=step_idx,
                    nfe=nfe,
                    total_positions=total_positions,
                )

                flat_locked[batch_idx, lock_positions] = True
                flat_state[batch_idx, lock_positions] = proposal[batch_idx, lock_positions]

            state = flat_state.view(batch_size, height, width)
            locked_mask = flat_locked.view(batch_size, height, width)
            proposal_state = proposal.view(batch_size, height, width)
            if history is not None:
                history["proposal_history"].append(proposal_state.clone())
                history["state_history"].append(state.clone())
                history["locked_mask_history"].append(locked_mask.clone())

        unresolved_mask = state == self.mask_token_id
        if torch.any(unresolved_mask):
            logits, _ = self._predict_logits(
                model_fn=model_fn,
                state=state,
                condition=condition,
                height=height,
                width=width,
            )
            flat_state = state.view(batch_size, -1)
            flat_unresolved = unresolved_mask.view(batch_size, -1)
            flat_logits = logits.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            proposal = flat_state.clone()
            for batch_idx in range(batch_size):
                unresolved_positions = torch.nonzero(flat_unresolved[batch_idx], as_tuple=False).flatten()
                if int(unresolved_positions.numel()) <= 0:
                    continue
                pixel_logits = flat_logits[batch_idx, unresolved_positions]
                pixel_uniform = self._step_uniform(
                    flat_noise[batch_idx, unresolved_positions],
                    step_idx=nfe,
                    offset=1,
                )
                sampled_class, _, _ = self._sample_classes(
                    pixel_logits,
                    pixel_uniform=pixel_uniform,
                    step_idx=nfe,
                    device=device,
                )
                proposal[batch_idx, unresolved_positions] = sampled_class.to(dtype=proposal.dtype)
                flat_state[batch_idx, unresolved_positions] = proposal[batch_idx, unresolved_positions]
            state = flat_state.view(batch_size, height, width)
            locked_mask = torch.ones_like(state, dtype=torch.bool)
            if history is not None:
                history["proposal_history"].append(proposal.view(batch_size, height, width).clone())
                history["state_history"].append(state.clone())
                history["locked_mask_history"].append(locked_mask.clone())

        return history if return_history else state

    def _proposal_visible_refine(
        self,
        *,
        model_fn,
        batch_size,
        height,
        width,
        device,
        condition,
        base_uniform,
        ratio_schedule,
        nfe,
        return_history,
    ):
        total_positions = int(height * width)
        state = torch.full(
            (batch_size, height, width),
            fill_value=self.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        locked_mask = torch.zeros_like(state, dtype=torch.bool)
        history = None
        if return_history:
            history = {
                "state_history": [state.clone()],
                "proposal_history": [],
                "locked_mask_history": [locked_mask.clone()],
            }

        flat_noise = base_uniform.view(batch_size, -1)

        for step_idx in range(nfe):
            editable_mask = ~locked_mask
            editable_counts = editable_mask.view(batch_size, -1).sum(dim=1)
            if int(editable_counts.max().item()) <= 0:
                if history is not None:
                    history["proposal_history"].append(state.clone())
                    history["state_history"].append(state.clone())
                    history["locked_mask_history"].append(locked_mask.clone())
                continue

            logits, _, _ = self._predict_logits_from_editable_fraction(
                model_fn=model_fn,
                state=state,
                locked_mask=locked_mask,
                condition=condition,
                height=height,
                width=width,
            )
            flat_state = state.view(batch_size, -1)
            flat_locked = locked_mask.view(batch_size, -1)
            flat_logits = logits.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            proposal = flat_state.clone()

            for batch_idx in range(batch_size):
                unlocked_positions = torch.nonzero(~flat_locked[batch_idx], as_tuple=False).flatten()
                if int(unlocked_positions.numel()) <= 0:
                    continue

                pixel_logits = flat_logits[batch_idx, unlocked_positions]
                pixel_uniform = self._step_uniform(
                    flat_noise[batch_idx, unlocked_positions],
                    step_idx=step_idx,
                    offset=1,
                )
                sampled_class, sampled_confidence, _ = self._sample_classes(
                    pixel_logits,
                    pixel_uniform=pixel_uniform,
                    step_idx=step_idx,
                    device=device,
                )
                proposal[batch_idx, unlocked_positions] = sampled_class.to(dtype=proposal.dtype)

                lock_positions = self._select_lock_positions(
                    unlocked_positions=unlocked_positions,
                    sampled_confidence=sampled_confidence,
                    flat_locked_row=flat_locked[batch_idx],
                    flat_noise_row=flat_noise[batch_idx],
                    ratio_schedule=ratio_schedule,
                    step_idx=step_idx,
                    nfe=nfe,
                    total_positions=total_positions,
                )
                flat_locked[batch_idx, lock_positions] = True

            proposal_state = proposal.view(batch_size, height, width)
            state = proposal_state
            locked_mask = flat_locked.view(batch_size, height, width)
            if history is not None:
                history["proposal_history"].append(proposal_state.clone())
                history["state_history"].append(state.clone())
                history["locked_mask_history"].append(locked_mask.clone())

        return history if return_history else state

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
        ratio_schedule = self._build_ratio_schedule(nfe=nfe, device=device)

        base_noise = self._resolve_noise(
            noise,
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=device,
        )
        base_uniform = _gaussian_to_uniform(base_noise)

        if self.refinement_mode == "progressive_reveal":
            return self._progressive_reveal(
                model_fn=model_fn,
                batch_size=batch_size,
                height=height,
                width=width,
                device=device,
                condition=condition,
                base_uniform=base_uniform,
                ratio_schedule=ratio_schedule,
                nfe=nfe,
                return_history=return_history,
            )
        if self.refinement_mode == "remask_low_confidence":
            return self._remask_low_confidence(
                model_fn=model_fn,
                batch_size=batch_size,
                height=height,
                width=width,
                device=device,
                condition=condition,
                base_uniform=base_uniform,
                ratio_schedule=ratio_schedule,
                nfe=nfe,
                return_history=return_history,
            )
        if self.refinement_mode == "proposal_visible_refine":
            return self._proposal_visible_refine(
                model_fn=model_fn,
                batch_size=batch_size,
                height=height,
                width=width,
                device=device,
                condition=condition,
                base_uniform=base_uniform,
                ratio_schedule=ratio_schedule,
                nfe=nfe,
                return_history=return_history,
            )
        raise ValueError(f"Unsupported refinement_mode: {self.refinement_mode}")

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

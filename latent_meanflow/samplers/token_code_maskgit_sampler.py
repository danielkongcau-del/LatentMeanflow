import math

import torch
import torch.nn as nn

from latent_meanflow.models.backbones.token_code_maskgit_transformer import get_mask_scheduling_fn


class TokenCodeMaskGitSampler(nn.Module):
    def __init__(
        self,
        *,
        codebook_size,
        mask_token_id,
        default_nfe=8,
        mask_schedule_type="cosine",
        sample_temperature=1.0,
        top_k=None,
        base_gumbel_temp=4.5,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.mask_token_id = int(mask_token_id)
        self.default_nfe = int(default_nfe)
        self.mask_schedule_type = str(mask_schedule_type)
        self.sample_temperature = float(sample_temperature)
        self.top_k = None if top_k in {None, ""} else max(1, int(top_k))
        self.base_gumbel_temp = float(base_gumbel_temp)
        self.gamma = get_mask_scheduling_fn(self.mask_schedule_type)

    def _apply_top_k(self, logits):
        if self.top_k is None or self.top_k >= int(logits.shape[-1]):
            return logits
        threshold = torch.topk(logits, k=self.top_k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _sample_one_step(self, *, idx, n_masked_after_step, model_fn, generator, step_idx, total_steps):
        batch_size, sequence_length = idx.shape
        masked_positions = idx.eq(self.mask_token_id)

        logits = model_fn(idx)
        if tuple(logits.shape) != (batch_size, sequence_length, self.codebook_size):
            raise ValueError(
                "TokenCodeMaskGitSampler expected logits with shape "
                f"{(batch_size, sequence_length, self.codebook_size)}, got {tuple(logits.shape)}"
            )

        logits = self._apply_top_k(logits.to(dtype=torch.float32))
        logits = logits / float(max(self.sample_temperature, 1.0e-6))
        probs = torch.softmax(logits, dim=-1)

        sampled_idx = torch.multinomial(
            probs.reshape(batch_size * sequence_length, self.codebook_size),
            num_samples=1,
            generator=generator,
        ).reshape(batch_size, sequence_length)
        sampled_probs = torch.gather(probs, dim=-1, index=sampled_idx.unsqueeze(-1)).squeeze(-1)

        sampled_idx = torch.where(masked_positions, sampled_idx, idx)
        sampled_probs = torch.where(
            masked_positions,
            sampled_probs,
            torch.full_like(sampled_probs, float("inf")),
        )

        if n_masked_after_step >= sequence_length:
            return torch.full_like(idx, fill_value=self.mask_token_id)

        gumbel_temp = self.base_gumbel_temp * (1.0 - float(step_idx + 1) / float(max(total_steps, 1)))
        uniform = torch.rand(
            sampled_probs.shape,
            device=sampled_probs.device,
            generator=generator,
            dtype=sampled_probs.dtype,
        ).clamp_(1.0e-6, 1.0 - 1.0e-6)
        gumbel = -torch.log(-torch.log(uniform))
        confidence = torch.log(sampled_probs.clamp_min(1.0e-12)) + gumbel_temp * gumbel

        reveal_count = max(0, int(sequence_length - n_masked_after_step))
        reveal_count = min(reveal_count, int(sequence_length))
        if reveal_count > 0:
            indices = confidence.topk(reveal_count, dim=1).indices
            masked_positions = masked_positions.scatter(
                dim=1,
                index=indices,
                src=torch.zeros_like(masked_positions, dtype=torch.bool),
            )
        sampled_idx = torch.where(masked_positions, self.mask_token_id, sampled_idx)
        return sampled_idx

    @torch.no_grad()
    def sample(self, *, model_fn, batch_size, sequence_length, device, generator=None, nfe=None):
        total_steps = self.default_nfe if nfe is None else max(1, int(nfe))
        sequence_length = int(sequence_length)
        idx = torch.full(
            (int(batch_size), sequence_length),
            fill_value=self.mask_token_id,
            device=device,
            dtype=torch.long,
        )

        for step_idx in range(total_steps):
            n_masked_after_step = math.floor(float(self.gamma(float(step_idx + 1) / float(total_steps))) * sequence_length)
            n_masked_after_step = min(n_masked_after_step, sequence_length - 1 - step_idx)
            n_masked_after_step = max(0, int(n_masked_after_step))
            idx = self._sample_one_step(
                idx=idx,
                n_masked_after_step=n_masked_after_step,
                model_fn=model_fn,
                generator=generator,
                step_idx=step_idx,
                total_steps=total_steps,
            )
        return idx

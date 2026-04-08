import math
import warnings

import torch
import torch.nn as nn
from ldm.util import instantiate_from_config

from .common import (
    build_time_sampler,
    expand_time_like,
    rectified_path,
    sample_interval,
    weighted_regression_loss,
)
from .meanflow import meanflow_jvp


def _sigmoid_alpha_value(step, *, start_step, end_step, gamma, clamp_eta):
    if end_step <= start_step:
        raise ValueError(
            f"end_step must be greater than start_step, got {start_step} and {end_step}"
        )
    step = float(step)
    scale = 1.0 / float(end_step - start_step)
    offset = -float(start_step + end_step) / 2.0 / float(end_step - start_step)
    alpha = 1.0 - 1.0 / (1.0 + math.exp(-((scale * step + offset) * gamma)))
    if alpha > (1.0 - clamp_eta):
        return 1.0
    if alpha < clamp_eta:
        return 0.0
    return alpha


class SigmoidAlphaScheduler(nn.Module):
    def __init__(self, start_step, end_step, gamma=25.0, clamp_eta=0.05):
        super().__init__()
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.gamma = float(gamma)
        self.clamp_eta = float(clamp_eta)
        if self.end_step <= self.start_step:
            raise ValueError(
                f"end_step must be greater than start_step, got {self.start_step} and {self.end_step}"
            )

    def forward(self, step):
        return _sigmoid_alpha_value(
            step,
            start_step=self.start_step,
            end_step=self.end_step,
            gamma=self.gamma,
            clamp_eta=self.clamp_eta,
        )


class BudgetSigmoidAlphaScheduler(nn.Module):
    def __init__(
        self,
        transition_start_fraction=0.0,
        transition_end_fraction=1.0,
        gamma=25.0,
        clamp_eta=0.05,
        total_steps=None,
        total_epochs=None,
        optimizer_steps_per_epoch=None,
    ):
        super().__init__()
        self.transition_start_fraction = float(transition_start_fraction)
        self.transition_end_fraction = float(transition_end_fraction)
        self.gamma = float(gamma)
        self.clamp_eta = float(clamp_eta)
        self.configured_total_steps = None if total_steps is None else int(total_steps)
        self.configured_total_epochs = None if total_epochs is None else int(total_epochs)
        self.configured_optimizer_steps_per_epoch = (
            None if optimizer_steps_per_epoch is None else int(optimizer_steps_per_epoch)
        )

        if not 0.0 <= self.transition_start_fraction < 1.0:
            raise ValueError(
                "transition_start_fraction must lie in [0, 1), "
                f"got {self.transition_start_fraction}"
            )
        if not 0.0 < self.transition_end_fraction <= 1.0:
            raise ValueError(
                "transition_end_fraction must lie in (0, 1], "
                f"got {self.transition_end_fraction}"
            )
        if self.transition_end_fraction <= self.transition_start_fraction:
            raise ValueError(
                "transition_end_fraction must be greater than transition_start_fraction, "
                f"got {self.transition_start_fraction} and {self.transition_end_fraction}"
            )
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if not 0.0 <= self.clamp_eta < 0.5:
            raise ValueError(f"clamp_eta must lie in [0, 0.5), got {self.clamp_eta}")

        self._resolved_total_steps = None
        self.set_training_budget(
            total_steps=self.configured_total_steps,
            total_epochs=self.configured_total_epochs,
            optimizer_steps_per_epoch=self.configured_optimizer_steps_per_epoch,
            allow_unresolved=True,
        )

    def _resolve_total_steps(self, *, total_steps=None, total_epochs=None, optimizer_steps_per_epoch=None):
        if total_steps is not None:
            total_steps = int(total_steps)
            if total_steps <= 1:
                raise ValueError(f"total_steps must be greater than 1, got {total_steps}")
            return total_steps
        if total_epochs is None and optimizer_steps_per_epoch is None:
            return None
        if total_epochs is None or optimizer_steps_per_epoch is None:
            raise ValueError(
                "BudgetSigmoidAlphaScheduler requires either total_steps or "
                "both total_epochs and optimizer_steps_per_epoch."
            )
        total_epochs = int(total_epochs)
        optimizer_steps_per_epoch = int(optimizer_steps_per_epoch)
        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be positive, got {total_epochs}")
        if optimizer_steps_per_epoch <= 0:
            raise ValueError(
                f"optimizer_steps_per_epoch must be positive, got {optimizer_steps_per_epoch}"
            )
        total_steps = total_epochs * optimizer_steps_per_epoch
        if total_steps <= 1:
            raise ValueError(
                "Resolved total_steps must be greater than 1, "
                f"got {total_steps} from total_epochs={total_epochs} "
                f"and optimizer_steps_per_epoch={optimizer_steps_per_epoch}"
            )
        return total_steps

    def set_training_budget(
        self,
        *,
        total_steps=None,
        total_epochs=None,
        optimizer_steps_per_epoch=None,
        allow_unresolved=False,
    ):
        resolved_total_steps = self._resolve_total_steps(
            total_steps=total_steps,
            total_epochs=total_epochs,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        )
        if resolved_total_steps is None:
            if allow_unresolved:
                self._resolved_total_steps = None
                return
            raise ValueError(
                "BudgetSigmoidAlphaScheduler could not resolve a training budget. "
                "Provide total_steps directly, provide total_epochs with "
                "optimizer_steps_per_epoch, or let the trainer inject the fit budget."
            )
        self._resolved_total_steps = int(resolved_total_steps)

    def _resolve_window(self):
        if self._resolved_total_steps is None:
            raise ValueError(
                "BudgetSigmoidAlphaScheduler has no resolved training budget yet. "
                "Call set_training_budget(...) or use it through LatentFlowTrainer.fit."
            )
        horizon = max(self._resolved_total_steps - 1, 1)
        start_step = int(round(self.transition_start_fraction * horizon))
        end_step = int(round(self.transition_end_fraction * horizon))
        if end_step <= start_step:
            raise ValueError(
                "Resolved schedule window is empty. "
                f"Got start_step={start_step}, end_step={end_step}, "
                f"resolved_total_steps={self._resolved_total_steps}"
            )
        return start_step, end_step

    def forward(self, step):
        start_step, end_step = self._resolve_window()
        return _sigmoid_alpha_value(
            step,
            start_step=start_step,
            end_step=end_step,
            gamma=self.gamma,
            clamp_eta=self.clamp_eta,
        )


class ConstantAlphaScheduler(nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.value = float(value)

    def forward(self, step):
        _ = step
        return self.value


class AlphaFlowObjective(nn.Module):
    name = "alphaflow"
    prediction_type = "average_velocity"

    def __init__(
        self,
        time_eps=1.0e-4,
        min_delta=1.0e-3,
        loss_type="mse",
        time_sampler_config=None,
        r_equals_t_ratio=0.0,
        border_fm_ratio=None,
        trajectory_fm_ratio=0.0,
        flow_matching_ratio=None,
        weighting_mode="alpha_adaptive_exact",
        adaptive_weight_power=1.0,
        adaptive_weight_bias=1.0e-4,
        alpha_schedule_config=None,
        meanflow_alpha_threshold=1.0e-8,
        alpha_inverse_eps=1.0e-6,
    ):
        super().__init__()
        self.time_eps = float(time_eps)
        self.min_delta = float(min_delta)
        self.loss_type = str(loss_type)
        if border_fm_ratio is not None:
            if float(r_equals_t_ratio) != 0.0 and float(r_equals_t_ratio) != float(border_fm_ratio):
                raise ValueError("r_equals_t_ratio and border_fm_ratio disagree")
            r_equals_t_ratio = border_fm_ratio
        if flow_matching_ratio is not None:
            if float(trajectory_fm_ratio) != 0.0 and float(trajectory_fm_ratio) != float(flow_matching_ratio):
                raise ValueError("trajectory_fm_ratio and flow_matching_ratio disagree")
            warnings.warn(
                "flow_matching_ratio is deprecated because it is semantically ambiguous. "
                "Use trajectory_fm_ratio for alpha=1 trajectory flow matching.",
                stacklevel=2,
            )
            trajectory_fm_ratio = flow_matching_ratio
        self.r_equals_t_ratio = float(r_equals_t_ratio)
        self.trajectory_fm_ratio = float(trajectory_fm_ratio)
        self.weighting_mode = str(weighting_mode)
        self.adaptive_weight_power = float(adaptive_weight_power)
        self.adaptive_weight_bias = float(adaptive_weight_bias)
        self.meanflow_alpha_threshold = float(meanflow_alpha_threshold)
        self.alpha_inverse_eps = float(alpha_inverse_eps)
        self.time_sampler = build_time_sampler(time_sampler_config=time_sampler_config, time_eps=self.time_eps)

        if alpha_schedule_config is None:
            alpha_schedule_config = {
                "target": "latent_meanflow.objectives.alphaflow.SigmoidAlphaScheduler",
                "params": {
                    "start_step": 0,
                    "end_step": 1_200_000,
                    "gamma": 25.0,
                    "clamp_eta": 0.25,
                },
            }
        self.alpha_schedule = instantiate_from_config(alpha_schedule_config)

    def set_training_budget(self, total_steps=None, total_epochs=None, optimizer_steps_per_epoch=None):
        setter = getattr(self.alpha_schedule, "set_training_budget", None)
        if setter is None:
            return
        setter(
            total_steps=total_steps,
            total_epochs=total_epochs,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        )

    def get_alpha(self, global_step):
        if global_step is None:
            global_step = 0
        if isinstance(global_step, torch.Tensor):
            global_step = int(global_step.item())
        return float(self.alpha_schedule(global_step))

    def _combine_branch_losses(self, branch_items, device, dtype):
        total_count = sum(item["count"] for item in branch_items)
        if total_count <= 0:
            raise ValueError("AlphaFlowObjective received an empty batch")

        total_loss = torch.zeros((), device=device, dtype=dtype)
        combined_stats = {}
        for item in branch_items:
            if item["count"] == 0:
                continue
            branch_scale = float(item["count"]) / float(total_count)
            total_loss = total_loss + item["loss"] * branch_scale
            for name, value in item["stats"].items():
                if name not in combined_stats:
                    combined_stats[name] = torch.zeros((), device=device, dtype=dtype)
                combined_stats[name] = combined_stats[name] + value * branch_scale
        return total_loss, combined_stats

    def forward(self, model_fn, x_lat, condition=None, global_step=None, **kwargs):
        batch_size = x_lat.shape[0]
        r, t, delta_t = sample_interval(
            batch_size=batch_size,
            device=x_lat.device,
            time_sampler=self.time_sampler,
            min_delta=self.min_delta,
            r_equals_t_ratio=self.r_equals_t_ratio,
            dtype=x_lat.dtype,
        )
        z_t, velocity, noise = rectified_path(x_lat, t=t)
        target_model_fn = kwargs.get("target_model_fn", model_fn)

        alpha_value = self.get_alpha(global_step)
        alpha = torch.full((batch_size,), alpha_value, device=x_lat.device, dtype=x_lat.dtype)
        if self.trajectory_fm_ratio > 0.0:
            trajectory_fm_mask = torch.rand(batch_size, device=x_lat.device) < self.trajectory_fm_ratio
            alpha = torch.where(trajectory_fm_mask, torch.ones_like(alpha), alpha)
        else:
            trajectory_fm_mask = torch.zeros(batch_size, device=x_lat.device, dtype=torch.bool)

        pred_field = torch.zeros_like(x_lat)
        target_field = torch.zeros_like(x_lat)
        base_weight = torch.ones(batch_size, device=x_lat.device, dtype=x_lat.dtype)
        branch_losses = []
        total_derivative = None
        s = t.clone()
        objective_branch = torch.full((batch_size,), 1, device=x_lat.device, dtype=torch.long)

        meanflow_mask = alpha <= self.meanflow_alpha_threshold
        if torch.any(meanflow_mask):
            pred_meanflow, total_derivative_meanflow = meanflow_jvp(
                model_fn=model_fn,
                z_t=z_t[meanflow_mask],
                r=r[meanflow_mask],
                t=t[meanflow_mask],
                velocity=velocity[meanflow_mask],
                condition=None if condition is None else condition[meanflow_mask],
            )
            pred_field[meanflow_mask] = pred_meanflow
            target_field[meanflow_mask] = (
                velocity[meanflow_mask]
                - expand_time_like(delta_t[meanflow_mask], x_lat[meanflow_mask]) * total_derivative_meanflow
            )
            total_derivative = torch.zeros_like(x_lat)
            total_derivative[meanflow_mask] = total_derivative_meanflow
            objective_branch[meanflow_mask] = 0
            meanflow_loss, meanflow_stats = weighted_regression_loss(
                pred_meanflow,
                target_field[meanflow_mask].detach(),
                loss_type=self.loss_type,
                base_weight=torch.ones_like(alpha[meanflow_mask]),
                weighting_mode="paper_like",
                adaptive_power=self.adaptive_weight_power,
                adaptive_bias=self.adaptive_weight_bias,
            )
            branch_losses.append(
                {
                    "name": "meanflow",
                    "count": int(meanflow_mask.sum().item()),
                    "loss": meanflow_loss,
                    "stats": meanflow_stats,
                }
            )

        alphaflow_mask = ~meanflow_mask
        if torch.any(alphaflow_mask):
            alpha_subset = alpha[alphaflow_mask]
            pred_subset = model_fn(
                z_t[alphaflow_mask],
                r=r[alphaflow_mask],
                t=t[alphaflow_mask],
                delta_t=delta_t[alphaflow_mask],
                condition=None if condition is None else condition[alphaflow_mask],
            )
            s_subset = alpha_subset * r[alphaflow_mask] + (1.0 - alpha_subset) * t[alphaflow_mask]
            z_s_subset = z_t[alphaflow_mask] - expand_time_like(t[alphaflow_mask] - s_subset, z_t[alphaflow_mask]) * velocity[
                alphaflow_mask
            ]
            with torch.no_grad():
                shifted_subset = target_model_fn(
                    z_s_subset,
                    r=r[alphaflow_mask],
                    t=s_subset,
                    delta_t=s_subset - r[alphaflow_mask],
                    condition=None if condition is None else condition[alphaflow_mask],
                )
            target_subset = (
                expand_time_like(alpha_subset, z_t[alphaflow_mask]) * velocity[alphaflow_mask]
                + expand_time_like(1.0 - alpha_subset, z_t[alphaflow_mask]) * shifted_subset
            )

            pred_field[alphaflow_mask] = pred_subset
            target_field[alphaflow_mask] = target_subset
            s[alphaflow_mask] = s_subset
            base_weight[alphaflow_mask] = 1.0 / alpha_subset.clamp_min(self.alpha_inverse_eps)
            objective_branch[alphaflow_mask] = torch.where(
                trajectory_fm_mask[alphaflow_mask],
                torch.full_like(alpha_subset, 2, dtype=torch.long),
                torch.full_like(alpha_subset, 1, dtype=torch.long),
            )
            # alpha_adaptive_exact only applies to alpha>0 discrete AlphaFlow samples.
            # When alpha=0, the objective reduces to MeanFlow and must use
            # MeanFlow's paper_like weighting instead of alpha-based weighting.
            alphaflow_loss, alphaflow_stats = weighted_regression_loss(
                pred_subset,
                target_subset.detach(),
                loss_type=self.loss_type,
                base_weight=1.0 / alpha_subset.clamp_min(self.alpha_inverse_eps),
                weighting_mode=self.weighting_mode,
                adaptive_power=self.adaptive_weight_power,
                adaptive_bias=self.adaptive_weight_bias,
                alpha=alpha_subset,
            )
            branch_losses.append(
                {
                    "name": "alphaflow",
                    "count": int(alphaflow_mask.sum().item()),
                    "loss": alphaflow_loss,
                    "stats": alphaflow_stats,
                }
            )

        loss, weighting_stats = self._combine_branch_losses(
            branch_losses,
            device=x_lat.device,
            dtype=x_lat.dtype,
        )
        border_fm_mask = delta_t == 0
        loss_dict = {
            "alphaflow_loss": loss,
            "total_loss": loss,
            "alpha": alpha.mean(),
            "delta_t_mean": delta_t.mean(),
            "trajectory_fm_ratio": trajectory_fm_mask.float().mean(),
            "border_fm_ratio": border_fm_mask.float().mean(),
            "r_equals_t_ratio": border_fm_mask.float().mean(),
            "meanflow_branch_ratio": meanflow_mask.float().mean(),
            "alphaflow_branch_ratio": alphaflow_mask.float().mean(),
            **weighting_stats,
        }
        return {
            "loss": loss,
            "alphaflow_loss": loss,
            "loss_dict": loss_dict,
            "alpha": alpha,
            "alpha_value": alpha.mean(),
            "r": r,
            "t": t,
            "s": s,
            "delta_t": delta_t,
            "noise": noise,
            "z_t": z_t,
            "velocity": velocity,
            "pred_field": pred_field,
            "target_field": target_field.detach(),
            "total_derivative": total_derivative,
            "objective_branch": objective_branch,
            "weighting_stats": weighting_stats,
            "base_weight": base_weight,
            "trajectory_fm_mask": trajectory_fm_mask,
            "border_fm_mask": border_fm_mask,
        }

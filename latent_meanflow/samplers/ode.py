import torch


class EulerFlowSampler:
    def __init__(self, default_nfe=32, time_start=1.0, time_end=0.0):
        self.default_nfe = int(default_nfe)
        self.time_start = float(time_start)
        self.time_end = float(time_end)

    def sample(self, model_fn, batch_size, latent_shape, device, condition=None, noise=None, nfe=None):
        nfe = self.default_nfe if nfe is None else int(nfe)
        if nfe <= 0:
            raise ValueError(f"nfe must be positive, got {nfe}")
        if noise is None:
            noise = torch.randn(batch_size, *latent_shape, device=device)

        z = noise
        time_grid = torch.linspace(self.time_start, self.time_end, nfe + 1, device=device)
        for step_idx in range(len(time_grid) - 1):
            t_curr = time_grid[step_idx].expand(batch_size)
            delta_t = time_grid[step_idx + 1] - time_grid[step_idx]
            velocity = model_fn(z, t=t_curr, condition=condition)
            z = z + delta_t * velocity
        return z

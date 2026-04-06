import torch


class IntervalFlowSampler:
    def __init__(self, default_nfe=1, two_step_time=0.5, time_start=1.0, time_end=0.0):
        self.default_nfe = int(default_nfe)
        self.two_step_time = float(two_step_time)
        self.time_start = float(time_start)
        self.time_end = float(time_end)
        if not self.time_end <= self.two_step_time <= self.time_start:
            raise ValueError(
                f"two_step_time must be in [{self.time_end}, {self.time_start}], got {self.two_step_time}"
            )

    def build_time_grid(self, nfe, device):
        nfe = int(nfe)
        if nfe <= 0:
            raise ValueError(f"nfe must be positive, got {nfe}")
        if nfe == 1:
            values = [self.time_start, self.time_end]
        elif nfe == 2:
            values = [self.time_start, self.two_step_time, self.time_end]
        else:
            values = torch.linspace(self.time_start, self.time_end, nfe + 1, device=device)
            return values
        return torch.tensor(values, device=device, dtype=torch.float32)

    def sample(self, model_fn, batch_size, latent_shape, device, condition=None, noise=None, nfe=None):
        nfe = self.default_nfe if nfe is None else int(nfe)
        if noise is None:
            noise = torch.randn(batch_size, *latent_shape, device=device)

        z = noise
        time_grid = self.build_time_grid(nfe=nfe, device=device)
        for step_idx in range(len(time_grid) - 1):
            t_curr = time_grid[step_idx].expand(batch_size)
            r_next = time_grid[step_idx + 1].expand(batch_size)
            average_velocity = model_fn(
                z,
                r=r_next,
                t=t_curr,
                delta_t=t_curr - r_next,
                condition=condition,
            )
            z = z - (time_grid[step_idx] - time_grid[step_idx + 1]) * average_velocity
        return z

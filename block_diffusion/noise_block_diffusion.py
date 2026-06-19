import abc

import torch
import torch.nn as nn


def get_noise(config, noise_type=None):
    if noise_type is None:
        noise_type = config.noise.type

    if noise_type == "loglinear":
        return LogLinearNoise()
    elif noise_type == "square":
        return ExpNoise(2)
    elif noise_type == "square_root":
        return ExpNoise(0.5)
    elif noise_type == "log":
        return LogarithmicNoise()
    elif noise_type == "cosine":
        return CosineNoise()
    else:
        raise ValueError(f"{noise_type} is not a valid noise")


class Noise(abc.ABC, nn.Module):
    """Baseline forward method to get the total + rate of noise at a timestep."""

    def forward(self, t):
        return self.compute_loss_scaling_and_move_chance(t)


class CosineNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def compute_loss_scaling_and_move_chance(self, t):
        cos = -(1 - self.eps) * torch.cos(t * torch.pi / 2)
        sin = -(1 - self.eps) * torch.sin(t * torch.pi / 2)
        move_chance = cos + 1
        loss_scaling = sin / (move_chance + self.eps) * torch.pi / 2
        return loss_scaling, move_chance


class ExpNoise(Noise):
    def __init__(self, exp=2, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.exp = exp

    def compute_loss_scaling_and_move_chance(self, t):
        move_chance = torch.pow(t, self.exp)
        move_chance = torch.clamp(move_chance, min=self.eps)
        loss_scaling = -(self.exp * torch.pow(t, self.exp - 1)) / move_chance
        return loss_scaling, move_chance


class LogarithmicNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def compute_loss_scaling_and_move_chance(self, t):
        move_chance = torch.log1p(t) / torch.log(torch.tensor(2.0))
        loss_scaling = -1 / (
            move_chance * torch.log(torch.tensor(2.0)) * (1 + t)
        )
        return loss_scaling, move_chance


class LogLinearNoise(Noise):
    """Log Linear noise schedule.

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and ~1 when t varies
    from 0 to 1. Total noise is -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def compute_loss_scaling_and_move_chance(self, t):
        loss_scaling = -1 / t
        return loss_scaling, t


class BlockDiffusionNoise(nn.Module):
    def __init__(self, noise_type=LogLinearNoise, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.schedule = self._build_schedule(noise_type)

    def _build_schedule(self, noise_type):
        if isinstance(noise_type, str):
            mapping = {
                "loglinear": LogLinearNoise,
                "LogLinearNoise": LogLinearNoise,
                "square": lambda eps: ExpNoise(2, eps=eps),
                "square_root": lambda eps: ExpNoise(0.5, eps=eps),
                "log": LogarithmicNoise,
                "cosine": CosineNoise,
            }
            if noise_type not in mapping:
                raise ValueError(f"{noise_type} is not a valid noise")
            noise_type = mapping[noise_type]

        if isinstance(noise_type, type):
            return noise_type(eps=self.eps)
        return noise_type

    def sample_t(self, batch_size, device):
        t = torch.rand(batch_size, device=device)
        return (1.0 - self.eps) * t + self.eps

    def _sample_t(
        self,
        batch_dims,
        device: torch.device,
        sampling_eps_min: float,
        sampling_eps_max: float,
        block_size: int | None = None,
    ):
        if block_size is None:
            block_size = batch_dims[-1]
        n = batch_dims[-1]
        num_blocks = max(1, n // block_size)
        _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)
        t = _eps_b
        if block_size != n:
            t = t.repeat_interleave(block_size, dim=-1)
        if sampling_eps_max >= 1 and sampling_eps_min >= 1:
            return torch.ones_like(t)
        t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
        return t

    def _sigma_from_p(self, p: torch.Tensor) -> torch.Tensor:
        sigma_max = self.total_noise(torch.tensor(1.0, device=p.device))
        return torch.min(-torch.log(1 - p), sigma_max)

    def _q_xt(
        self,
        x0: torch.Tensor,
        p: torch.Tensor,
        pad_id: int,
        mask_id: int,
        ignore_bos: bool = False,
    ) -> torch.Tensor:
        xt = x0.clone()
        move = torch.rand_like(x0.float()) <= p
        move = move & (x0 != pad_id) & (x0 != mask_id)
        if ignore_bos:
            move[:, 0] = False
        xt[move] = mask_id
        return xt

    def forward(
        self,
        x,
        sampling_eps_min: float | None = None,
        sampling_eps_max: float | None = None,
        block_size: int | None = None,
        pad_id: int | None = None,
        mask_id: int | None = None,
        ignore_bos: bool = False,
        mdlm_loss_scale: bool = False,
    ):
        

        if sampling_eps_min is None:
            sampling_eps_min = 1e-3
        if sampling_eps_max is None:
            sampling_eps_max = 1.0

        t = self._sample_t(
            x.shape,
            device=x.device,
            sampling_eps_min=sampling_eps_min,
            sampling_eps_max=sampling_eps_max,
            block_size=block_size,
        )
        loss_scale, p = self.schedule.compute_loss_scaling_and_move_chance(t)
        sigma = self._sigma_from_p(p)
        xt = self._q_xt(
            x,
            p,
            pad_id=pad_id,
            mask_id=mask_id,
            ignore_bos=ignore_bos,
        )
        if sampling_eps_min is not None and sampling_eps_min > 0.5:
            loss_scale = -torch.ones_like(loss_scale)
        return xt, sigma, p, loss_scale, t

    def noise(self, t):
        return self.schedule.compute_loss_scaling_and_move_chance(t)

    def rate_noise(self, t):
        return self.schedule.rate_noise(t)

    def total_noise(self, t):
        return self.schedule.total_noise(t)

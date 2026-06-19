"""Predictor implementation for BlockDiffusion.

This is a small semi-autoregressive block sampler.
"""

from typing import Any, Dict, List, Optional

import torch

from xlm.datamodule import Tokenizer
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from xlm.utils.nn import sample_categorical

from .types_block_diffusion import (
    BlockDiffusionBatch,
    BlockDiffusionModel,
    BlockDiffusionPredictionDict,
)


class BlockDiffusionPredictor(
    torch.nn.Module, Predictor[BlockDiffusionBatch, BlockDiffusionPredictionDict]
):
    """Semi-AR block sampler for BlockDiffusion."""

    def __init__(
        self,
        model: BlockDiffusionModel = None,
        tokenizer: Tokenizer = None,
        noise_schedule: NoiseSchedule = None,
        max_steps: int = 8,
        max_length: int = 512,
        sampling_method: str = "sample_top_p",
        p: float = 0.9,
        top_k: int = 50,
        temperature: float = 1.0,
        block_size: Optional[int] = None,
        first_hitting: bool = False,
        kv_cache: bool = False,
        var_length: bool = False,
        context_size: int = 1024,
        **kwargs,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.max_steps = max_steps
        self.max_length = max_length
        self.sampling_method = sampling_method
        self.p = p
        self.top_k = top_k
        self.temperature = temperature
        self.block_size = block_size
        self.first_hitting = first_hitting
        self.kv_cache = kv_cache
        self.var_length = var_length
        self.context_size = context_size

    def _mask_id(self) -> int:
        if hasattr(self.tokenizer, "mask_token_id") and self.tokenizer.mask_token_id is not None:
            return int(self.tokenizer.mask_token_id)
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        return len(self.tokenizer) - 1

    def _get_block_size(self) -> int:
        if self.block_size is not None:
            return max(1, int(self.block_size))
        if self.model is not None and hasattr(self.model, "config") and hasattr(self.model.config, "block_size"):
            return max(1, int(self.model.config.block_size))
        return 1

    @staticmethod
    def _round_up_to_multiple(length: int, multiple: int) -> int:
        if multiple <= 1:
            return max(1, length)
        return ((length + multiple - 1) // multiple) * multiple

    def _reset_kv_cache(self, batch_size: int) -> None:
        if self.kv_cache and self.model is not None and hasattr(self.model, "reset_kv_cache"):
            self.model.reset_kv_cache(eval_batch_size=batch_size)

    def _sample_prior(self, *batch_dims: int, device: torch.device) -> torch.Tensor:
        return torch.full(batch_dims, self._mask_id(), dtype=torch.long, device=device)

    def _nucleus_sample(self, probs: torch.Tensor) -> torch.Tensor:
        if self.p >= 1.0:
            return probs
        block_size = self._get_block_size()
        probs_out = probs.clone()
        block_probs = probs_out[:, -block_size:].clone()
        sorted_probs, sorted_indices = block_probs.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        nucleus_mask = cum_probs <= self.p
        nucleus_mask[..., 0] = True
        sorted_probs = sorted_probs * nucleus_mask
        block_probs.zero_().scatter_(-1, sorted_indices, sorted_probs)
        block_probs /= block_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs_out[:, -block_size:] = block_probs
        return probs_out

    def _sigma_from_p(self, p: torch.Tensor) -> torch.Tensor:
        sigma_max = self.noise_schedule.total_noise(
            torch.tensor(1.0, device=p.device, dtype=p.dtype)
        )
        return torch.minimum(-torch.log(1 - p), sigma_max)

    def _ddpm_caching_update(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        p_x0: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        mask_id = self._mask_id()
        block_size = self._get_block_size()
        _, move_chance_t = self.noise_schedule.noise(t)
        _, move_chance_s = self.noise_schedule.noise(t - dt)
        sigma_t = self._sigma_from_p(move_chance_t)

        move_chance_t = move_chance_t[:, None]
        move_chance_s = move_chance_s[:, None]
        mask_prob = move_chance_s / move_chance_t

        if p_x0 is None:
            if self.kv_cache:
                logits = self.model(
                    x[:, -block_size:],
                    sigma_t,
                    sample_mode=True,
                    store_kv=False,
                )
            else:
                logits = self.model(
                    x,
                    sigma_t,
                    sample_mode=True,
                    store_kv=False,
                )
                logits = logits[:, -block_size:]
            p_x0 = torch.softmax(logits.to(torch.float64), dim=-1)
            p_x0 = self._nucleus_sample(p_x0)

        current_block = x[:, -block_size:]
        if self.first_hitting:
            x_block = sample_categorical(p_x0)
            masked_positions = current_block == mask_id
            selected = torch.multinomial(
                masked_positions.to(torch.float32),
                num_samples=1,
            ).squeeze(-1)
            mask = (
                torch.arange(block_size, device=x.device) == selected[:, None]
            ).to(x_block.dtype)
            x_block = x_block * mask + current_block * (1 - mask)
        else:
            q_xs = p_x0 * (1 - mask_prob)
            q_xs[:, :, mask_id] = mask_prob.squeeze(-1)
            x_block = sample_categorical(q_xs)

        copy_flag = (current_block != mask_id).to(x.dtype)
        x_block = copy_flag * current_block + (1 - copy_flag) * x_block
        x_new = torch.cat((x[:, :-block_size], x_block), dim=-1)

        if self.kv_cache and mask_id not in x_block:
            _ = self.model(
                x_block,
                sigma_t,
                sample_mode=True,
                store_kv=True,
            )

        if not torch.allclose(x_new, x):
            return None, x_new
        return p_x0, x_new

    def _compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        _, counts = torch.unique(x, return_counts=True, sorted=False)
        entropy = torch.special.entr(counts.float() / counts.sum()).sum()
        return entropy

    def _check_stop_conds(self, x: torch.Tensor) -> tuple[bool, torch.Tensor]:
        """Apply the original BD3-LM entropy and variable-length stop rules."""
        stop = False
        truncate_idx = None

        entropy = self._compute_entropy(x[:, -256:])
        if entropy < 4:
            stop = True

        if self.var_length:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None and len(torch.where(x == eos_id)[0]) > 1:
                stop = True
                eos_idx = torch.where(x == eos_id)
                if len(eos_idx[0]) > 1:
                    truncate_idx = min(eos_idx[1][1] + 1, x.shape[1])

            if entropy < 4:
                stop = True
                truncate_idx = x.shape[1] - 256

        if truncate_idx is not None:
            x = x[:, :truncate_idx]
            if x.ndim == 1:
                x = x.unsqueeze(0)
        return stop, x

    def _sample(
        self,
        seqlen: Optional[int] = None,
        num_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        block_size = self._get_block_size()
        seqlen = self._round_up_to_multiple(int(seqlen), block_size)
        num_strides = max(1, seqlen // block_size)

        for _ in range(10):
            sample_i, _ = self._semi_ar_sampler(
                n_samples=batch_size,
                num_steps=max(1, num_steps),
                num_strides=num_strides,
                seqlen=seqlen,
                device=device,
            )
            if sample_i is not None:
                return sample_i
        raise ValueError("Sampling failed.")

    def _semi_ar_sampler(
        self,
        n_samples: int,
        num_steps: int,
        num_strides: int,
        seqlen: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[int]]:
        mask_id = self._mask_id()
        block_size = self._get_block_size()
        sampling_steps = 0
        ones = torch.ones((n_samples, 1), dtype=torch.float32, device=device)

        if self.kv_cache:
            self._reset_kv_cache(n_samples)

        for stride_num in range(num_strides):
            if stride_num == 0:
                x_accum = self._sample_prior(
                    n_samples, block_size, device=device
                )
                if (
                    hasattr(self.tokenizer, "bos_token_id")
                    and self.tokenizer.bos_token_id is not None
                ):
                    x_accum[:, 0] = int(self.tokenizer.bos_token_id)
            else:
                x = self._sample_prior(
                    n_samples, block_size, device=device
                )
                x_accum = torch.cat((x_accum, x), dim=1)

            end_idx = (stride_num + 1) * block_size
            start_idx = max(end_idx - self.context_size, 0)
            fwd_idx = torch.arange(start_idx, end_idx, device=device)

            dt = 1.0 / num_steps
            p_x0_cache = None
            timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)
            t = ones.clone()
            for i in range(num_steps):
                if mask_id not in x_accum:
                    break

                if self.first_hitting:
                    u = torch.rand((n_samples, 1), device=device)
                    num_masked = (
                        x_accum[:, fwd_idx] == mask_id
                    ).sum(-1, keepdim=True).clamp_min(1)
                    t = t * u.pow(1.0 / num_masked)
                else:
                    t = timesteps[i] * ones

                p_x0_cache, x_next = self._ddpm_caching_update(
                    x=x_accum[:, fwd_idx],
                    t=t,
                    dt=dt,
                    p_x0=p_x0_cache,
                )
                if p_x0_cache is None:
                    sampling_steps += 1
                x_accum[:, fwd_idx] = x_next

            if x_accum.shape[1] > 256:
                stop, x_accum = self._check_stop_conds(x_accum)
                if stop and not self.var_length:
                    return None, None
                if stop and x_accum.shape[-1] == 1:
                    return None, None
                if stop:
                    break

        return x_accum, sampling_steps

    @torch._dynamo.disable()
    def predict(
        self,
        batch: Dict[str, Any],  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
        max_len: int = 0,
    ) -> BlockDiffusionPredictionDict:
        assert self.model is not None, "Model is not initialized"

        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]
        device = input_ids.device

        model_length = self.max_length

        sampled = self._sample(
            seqlen=model_length,
            num_steps=max(1, self.max_steps),
            batch_size=batch_size,
            device=device,
        )
        if sampled is None:
            raise ValueError("Sampling failed.")

        current_ids = sampled
        generated_text = self.tokenizer.batch_decode(current_ids)

        return {
            "text": generated_text,
        }

    def to_dict(
        self,
        batch: BlockDiffusionBatch,
        preds: BlockDiffusionPredictionDict,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        results = []
        for i in range(len(preds["text"])):
            results.append(
                {
                    "generated_text": preds["text"][i],
                }
            )
        return results

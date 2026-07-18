

from typing import Optional
import torch
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer
from .types_block_diffusion import BlockDiffusionBatch, BlockDiffusionLossDict, BlockDiffusionModel
import sys

class BlockDiffusionLoss(LossFunction[BlockDiffusionBatch, BlockDiffusionLossDict]):
    def __init__(
        self,
        model: Optional[BlockDiffusionModel] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def loss_fn(
        self,
        batch: BlockDiffusionBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> BlockDiffusionLossDict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch.get("diffusion_mask", attention_mask)
        target_ids = batch["target_ids"]
        sigma = batch["sigma"]
        t = batch["t"]
        loss_scale = batch["loss_scale"]

        assert self.model is not None

        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = self.tokenizer.vocab_size
        x_input = input_ids
        model_attention_mask = attention_mask
        model_cfg = getattr(self.model, "config", None)
        if model_cfg is not None and getattr(model_cfg, "cross_attn", False):
            x_input = torch.cat((input_ids, target_ids), dim=-1)
            model_attention_mask = torch.cat((attention_mask, attention_mask), dim=-1)
        
        logits = self.model(x_input, sigma, attention_mask=model_attention_mask)

        
        if logits.shape[1] != target_ids.shape[1]:
            logits = logits[:, : target_ids.shape[1]]

        logits = logits.clone()
        logits[:, :, mask_token_id] = float("-inf")
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        unmasked_positions = input_ids != mask_token_id
        logits[unmasked_positions] = float("-inf")
        logits[unmasked_positions, input_ids[unmasked_positions]] = 0.0
        
        log_p_theta = torch.gather(
            input=logits,
            dim=-1,
            index=target_ids[:, :, None],
        ).squeeze(-1)

        if t.shape != log_p_theta.shape or loss_scale.shape != log_p_theta.shape:
            raise ValueError(
                "Expected one diffusion timestep and loss weight per target "
                "token, but got "
                f"t.shape={tuple(t.shape)} and "
                f"loss_scale.shape={tuple(loss_scale.shape)} and "
                f"target shape={tuple(log_p_theta.shape)}."
            )

        # The original BD3-LM objective applies the schedule-derived weight
        # independently at every token. Since t is constant within each block,
        # this preserves the timestep sampled for each block.
        loss = loss_scale * log_p_theta
        loss = loss * loss_mask
        denom = loss_mask.sum()
        if denom.item() == 0:
            return {"loss": loss.new_zeros(())}
        
        return {"loss": loss.sum() / denom}

    def __call__(
        self,
        batch: BlockDiffusionBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> BlockDiffusionLossDict:
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def configure(self, pl_module: Harness) -> None:
        pass

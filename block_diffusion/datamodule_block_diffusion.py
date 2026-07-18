"""Data collation logic for BlockDiffusion model.

This file implements the data preprocessing and batching logic.
Based on ARLM implementation - modify as needed.
"""

from typing import List, Dict, Any, Optional, Literal
import torch
from xlm.datamodule import Collator, Tokenizer, Seq2SeqCollatorInput, BaseCollatorInput
from xlm.noise import NoiseSchedule
from xlm.utils.nn import pad_truncate_list
from torch.utils.data import IterableDataset
from .types_block_diffusion import BlockDiffusionBatch, BlockDiffusionSeq2SeqBatch
import sys

class BlockDiffusionEmptyDataset(IterableDataset):

    def __init__(
        self,
        tokenizer: Tokenizer,
        num_examples: int,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_length = max_length

    def __iter__(self):
        for _ in range(self.num_examples):
            yield {"input_ids": []}


class DefaultBlockDiffusionCollator(Collator):
    """Create a noisy input batch and a clean target batch.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        max_length: Optional[int] = None,
        truncate: Literal["max", "block", None] = "block",
        add_eos: bool = False,
        sampling_eps_min: float = 1e-3,
        sampling_eps_max: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.max_length = max_length
        self.truncate = truncate
        self.add_eos = add_eos
        self._vocab_size = len(tokenizer)
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max

        if self.max_length is not None:
            if self.max_length < 1 + int(self.add_eos):
                raise ValueError(
                    "max_length must leave room for the BOS token and optional EOS token."
                )
            if self.max_length % self.block_size != 0:
                raise ValueError(
                    f"max_length={self.max_length} must be divisible by "
                    f"block_size={self.block_size}."
                )

    @staticmethod
    def _round_up_to_block_length(length: int, block_size: int) -> int:
        return ((length + block_size - 1) // block_size) * block_size

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not set")
            self._vocab_size = len(self.tokenizer)
        return self._vocab_size

    def __call__(self, examples: List[BaseCollatorInput]) -> BlockDiffusionBatch:
        token_count = 1 + int(self.add_eos)
        longest = max(len(example["input_ids"]) for example in examples)
        max_len = self._round_up_to_block_length(longest + token_count, self.block_size)
        if self.max_length is not None:
            max_len = self.max_length
        max_tokens = max_len - token_count

        x0_batch: List[List[int]] = []
        attention_mask: List[List[int]] = []

        for example in examples:
            seq = example["input_ids"][:max_tokens]
            x0 = [self.tokenizer.bos_token_id] + seq
            if self.add_eos:
                x0 = x0 + [self.tokenizer.eos_token_id]

            x0 = pad_truncate_list(
                x0,
                max_len,
                self.tokenizer.pad_token_id,
                pad_left=False,
            )

            x0_batch.append(x0)
            attention_mask.append(
                [1] * len([tok for tok in x0 if tok != self.tokenizer.pad_token_id]) +
                [0] * (max_len - len([tok for tok in x0 if tok != self.tokenizer.pad_token_id]))
            )

        x0_tensor = torch.tensor(x0_batch, dtype=torch.long)
       
        xt, sigma, p, loss_scale, t = self.noise_schedule(
            x0_tensor,
            sampling_eps_min=self.sampling_eps_min,
            sampling_eps_max=self.sampling_eps_max,
            block_size=self.block_size,
            pad_id=self.tokenizer.pad_token_id,
            mask_id=self.tokenizer.mask_token_id,
        )
        
        return {
            "input_ids": xt.to(torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "target_ids": x0_tensor,
            # Match the original BD3-LM implementation: the model is
            # conditioned with the first block's sigma, while loss scaling
            # keeps the timestep assigned to every token/block.
            "sigma": sigma[:, 0] if sigma.ndim > 1 else sigma,
            "t": t,
            "loss_scale": loss_scale,
        }


class BlockDiffusionUnconditionalPredCollator(Collator):

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        max_length: int,
        add_bos: bool = True,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_length = max_length
        self.add_bos = add_bos

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> Dict[str, Any]:
        batch_size = len(examples)
        start_token = self.tokenizer.bos_token_id
        mask_token = self.tokenizer.mask_token_id

        input_ids = torch.full(
            (batch_size, self.max_length),
            int(mask_token),
            dtype=torch.long,
        )
        input_ids[:, 0] = int(start_token)

        return {
            "input_ids": input_ids,
        }


################################################################################
# region: Helper Functions


def prepare_prefix_ids_block_diffusion(
    prefix_ids: List[List[int]],
    pad_token_id: int,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
    add_bos: Optional[str] = None,
    add_eos: bool = False,
) -> Dict[str, List[List[int]]]:
    """
    Prepare prefix ids for BlockDiffusion seq2seq tasks.

    Args:
        prefix_ids: List of prefix token sequences.
        pad_token_id: Padding token ID.
        bos_token_id: BOS token ID.
        eos_token_id: EOS token ID.
        max_seq_len: Maximum sequence length.
        truncate: Truncation strategy.
        add_bos: Where to add BOS token ("input" for prefix, "output" for after prefix, None for no BOS).
        add_eos: Whether to add EOS token at the end of the prefix.

    Returns:
        Dictionary with input_ids and attention_mask as lists.
    """
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []

    # Determine max length
    if truncate in ["max", None]:
        max_len = max(len(_prefix_ids) for _prefix_ids in prefix_ids)
        if truncate == "max" and max_seq_len is not None:
            max_len = max(max_len, max_seq_len)
    elif truncate == "block" and max_seq_len is not None:
        max_len = max_seq_len
    else:
        raise ValueError(f"Invalid truncate, max_seq_len: {max_seq_len}")

    assert max_len is not None

    for _prefix_ids in prefix_ids:
        # Add BOS to prefix if requested
        if add_bos == "input" and bos_token_id is not None:
            temp = [bos_token_id] + _prefix_ids
        elif add_bos == "output" and bos_token_id is not None:
            temp = _prefix_ids + [bos_token_id]  # Add BOS to the right
        else:
            temp = _prefix_ids

        # Add EOS token at the end if requested
        if add_eos and eos_token_id is not None:
            temp = temp + [eos_token_id]

        # Pad/truncate
        padded_seq = pad_truncate_list(
            temp, max_len, pad_token_id, pad_left=True
        )
        input_ids.append(padded_seq)

        # Create attention mask (1 for real tokens, 0 for padding on the left)
        mask = [0] * (max_len - len(temp)) + [1] * len(temp)
        attention_mask.append(mask)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def prepare_suffix_ids_block_diffusion(
    suffix_ids: List[List[int]],
    pad_token_id: int,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
    add_bos: Optional[str] = None,
    add_eos: bool = False,
) -> Dict[str, List[List[int]]]:
    """
    Prepare suffix ids for BlockDiffusion seq2seq tasks.

    Args:
        suffix_ids: List of suffix token sequences.
        pad_token_id: Padding token ID.
        bos_token_id: BOS token ID.
        eos_token_id: EOS token ID.
        max_seq_len: Maximum sequence length.
        truncate: Truncation strategy.
        add_bos: Where to add BOS token ("input" for prefix, "output" for after prefix, None for no BOS).
        add_eos: Whether to add EOS token at the end of the suffix.

    Returns:
        Dictionary with input_ids, attention_mask, and target_ids as lists.
    """
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    target_ids: List[List[int]] = []

    # Determine max length
    if truncate in ["max", None]:
        max_len = max(len(_suffix_ids) for _suffix_ids in suffix_ids)
        if truncate == "max" and max_seq_len is not None:
            max_len = max(max_len, max_seq_len)
    elif truncate == "block" and max_seq_len is not None:
        max_len = max_seq_len
    else:
        raise ValueError(f"Invalid truncate, max_seq_len: {max_seq_len}")

    assert max_len is not None

    for _suffix_ids in suffix_ids:
        # Add BOS before suffix if requested
        if add_bos == "output" and bos_token_id is not None:
            temp = [bos_token_id] + _suffix_ids
        else:
            temp = _suffix_ids

        # Add EOS token at the end if requested
        if add_eos and eos_token_id is not None:
            temp = temp + [eos_token_id]

        # Pad/truncate
        padded_seq = pad_truncate_list(
            temp, max_len, pad_token_id, pad_left=False
        )
        input_ids.append(padded_seq)

        # Create attention mask
        mask = [1] * len(temp) + [0] * (max_len - len(temp))
        attention_mask.append(mask)

        # Create target_ids (unshifted - will be shifted in collator if needed)
        target_ids.append(padded_seq)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
    }


################################################################################
# region: Collators


class BlockDiffusionSeq2SeqCollator:
    """Seq2seq collator for BlockDiffusion model.
    
    Based on ARLM implementation - modify as needed.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        noise_schedule: NoiseSchedule,
        block_size: Optional[int] = None,
        input_length: Optional[int] = None,
        prompt_length: Optional[int] = None,
        input_block_size: Optional[int] = None,
        target_length: Optional[int] = None,
        add_bos: Optional[str] = None,
        add_eos: bool = False,
        truncate: Literal["max", "block", None] = "block",
    ):
        """Initialize the BlockDiffusion sequence-to-sequence collator.

        Args:
            tokenizer: The tokenizer to use.
            noise_schedule: Noise schedule (not used but kept for interface consistency).
            block_size: Diffusion block size for the target.
            input_length: Maximum sequence length for the input/prompt.
            prompt_length: Alias for input_length.
            input_block_size: Backward-compatible alias for input_length.
            target_length: Maximum padded sequence length for the target.
            add_bos: Where to add BOS token ("input" for prefix, "output" for after prefix, None for no BOS).
            add_eos: Whether to add EOS token at the end of the suffix.
            truncate: Truncation strategy.
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.input_length = (
            input_length
            if input_length is not None
            else prompt_length
            if prompt_length is not None
            else input_block_size
        )
        self.input_block_size = self.input_length
        self.target_length = target_length if target_length is not None else block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncate = truncate
        self._vocab_size = (
            len(self.tokenizer) if self.tokenizer is not None else None
        )

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not set")
            self._vocab_size = len(self.tokenizer)
        return self._vocab_size

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> BlockDiffusionSeq2SeqBatch:
        """Collate examples into a batch for BlockDiffusion sequence-to-sequence training.

        Args:
            examples: List of examples with prompt_ids and input_ids.

        Returns:
            BlockDiffusionSeq2SeqBatch with input_ids, attention_mask, target_ids.
        """
        prefix_token_count = 0
        if self.add_bos is not None and self.tokenizer.bos_token_id is not None:
            prefix_token_count += 1
        longest_prefix = max(len(e["prompt_ids"]) for e in examples)
        prefix_len = longest_prefix + prefix_token_count
        prefix_max_seq_len = (
            (prefix_len + self.block_size - 1) // self.block_size
        ) * self.block_size
        if self.input_length is not None:
            prefix_max_seq_len = max(self.input_length, prefix_max_seq_len)

        # Prepare prefix (prompt)
        prefix = prepare_prefix_ids_block_diffusion(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=prefix_max_seq_len,
            truncate=self.truncate,
            add_bos=self.add_bos,
            add_eos=False,  # No EOS in prefix for seq2seq
        )

        suffix_token_count = 0
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            suffix_token_count += 1
        longest_suffix = max(len(e["input_ids"]) for e in examples)
        suffix_len = longest_suffix + suffix_token_count
        suffix_max_seq_len = (
            (suffix_len + self.block_size - 1) // self.block_size
        ) * self.block_size
        if self.target_length is not None:
            suffix_max_seq_len = max(self.target_length, suffix_max_seq_len)

        # Prepare suffix (target)
        suffix = prepare_suffix_ids_block_diffusion(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=suffix_max_seq_len,
            truncate=self.truncate,
            add_bos=None,  # BOS through prefix
            add_eos=self.add_eos,
        )

        clean_suffix = torch.tensor(suffix["input_ids"], dtype=torch.long)
        noisy_suffix_tensor, sigma, p, suffix_loss_scale, suffix_t = self.noise_schedule(
            clean_suffix,
            block_size=self.block_size,
            pad_id=self.tokenizer.pad_token_id,
            mask_id=self.tokenizer.mask_token_id,
        )
        noisy_suffix = {
            "input_ids": noisy_suffix_tensor.tolist(),
            "attention_mask": suffix["attention_mask"],
            "target_ids": suffix["target_ids"],
        }

        # Concatenate prefix and suffix as lists
        input_ids = [
            p + s for p, s in zip(prefix["input_ids"], suffix["input_ids"])
        ]
        noisy_input_ids = [
            p + s for p, s in zip(prefix["input_ids"], noisy_suffix["input_ids"])
        ]
        prefix_attention_mask = torch.tensor(
            prefix["attention_mask"], dtype=torch.bool
        )
        suffix_attention_mask = torch.tensor(
            suffix["attention_mask"], dtype=torch.bool
        )
        attention_mask = torch.cat(
            [prefix_attention_mask, suffix_attention_mask],
            dim=-1,
        )
        diffusion_mask = torch.cat(
            [torch.zeros_like(prefix_attention_mask), suffix_attention_mask],
            dim=-1,
        )

        # Block diffusion predicts the clean token at the same position as the
        # noisy token, so targets are not shifted like autoregressive LM labels.
        target_ids = input_ids

        prefix_len = len(prefix["input_ids"][0])
        prefix_t = torch.zeros(
            clean_suffix.shape[0],
            prefix_len,
            dtype=suffix_t.dtype,
            device=suffix_t.device,
        )
        prefix_loss_scale = torch.zeros(
            clean_suffix.shape[0],
            prefix_len,
            dtype=suffix_loss_scale.dtype,
            device=suffix_loss_scale.device,
        )
        t = torch.cat([prefix_t, suffix_t], dim=-1)
        loss_scale = torch.cat([prefix_loss_scale, suffix_loss_scale], dim=-1)

        return {
            "input_ids": torch.tensor(noisy_input_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "diffusion_mask": diffusion_mask,
            "token_type_ids": torch.zeros(
                len(input_ids),
                max(len(seq) for seq in input_ids),
                dtype=torch.long,
            ),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "sigma": sigma[:, 0] if sigma.ndim > 1 else sigma,
            "t": t,
            "loss_scale": loss_scale,
        }


class BlockDiffusionSeq2SeqPredCollator(BlockDiffusionSeq2SeqCollator):
    """Seq2seq prediction collator for BlockDiffusion.

    For prediction, the model receives only the padded prompt:

        input_ids  = [prefix]
        target_ids = [clean suffix]

    The clean suffix target is kept only for evaluation/logging.
    """

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> BlockDiffusionSeq2SeqBatch:
        """Collate examples into a batch for BlockDiffusion sequence-to-sequence prediction.

        Args:
            examples: List of examples with prompt_ids and input_ids.

        Returns:
            BlockDiffusionSeq2SeqBatch with input_ids, attention_mask, target_ids.
        """
        prefix_token_count = 0
        if self.add_bos is not None and self.tokenizer.bos_token_id is not None:
            prefix_token_count += 1
        longest_prefix = max(len(e["prompt_ids"]) for e in examples)
        prefix_len = longest_prefix + prefix_token_count
        batch_prefix_block_len = (
            (prefix_len + self.block_size - 1) // self.block_size
        ) * self.block_size
        if self.input_length is not None:
            prefix_max_seq_len = max(self.input_length, batch_prefix_block_len)
        else:
            prefix_max_seq_len = batch_prefix_block_len

        prefix = prepare_prefix_ids_block_diffusion(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=prefix_max_seq_len,
            truncate=self.truncate,
            add_bos=self.add_bos,
            add_eos=False,  # No EOS in prefix for seq2seq
        )

        suffix = prepare_suffix_ids_block_diffusion(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.target_length,
            truncate=self.truncate,
            add_bos=None,
            add_eos=self.add_eos,
        )
        
        prefix_attention_mask = torch.tensor(
            prefix["attention_mask"], dtype=torch.bool
        )
       

        return {
            "input_ids": torch.tensor(prefix["input_ids"], dtype=torch.long),
            "target_ids": torch.tensor(suffix["target_ids"], dtype=torch.long),
            "attention_mask": prefix_attention_mask,
        }


# endregion: Collators
################################################################################


################################################################################
# region: Utilities


def _replace_100_with_pad(ids: torch.Tensor, tokenizer: Tokenizer):
    _ids = ids.clone()
    _ids[_ids == -100] = tokenizer.pad_token_id
    return _ids


def print_batch_block_diffusion(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
):
    """Print batch information for debugging BlockDiffusion batches.

    Args:
        batch: The batch to print.
        split: The split name.
        tokenizer: The tokenizer to decode tokens.
        dataloader_name: Name of the dataloader.
    """
    print(
        f"Printing first entries of the tensors in batch for {split}/{dataloader_name}..."
    )
    print("input tokens:")
    # replace -100 with <pad>
    _input_ids = _replace_100_with_pad(batch["input_ids"][0], tokenizer)
    print(tokenizer.decode(_input_ids))
    print("input_ids:")
    print(batch["input_ids"][0])
    if "attention_mask" in batch:
        print("attention_mask (int):")
        print(batch["attention_mask"][0].int())
    if "target_ids" in batch:
        print("target_ids:")
        print(batch["target_ids"][0])
        print("target tokens:")
        _target_ids = _replace_100_with_pad(batch["target_ids"][0], tokenizer)
        print(tokenizer.decode(_target_ids))


# endregion: Utilities
################################################################################

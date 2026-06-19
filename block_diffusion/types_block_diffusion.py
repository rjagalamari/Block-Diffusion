"""Type definitions for BlockDiffusion model.

This file defines the data structures used throughout the BlockDiffusion implementation.
Based on ARLM types - modify as needed for your specific model.
"""

from typing import Optional, Protocol, List, TypedDict
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


class BlockDiffusionBatch(TypedDict):
    """Input to the BlockDiffusion model.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The noisy input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        target_ids (Integer[TT, " batch seq_len"]): The clean target ids for supervision.
        sigma (Float[TT, " batch"]): Diffusion noise level for each example.
        t (Float[TT, " batch"]): Diffusion time sampled for each example.
    """
    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Bool[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len"]
    sigma: Float[TT, " batch"]
    t: Float[TT, " batch"]


class BlockDiffusionSeq2SeqBatch(TypedDict):
    """Input to the BlockDiffusion for sequence-to-sequence training.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model (prompt + target).
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): Token type ids (not used but kept for interface consistency).
        target_ids (Integer[TT, " batch seq_len"]): The target ids for language modeling (shifted by 1).
            Positions with -100 are ignored during loss computation (prompt tokens or padding).
    """
    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Bool[TT, " batch seq_len"]
    token_type_ids: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len"]


class BlockDiffusionLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
    """
    loss: Float[TT, ""]


class BlockDiffusionPredictionDict(TypedDict):
    """Output of the Predictor for BlockDiffusion.

    Attributes:
        text (List[str]): The batch of generated text.
        ids (Integer[TT, " batch seq_len"]): The batch of generated token_ids.
    """
    text: List[str]
    ids: Integer[TT, " batch seq_len"]


class BlockDiffusionModel(Protocol):
    """Protocol defining the interface for BlockDiffusion models."""
    
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        sigma: Float[TT, " batch"],
        attention_mask: Optional[Bool[TT, " batch seq_len seq_len"]] = None,
        sample_mode: bool = False,
        store_kv: bool = False,
        positions: Optional[Integer[TT, " batch seq_len"]] = None,
        **kwargs
    ) -> Float[TT, " batch seq_len vocab_size"]:
        """Forward pass of the model.
        
        Args:
            x_t: The input tokens of shape (batch, seq_len)
            attention_mask: The attention mask of shape (batch, seq_len, seq_len) for full attention matrix,
                          or (batch, seq_len) for simple mask. True for non-padding tokens.
            positions: The positions of the tokens of shape (batch, seq_len)
            **kwargs: Additional model-specific arguments
            
        Returns:
            vocab_logits: The vocabulary logits of shape (batch, seq_len, vocab_size)
        """
        ...

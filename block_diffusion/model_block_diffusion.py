"""Neural network implementation for BlockDiffusion model.

This file contains the main model architecture. Based on ARLM implementation -
modify as needed for your specific model.
"""

from typing import Optional
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT
from xlm.modules.rotary_transformer import (
    RotaryTransformerFinalLayer,
    RotaryTransformerLayer,
    RotaryTransformerLayerList,
    RotaryEmbedding,
)
import math
import typing

import einops
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import modeling_outputs
try:
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask
  FLEX_ATTN_AVAILABLE = True
except:
  FLEX_ATTN_AVAILABLE = False

class BD3LMConfig:
  def __init__(
      self,
      name: str,
      type: str,
      hidden_size: int,
      cond_dim: int,
      length: int,
      n_blocks: int,
      n_heads: int,
      scale_by_sigma: bool,
      dropout: float,
      tie_word_embeddings: bool,
      adaln: bool,
      attn_backend: str,
      vocab_size: int,
      block_size: int,
      causal_attention: bool,
      cross_attn: bool,
      time_conditioning: bool,
      var_min: bool,
      sampling_eps_min: float = 1e-3,
      sampling_eps_max: float = 1.0,
  ):
    self.name = name
    self.type = type

    self.hidden_dim = hidden_size
    self.cond_dim = cond_dim
    self.model_length = length
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    self.scale_by_sigma = scale_by_sigma
    self.dropout = dropout
    self.tie_word_embeddings = tie_word_embeddings
    self.adaln = adaln
    self.attn_backend = attn_backend
    self.vocab_size = vocab_size
    self.block_size = block_size
    self.causal_attention = causal_attention
    self.causal = causal_attention
    self.cross_attn = cross_attn
    self.time_conditioning = time_conditioning
    self.var_min = var_min
    self.sampling_eps_min = sampling_eps_min
    self.sampling_eps_max = sampling_eps_max

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
  """
  Constructs the specialized block diffusion attention mask for training
  composed of three masks:
  - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
  - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
  - **Block Causal Mask (M_BC)**: Attention to update x0

  Args:
      b, h: Batch and head indices (ignored for mask logic).
      q_idx, kv_idx: Query and Key indices.
      seq_len: Total sequence length.
      block_size: Defines the block structure.

  Returns:
      A boolean attention mask.
  """

  # Indicate whether token belongs to xt or x0
  x0_flag_q = (q_idx >= n)
  x0_flag_kv = (kv_idx >= n)

  # Compute block indices
  block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
  block_kv = torch.where(x0_flag_kv == 1,
                        (kv_idx - n) // block_size,
                        kv_idx // block_size)

  # **1. Block Diagonal Mask (M_BD) **
  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

  # **2. Offset Block-Causal Mask (M_OBC) **
  offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0)
  )

  # **3. Block-Causal Mask (M_BC) **
  block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

  # **4. Combine Masks **
  return block_diagonal | offset_block_causal | block_causal

@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask=None):
    return flex_attention(q, k, v, block_mask=mask)

def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift

@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)

@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)

@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)

class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)

# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

def regular_attention_multi_headed(qkv):
  # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
  # where the 3 represents Q, K, V packed in that order
  batch_size, seq_len, _, num_heads, head_dim = qkv.shape
  # Separate Q, K, V from the packed qkv tensor
  # [batch_size, seq_len, num_heads, head_dim]
  q = qkv[:, :, 0, :, :]
  k = qkv[:, :, 1, :, :]
  v = qkv[:, :, 2, :, :]  

  # Transpose and reshape Q and K for batched matrix multiplication:
  # [batch_size, num_heads, seq_len, head_dim]
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)

  # Compute scaled dot-product attention
  # [batch_size, num_heads, seq_len, seq_len]
  attention_scores = torch.matmul(
    q, k.transpose(-2, -1)) / math.sqrt(head_dim)

  # Apply softmax to calculate the attention weights
  attention_probs = F.softmax(attention_scores, dim=-1)

  # [batch_size, num_heads, seq_len, head_dim]
  attention_output = torch.matmul(attention_probs, v)

  # [batch_size, seq_len, num_heads, head_dim]
  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output,
                          'b s h d -> b s (h d)')


class DDiTBlock(nn.Module):
  def __init__(self, n, block_size, dim, n_heads, cond_dim, causal=False,
               mlp_ratio=4, dropout=0.1, adaln=True, attn_backend='sdpa'):
    super().__init__()
    self.n = n
    self.block_size = block_size
    self.n_heads = n_heads
    self.attn_backend = attn_backend
    self.kv_cache = None
    self.cache_idx = 0
    self.causal = causal

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.adaln = adaln
    if self.adaln:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference
  
  def get_qkv(self, x, rotary_cos_sin, store_kv=False, use_kv_cache=False):
    # compute qkv (potentially use cache)
    if use_kv_cache and self.kv_cache is not None:
      new_qkv = self.attn_qkv(x)
      self.kv_cache[:, self.cache_idx:self.cache_idx+self.block_size] = new_qkv
      qkv = self.kv_cache[:, :self.cache_idx+self.block_size].clone()
    else:
      qkv = self.attn_qkv(x)
    # store kv cache in a sliding window (can't exceed context len)
    if store_kv:
      self.cache_idx += self.block_size
      if self.cache_idx >= self.n:
        # left-shift the cache
        self.cache_idx = self.n - self.block_size
        self.kv_cache[:, :-self.block_size] = self.kv_cache[:, self.block_size:].clone()

    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb_torchscript(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    return qkv

  def cross_attn(self, x, qkv, mask=None):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    mask = mask.bool() if mask is not None else None
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=mask,
      is_causal=self.causal,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = einops.rearrange(x, 'b s h d -> b s (h d)')
    return x
  
  def cross_attn_flex(self, qkv, mask=None):
    qkv = einops.rearrange(qkv, 'b s three h d -> b h three s d', h=self.n_heads)
    x = fused_flex_attention(
      qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask)
    x = einops.rearrange(x, 'b h s d -> b s (h d)')
    return x
  
  def forward(self, x, rotary_cos_sin, c, mask=None,
              sample_mode=False, store_kv=False):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    if self.adaln:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # attention operation
    x_skip = x
    if self.adaln:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)

    # get qkvs
    if mask is not None and not sample_mode:
      n = mask.shape[-1] // 2
      qkv_x = self.get_qkv(x[:,:n], rotary_cos_sin, use_kv_cache=False)
      qkv_x0 = self.get_qkv(x[:,n:], rotary_cos_sin, use_kv_cache=False)
      qkv = torch.cat((qkv_x, qkv_x0), dim=1)
    else:
      qkv = self.get_qkv(
        x,
        rotary_cos_sin,
        store_kv=store_kv,
        use_kv_cache=sample_mode,
      )

    if self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      x = self.cross_attn_flex(qkv, mask=mask)
    elif self.attn_backend == 'sdpa' or not FLEX_ATTN_AVAILABLE:
      x = self.cross_attn(x, qkv, mask=mask)
    else:
      raise ValueError('Unknown attention backend')
    
    if sample_mode and self.kv_cache is not None:
      x = x[:, -self.block_size:]

    # mlp operation
    if self.adaln:
      x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      x = bias_dropout_scale_fn(self.attn_out(x),
                              None, torch.ones_like(x), x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)),
        None, torch.ones_like(x), x, self.dropout)

    return x


class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim, adaln=True):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaln = adaln
    if self.adaln:
      self.adaLN_modulation = nn.Linear(cond_dim,
                                        2 * hidden_size,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    if self.adaln:
      shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      x = modulate_fused(self.norm_final(x), shift, scale)
    else:
      x = self.norm_final(x)
    x = self.linear(x)
    return x


class DITBackbone(nn.Module):
  def __init__(
      self,
      config: BD3LMConfig):
    super().__init__()

    self.config = config
    self.cross_attn = config.cross_attn
    self.block_size = config.block_size
    self.vocab_size = config.vocab_size
    self.n = config.model_length

    self.vocab_embed = EmbeddingLayer(
      config.hidden_dim,
      config.vocab_size)
    self.adaln = config.adaln
    if self.adaln:
      self.sigma_map = TimestepEmbedder(
        config.cond_dim)
    self.rotary_emb = Rotary(
      config.hidden_dim // config.n_heads)

    blocks = []
    for _ in range(config.n_blocks):
      blocks.append(DDiTBlock(self.n,
                              self.block_size,
                              config.hidden_dim,
                              config.n_heads,
                              config.cond_dim,
                              causal=config.causal,
                              dropout=config.dropout,
                              adaln=config.adaln,
                              attn_backend=config.attn_backend,))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.hidden_dim,
      config.vocab_size,
      config.cond_dim,
      adaln=config.adaln)
    if self.cross_attn:
      self.gen_mask(config.model_length, self.block_size, attn_backend=config.attn_backend)
    self.precision = torch.float32

  def reset_kv_cache(self, eval_batch_size=1):
    parameter = next(self.parameters())
    for block in self.blocks:
      block.kv_cache = torch.zeros(
        eval_batch_size,
        self.n,
        self.config.hidden_dim * 3,
        device=parameter.device,
        dtype=parameter.dtype)
      block.cache_idx = 0

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference
    
  def gen_mask(self, seqlen, block_size, attn_backend='sdpa'):
    """Genererates attention mask"""
    if attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      self.mask = create_block_mask(
        partial(block_diff_mask, block_size=block_size, n=seqlen),
        B=None, H=None, Q_LEN=seqlen*2, KV_LEN=seqlen*2)
    elif attn_backend == 'sdpa' or not FLEX_ATTN_AVAILABLE:
      self.mask = block_diff_mask(
        b=None, h=None, q_idx=torch.arange(seqlen*2)[:, None], 
        kv_idx=torch.arange(seqlen*2)[None, :], block_size=block_size, n=seqlen)
    else:
      raise ValueError('Unknown attention backend')

  def forward(self, indices, sigma, attention_mask=None, sample_mode=False,
             store_kv=False, output_hidden_states=False):
    if not self.config.time_conditioning and self.adaln:
      sigma = torch.zeros_like(sigma)
    all_hidden_states = []
    x = self.vocab_embed(indices)
    if output_hidden_states:
      all_hidden_states.append(x)
    c = None
    if self.adaln:
      c = F.silu(self.sigma_map(sigma))
    if self.cross_attn:
      # use block-causal mask only during sampling
      if not sample_mode:
        if x.shape[1] % 2 != 0:
          raise ValueError(
            "Cross-attention training expects concatenated xt/x0 streams "
            f"with equal lengths, but received sequence length {x.shape[1]}."
          )
        n = x.shape[1] // 2
        rotary_cos_sin = self.rotary_emb(x[:, :n])
        if self.config.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
          mask = create_block_mask(
            partial(block_diff_mask, block_size=self.block_size, n=n),
            B=None, H=None, Q_LEN=n * 2, KV_LEN=n * 2,
            device=x.device,
          )
        else:
          positions = torch.arange(n * 2, device=x.device)
          mask = block_diff_mask(
            b=None,
            h=None,
            q_idx=positions[:, None],
            kv_idx=positions[None, :],
            block_size=self.block_size,
            n=n,
          )
      else:
        mask = self.mask.to(x.device)
        n = self.n
        if self.blocks[0].kv_cache is not None:
          mask = None
          accum_length = self.blocks[0].cache_idx + self.block_size
          # positional encodings for cache
          x_full = torch.zeros((
            x.shape[0], accum_length, x.shape[2]), device=x.device)
          rotary_cos_sin = self.rotary_emb(x_full)
        else:
          rotary_cos_sin = self.rotary_emb(x[:, :n])
          mask = mask[
            n:n+x.shape[1], n:n+x.shape[1]]
    else:
      mask = None
      rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, 
                           rotary_cos_sin,
                           c,
                           mask=mask,
                           sample_mode=sample_mode,
                           store_kv=store_kv)
        if output_hidden_states:
          all_hidden_states.append(x)
      logits = self.output_layer(x, c)
    if self.cross_attn and not sample_mode:
      logits = logits[:, :n]
      all_hidden_states = [hidden_states[:, :n] for hidden_states in all_hidden_states]
    return logits, all_hidden_states


class BlockDiffusion(torch.nn.Module):
  def __init__(
      self,
      name: str,
      type: str,
      hidden_size: int,
      cond_dim: int,
      length: int,
      n_blocks: int,
      n_heads: int,
      scale_by_sigma: bool,
      dropout: float,
      tie_word_embeddings: bool,
      adaln: bool,
      attn_backend: str,
      vocab_size: int,
      block_size: int = 8,
      causal_attention: bool = False,
      cross_attn: bool = True,
      time_conditioning: bool = False,
      var_min: bool = True,
      sampling_eps_min: float = 1e-3,
      sampling_eps_max: float = 1.0,
  ):
    super().__init__()
    self.config = BD3LMConfig(
        name=name,
        type=type,
        hidden_size=hidden_size,
        cond_dim=cond_dim,
        length=length,
        n_blocks=n_blocks,
        n_heads=n_heads,
        scale_by_sigma=scale_by_sigma,
        dropout=dropout,
        tie_word_embeddings=tie_word_embeddings,
        adaln=adaln,
        attn_backend=attn_backend,
        vocab_size=vocab_size,
        block_size=block_size,
        causal_attention=causal_attention,
        cross_attn=cross_attn,
        time_conditioning=time_conditioning,
        var_min=var_min,
        sampling_eps_min=sampling_eps_min,
        sampling_eps_max=sampling_eps_max,
    )
    self.backbone = DITBackbone(self.config)

  def forward(self, x, sigma, attention_mask=None, sample_mode=False, store_kv=False):
    logits, _ = self.backbone(
        indices=x,
        sigma=sigma,
        attention_mask=attention_mask,
        sample_mode=sample_mode,
        store_kv=store_kv,
    )
    return logits

  def reset_kv_cache(self, eval_batch_size=1):
    self.backbone.reset_kv_cache(eval_batch_size=eval_batch_size)

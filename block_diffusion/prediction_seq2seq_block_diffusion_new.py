"""Seq2seq predictor implementation for BlockDiffusion.

This file starts from the existing unconditional BlockDiffusion predictor and
separates the seq2seq prediction path so we can modify it without touching the
original unconditional generation code.
"""

from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
import numpy as np
from .predictor_block_diffusion import BlockDiffusionPredictor
from .types_block_diffusion import BlockDiffusionPredictionDict
from xlm.utils.nn import sample_categorical as _sample_categorical
import random
import sys
class BlockDiffusionSeq2SeqPredictor(BlockDiffusionPredictor):
    """Seq2seq predictor for BlockDiffusion.

    Current starting point:
      - ``batch["input_ids"]`` is the prefix/prompt only.
      - ``batch["target_ids"]`` is the clean answer, used for logging/eval.

    The body below intentionally starts from the original unconditional
    ``BlockDiffusionPredictor.predict`` logic. We will replace the unconditional
    sampler with prefix-conditioned seq2seq sampling step by step.
    """

    def _decode_without_pad(self, ids: Any) -> List[str]:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        sequences = []
        raw_sequences = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        for seq in raw_sequences:
            if pad_id is not None:
                seq = [token_id for token_id in seq if token_id != int(pad_id)]
            sequences.append(seq)
        return self.tokenizer.batch_decode(sequences)

    def _ddpm_caching_update(self, x, t, dt, p_x0=None,attention_mask=None):
        
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
                p_x0 = self.model(
                    x[:, -block_size:],
                    sigma_t,
                    attention_mask = attention_mask,
                    sample_mode=True,
                ).to(torch.float64)
            else:
                p_x0 = self.model(
                    x,
                    sigma_t,
                    attention_mask = attention_mask,
                    sample_mode=True,
                ).to(torch.float64)
                p_x0 = p_x0[:, -block_size:]
            p_x0 = p_x0.exp()
            p_x0 = self._nucleus_sample(p_x0)

        if self.first_hitting:
            x_block = _sample_categorical(p_x0)
            # randomly and uniformly select an index in the block (among masked tokens)
            
            num_masked = (x[:, -block_size:] == mask_id).sum(-1)
            
            ind = torch.randint(0, num_masked, (x_block.shape[0],))
            ind = (x[:, -block_size:] == mask_id).nonzero()[ind, 1]
            mask = (
                torch.arange(block_size, device=x.device) == ind[:, None]
            ).to(x_block.dtype)
            x_block = x_block * mask + x[:, -block_size:] * (1 - mask)
        else:
            q_xs = p_x0 * (1 - mask_prob)
            q_xs[:, :, mask_id] = mask_prob.squeeze(-1)
            x_block = _sample_categorical(q_xs)

        copy_flag = (x[:, -block_size:] != mask_id).to(x.dtype)
        x_block = copy_flag * x[:, -block_size:] + (1 - copy_flag) * x_block
        x_new = torch.cat((x[:, :-block_size], x_block), dim=-1)

        # compute kv cache if all tokens in a block are sampled
        if self.kv_cache and mask_id not in x_block:
            _ = self.model(x_block, sigma_t,attention_mask=attention_mask, sample_mode=True, store_kv=True)
        
        if not torch.allclose(x_new, x):
            return None, x_new
       
        return p_x0, x_new

    def _sample(
        self,
        input_ids: torch.Tensor,
        seqlen: Optional[int] = None,
        num_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        block_size = self._get_block_size()
        seqlen = self._round_up_to_multiple(int(seqlen), block_size)
        num_strides = max(1, seqlen // block_size)

        for _ in range(10):
            sample_i, _ = self._semi_ar_sampler(
                input_ids=input_ids,
                n_samples=batch_size,
                num_steps=num_steps,
                num_strides=num_strides,
                seqlen=seqlen,
                device=device,
                attention_mask = attention_mask,
            )
            if sample_i is not None:
                return sample_i
        raise ValueError("Sampling failed.")

   


    def _semi_ar_sampler(
        self,
        input_ids,
        n_samples,
        num_steps,
        num_strides,
        seqlen,
        device,
        context_size=1024,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        block_size = self._get_block_size()
        mask_index = self._mask_id()
        if seqlen is None:
            seqlen = self.config.model.length
        sampling_steps = 0

        '''mdlm_semi_ar = (
            self.config.algo.name == "mdlm"
            and self.config.model.length > self.block_size
        )
        if mdlm_semi_ar:
            # sliding window of length 512 for mdlm semi-ar decoding
            num_strides = self.config.model.length // 512
            num_strides -= 1'''
        if attention_mask is not None:
            target_mask = torch.ones(
             (attention_mask.shape[0], 12),
             dtype=torch.bool,
             device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, target_mask], dim=1)
        
        ones = torch.ones((n_samples, 1), dtype=torch.float32, device=device)
        batch_size = input_ids.shape[0]
        # reset kvs
        if self.kv_cache:
            self.model.reset_kv_cache(
                eval_batch_size=batch_size
            )
        ### prefix filling phase prompt filling..
        num_prefix_blocks = input_ids.shape[1] // block_size
        
        if self.kv_cache:
                for i in range(num_prefix_blocks):
                    sigma_t = torch.zeros((input_ids.shape[0],), device=device, dtype=torch.float32)
                    if attention_mask is not None:
                        pre_x = self.model(
                        input_ids[:, i * block_size : (i + 1) * block_size],
                        sigma_t,
                        attention_mask=attention_mask[:, 0: (i + 1) * block_size],
                        sample_mode=True,
                        store_kv=True,
                        ).to(torch.float64)
                    else:
                        pre_x = self.model(
                        input_ids[:, i * block_size : (i + 1) * block_size],
                        sigma_t,
                        attention_mask=attention_mask,
                        sample_mode=True,
                        store_kv=True,
                        ).to(torch.float64)
        else:
                sigma_t = torch.zeros((input_ids.shape[0],), device=device, dtype=torch.float32)
                per_x = self.model(
                    input_ids,
                    sigma_t,
                    attention_mask=attention_mask,
                    sample_mode=True,
                ).to(torch.float64)
        start = input_ids.shape[1]
        #print("this is the block size",block_size)
        #exit()
        j = 1 
        k=0
        for stride_num in tqdm(range(num_strides)):
            
           
            # sample next block
            if stride_num == 0:
                x_accum= self._sample_prior(n_samples, block_size,device=device).to(device)
                #x_accum[:, 0] = self.tokenizer.bos_token_id
            else:
                '''if mdlm_semi_ar:
                    x = self._sample_prior(n_samples, 512).to(device)
                else:'''
                x = self._sample_prior(n_samples, block_size,device=device).to(device)
                x_accum = torch.cat((x_accum, x), dim=1)

            # compute logits in a sliding window (context passed to model can't exceed context_size)
            end_idx = (stride_num + 1) * block_size
            start_idx = max(end_idx - context_size, 0)
            fwd_idx = torch.arange(start_idx, end_idx)
            '''if mdlm_semi_ar and stride_num > 0:  # MDLM
                fwd_idx = torch.arange(
                    512 * (stride_num),
                    (512 * (stride_num)) +block_size,
                )'''
            #print("this is teh block size here",block_size)
            #exit()
            dt = 1 / num_steps
            p_x0_cache = None
            timesteps = torch.linspace(1, 0, num_steps, device=device)
            t = 1
            for i in range(num_steps):
                
                if mask_index not in x_accum:
                    break

                # faster (equivalent) sampler from zheng et al (2025)
                if self.first_hitting:
                    u = np.random.rand()
                    num_masked = (x_accum[:, fwd_idx] == mask_index).sum(-1).item()
                    t *= u ** (1 / num_masked)
                elif not self.first_hitting:
                    t = timesteps[i]
                
                if attention_mask is not None:
                    #cur_inp = torch.cat([input_ids, x_accum[:, fwd_idx]], dim=1)
                    p_x0_cache, x_next = self._ddpm_caching_update(
                        x=x_accum[:, fwd_idx],
                        t=t * ones,
                        dt=dt,
                        p_x0=p_x0_cache,
                        attention_mask=attention_mask[:, :start+(j*block_size)],
                    )
                else:
                    #cur_inp = torch.cat([input_ids, x_accum[:, fwd_idx]], dim=1)
                    
                    p_x0_cache, x_next = self._ddpm_caching_update(
                        x=x_accum[:, fwd_idx],
                        t=t * ones,
                        dt=dt,
                        p_x0=p_x0_cache,
                    )
                if p_x0_cache is None:
                    sampling_steps += 1
                #print("x_next is ", x_next)
                if self.kv_cache is False:
                    x_accum[:, fwd_idx] = x_next[:,start:]
                else:
                    x_accum[:, fwd_idx] = x_next
            j+=1
            # check if we need to resample (or stop sampling for variable-length sampling)
            if x_accum.shape[1] > 256:
                stop, x_accum = self._check_stop_conds(x_accum)
                if (stop and not self.config.sampling.var_length) or (
                    stop and x.shape[-1] == 1
                ):
                    return None, None
                elif stop:
                    break
            '''eos_id = getattr(self.tokenizer, "eos_token_id", None)
            mask_id = mask_index  # or self._mask_id()

            if eos_id is not None:
                block_tokens = x_accum[:, fwd_idx]

                # For batch size 1 this is the common case
                eos_pos = (block_tokens == eos_id).nonzero(as_tuple=False)

                if eos_pos.numel() > 0:
                    first_eos_col = int(eos_pos[0, 1].item())

                    # If there is any MASK before EOS, keep sampling
                    has_mask_before_eos = (block_tokens[:, :first_eos_col] == mask_id).any()

                    if not has_mask_before_eos:
                        x_accum = x_accum[:, : start_idx + first_eos_col + 1]
                        return x_accum, sampling_steps'''
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
        target_ids = batch.get("target_ids")
        batch_size = input_ids.shape[0]
        device = input_ids.device
        attention_mask = batch.get("attention_mask", None)
        
        sampled = self._sample(
            input_ids=input_ids,
            seqlen=12,
            num_steps=max(1, (self.max_steps+1)),
            batch_size=batch_size,
            device=device,
            attention_mask=attention_mask,
        )
        
        if sampled is None:
            raise ValueError("Sampling failed.")

        current_ids = sampled

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and eos_id in current_ids[0]:
            eos_idx = (current_ids[0] == eos_id).nonzero()[0].item()
            current_ids = current_ids[:, :eos_idx + 1]
        generated_text = self._decode_without_pad(current_ids)
        input_text = self._decode_without_pad(input_ids)
        target_text = self._decode_without_pad(target_ids)
        
        return {
            "text": generated_text,
            "input_text": input_text,
            "target_text": target_text,
        }

    def to_dict(
        self,
        batch: Dict[str, Any],
        preds: BlockDiffusionPredictionDict,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        results = []
        input_text = preds.get("input_text", [""] * len(preds["text"]))
        target_text = preds.get("target_text", [""] * len(preds["text"]))
        for i in range(len(preds["text"])):
            results.append(
                {
                    "input_text": input_text[i],
                    "generated_text": preds["text"][i],
                    "target_text": target_text[i],
                }
            )
        return results

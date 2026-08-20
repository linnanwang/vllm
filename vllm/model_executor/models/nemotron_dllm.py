# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron Ministral-based masked block-diffusion LM (LLaDA-style).

A Ministral (Llama-like) backbone run in two modes over a fixed-length
canvas that rides vLLM's spec-decode path (like DiffusionGemma):
- encoder mode (prefill/commit): causal attention, writes KV
- denoise mode: bidirectional attention over the canvas, reads prefix KV

Unlike DiffusionGemma (random-state diffusion), this model uses MASKED
diffusion: the canvas starts as ``mask_token_id`` tokens and per step the
most confident masked positions are replaced by the predicted token —
greedy argmax by default, or Gumbel-max sampling when a sampling
temperature is configured (absorbing state; tokens are never re-masked).
The block is committed once no masked positions remain or the step budget
is exhausted.

Checkpoint layout: ``encoder.*`` -> backbone, ``diffusion_head.*`` -> LM head
(untied).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor, async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import use_penalty
from vllm.v1.worker.gpu.states import RequestState

from .interfaces import SupportsPP, SupportsQuant

logger = init_logger(__name__)


def _hf_yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    """``get_mscale`` from HF ``_compute_yarn_parameters``."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def is_ar_mode(hf_config: PretrainedConfig) -> bool:
    """Whether to run the checkpoint as a plain causal LM.

    The backbone is trained for both generation modes, so setting
    ``ar_mode`` on the HF config (``--hf-overrides '{"ar_mode": true}'``)
    selects fully causal attention and vLLM's ordinary autoregressive
    decode instead of the masked block-diffusion canvas. Matches the
    SGLang port's ``ar_mode`` override.
    """
    return bool(getattr(hf_config, "ar_mode", False))


class NemotronLabsDiffusionAttention(LlamaAttention):
    """Llama attention + YaRN RoPE (HF attention factor) + per-token
    llama-4 Q scaling applied after RoPE."""

    def __init__(self, *, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.llama4_scaling_beta: float | None = rope_parameters.get(
            "llama_4_scaling_beta"
        )
        self.llama4_original_max_pos: int = rope_parameters.get(
            "original_max_position_embeddings",
            getattr(config, "max_position_embeddings", 16384),
        )

    def _init_rotary_emb(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        rope_parameters = dict(getattr(config, "rope_parameters", None) or {})
        if rope_parameters.get("rope_type") == "yarn":
            # The HF reference scales cos/sin by
            # get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            # (== 1.0 for this model), while vLLM's YaRN would apply
            # 0.1*ln(factor)+1. Compute the HF factor and pass it via
            # attn_factor with vLLM's own yarn scaling disabled.
            if "attn_factor" not in rope_parameters:
                factor = rope_parameters["factor"]
                mscale = rope_parameters.get("mscale")
                mscale_all_dim = rope_parameters.get("mscale_all_dim")
                if mscale and mscale_all_dim:
                    attn_factor = _hf_yarn_get_mscale(
                        factor, mscale
                    ) / _hf_yarn_get_mscale(factor, mscale_all_dim)
                else:
                    attn_factor = _hf_yarn_get_mscale(factor)
                rope_parameters["attn_factor"] = float(attn_factor)
            rope_parameters["apply_yarn_scaling"] = False
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        if self.llama4_scaling_beta is not None:
            # scale = 1 + beta * log(1 + floor(pos / original_max_pos)),
            # applied AFTER RoPE (matches HF `_get_llama_4_attn_scale`).
            scale = 1.0 + self.llama4_scaling_beta * torch.log(
                1.0 + torch.floor(positions.float() / self.llama4_original_max_pos)
            )
            q = q * scale.unsqueeze(-1).to(q.dtype)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class NemotronLabsDiffusionDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: PretrainedConfig | None = None,
    ) -> None:
        super().__init__(
            vllm_config,
            prefix=prefix,
            config=config,
            attn_layer_type=NemotronLabsDiffusionAttention,
        )


class NemotronLabsDiffusionForBlockDiffusion(nn.Module, SupportsPP, SupportsQuant):
    """Ministral masked block-diffusion LM for vLLM.

    The backbone is a stock ``LlamaModel`` with Ministral parameters; the
    only architectural deltas live in the attention layer (YaRN factor and
    llama-4 Q scale). Mixed causal/bidirectional attention is driven by
    ``NemotronLabsDiffusionModelState.prepare_attn`` via a per-request causal tensor.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "encoder.": "model.",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def get_model_state_cls(self):
        if self.ar_mode:
            from vllm.v1.worker.gpu.model_states.default import DefaultModelState

            return DefaultModelState
        return NemotronLabsDiffusionModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.ar_mode = is_ar_mode(config)

        self.model = LlamaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=NemotronLabsDiffusionDecoderLayer,
        )
        # The LM head is named diffusion_head in the checkpoint and is NOT
        # tied to the embeddings.
        self.diffusion_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "diffusion_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.diffusion_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class NemotronLabsDiffusionForCausalLM(nn.Module, SupportsPP, SupportsQuant):
    """AR (causal) variant of the Nemotron masked block-diffusion LM.

    Identical backbone (LlamaModel with NemotronLabsDiffusionDecoderLayer: YaRN
    factor + llama-4 Q scale) and untied ``diffusion_head`` LM head as
    NemotronLabsDiffusionForBlockDiffusion, and the SAME ``encoder.`` -> ``model.``
    weight mapper -- so weights load bit-identically. The ONLY difference is that
    this class does NOT define ``get_model_state_cls``, so the v1 runner falls
    back to the default causal AR state (standard causal attention + normal
    KV-cache decode) instead of the masked-canvas denoising path. Used for AR
    GRPO on the ORIGINAL diffusion checkpoint: select via architectures override
    ["NemotronLabsDiffusionForCausalLM"]; ModelConfig.is_diffusion returns False
    for it (see config/model.py).
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "encoder.": "model.",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # NOTE: intentionally NO get_model_state_cls -> DefaultModelState (AR path).

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = LlamaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=NemotronLabsDiffusionDecoderLayer,
        )
        # LM head is named diffusion_head in the checkpoint and is NOT tied.
        self.diffusion_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "diffusion_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.diffusion_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@torch.compile(dynamic=True)
def _compute_num_rejected(
    num_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_rejected = num_logits - num_sampled
    is_denoise = (num_logits > 0) & (num_sampled == 0)
    return torch.where(is_denoise, query_lens, num_rejected)


@torch.compile(dynamic=True)
def _compiled_masked_step(
    # Logits from the model [num_decode * CL, vocab]
    logits: torch.Tensor,
    # Request mapping
    decode_slots: torch.Tensor,  # [num_decode] int64 -> slot indices
    decode_idx: torch.Tensor,  # [num_decode] int64 -> position in num_reqs
    all_slots: torch.Tensor,  # [num_reqs] int64 -> all slot indices
    valid_canvas_len: torch.Tensor,  # [num_decode] int64 real canvas len (<=CL)
    req_temps: torch.Tensor,  # [num_decode] fp32 per-request temperature (0=greedy)
    # State tensors (modified in-place)
    canvas: torch.Tensor,  # [max_num_reqs, CL]
    step_tensor: torch.Tensor,  # [max_num_reqs]
    is_encoder_phase: torch.Tensor,  # [max_num_reqs]
    unmask_logprobs: torch.Tensor,  # [max_num_reqs, CL] fp32
    unmask_ranks: torch.Tensor,  # [max_num_reqs, CL] int32
    unmask_entropy: torch.Tensor,  # [max_num_reqs, CL] fp32
    unmask_step: torch.Tensor,  # [max_num_reqs, CL] int32
    # Output tensors (modified in-place)
    sampled: torch.Tensor,  # [num_reqs, CL]
    num_sampled: torch.Tensor,  # [num_reqs]
    draft_tokens: torch.Tensor,  # [max_num_reqs, >=CL]
    reveal_steps: torch.Tensor,  # [num_reqs, CL] int32
    # Scalar config
    max_denoising_steps: int,
    mask_token_id: int,
    CL: int,
    temperature: float,
    capture_logprobs: bool,
    capture_entropy: bool,
    capture_steps: bool,
    leftmost: bool,
    random_mode: bool,
    threshold_mode: bool,
    log_threshold: float,
) -> torch.Tensor:
    """Compiled masked-diffusion step (low-confidence, leftmost, or
    confidence-threshold selection, absorbing state).

    Per denoise step: x0 over the whole canvas — greedy argmax when
    ``temperature == 0``, otherwise sampled from ``softmax(logits / T)``
    via the Gumbel-max trick (LLaDA's ``add_gumbel_noise``). Confidence is
    the prob of the chosen token under the UNSCALED distribution (LLaDA
    low-confidence remasking); the top-k most confident masked positions
    are unmasked where k follows the even transfer schedule (remaining
    masks split evenly over remaining steps). With ``leftmost``, position
    order replaces confidence order (strictly left-to-right reveal — the
    leftmost-reveal RL rollout policy). With ``random_mode``, a uniform random
    order replaces confidence order (the reveal choice is independent of the
    model scores, so the recorded logprobs are unbiased policy logprobs). With ``threshold_mode``, the even
    schedule is replaced by SGLang FastDiffuser thresholding: every masked
    position whose chosen-token logprob is >= ``log_threshold`` is unmasked,
    plus always the single most confident one so each step makes progress
    (worst case degrades to one-token-per-step and the ``canvas_length``
    default step budget still terminates). Unmasked positions are never
    re-masked. A request converges when no masks remain or the step budget
    is reached; the next step commits the block (causal, writes KV) and
    resets the canvas to masks.

    When ``capture_logprobs``, the chosen token's logprob and rank under
    the sampling distribution ``softmax(logits / max(T, 1))`` are written
    into ``unmask_logprobs``/``unmask_ranks`` at the step each position is
    unmasked, so the logprobs emitted at commit reflect the distribution
    each token was actually sampled from (not the final denoise step's).
    With ``capture_entropy`` (requires ``capture_logprobs``), the entropy
    ``H = -sum_v p_v log p_v`` of that same at-unmask sampling distribution
    is written into ``unmask_entropy`` (the DiffusionConfig.return_entropy
    channel, consumed by the JustGRPO-Fast trainer).
    With ``capture_steps`` (independent of ``capture_logprobs``), the
    block-relative denoise step that unmasked each position is written into
    ``unmask_step`` and emitted through ``reveal_steps`` on the commit step
    (the DiffusionConfig.return_reveal_steps channel, consumed by the
    trajectory-replay "Trace" trainer).

    Returns the fp32 logits ``[num_decode, CL, vocab]`` so the caller can
    compute top-k logprobs outside the compiled region.
    """
    num_decode = decode_slots.shape[0]
    device = decode_slots.device

    logits_3d = logits.reshape(num_decode, CL, -1).float()

    is_commit = is_encoder_phase[decode_slots]  # [num_decode]
    is_denoise = ~is_commit

    cur_canvas = canvas[decode_slots]  # [num_decode, CL]
    pos = torch.arange(CL, device=device)
    # A canvas truncated near max_model_len is zero-padded up to CL by the
    # caller; the padded tail is treated as non-masked so it is never
    # transferred or committed.
    valid = pos.unsqueeze(0) < valid_canvas_len.unsqueeze(1)  # [num_decode, CL]
    mask_index = (cur_canvas == mask_token_id) & valid

    # ---- x0 + confidence (prob of chosen token) at masked positions ----
    log_probs = logits_3d.log_softmax(dim=-1)
    if temperature > 0:
        # Gumbel-max: argmax(logits + T*g) ~ softmax(logits / T).
        # Rows whose request asked for greedy (per-request temperature 0)
        # get zero noise and reduce to argmax.
        u = torch.rand_like(logits_3d).clamp_min(1e-20)
        gumbel = -torch.log(-torch.log(u))
        row_scale = temperature * (req_temps > 0).to(logits_3d.dtype)
        x0 = (logits_3d + row_scale[:, None, None] * gumbel).argmax(dim=-1)
    else:
        x0 = logits_3d.argmax(dim=-1)  # [num_decode, CL]
    x0_logprob = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    neg_inf = torch.full_like(x0_logprob, float("-inf"))
    confidence = torch.where(mask_index, x0_logprob, neg_inf)

    # ---- Even transfer schedule: k = ceil(#masks / remaining steps) ----
    num_masks = mask_index.sum(dim=-1)  # [num_decode] int64
    cur_step = step_tensor[decode_slots].to(torch.int64)
    remaining_steps = (max_denoising_steps - cur_step).clamp(min=1)
    k = -(-num_masks // remaining_steps)  # ceil div

    if leftmost:
        # Leftmost-reveal: unmask the first k masked positions (strict
        # left-to-right order instead of confidence order).
        csum = torch.cumsum(mask_index.to(torch.int64), dim=-1)
        transfer = mask_index & (csum <= k.unsqueeze(1))
    elif random_mode:
        # Uniform random reveal order (SGLang FastDiffuser "random"): rank the
        # masked positions by uniform noise instead of confidence, so the
        # choice carries no information about the model scores. k still comes
        # from the even transfer schedule. Non-masked positions get -inf, so
        # they can never be selected regardless of their noise draw.
        rand_score = torch.where(mask_index, torch.rand_like(x0_logprob), neg_inf)
        order = torch.argsort(rand_score, dim=-1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, pos.expand(num_decode, CL))
        transfer = mask_index & (ranks < k.unsqueeze(1))
    elif threshold_mode:
        # FastDiffuser thresholding: unmask every masked position whose
        # chosen-token logprob clears the threshold, and always the single
        # most confident one (guaranteed progress; also covers the case
        # where nothing clears the bar). `confidence` is -inf outside
        # mask_index, so argmax lands on a masked position whenever one
        # exists and both terms are automatically gated by mask_index.
        above = confidence >= log_threshold
        top1 = confidence.argmax(dim=-1, keepdim=True)  # [num_decode, 1]
        force = torch.zeros_like(mask_index).scatter_(1, top1, True)
        transfer = mask_index & (above | force)
    else:
        # Top-k per row with per-row k: rank positions by confidence (desc).
        order = torch.argsort(confidence, dim=-1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, pos.expand(num_decode, CL))
        transfer = mask_index & (ranks < k.unsqueeze(1))

    # ---- At-unmask logprob/rank capture (sampling distribution) ----
    if capture_logprobs:
        if temperature > 0 and temperature != 1.0:
            sample_lp = (logits_3d / temperature).log_softmax(dim=-1)
            chosen_lp = sample_lp.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        else:
            sample_lp = log_probs
            chosen_lp = x0_logprob
        # Rank convention matches _ranks_kernel: #tokens with lp >= chosen.
        chosen_rank = (
            (sample_lp >= chosen_lp.unsqueeze(-1)).sum(dim=-1).to(torch.int32)
        )
        write = transfer & is_denoise.unsqueeze(1)
        unmask_logprobs[decode_slots] = torch.where(
            write, chosen_lp, unmask_logprobs[decode_slots]
        )
        unmask_ranks[decode_slots] = torch.where(
            write, chosen_rank, unmask_ranks[decode_slots]
        )
        if capture_entropy:
            # Entropy of the same at-unmask sampling distribution the
            # chosen token's logprob is scored under.
            token_entropy = -(sample_lp.exp() * sample_lp).sum(dim=-1)
            unmask_entropy[decode_slots] = torch.where(
                write, token_entropy, unmask_entropy[decode_slots]
            )

    # ---- At-unmask reveal-step capture (block-relative denoise step) ----
    # Deliberately outside the ``capture_logprobs`` block: the reveal-step
    # channel does not ride the logprobs output, so it must work at
    # ``logprobs=0`` and when logprobs are off entirely. ``cur_step`` is the
    # pre-increment counter, which resets to 0 at every commit, so the
    # recorded value is block-relative and 0-based -- the same convention as
    # the SGLang fork's ``commit_step``.
    if capture_steps:
        step_write = transfer & is_denoise.unsqueeze(1)
        unmask_step[decode_slots] = torch.where(
            step_write,
            cur_step.to(unmask_step.dtype).unsqueeze(1),
            unmask_step[decode_slots],
        )

    # ---- Absorbing-state update (denoise rows only) ----
    new_canvas = torch.where(transfer & is_denoise.unsqueeze(1), x0, cur_canvas)

    # Commit rows: reset the canvas to masks for the next block.
    mask_canvas = torch.full_like(cur_canvas, mask_token_id)
    canvas[decode_slots] = torch.where(is_commit.unsqueeze(1), mask_canvas, new_canvas)

    # Step counter: +1 for denoise, reset to 0 for commit.
    new_step = torch.where(
        is_denoise,
        (cur_step + 1).to(step_tensor.dtype),
        step_tensor.new_zeros(num_decode),
    )
    step_tensor[decode_slots] = new_step

    # Sampled output: the commit step emits the canvas that was fed to the
    # model this step; denoise steps emit nothing. A canvas truncated by
    # max_model_len can converge with unresolved masks in its tail (only the
    # scheduled prefix is denoised); never emit mask tokens — commit stops at
    # the first remaining mask (for leftmost reveal this is exactly the clean
    # prefix).
    mask_in_canvas = (cur_canvas == mask_token_id) & valid
    first_mask = torch.where(
        mask_in_canvas.any(dim=-1),
        mask_in_canvas.to(torch.int64).argmax(dim=-1),
        valid_canvas_len,
    )
    emit_len = torch.minimum(valid_canvas_len, first_mask)
    sampled[decode_idx] = cur_canvas.to(sampled.dtype) * is_commit.unsqueeze(1).to(
        sampled.dtype
    )
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * emit_len.to(
        num_sampled.dtype
    )
    if capture_steps:
        # Parallel to ``sampled``, emitted on the commit step only. Trimmed to
        # ``num_sampled`` downstream, exactly like ``sampled_token_ids``.
        reveal_steps[decode_idx] = unmask_step[decode_slots].to(
            reveal_steps.dtype
        ) * is_commit.unsqueeze(1).to(reveal_steps.dtype)

    # ---- Convergence: no masks remain, or step budget exhausted ----
    masks_left = ((new_canvas == mask_token_id) & valid).sum(dim=-1)
    converged = (masks_left == 0) | (new_step.to(torch.int64) >= max_denoising_steps)
    # Commit done -> denoise next (False); denoise converged -> commit (True).
    is_encoder_phase[decode_slots] = torch.where(
        is_commit, is_commit.new_zeros(num_decode), converged
    )

    # Feed the canvas back to the scheduler as next step's draft tokens.
    draft_tokens[all_slots, :CL] = canvas[all_slots]

    return logits_3d


class NemotronLabsDiffusionRequestStates:
    """Pre-allocated GPU tensors for per-request masked-diffusion state."""

    def __init__(
        self,
        max_num_reqs: int,
        canvas_length: int,
        mask_token_id: int,
        max_denoising_steps: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.mask_token_id = mask_token_id
        self.max_denoising_steps = max_denoising_steps
        self.device = device

        self.is_encoder_phase = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        # Canvas tokens [max_num_reqs, canvas_length], mask_token_id when
        # not yet unmasked.
        self.canvas = torch.full(
            (max_num_reqs, canvas_length),
            mask_token_id,
            dtype=torch.int64,
            device=device,
        )
        # Denoising step counter (0..max_denoising_steps).
        self.step = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # Per-position logprob/rank of the chosen token, captured at the
        # denoise step each position was unmasked; emitted at commit.
        self.unmask_logprobs = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.float32, device=device
        )
        self.unmask_ranks = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int32, device=device
        )
        # Per-position entropy of the at-unmask sampling distribution;
        # populated only when the sampler runs with return_entropy.
        self.unmask_entropy = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.float32, device=device
        )
        # Per-position block-relative denoise step at which the position was
        # unmasked; populated only when the sampler runs with
        # return_reveal_steps.
        self.unmask_step = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int32, device=device
        )

    def init_canvas(self, slot_indices) -> None:
        """Reset the canvas to all-mask for the given slots."""
        self.canvas[slot_indices] = self.mask_token_id

    def add_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = True
        self.canvas[slot_idx] = self.mask_token_id
        self.step[slot_idx] = 0
        self.unmask_logprobs[slot_idx] = 0.0
        self.unmask_ranks[slot_idx] = 0
        self.unmask_entropy[slot_idx] = 0.0
        self.unmask_step[slot_idx] = 0

    def remove_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = False


class NemotronLabsDiffusionModelState(ModelState):
    """ModelState for the Ministral masked block-diffusion LM.

    Single backbone in two modes:
    - encoder mode (prefill/commit): causal attention, writes KV
    - denoise mode: bidirectional attention over the canvas
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: Any,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)

        hf_config = self.model_config.hf_config
        diffusion_config = vllm_config.diffusion_config
        canvas_length = (
            diffusion_config.canvas_length
            if diffusion_config is not None
            else getattr(hf_config, "canvas_length", 32)
        )
        gen_config = self.model_config.try_get_generation_config()
        # Default to one unmasked token per step (HF steps == gen_length).
        max_denoising_steps = (
            (diffusion_config.max_denoising_steps if diffusion_config else None)
            or gen_config.get("max_denoising_steps")
            or canvas_length
        )
        # Denoising sampler temperature (engine-wide; per-request temperature
        # is rejected for diffusion models). 0 = greedy. None-checks rather
        # than `or` since 0.0 is a valid explicit setting.
        temperature = diffusion_config.temperature if diffusion_config else None
        if temperature is None:
            temperature = gen_config.get("diffusion_temperature")
        if temperature is None:
            temperature = 0.0
        self.sampling_temperature = float(temperature)
        selection = (
            diffusion_config.selection_policy if diffusion_config else None
        ) or gen_config.get("diffusion_selection_policy") or "low_confidence"
        if selection not in (
            "low_confidence",
            "leftmost",
            "confidence_threshold",
            "random",
        ):
            raise ValueError(
                f"Unknown diffusion selection_policy: {selection!r}"
            )
        self.selection_policy = selection
        # Probability threshold for selection_policy="confidence_threshold"
        # (SGLang FastDiffuser thresholding). None-checks rather than `or`
        # so an explicit small value is honored.
        threshold = (
            diffusion_config.confidence_threshold if diffusion_config else None
        )
        if threshold is None:
            threshold = gen_config.get("diffusion_confidence_threshold")
        if threshold is None:
            threshold = 0.9
        self.confidence_threshold = float(threshold)
        # Opt-in per-token entropy channel (JustGRPO-Fast): emitted through
        # the sample-logprobs output as a second column keyed by
        # mask_token_id; requests must ask for logprobs>=1 to receive it.
        return_entropy = (
            diffusion_config.return_entropy if diffusion_config else None
        )
        if return_entropy is None:
            return_entropy = gen_config.get("diffusion_return_entropy")
        self.return_entropy = bool(return_entropy)
        # Opt-in per-token reveal-step channel (trajectory replay / "Trace"):
        # surfaced as its own CompletionOutput.reveal_steps field rather than
        # through the logprobs output, so it is independent of ``logprobs``.
        return_reveal_steps = (
            diffusion_config.return_reveal_steps if diffusion_config else None
        )
        if return_reveal_steps is None:
            return_reveal_steps = gen_config.get("diffusion_return_reveal_steps")
        self.return_reveal_steps = bool(return_reveal_steps)
        # emit_full_blocks is consumed by the scheduler (it governs stop-token
        # trimming, not sampling); read here only so the value is validated and
        # visible in the engine log alongside the other diffusion settings.
        emit_full_blocks = (
            diffusion_config.emit_full_blocks if diffusion_config else None
        )
        if emit_full_blocks is None:
            emit_full_blocks = gen_config.get("diffusion_emit_full_blocks")
        self.emit_full_blocks = bool(emit_full_blocks)
        self.diffusion_states = NemotronLabsDiffusionRequestStates(
            max_num_reqs=self.max_num_reqs,
            canvas_length=canvas_length,
            mask_token_id=getattr(hf_config, "mask_token_id", 100),
            max_denoising_steps=max_denoising_steps,
            device=device,
        )
        self._req_id_to_index: dict[str, int] = {}

        # Persistent buffer for per-request causal flags, updated in-place
        # so FULL CUDA graph replay sees the latest values.
        self._causal_buf = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        return MaskedDiffusionSampler(
            sampler=sampler,
            vocab_size=self.model_config.get_vocab_size(),
            diffusion_states=self.diffusion_states,
            temperature=self.sampling_temperature,
            selection_policy=self.selection_policy,
            confidence_threshold=self.confidence_threshold,
            return_entropy=self.return_entropy,
            return_reveal_steps=self.return_reveal_steps,
        ), None

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        return None

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        # Text-only model, no self-conditioning: use the default
        # input_ids path.
        return {}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {}

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        # Per-request causal mode: encoder (prefill/commit) = causal,
        # denoise = bidirectional. Pass a GPU tensor so the attention
        # backend can handle mixed batches.
        actual_num_reqs = input_batch.num_reqs
        slots = input_batch.idx_mapping[:actual_num_reqs]
        # Invariant: the sampler flips is_encoder_phase to False only after a
        # request's FINAL prompt chunk, so a prompt spanning multiple chunks
        # stays causal for every chunk.
        self._causal_buf[:actual_num_reqs] = self.diffusion_states.is_encoder_phase[
            slots
        ]
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = False
        causal: bool | torch.Tensor = self._causal_buf[:num_reqs]

        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            causal=causal,
        )

    num_new_sampled_tokens_per_step: int = 0


# Penalty stub for the diffusion path: the runner reads
# penalties_state.output_bin_counts, and post_update treats None as
# "no penalty bookkeeping".
_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)


# Set when a MaskedDiffusionSampler is constructed, so out-of-band callers
# (the NeMo-RL worker extension polling TPF over collective_rpc) can reach the
# live sampler without walking the model-runner object graph. One diffusion
# sampler exists per engine process.
_ACTIVE_DIFFUSION_SAMPLER: "MaskedDiffusionSampler | None" = None


def get_diffusion_tpf_stats(reset: bool = False) -> dict[str, int]:
    """Return cumulative denoising forwards and committed tokens.

    ``TPF = committed_tokens / nfe``. Returns zeros when the process is not
    running a diffusion sampler (e.g. an ``ar_mode`` rollout engine).

    Args:
        reset: Zero the counters after reading, so the next call reports a
            fresh window (one validation pass, say) rather than a running
            total.
    """
    sampler = _ACTIVE_DIFFUSION_SAMPLER
    if sampler is None:
        return {"nfe": 0, "committed_tokens": 0}
    return sampler.tpf_stats(reset=reset)


class MaskedDiffusionSampler:
    """Batched masked-diffusion sampler (LLaDA-style).

    Greedy when ``temperature == 0`` (the default), Gumbel-max sampling from
    ``softmax(logits / T)`` otherwise. Per-request SamplingParams temperature
    is rejected for diffusion models, so the temperature is engine-wide
    (``DiffusionConfig.temperature`` or the ``diffusion_temperature``
    generation-config key). Unmask order is ``selection_policy``:
    ``low_confidence`` (LLaDA, default), ``leftmost`` (strict left-to-right
    reveal for leftmost-reveal RL rollouts), ``random`` (uniform random reveal
    order, unbiased policy logprobs), or ``confidence_threshold``
    (SGLang FastDiffuser thresholding: every position whose chosen-token
    probability exceeds ``confidence_threshold``, at least one per step).

    With ``return_entropy`` (``DiffusionConfig.return_entropy`` or the
    ``diffusion_return_entropy`` generation-config key), commits emit the
    at-unmask entropy of each token's sampling distribution as a second
    logprob column keyed by ``mask_token_id``; this replaces top-k logprob
    introspection while enabled and requires requests to set ``logprobs>=1``.

    With ``return_reveal_steps`` (``DiffusionConfig.return_reveal_steps`` or
    the ``diffusion_return_reveal_steps`` generation-config key), each commit
    also emits the block-relative denoise step that revealed each token via
    ``SamplerOutput.reveal_steps``, which the engine surfaces as
    ``CompletionOutput.reveal_steps``. Unlike the entropy channel this rides
    its own field, so it needs no logprobs and composes with either.

    Logprobs: the sampled token's logprob/rank (the ``logprobs=0`` path, used
    by RL rollouts) are captured at the denoise step each position was
    unmasked, i.e. under the distribution the token was actually sampled
    from. Top-k introspection columns (``logprobs>0``) are computed from the
    final converged denoise step instead.

    Follows the same structure as ``DiffusionSampler`` in diffusion_gemma:
    decomposed into named methods, all GPU state in pre-allocated buffers,
    no GPU->CPU syncs on the hot path.
    """

    def __init__(
        self,
        sampler: Any,
        vocab_size: int,
        diffusion_states: NemotronLabsDiffusionRequestStates,
        temperature: float = 0.0,
        selection_policy: str = "low_confidence",
        confidence_threshold: float = 0.9,
        return_entropy: bool = False,
        return_reveal_steps: bool = False,
    ):
        self.sampling_states = sampler.sampling_states
        self.req_states = sampler.req_states
        self.vocab_size = vocab_size
        self.diffusion_states = diffusion_states
        self.canvas_length = diffusion_states.canvas_length
        self.mask_token_id = diffusion_states.mask_token_id
        self.temperature = temperature
        self.leftmost = selection_policy == "leftmost"
        self.random_mode = selection_policy == "random"
        self.threshold_mode = selection_policy == "confidence_threshold"
        # Compared against per-position chosen-token logprobs (natural log).
        self.log_threshold = math.log(confidence_threshold)
        # Emit at-unmask entropy as a second logprob column keyed by
        # mask_token_id (replaces top-k introspection while enabled).
        self.return_entropy = return_entropy
        # Emit the block-relative reveal step of each token on its own
        # SamplerOutput field (independent of the logprobs channel).
        self.return_reveal_steps = return_reveal_steps

        # TPF accounting. Kept on GPU and only synced when read, so the hot
        # path stays free of device->host transfers. `nfe` counts
        # (request, denoising-step) pairs rather than engine steps, matching
        # the per-prompt definition the offline Megatron-Bridge harness uses,
        # so batched and single-stream numbers are comparable. The per-block
        # causal KV-update forward is excluded, again matching the harness.
        self._tpf_nfe = torch.zeros(
            (), dtype=torch.int64, device=diffusion_states.device
        )
        self._tpf_tokens = torch.zeros(
            (), dtype=torch.int64, device=diffusion_states.device
        )

        global _ACTIVE_DIFFUSION_SAMPLER
        _ACTIVE_DIFFUSION_SAMPLER = self

        max_num_reqs = diffusion_states.max_num_reqs
        device = diffusion_states.device
        self._sampled = torch.zeros(
            max_num_reqs, self.canvas_length, dtype=torch.int32, device=device
        )
        self._num_sampled = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # Parallel to ``_sampled``; only written when return_reveal_steps.
        self._reveal_steps = torch.zeros(
            max_num_reqs, self.canvas_length, dtype=torch.int32, device=device
        )
        self._decode_slots = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._decode_idx = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._query_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self._num_logits = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

        # Per-slot stash for top-k (logprobs>0) tensors computed on the
        # converging denoise step; consumed on the subsequent commit step.
        self._pending_logprobs: dict[int, LogprobsTensors] = {}

        # ---- Thinking budget -------------------------------------------
        # Reuse the V2 sampler's ThinkingBudgetState for the parts that are not
        # decode-specific: it parses ReasoningConfig into token ids and stores
        # per-request SamplingParams.thinking_token_budget. Its own apply() is
        # unreachable here (custom_sampler replaces the V2 sampler entirely), so
        # enforcement below is ours.
        self._budget = getattr(sampler, "thinking_budget_state", None)
        self._budget_end_id: int | None = None
        if self._budget is not None and getattr(self._budget, "enabled", False):
            end_ids = self._budget.reasoning_end_token_ids
            if end_ids is not None and len(end_ids) == 1:
                self._budget_end_id = int(end_ids[0].item())
            else:
                # A multi-token delimiter cannot be seeded into a single canvas
                # position; refuse rather than inject a partial tag.
                logger.warning_once(
                    "Masked block-diffusion thinking budget requires a "
                    "single-token reasoning end delimiter; got %s tokens. "
                    "Thinking budget disabled for this model.",
                    "None" if end_ids is None else len(end_ids),
                )
        # Cumulative committed tokens per request. Generation starts inside an
        # open <think> block, so until the tag closes this is the thinking count.
        self._think_total_gen = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        # end delimiter committed (naturally OR forced) -> stop consulting budget
        self._think_end_seen = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        # Set ONLY by our injection, so a forced close is distinguishable from a
        # natural one. Deliberately not plumbed to the client yet: doing so means
        # threading a per-request field out through vLLM + Gym (the return_entropy
        # precedent was a six-layer chain), and generation is correctly bounded
        # without it. Kept because the distinction is what separates "reasoned
        # badly" from "ran out of budget" when reading eval failures.
        self._think_forced = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:
        if use_penalty(sampling_params):
            logger.warning_once(
                "Masked block-diffusion does not support repetition/frequency/"
                "presence penalties; ignoring them for this request."
            )
        # Purge any stale logprobs stashed under this slot by a prior request
        # that was aborted between its converging denoise and commit steps.
        self._pending_logprobs.pop(req_idx, None)
        self.sampling_states.add_request(req_idx, sampling_params)
        if self._budget is not None:
            # Populates thinking_token_budget/use_thinking_budget for this slot.
            self._budget.add_request(req_idx, sampling_params)
        self._think_total_gen[req_idx] = 0
        self._think_end_seen[req_idx] = False
        self._think_forced[req_idx] = False

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        if self._budget is not None:
            self._budget.apply_staged_writes()

    @property
    def penalties_state(self):
        return _NO_PENALTIES_STATE

    def tpf_stats(self, reset: bool = False) -> dict[str, int]:
        """Cumulative denoising forwards and committed tokens.

        This is the only device->host sync in the TPF path; the counters are
        accumulated on GPU during decoding.
        """
        stats = {
            "nfe": int(self._tpf_nfe.item()),
            "committed_tokens": int(self._tpf_tokens.item()),
        }
        if reset:
            self._tpf_nfe.zero_()
            self._tpf_tokens.zero_()
        return stats

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _finish_prefills(
        self, input_batch: Any, prefill_indices_np: np.ndarray
    ) -> None:
        """Transition requests whose prompt completes this step to denoising.

        Initializes their all-mask canvas, seeds draft tokens, and flips
        is_encoder_phase to False. Mid-chunk requests (prompt longer than the
        token budget) are left untouched so is_encoder_phase stays True and
        prepare_attn keeps causal attention for their remaining chunks.
        """
        states = self.diffusion_states
        done_prefill_np = (
            input_batch.num_computed_prefill_tokens_np[prefill_indices_np]
            + input_batch.num_scheduled_tokens[prefill_indices_np]
            >= input_batch.prefill_len_np[prefill_indices_np]
        )
        ps = input_batch.idx_mapping_np[prefill_indices_np[done_prefill_np]]
        if len(ps) == 0:
            return
        states.init_canvas(ps)
        self.req_states.draft_tokens[ps, : self.canvas_length] = states.canvas[ps]
        ps_gpu = async_copy_to_gpu(
            ps.astype(np.int64), device=states.is_encoder_phase.device
        )
        states.is_encoder_phase.index_fill_(0, ps_gpu, False)

    def _handle_prefill(self, input_batch: Any) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        self._finish_prefills(input_batch, np.arange(num_reqs))
        sampled = self._sampled[:num_reqs, :1]
        sampled.zero_()
        num_sampled = self._num_sampled[:num_reqs]
        num_sampled.zero_()
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_sampled,
        )

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def _build_output(
        self,
        input_batch: Any,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        per_req_nlogits_np: np.ndarray,
        logprobs_tensors: LogprobsTensors | None = None,
        reveal_steps: torch.Tensor | None = None,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs

        self._query_lens.np[:num_reqs] = np.diff(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        self._num_logits.np[:num_reqs] = per_req_nlogits_np
        self._query_lens.copy_to_uva()
        self._num_logits.copy_to_uva()

        num_rejected = _compute_num_rejected(
            self._num_logits.gpu[:num_reqs],
            num_sampled,
            input_batch.query_start_loc[: num_reqs + 1],
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            reveal_steps=reveal_steps,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        device = logits.device

        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch)

        # --- CPU/NumPy setup (outside compile): split decode vs prefill,
        # init canvas for any new prefills, stage decode slot indices. ---
        states = self.diffusion_states
        CL = self.canvas_length
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])

        decode_indices_np = np.where(per_req_nlogits_np > 0)[0]
        prefill_indices_np = np.where(per_req_nlogits_np == 0)[0]
        decode_slots_np = slots_np[decode_indices_np]

        if len(prefill_indices_np) > 0:
            self._finish_prefills(input_batch, prefill_indices_np)

        num_decode = len(decode_indices_np)
        self._decode_slots.np[:num_decode] = decode_slots_np
        self._decode_idx.np[:num_decode] = decode_indices_np
        self._decode_slots.copy_to_uva()
        self._decode_idx.copy_to_uva()
        decode_slots = self._decode_slots.gpu[:num_decode]
        decode_idx = self._decode_idx.gpu[:num_decode]

        # Real canvas length per decode request; == CL except for a canvas
        # truncated near max_model_len.
        valid_canvas_len_np = per_req_nlogits_np[per_req_nlogits_np > 0]
        valid_canvas_len = async_copy_to_gpu(
            valid_canvas_len_np.astype(np.int64), device=device
        )

        # Pad any truncated canvas back to CL so the uniform-CL sampler math
        # holds. Padded positions are zeroed (uniform logits) and are treated
        # as non-masked (never transferred or committed).
        if num_decode > 0 and valid_canvas_len_np.min() < CL:
            ar = torch.arange(CL, device=device)
            starts = valid_canvas_len.cumsum(0) - valid_canvas_len
            valid = ar.unsqueeze(0) < valid_canvas_len.unsqueeze(1)
            src = (starts.unsqueeze(1) + ar.unsqueeze(0)).clamp_max(logits.shape[0] - 1)
            logits = logits[src.reshape(-1)] * valid.reshape(-1, 1).to(logits.dtype)

        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]
        sampled.zero_()
        num_sampled.zero_()
        reveal_steps = self._reveal_steps[:num_reqs]
        if self.return_reveal_steps:
            reveal_steps.zero_()

        all_slots = input_batch.idx_mapping[:num_reqs]

        # Snapshot which slots are committing BEFORE the compiled step runs,
        # since it mutates is_encoder_phase (commit->False, converge->True).
        is_committing = states.is_encoder_phase[decode_slots].clone()

        max_num_logprobs = self.sampling_states.max_num_logprobs(slots_np)

        req_temps = self.sampling_states.temperature.gpu[decode_slots]

        logits_3d = _compiled_masked_step(
            logits[: num_decode * CL],
            decode_slots,
            decode_idx,
            all_slots,
            valid_canvas_len,
            req_temps,
            # State
            states.canvas,
            states.step,
            states.is_encoder_phase,
            states.unmask_logprobs,
            states.unmask_ranks,
            states.unmask_entropy,
            states.unmask_step,
            # Output
            sampled,
            num_sampled,
            self.req_states.draft_tokens,
            reveal_steps,
            # Config
            max_denoising_steps=int(states.max_denoising_steps),
            mask_token_id=int(states.mask_token_id),
            CL=CL,
            temperature=self.temperature,
            capture_logprobs=max_num_logprobs >= 0,
            capture_entropy=self.return_entropy and max_num_logprobs >= 0,
            capture_steps=self.return_reveal_steps,
            leftmost=self.leftmost,
            random_mode=self.random_mode,
            threshold_mode=self.threshold_mode,
            log_threshold=self.log_threshold,
        )

        # TPF accounting (tensor ops only -- no host sync). Rows that were
        # denoising this step each consumed one forward; rows that committed
        # emitted `num_sampled` tokens. Accumulated separately because a block
        # spans many denoise steps and exactly one commit.
        if num_decode > 0:
            self._tpf_nfe += (~is_committing).sum()
            self._tpf_tokens += num_sampled[decode_idx].sum()

        # ---- Thinking budget enforcement -------------------------------
        # Runs AFTER the compiled step, which has already reset every committing
        # row's canvas to all-mask for the next block -- so writing position 0
        # seeds the NEXT block. Safe because every reveal path masks its writes
        # with `mask_index`: a non-mask position is absorbing and is never
        # overwritten by denoising, so the injected token survives as context.
        #
        # Block-granular: checked once per commit. Overshoot is therefore
        # bounded by (canvas_length - 1) and is ZERO when the budget is a
        # multiple of canvas_length. The AR implementation is likewise
        # step-granular.
        #
        # budget=0 is NOT the way to disable reasoning here. The first block
        # commits before this check can run -- there is no commit at which
        # total_gen is still 0 -- so a request gets exactly canvas_length (~32)
        # thinking tokens (measured 32/32 over 16 AIME problems), where the AR
        # path force-indexes token 0 and yields a true zero. This is deliberate,
        # not an oversight: to disable reasoning, use enable_thinking=false in
        # the chat template, which prefills a closed <think></think> so the
        # block is never opened and the engine is not involved at all.
        if self._budget_end_id is not None and num_decode > 0:
            commit_rows = is_committing
            if commit_rows.any():
                commit_slots = decode_slots[commit_rows]
                emitted = num_sampled[decode_idx][commit_rows].to(
                    self._think_total_gen.dtype
                )
                self._think_total_gen[commit_slots] += emitted
                # Scan the whole committed row: under out-of-order reveal the
                # delimiter need not be last. Positions past the emitted length
                # hold stale values, so mask the comparison by column index.
                rows = sampled[decode_idx][commit_rows]
                cols = torch.arange(rows.shape[1], device=rows.device)
                valid = cols.unsqueeze(0) < emitted.unsqueeze(1)
                saw_end = ((rows == self._budget_end_id) & valid).any(dim=1)
                self._think_end_seen[commit_slots] |= saw_end
                # budget < 0 means unset for that request (upstream sentinel).
                budgets = self._budget.thinking_token_budget.gpu[commit_slots]
                # >= (not >): a budget that is a multiple of canvas_length then
                # closes exactly on it -- e.g. 512 with canvas 32 stops at 512, not
                # 544. For a budget that is not a multiple, both forms close on the
                # same commit, so >= is never worse.
                needs_close = (
                    (budgets >= 0)
                    & (self._think_total_gen[commit_slots] >= budgets)
                    & ~self._think_end_seen[commit_slots]
                )
                if needs_close.any():
                    close_slots = commit_slots[needs_close]
                    self.diffusion_states.canvas[close_slots, 0] = (
                        self._budget_end_id
                    )
                    self._think_end_seen[close_slots] = True
                    self._think_forced[close_slots] = True

        # Top-k introspection (logprobs>0): stash the converged denoise
        # step's top-k (is_encoder_phase flipped False->True); attached on
        # the commit. The logprobs==0 path reads the at-unmask buffers
        # instead and needs no stash. return_entropy replaces top-k
        # introspection entirely (the extra column carries entropy).
        if max_num_logprobs > 0 and not self.return_entropy and num_decode > 0:
            converged_mask = states.is_encoder_phase[decode_slots]
            just_converged = converged_mask & ~is_committing
            if just_converged.any():
                flat_logits = logits_3d.reshape(-1, logits_3d.shape[-1])
                final_canvas = states.canvas[decode_slots]
                for local_idx in just_converged.nonzero(as_tuple=True)[0]:
                    li = int(local_idx.item())
                    slot = int(decode_slots[local_idx].item())
                    k_i = int(valid_canvas_len_np[li])
                    row = li * CL
                    self._pending_logprobs[slot] = compute_topk_logprobs(
                        flat_logits[row : row + k_i],
                        max_num_logprobs,
                        final_canvas[local_idx][:k_i],
                    )

        # Commit steps: emit logprobs for the committed block. The sampled
        # token's logprob/rank come from the at-unmask buffers when only
        # they are needed (logprobs==0); top-k rows come from the stash.
        logprobs_tensors = None
        emit = max_num_logprobs >= 0 and num_decode > 0
        committing_np = is_committing.cpu().numpy() if emit else None
        if emit and committing_np.any():
            decode_row_of_req = {
                int(decode_indices_np[li]): li for li in range(num_decode)
            }
            parts_ids, parts_lp, parts_ranks = [], [], []
            cu_gen: list[int] = []
            flat_offset = 0
            for i in range(num_reqs):
                cu_gen.append(flat_offset)
                li = decode_row_of_req.get(i)
                if li is None or not committing_np[li]:
                    continue
                slot = int(slots_np[i])
                k_i = int(valid_canvas_len_np[li])
                if self.return_entropy:
                    # Entropy mode (JustGRPO-Fast): two columns per token —
                    # col 0 is the sampled token's at-unmask logprob, col 1
                    # carries the at-unmask entropy keyed by mask_token_id
                    # (never a committed token, so the key cannot collide).
                    # Requests need logprobs>=1 for the extra column to
                    # survive per-request truncation in the logprobs
                    # processor downstream.
                    ids = sampled[i, :k_i].to(torch.int64)
                    parts_ids.append(
                        torch.stack(
                            [ids, torch.full_like(ids, self.mask_token_id)],
                            dim=-1,
                        )
                    )
                    parts_lp.append(
                        torch.stack(
                            [
                                states.unmask_logprobs[slot, :k_i],
                                states.unmask_entropy[slot, :k_i],
                            ],
                            dim=-1,
                        )
                    )
                    parts_ranks.append(states.unmask_ranks[slot, :k_i])
                    flat_offset += k_i
                elif max_num_logprobs > 0:
                    lp = self._pending_logprobs.pop(slot, None)
                    if lp is None:
                        continue
                    parts_ids.append(lp.logprob_token_ids)
                    parts_lp.append(lp.logprobs)
                    parts_ranks.append(lp.selected_token_ranks)
                    flat_offset += lp.logprobs.shape[0]
                else:
                    parts_ids.append(
                        sampled[i, :k_i].to(torch.int64).unsqueeze(-1)
                    )
                    parts_lp.append(
                        states.unmask_logprobs[slot, :k_i].unsqueeze(-1)
                    )
                    parts_ranks.append(states.unmask_ranks[slot, :k_i])
                    flat_offset += k_i
            if parts_ids:
                logprobs_tensors = LogprobsTensors(
                    logprob_token_ids=torch.cat(parts_ids),
                    logprobs=torch.cat(parts_lp),
                    selected_token_ranks=torch.cat(parts_ranks),
                    cu_num_generated_tokens=cu_gen,
                )

        return self._build_output(
            input_batch,
            sampled,
            num_sampled,
            per_req_nlogits_np,
            logprobs_tensors=logprobs_tensors,
            reveal_steps=reveal_steps if self.return_reveal_steps else None,
        )

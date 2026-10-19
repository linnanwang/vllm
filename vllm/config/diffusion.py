# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for discrete diffusion (dLLM) models."""

from typing import Literal

from pydantic import Field

from vllm.config.utils import config


@config
class DiffusionConfig:
    """Configuration for discrete diffusion language models (dLLMs).

    dLLMs generate tokens via iterative denoising over a fixed-length canvas
    rather than left-to-right autoregressive decoding. They reuse the
    speculative-decoding data path (draft token ids, scheduled spec decode
    tokens) with overloaded semantics for block-based generation.
    """

    canvas_length: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """Length of the denoising canvas (block).  Also determines the number of
    speculative tokens scheduled per step."""

    max_denoising_steps: int | None = None
    """Maximum number of denoising iterations per canvas block.
    If not set, read from the model's generation_config.json."""

    temperature: float | None = Field(default=None, ge=0)
    """Sampling temperature for the denoising sampler (engine-wide, since
    per-request sampling parameters are not supported for diffusion models).
    0 means greedy. If not set, read from the model's generation_config.json
    (model-specific key, e.g. ``diffusion_temperature``), defaulting to
    greedy for masked-diffusion models."""

    selection_policy: (
        Literal[
            "low_confidence",
            "leftmost",
            "confidence_threshold",
            "confidence_threshold_leftmost",
            "random",
        ]
        | None
    ) = None
    """Which masked positions to unmask each denoising step (masked-diffusion
    models). ``low_confidence``: the most confident predictions first (LLaDA),
    top-k per step with k from the even transfer schedule
    (``ceil(masks / remaining_steps)``), so tokens-per-forward is fixed at
    ``canvas_length / max_denoising_steps``.
    ``leftmost``: strictly left-to-right — with one token per step this yields
    clean per-token policy logprobs matching a leftmost-reveal RL recompute.
    ``random``: uniform random reveal order among the still-masked positions,
    with k from the even transfer schedule. The choice is independent of the
    model's scores, so each committed token's recorded logprob is an unbiased
    policy logprob (unlike confidence selection) -- the SGLang FastDiffuser
    ``random`` decode, used by the trajectory-replay baselines.
    ``confidence_threshold_leftmost``: unmask all masked candidates at or above
    the threshold; if none pass, reveal only the leftmost remaining mask.
    Requires at least ``canvas_length`` denoising steps.
    ``confidence_threshold``: unmask every position whose chosen-token
    probability exceeds ``confidence_threshold`` (always at least the single
    most confident one, so each step makes progress). Tokens-per-forward is
    then adaptive rather than scheduled, matching the SGLang FastDiffuser and
    Megatron-Bridge thresholding decode.
    If not set, read from the model's generation_config.json
    (``diffusion_selection_policy``), defaulting to ``low_confidence``."""

    confidence_threshold: float | None = Field(default=None, gt=0, lt=1)
    """Probability bar a masked position must clear to be unmasked under
    ``confidence_threshold`` or ``confidence_threshold_leftmost`` selection.
    If not set, read from the
    model's generation_config.json (``diffusion_confidence_threshold``),
    defaulting to 0.9. Ignored by the other selection policies."""

    return_entropy: bool | None = None
    """Emit per-token entropy ``H = -sum_v p_v log p_v`` of the sampling
    distribution, captured at the denoise step each token was unmasked
    (masked-diffusion models) — the same distribution the token's logprob is
    scored under. The entropy rides the sample-logprobs channel as an extra
    logprob column keyed by the mask token id (which is never a committed
    token), so requests must set ``logprobs>=1`` to receive it; while enabled
    it replaces top-k logprob introspection. Consumed by the
    entropy-sparsified JustGRPO-Fast trainer. If not set, read from the
    model's generation_config.json (``diffusion_return_entropy``),
    defaulting to off."""

    return_reveal_steps: bool | None = None
    """Emit the block-relative denoising step that revealed each generated
    token (masked-diffusion models), so a trainer can replay the decoding
    trajectory and score every token on exactly the context inference
    conditioned on. Surfaced as ``CompletionOutput.reveal_steps``, a plain
    ``list[int]`` parallel to ``token_ids`` -- the structural twin of the
    SGLang fork's ``meta_info["output_token_reveal_steps"]``. Steps are
    0-based and reset at each block commit. Independent of ``logprobs``:
    the channel does not ride the logprobs output, so requests need not ask
    for logprobs to receive it. Consumed by the inference-trajectory-replay
    ("Trace" / Amortized GRPO) trainer. If not set, read from the model's
    generation_config.json (``diffusion_return_reveal_steps``), defaulting
    to off."""

    emit_full_blocks: bool | None = None
    """Keep the whole final block when a request stops on a stop token,
    instead of trimming the output at that token (masked-diffusion models).

    One engine step commits an entire block at once, and under out-of-order
    reveal (``random`` / confidence selection) the stop token can be committed
    before positions that were revealed EARLIER in the trajectory. The default
    trim would silently drop those tokens, leaving a trajectory consumer with
    a reveal-step sequence that is missing its own history. Enabling this
    extends the token ids, the logprobs and the reveal steps consistently,
    since all three are sliced to the same length downstream.

    Only stop-token stops are extended; a length-capped stop is never pushed
    past ``max_model_len`` / ``max_tokens``. This is the vLLM port of the
    SGLang FastDiffuser ``emit_full_blocks`` option. If not set, read from the
    model's generation_config.json (``diffusion_emit_full_blocks``),
    defaulting to off."""

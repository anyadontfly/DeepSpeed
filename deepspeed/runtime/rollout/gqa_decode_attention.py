# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""SDPA decode attention for grouped-query models during rollout.

Transformers' ``sdpa`` attention copies every key/value head once per query
head (``repeat_kv``) whenever an attention mask is passed, because masked
grouped-query SDPA used to fall back to the math kernel on CUDA. Rollout decode
always passes a mask over each row's whole KV cache, so every decode step
copies the full cache to ``num_key_value_groups`` times its size before
attention reads it again.

Recent PyTorch runs ``enable_gqa=True`` with a mask on the memory-efficient and
cuDNN kernels. During a rollout, :func:`gqa_decode_attention` switches models
that use ``sdpa`` to an attention implementation that passes
``enable_gqa=True`` for one-token queries and calls Transformers' ``sdpa``
otherwise, then restores the model's implementation so training is unchanged.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

GQA_DECODE_ATTENTION = "deepspeed_sdpa_gqa_decode"

_registered = None
_kernel_support = {}


def sdpa_gqa_decode_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Transformers attention function: SDPA with ``enable_gqa`` for one-token queries, else Transformers' SDPA."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if (query.shape[2] != 1 or getattr(module, "num_key_value_groups", 1) <= 1
            or kwargs.get("position_bias") is not None or kwargs.get("cache") is not None
            or kwargs.get("output_attentions", False) or key.shape[-1] != value.shape[-1]):
        return ALL_ATTENTION_FUNCTIONS["sdpa"](module,
                                               query,
                                               key,
                                               value,
                                               attention_mask,
                                               dropout=dropout,
                                               scaling=scaling,
                                               **kwargs)

    if attention_mask is not None and attention_mask.shape[-1] > key.shape[-2]:
        attention_mask = attention_mask[..., :key.shape[-2]]
    # A single query token attends to every unmasked key, so the attention is never causal here.
    attn_output = F.scaled_dot_product_attention(query,
                                                 key,
                                                 value,
                                                 attn_mask=attention_mask,
                                                 dropout_p=dropout,
                                                 scale=scaling,
                                                 is_causal=False,
                                                 enable_gqa=True)
    return attn_output.transpose(1, 2).contiguous(), None


def _register():
    """Register the attention and mask functions with Transformers; False if this version can't."""
    global _registered
    if _registered is None:
        try:
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
            from transformers.modeling_utils import AttentionInterface
        except ImportError:
            _registered = False
        else:
            AttentionInterface.register(GQA_DECODE_ATTENTION, sdpa_gqa_decode_attention)
            # The attention takes the same masks as sdpa.
            AttentionMaskInterface.register(GQA_DECODE_ATTENTION, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
            _registered = True
    return _registered


def _supports_masked_gqa(device, dtype, head_dim, groups):
    """Whether SDPA runs masked grouped-query attention on a kernel other than math on this device.

    Older PyTorch versions only run ``enable_gqa`` with a mask on the math kernel, which is slower than copying
    the heads, so the rollout keeps Transformers' path there.
    """
    key = (device.type, device.index, dtype, head_dim, groups)
    if key not in _kernel_support:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        query = torch.zeros(1, groups, 1, head_dim, device=device, dtype=dtype)
        kv = torch.zeros(1, 1, 8, head_dim, device=device, dtype=dtype)
        mask = torch.ones(1, 1, 1, 8, device=device, dtype=torch.bool)
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
                F.scaled_dot_product_attention(query, kv, kv, attn_mask=mask, enable_gqa=True)
            _kernel_support[key] = True
        except (RuntimeError, TypeError):
            _kernel_support[key] = False
    return _kernel_support[key]


def _grouped_attention(model):
    """Return ``(head_dim, groups, parameter)`` of the first grouped-query attention module, or None."""
    for submodule in model.modules():
        groups = getattr(submodule, "num_key_value_groups", 1)
        head_dim = getattr(submodule, "head_dim", None)
        if isinstance(groups, int) and groups > 1 and isinstance(head_dim, int):
            parameter = next(submodule.parameters(), None)
            if parameter is not None:
                return head_dim, groups, parameter
    return None


def _sdpa_configs(model):
    """Distinct configs of ``model`` and its submodules whose attention implementation is ``sdpa``."""
    configs = {}
    for submodule in model.modules():
        config = getattr(submodule, "config", None)
        # Read and later set the stored value rather than the ``_attn_implementation`` property, whose setter
        # also overwrites sub-configs that the model may have configured differently.
        if getattr(config, "__dict__", {}).get("_attn_implementation_internal") == "sdpa":
            configs[id(config)] = config
    return list(configs.values())


@contextmanager
def gqa_decode_attention(model, device):
    """Run grouped-query ``sdpa`` models with :func:`sdpa_gqa_decode_attention` inside the block.

    Models on other attention implementations, without grouped-query heads, off CUDA, or on PyTorch or
    Transformers versions without the needed support are left unchanged.
    """
    configs = []
    if device.type == "cuda":
        configs = _sdpa_configs(model)
        grouped = _grouped_attention(model) if configs else None
        if grouped is None or not _register():
            configs = []
        else:
            head_dim, groups, parameter = grouped
            if not _supports_masked_gqa(device, parameter.dtype, head_dim, groups):
                configs = []
    for config in configs:
        config._attn_implementation_internal = GQA_DECODE_ATTENTION
    try:
        yield
    finally:
        for config in configs:
            config._attn_implementation_internal = "sdpa"

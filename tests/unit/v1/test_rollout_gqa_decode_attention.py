# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""GQA-aware SDPA decode attention in HybridEngineRollout."""

from types import SimpleNamespace

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.gqa_decode_attention import GQA_DECODE_ATTENTION, gqa_decode_attention
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig


def _cuda_device():
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("the GQA decode attention is used only on CUDA")
    return torch.device(accelerator.device_name())


def _tiny_llama(device, num_key_value_heads=2, attn_implementation="sdpa", dtype=torch.float32):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64,
                         hidden_size=64,
                         intermediate_size=128,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=num_key_value_heads,
                         max_position_embeddings=128)
    config._attn_implementation = attn_implementation
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        model.lm_head.weight.mul_(30)
    return model.to(device=device, dtype=dtype).eval()


def _padded_request(device, batch_size=4, prompt_len=8):
    prompt_ids = torch.randint(1, 64, (batch_size, prompt_len), generator=torch.Generator().manual_seed(1))
    attention_mask = torch.ones_like(prompt_ids)
    for row in range(batch_size):
        attention_mask[row, :row] = 0
    prompt_ids = prompt_ids.masked_fill(attention_mask == 0, 0)
    return RolloutRequest(prompt_ids.to(device), attention_mask.to(device))


@pytest.fixture
def sdpa_calls(monkeypatch):
    """Record ``(batch, query length, key heads, enable_gqa)`` of every SDPA call."""
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def recorded(query, key, value, *args, **kwargs):
        calls.append((query.shape[0], query.shape[-2], key.shape[-3], kwargs.get("enable_gqa", False)))
        return original(query, key, value, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recorded)
    return calls


def _hf_generate(model, request, new_tokens):
    with torch.no_grad():
        return model.generate(request.prompt_ids,
                              attention_mask=request.prompt_attention_mask,
                              max_new_tokens=new_tokens,
                              do_sample=False,
                              eos_token_id=None,
                              pad_token_id=0)


@pytest.mark.parametrize("use_graph_capture, continuous_batch_size", [(False, None), (True, None), (True, 4),
                                                                      (False, 4)])
def test_rollout_decodes_with_gqa_and_matches_hf_sdpa(use_graph_capture, continuous_batch_size, sdpa_calls):
    device = _cuda_device()
    model = _tiny_llama(device)
    request = _padded_request(device)
    new_tokens = 12
    expected = _hf_generate(model, request, new_tokens)
    # Transformers' SDPA copies the KV heads under a mask.
    assert sdpa_calls and all(heads == 4 and not gqa for _, _, heads, gqa in sdpa_calls)
    sdpa_calls.clear()

    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=use_graph_capture))
    output = rollout.generate(
        request, SamplingConfig(max_new_tokens=new_tokens, temperature=0, continuous_batch_size=continuous_batch_size))

    assert output.input_ids.tolist() == expected.tolist()
    # Leave out the rollout's one-time kernel probe, which runs on a batch of one.
    model_calls = [call for call in sdpa_calls if call[0] > 1]
    decode_calls = [call for call in model_calls if call[1] == 1]
    prefill_calls = [call for call in model_calls if call[1] > 1]
    assert decode_calls and all(heads == 2 and gqa for _, _, heads, gqa in decode_calls)
    # Prefill keeps Transformers' SDPA path.
    assert prefill_calls and all(heads == 4 and not gqa for _, _, heads, gqa in prefill_calls)
    assert model.config._attn_implementation == "sdpa"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("four_d_mask", [True, False])
def test_decode_logits_match_hf_sdpa(dtype, four_d_mask):
    device = _cuda_device()
    from transformers import DynamicCache

    model = _tiny_llama(device, dtype=dtype)
    request = _padded_request(device)
    prompt_len = request.prompt_ids.shape[1]
    position_ids = (request.prompt_attention_mask.cumsum(-1) - 1).clamp(min=0)
    step = torch.randint(1, 64, (request.prompt_ids.shape[0], 1), device=device)
    attention = torch.cat((request.prompt_attention_mask, torch.ones_like(step)), dim=1)
    if four_d_mask:
        mask = torch.zeros(attention.shape[0], 1, 1, attention.shape[1], device=device, dtype=dtype)
        mask.masked_fill_(attention[:, None, None, :] == 0, torch.finfo(dtype).min)
    else:
        mask = attention

    @torch.no_grad()
    def decode_logits():
        cache = DynamicCache(config=model.config)
        model(request.prompt_ids,
              attention_mask=request.prompt_attention_mask,
              position_ids=position_ids,
              past_key_values=cache,
              use_cache=True)
        return model(step,
                     attention_mask=mask,
                     position_ids=position_ids[:, -1:] + 1,
                     past_key_values=cache,
                     use_cache=True,
                     cache_position=torch.tensor([prompt_len], device=device)).logits

    expected = decode_logits()
    with gqa_decode_attention(model, device):
        assert model.config._attn_implementation == GQA_DECODE_ATTENTION
        actual = decode_logits()
    torch.testing.assert_close(actual, expected)


def test_training_forward_unchanged_after_rollout():
    device = _cuda_device()
    model = _tiny_llama(device)
    request = _padded_request(device)
    with torch.no_grad():
        before = model(request.prompt_ids, attention_mask=request.prompt_attention_mask).logits
    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    rollout.generate(request, SamplingConfig(max_new_tokens=4, temperature=0, continuous_batch_size=4))
    assert model.config._attn_implementation == "sdpa"
    with torch.no_grad():
        after = model(request.prompt_ids, attention_mask=request.prompt_attention_mask).logits
    assert torch.equal(before, after)


@pytest.mark.parametrize("num_key_value_heads, attn_implementation", [(4, "sdpa"), (2, "eager")])
def test_models_without_grouped_sdpa_are_unchanged(num_key_value_heads, attn_implementation):
    device = _cuda_device()
    model = _tiny_llama(device, num_key_value_heads=num_key_value_heads, attn_implementation=attn_implementation)
    with gqa_decode_attention(model, device):
        assert model.config._attn_implementation == attn_implementation


def test_attention_restored_when_generation_fails():
    device = _cuda_device()
    model = _tiny_llama(device)
    with pytest.raises(RuntimeError, match="boom"):
        with gqa_decode_attention(model, device):
            assert model.config._attn_implementation == GQA_DECODE_ATTENTION
            raise RuntimeError("boom")
    assert model.config._attn_implementation == "sdpa"


def test_cpu_models_are_unchanged():
    model = _tiny_llama(torch.device("cpu"))
    with gqa_decode_attention(model, torch.device("cpu")):
        assert model.config._attn_implementation == "sdpa"

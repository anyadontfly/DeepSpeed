# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for HybridEngineRollout.

Most tests are CPU-only; the native shared-prefill cache test runs only when CUDA and
the transformer inference extension are available.
"""

import gc
import weakref
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PretrainedConfig

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.transformer.inference.op_binding.workspace import WorkspaceOp
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine
from deepspeed.runtime.rollout.base import RolloutRequest, Sample, SamplingConfig
from deepspeed.runtime.rollout.tool import Tool, ToolSet
from deepspeed.runtime.rollout.hybrid_engine_rollout import (
    HybridEngineRollout,
    HybridEngineRolloutConfig,
)
from deepspeed.utils.static_cache import DeepSpeedStaticCache


def _make_engine():
    engine = MagicMock()
    engine.module = MagicMock()
    engine.module.parameters.return_value = iter([])
    # Not ZeRO-3, so generation may stop at EOS.
    engine.zero_optimization_partition_weights.return_value = False
    return engine


def _make_engine_on_cpu():
    """An engine whose module has one CPU parameter, so rollout can resolve the model device."""
    engine = _make_engine()
    engine.module.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    return engine


def _make_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    tok.decode.side_effect = lambda ids, **_kwargs: " ".join(map(str, ids))
    return tok


def _prompt(tokens, attention_mask=None):
    """A pre-tokenized prompt sample; samples with tokens skip chat-template rendering."""
    if attention_mask is None:
        attention_mask = [1] * len(tokens)
    return Sample(tokens=list(tokens), attention_mask=list(attention_mask))


def _make_request():
    return RolloutRequest(samples=[_prompt([0, 1, 2], [0, 1, 1]), _prompt([0, 3, 4], [0, 1, 1])])


def _tensor_request(prompt_ids, attention_mask):
    """A request of pre-tokenized prompts given as left-padded ``[B, T]`` tensors."""
    return RolloutRequest(
        samples=[_prompt(ids.tolist(), mask.tolist()) for ids, mask in zip(prompt_ids, attention_mask)])


def _mock_cpu_accelerator(mock_get_accelerator):
    mock_get_accelerator.return_value.current_device_name.return_value = "cpu"


def _make_sampling(n_samples_per_prompt=1):
    return SamplingConfig(max_new_tokens=2, temperature=0, n_samples_per_prompt=n_samples_per_prompt)


# -- config defaults ----------------------------------------------------


def test_config_defaults():
    cfg = HybridEngineRolloutConfig()
    assert cfg.use_graph_capture is False
    assert cfg.enable_profiling is False
    assert cfg.use_shared_prefill is False
    assert cfg.align_decode_fronts is False
    assert cfg.enable_cache_trimming is False
    assert cfg.continuous_cache_capacity is None


# -- constructor --------------------------------------------------------


def test_constructor_stores_config():
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(use_graph_capture=True,
                                    enable_profiling=True,
                                    align_decode_fronts=True,
                                    enable_cache_trimming=True,
                                    continuous_cache_capacity=16)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    assert rollout.use_graph_capture is True
    assert rollout.enable_profiling is True
    assert rollout.use_shared_prefill is False
    assert rollout.align_decode_fronts is True
    assert rollout.enable_cache_trimming is True
    assert rollout.continuous_cache_capacity == 16
    assert rollout.engine is engine
    assert rollout.tokenizer is tok


def test_constructor_defaults_without_cfg():
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer())
    assert rollout.use_graph_capture is False
    assert rollout.enable_profiling is False
    assert rollout.use_shared_prefill is False
    assert rollout.align_decode_fronts is False
    assert rollout.enable_cache_trimming is False
    assert rollout.continuous_cache_capacity is None


def test_continuous_generation_rejects_unsupported_inputs():
    engine = _make_engine()
    engine.module = torch.nn.Linear(1, 1)
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    request = RolloutRequest(samples=[_prompt([0, 1, 2], [0, 1, 1])])

    empty = rollout.generate(RolloutRequest(samples=[]), SamplingConfig(max_new_tokens=2, continuous_batch_size=1))
    assert empty.samples == []
    with pytest.raises(ValueError, match="positive"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, continuous_batch_size=0))
    with pytest.raises(ValueError, match="one sample per prompt"):
        rollout.generate(
            request, SamplingConfig(max_new_tokens=2, temperature=0, n_samples_per_prompt=2, continuous_batch_size=1))
    with pytest.raises(ValueError, match="at least one prompt token"):
        rollout.generate(RolloutRequest(samples=[_prompt([0, 0], [0, 0])]),
                         SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (HybridEngineRolloutConfig(use_graph_capture=True, align_decode_fronts=True), "aligned decode fronts"),
        (HybridEngineRolloutConfig(use_graph_capture=True, enable_cache_trimming=True), "cache trimming"),
        (HybridEngineRolloutConfig(use_graph_capture=True, continuous_cache_capacity=4), "must fit"),
    ],
)
def test_continuous_graph_rejects_incompatible_fixed_layout_options(cfg, message):
    rollout = HybridEngineRollout(_make_engine_on_cpu(), _make_tokenizer(), cfg=cfg)
    request = _tensor_request(torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 1]]))

    with pytest.raises(ValueError, match=message):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


def test_continuous_graph_rejects_flash_attention():
    engine = _make_engine_on_cpu()
    engine.module.config = SimpleNamespace(_attn_implementation="flash_attention_2")
    rollout = HybridEngineRollout(engine, _make_tokenizer(), cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    request = _tensor_request(torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 1]]))

    with pytest.raises(ValueError, match="does not support flash_attention_2"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


def test_continuous_generation_routes_graph_capture_to_fixed_capacity_path():
    rollout = HybridEngineRollout(
        _make_engine_on_cpu(),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(use_graph_capture=True),
    )
    request = _make_request()
    sampling = SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=4)
    expected = MagicMock()

    with patch.object(rollout, "_generate_continuous_graph", return_value=expected) as generate_graph:
        assert rollout.generate(request, sampling) is expected

    generate_graph.assert_called_once()
    trajectories, requests, captured_sampling, max_batch_size = generate_graph.call_args.args
    assert [sample.tokens for sample in trajectories] == [sample.tokens for sample in request.samples]
    assert len(requests) == len(request.samples)
    assert captured_sampling is sampling
    assert max_batch_size == 4


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_continuous_graph_capture_uses_fixed_capacity_buffers(mock_get_accelerator):

    class DecodeModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, input_ids, **_kwargs):
            self.calls.append(input_ids)
            return SimpleNamespace(logits=torch.zeros((input_ids.shape[0], 1, 8)))

    stream = MagicMock()
    accelerator = mock_get_accelerator.return_value
    accelerator.Stream.return_value = stream
    accelerator.current_stream.return_value = stream
    accelerator.stream.side_effect = lambda _stream: nullcontext()
    accelerator.capture_to_graph.side_effect = lambda _graph: nullcontext()

    module = DecodeModule()
    static_input = torch.zeros((4, 1), dtype=torch.long)
    static_attention = torch.zeros((4, 6), dtype=torch.long)
    static_write_positions = torch.zeros(4, dtype=torch.long)
    static_cache_position = torch.tensor([5], dtype=torch.long)
    static_position_ids = torch.zeros((4, 1), dtype=torch.long)

    graph, logits = HybridEngineRollout._capture_continuous_graph(
        module,
        MagicMock(),
        static_input,
        static_attention,
        static_write_positions,
        static_cache_position,
        static_position_ids,
    )

    assert graph is accelerator.create_graph.return_value
    assert logits.shape == (4, 1, 8)
    assert len(module.calls) == 4
    assert all(call is static_input for call in module.calls)
    accelerator.capture_to_graph.assert_called_once_with(graph)


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_continuous_graph_capture_preserves_attention_implementation(mock_get_accelerator):

    class DecodeModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.attention_implementations = []

        def forward(self, input_ids, **_kwargs):
            self.attention_implementations.append(self.config._attn_implementation)
            return SimpleNamespace(logits=torch.zeros((input_ids.shape[0], 1, 8)))

    stream = MagicMock()
    accelerator = mock_get_accelerator.return_value
    accelerator.Stream.return_value = stream
    accelerator.current_stream.return_value = stream
    accelerator.stream.side_effect = lambda _stream: nullcontext()
    accelerator.capture_to_graph.side_effect = lambda _graph: nullcontext()

    module = DecodeModule()
    HybridEngineRollout._capture_continuous_graph(
        module,
        MagicMock(),
        torch.zeros((4, 1), dtype=torch.long),
        torch.zeros((4, 6), dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
        torch.zeros((4, 1), dtype=torch.long),
    )

    assert module.attention_implementations == ["sdpa"] * 4
    assert module.config._attn_implementation == "sdpa"


def test_continuous_graph_keeps_cache_position_alive_for_replay():
    accelerator = get_accelerator()
    if accelerator.device_name() != "cuda" or not accelerator.is_available():
        pytest.skip("requires CUDA graph capture")

    class DecodeModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=1, num_attention_heads=1, hidden_size=1)
            self.cache_position_ref = None

        def forward(self, input_ids, cache_position, **_kwargs):
            self.cache_position_ref = weakref.ref(cache_position)
            return SimpleNamespace(logits=(input_ids + cache_position).unsqueeze(-1))

    module = DecodeModule()
    rollout = HybridEngineRollout(SimpleNamespace(module=module), _make_tokenizer())
    state = rollout._get_continuous_graph_state(module, 1, 3, 5, torch.device(accelerator.current_device_name()),
                                                torch.float32, DeepSpeedStaticCache)
    gc.collect()

    # A captured kernel retains only the device pointer, not the Python tensor.
    assert module.cache_position_ref() is not None
    state.static_input.fill_(7)
    accelerator.replay_graph(state.graph)
    assert torch.equal(state.static_logits, torch.full_like(state.static_logits, 11))


def test_continuous_graph_state_reuses_matching_capture_and_rebuilds_for_weight_updates():

    class GraphCache:

        def __init__(self, *_args, **_kwargs):
            self.reset_calls = 0

        def set_write_position(self, _write_position):
            return None

        def reset(self):
            self.reset_calls += 1

    class DecodeModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace()

    module = DecodeModule()
    rollout = HybridEngineRollout(SimpleNamespace(module=module), _make_tokenizer())
    graph = MagicMock()
    logits = torch.empty((1, 1, 8))

    with patch.object(rollout, "_capture_continuous_graph", return_value=(graph, logits)) as capture:
        first = rollout._get_continuous_graph_state(module, 1, 3, 5, torch.device("cpu"), torch.float32, GraphCache)
        second = rollout._get_continuous_graph_state(module, 1, 3, 5, torch.device("cpu"), torch.float32, GraphCache)
        with torch.no_grad():
            module.weight.add_(1)
        third = rollout._get_continuous_graph_state(module, 1, 3, 5, torch.device("cpu"), torch.float32, GraphCache)

    assert first is second
    assert third is not first
    assert capture.call_count == 2


def test_static_cache_constructor_supports_max_batch_keyword():

    class MaxBatchStaticCache:

        def __init__(self, config, max_batch_size, max_cache_len, device, dtype):
            self.config = config
            self.max_batch_size = max_batch_size

    config = SimpleNamespace(num_attention_heads=4)
    cache = HybridEngineRollout._create_static_cache(MaxBatchStaticCache, config, 2, 8, "cpu", torch.float32)

    assert cache.max_batch_size == 2
    assert cache.config.num_key_value_heads == 4


def test_continuous_cache_span_does_not_sum_independent_requests():
    cache_len = HybridEngineRollout._estimate_continuous_cache_len(64, [64] * 100, 100)

    assert cache_len == 128
    assert cache_len < 64 + 64 * 100


def test_continuous_generation_validates_each_request_length():

    class LimitedModel(torch.nn.Module):

        _supports_cache_class = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=4)

    model = LimitedModel()
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None))
    request = _tensor_request(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long))

    with pytest.raises(ValueError, match="request exceeds"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


def test_continuous_generation_reports_cache_capacity_remedies():

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            # StaticCache on current Transformers main reads this before choosing layer types.
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            kwargs.pop("cache_position", None)
            kwargs.pop("position_ids", None)
            past_key_values.update(states, states, layer_idx=0, **kwargs)
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 4))
            logits[:, :, 1] = 1
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    rollout = HybridEngineRollout(
        SimpleNamespace(module=CacheClassModel()),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(align_decode_fronts=True, continuous_cache_capacity=2),
    )
    request = _tensor_request(torch.tensor([[1], [3]]), torch.ones((2, 1), dtype=torch.long))

    with pytest.raises(ValueError, match=r"capacity \(2\).*continuous_cache_capacity.*enable_cache_trimming"):
        rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=1))


def test_continuous_generation_rejects_legacy_cache_model():

    class LegacyModel(torch.nn.Module):
        _supports_cache_class = False

        _supports_cache_class = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=32)

    model = LegacyModel()
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None))
    request = _tensor_request(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long))

    with pytest.raises(ValueError, match="cache-class support"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


def test_continuous_graph_generation_rejects_legacy_cache_model():

    class LegacyModel(torch.nn.Module):
        _supports_cache_class = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=32)

    model = LegacyModel()
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        SimpleNamespace(pad_token_id=0, eos_token_id=None),
        cfg=HybridEngineRolloutConfig(use_graph_capture=True),
    )
    request = _tensor_request(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long))

    with pytest.raises(ValueError, match="cache-class support"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


def test_continuous_generation_covers_modern_static_cache_path():

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = []

            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            key_states = input_ids[:, None, :, None].to(dtype=torch.float32)
            _, cache_values = past_key_values.update(key_states, key_states, layer_idx=0, cache_kwargs=kwargs)
            cache_sums = cache_values[:, 0].sum(dim=(1, 2))
            next_tokens = torch.where(cache_sums == 6, 2, 7).long()
            self.calls.append((input_ids.shape[0], input_ids.shape[1]))
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model = CacheClassModel()
    rollout = HybridEngineRollout(SimpleNamespace(module=model), _make_tokenizer())
    request = _tensor_request(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]]),
        torch.ones((3, 3), dtype=torch.long),
    )
    output = rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=2))

    assert [sample.tokens for sample in output.samples] == [[1, 2, 3, 2], [1, 2, 4, 7, 7, 7], [1, 2, 5, 7, 7, 7]]
    assert [sample.loss_mask for sample in output.samples] == [[1], [1, 1, 1], [1, 1, 1]]
    assert [sample.status for sample in output.samples
            ] == [Sample.Status.COMPLETED, Sample.Status.TRUNCATED, Sample.Status.TRUNCATED]
    assert [sample.index for sample in output.samples] == [0, 1, 2]
    assert model.calls[0] == (2, 3)
    assert (1, 3) in model.calls


def test_continuous_prefill_requests_last_logits_only():

    class LastLogitsModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.prefill_logits_to_keep = []

            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        # Mirrors the Transformers contract: 0 returns logits for every position, k for the last k.
        def forward(self,
                    input_ids,
                    attention_mask,
                    past_key_values=None,
                    use_cache=True,
                    cache_position=None,
                    position_ids=None,
                    logits_to_keep=0):
            key_states = input_ids[:, None, :, None].to(dtype=torch.float32)
            past_key_values.update(key_states,
                                   key_states,
                                   layer_idx=0,
                                   cache_kwargs={"cache_position": cache_position})
            if input_ids.shape[1] > 1:
                self.prefill_logits_to_keep.append(logits_to_keep)
            # Each position predicts its own token + 5, so reading any position but the last changes the output.
            logits = torch.nn.functional.one_hot((input_ids + 5) % 16, num_classes=16).to(torch.float32)
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:]
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model = LastLogitsModel()
    rollout = HybridEngineRollout(SimpleNamespace(module=model), _make_tokenizer())
    request = _tensor_request(torch.tensor([[1, 2, 3], [1, 2, 4]]), torch.ones((2, 3), dtype=torch.long))
    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=2))

    assert [sample.tokens[3:] for sample in output.samples] == [[8, 13], [9, 14]]
    assert model.prefill_logits_to_keep == [1]


def test_aligned_continuous_generation_supports_mixed_effective_prompt_lengths():

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            # StaticCache on current Transformers main reads this before choosing layer types.
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.prefill_lengths = []
            self.decode_positions = []
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            if input_ids.shape[1] > 1:
                self.prefill_lengths.append(input_ids.shape[1])
            else:
                write_position = getattr(past_key_values, "_write_position", None)
                if write_position is not None:
                    self.decode_positions.append(int(write_position[0].item()))
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            kwargs.pop("cache_position", None)
            kwargs.pop("position_ids", None)
            _, values = past_key_values.update(states, states, layer_idx=0, **kwargs)
            cache_sums = values[:, 0].sum(dim=(1, 2))
            next_tokens = torch.where(cache_sums == 6, 2, 7).long()
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model = CacheClassModel()
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(align_decode_fronts=True, enable_cache_trimming=True),
    )
    request = _tensor_request(
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [0, 0, 1, 2]]),
        torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]]),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=2))

    assert [sample.tokens for sample in output.samples] == [
        [0, 1, 2, 3, 2],
        [1, 2, 3, 4, 7, 7],
        [0, 0, 1, 2, 7, 7],
    ]
    assert [sample.attention_mask for sample in output.samples] == [
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
    ]
    assert model.prefill_lengths == [4, 3, 2]
    assert model.decode_positions == [4, 3]
    stats = rollout.get_last_continuous_stats()
    assert stats["cache_capacity"] == 6
    assert stats["peak_cache_length"] == 5
    assert stats["trim_count"] == 1
    assert stats["trimmed_columns"] == 2
    assert stats["trim_frequency"] == pytest.approx(0.5)
    assert stats["trim_bytes_moved"] > 0
    assert stats["end_to_end_ms"] is None


def test_aligned_continuous_generation_reclaims_dead_prefix_when_cache_would_exhaust():

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            # StaticCache on current Transformers main reads this before choosing layer types.
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            kwargs.pop("cache_position", None)
            kwargs.pop("position_ids", None)
            _, values = past_key_values.update(states, states, layer_idx=0, **kwargs)
            cache_sums = values[:, 0].sum(dim=(1, 2))
            next_tokens = torch.where(cache_sums == 20, 2, 7).long()
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    rollout = HybridEngineRollout(
        SimpleNamespace(module=CacheClassModel()),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(align_decode_fronts=True),
    )
    request = _tensor_request(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]]),
        torch.ones((3, 3), dtype=torch.long),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=4, temperature=0, continuous_batch_size=2))

    assert [sample.tokens[3:] for sample in output.samples] == [[7, 7, 2], [7, 7, 7, 7], [7, 7, 7, 7]]
    assert [sample.attention_mask[3:] for sample in output.samples] == [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    assert rollout.get_last_continuous_stats()["trim_count"] == 1


def test_continuous_generation_trims_cache_after_staggered_eos():

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            _, values = past_key_values.update(states, states, layer_idx=0, cache_kwargs=kwargs)
            cache_sums = values[:, 0].sum(dim=(1, 2))
            eos_rows = (cache_sums == 6) | (cache_sums == 8) | (cache_sums == 10)
            next_tokens = torch.where(eos_rows, 2, 7).long()
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model = CacheClassModel()
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(enable_cache_trimming=True),
    )
    request = _tensor_request(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 2, 7], [1, 2, 8]]),
        torch.ones((6, 3), dtype=torch.long),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=4, temperature=0, continuous_batch_size=2))

    assert [sample.tokens[3:] for sample in output.samples] == [[2], [7, 7, 7, 7]] * 3
    assert [sample.status for sample in output.samples] == [Sample.Status.COMPLETED, Sample.Status.TRUNCATED] * 3


def test_continuous_generation_refills_padded_prompts_after_trim():

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            # StaticCache on current Transformers main reads this before choosing layer types.
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            kwargs.pop("cache_position", None)
            kwargs.pop("position_ids", None)
            _, values = past_key_values.update(states, states, layer_idx=0, **kwargs)
            cache_sums = values[:, 0].sum(dim=(1, 2))
            next_tokens = torch.where(cache_sums == 6, 2, 7).long()
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model = CacheClassModel()
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(enable_cache_trimming=True),
    )
    prompts = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]] * 16)
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]] * 16)
    request = _tensor_request(prompts, attention_mask)

    output = rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=8))

    assert len(output.samples) == 32
    assert all(sample.response_length > 0 for sample in output.samples)
    assert rollout.get_last_continuous_stats()["trim_count"] > 0


def _argmax(logits):
    return logits.argmax(dim=-1, keepdim=True)


@pytest.mark.parametrize("repetition_penalty, expected", [(1.5, [[1]]), (1.0, [[0]])])
def test_continuous_generation_applies_repetition_penalty(repetition_penalty, expected):
    row = SimpleNamespace(prompt_ids=torch.tensor([[0]]))
    responses = {0: [torch.tensor([[0]])]}

    # Pin the real-model regression: a bare argmax would select the repeated
    # token 0, while Transformers' repetition processor must select token 1.
    next_tokens = HybridEngineRollout._continuous_next_tokens(
        torch.tensor([[6.0, 5.0]]),
        (0, ),
        {0: row},
        responses,
        repetition_penalty,
        _argmax,
    )

    assert next_tokens.tolist() == expected


class _StubCacheConfig(SimpleNamespace):

    def get_text_config(self, **_kwargs):
        return self

    @property
    def per_layer_config(self):
        # StaticCache on current Transformers main reads this before choosing layer types.
        return [self]


class _LastTokenModel(torch.nn.Module):
    """Writes its inputs to the cache and scores each position's next token with ``score(input_ids)``."""

    _supports_cache_class = True

    def __init__(self, score, checkpoint_repetition_penalty=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.score = score
        self.generation_config = SimpleNamespace(repetition_penalty=checkpoint_repetition_penalty)
        self.config = _StubCacheConfig(
            max_position_embeddings=32,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            hidden_size=1,
            head_dim=1,
        )

    def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
        states = input_ids[:, None, :, None].to(dtype=torch.float32)
        kwargs.pop("cache_position", None)
        kwargs.pop("position_ids", None)
        past_key_values.update(states, states, layer_idx=0, **kwargs)
        return SimpleNamespace(logits=self.score(input_ids), past_key_values=past_key_values)


_VOCAB = 16


def _next_token_logits(previous):
    """Logits after token ``previous``: peaked at ``previous + 1``, sharper for larger tokens."""
    vocab = torch.arange(_VOCAB, dtype=torch.float32)
    return -((vocab - (previous + 1) % _VOCAB)**2) / (previous + 2)


def test_continuous_generation_records_greedy_log_probs():
    model = _LastTokenModel(lambda input_ids: _next_token_logits(input_ids[..., None].float()))
    rollout = HybridEngineRollout(SimpleNamespace(module=model), _make_tokenizer())
    request = RolloutRequest(samples=[_prompt([3, 4]), _prompt([0, 5], [0, 1]), _prompt([7, 8])])
    # One row at a time, so later prompts are admitted as earlier ones retire.
    sampling = SamplingConfig(max_new_tokens=3, temperature=0, return_log_probs=True, continuous_batch_size=1)

    samples = rollout.generate(request, sampling).samples

    assert [sample.tokens for sample in samples] == [[3, 4, 5, 6, 7], [0, 5, 6, 7, 8], [7, 8, 9, 10, 11]]
    for sample in samples:
        assert len(sample.rollout_log_probs) == sample.response_length == 3
        tokens = sample.tokens
        # Oracle: each greedy token's log-prob under the logits that follow the token before it.
        expected = [
            torch.log_softmax(_next_token_logits(torch.tensor(float(previous))), dim=-1)[token].item()
            for previous, token in zip(tokens[-4:-1], tokens[-3:])
        ]
        assert sample.rollout_log_probs == pytest.approx(expected, abs=1e-6)


def _repeat_preferring_logits(input_ids):
    """Prefer repeating the last token (2.0) over moving to the next one (1.5)."""
    logits = torch.zeros((*input_ids.shape, _VOCAB))
    logits.scatter_(-1, input_ids[..., None], 2.0)
    logits.scatter_(-1, ((input_ids + 1) % _VOCAB)[..., None], 1.5)
    return logits


def test_continuous_generation_reads_repetition_penalty_from_sampling():
    # A large checkpoint penalty must not apply: only SamplingConfig.repetition_penalty counts.
    model = _LastTokenModel(_repeat_preferring_logits, checkpoint_repetition_penalty=100.0)
    rollout = HybridEngineRollout(SimpleNamespace(module=model), _make_tokenizer())
    request = RolloutRequest(samples=[_prompt([5, 6])])

    def generate(repetition_penalty):
        sampling = SamplingConfig(max_new_tokens=3,
                                  temperature=0,
                                  repetition_penalty=repetition_penalty,
                                  continuous_batch_size=1)
        return rollout.generate(request, sampling).samples[0].tokens

    assert generate(1.0) == [5, 6, 6, 6, 6]
    assert generate(100.0) != generate(1.0)


def test_continuous_sampling_applies_repetition_penalty_before_warpers():
    row = SimpleNamespace(prompt_ids=torch.tensor([[0]]))
    responses = {0: [torch.tensor([[0]])]}
    select = HybridEngineRollout._token_selector(SamplingConfig(max_new_tokens=1, temperature=1.0, top_k=1))

    # generate() penalizes first: token 0 drops from 6 to 4, so top-k 1 keeps only token 1. Warping first would
    # keep only token 0.
    next_tokens = HybridEngineRollout._continuous_next_tokens(
        torch.tensor([[6.0, 5.0]]),
        (0, ),
        {0: row},
        responses,
        1.5,
        select,
    )

    assert next_tokens.tolist() == [[1]]


def _reference_probs(logits, temperature, top_k, top_p):
    """The distribution ``SamplingConfig`` describes, derived from ``logits`` independently of the rollout."""
    scores = logits / temperature
    if top_k > 0:
        kth_score = scores.topk(top_k).values[-1]
        scores = scores.masked_fill(scores < kth_score, float("-inf"))
    probs = scores.softmax(dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(descending=True)
        # A token is kept while the more likely tokens hold less than top_p of the mass.
        keep_sorted = (sorted_probs.cumsum(dim=-1) - sorted_probs) < top_p
        keep = torch.zeros_like(keep_sorted).scatter(0, order, keep_sorted)
        probs = torch.where(keep, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()
    return probs


@pytest.mark.parametrize("temperature, top_k, top_p", [(1.0, -1, 1.0), (0.5, -1, 1.0), (1.0, 3, 1.0), (1.0, -1, 0.7),
                                                       (0.7, 4, 0.9)])
def test_token_selector_samples_the_configured_distribution(temperature, top_k, top_p):
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0])
    expected = _reference_probs(logits, temperature, top_k, top_p)
    select = HybridEngineRollout._token_selector(
        SamplingConfig(max_new_tokens=1, temperature=temperature, top_k=top_k, top_p=top_p))
    draws = 20000

    torch.manual_seed(0)
    tokens = select(logits.expand(draws, -1)).squeeze(1)

    observed = torch.bincount(tokens, minlength=logits.numel()).float() / draws
    assert (observed[expected == 0] == 0).all(), "sampled a token outside the top-k / top-p support"
    tolerance = 4 * (expected * (1 - expected) / draws).sqrt() + 1e-6
    assert ((observed - expected).abs() <= tolerance).all(), f"observed {observed}, expected {expected}"


def test_token_selector_is_argmax_without_temperature():
    logits = torch.tensor([[0.1, 3.0, 2.0], [5.0, -1.0, 4.9]])
    select = HybridEngineRollout._token_selector(SamplingConfig(max_new_tokens=1, temperature=0))

    assert select(logits).tolist() == [[1], [0]]


def test_continuous_generation_samples_when_temperature_is_set():

    class TiedLogitsModel(torch.nn.Module):
        """Tokens 7 and 8 share the top logit at every position, so greedy decoding always picks 7."""
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            past_key_values.update(states, states, layer_idx=0, cache_kwargs=kwargs)
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits[..., 7] = 5.0
            logits[..., 8] = 5.0
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    rollout = HybridEngineRollout(SimpleNamespace(module=TiedLogitsModel()), _make_tokenizer())
    request = RolloutRequest(samples=[_prompt([1, 2, 3]) for _ in range(8)])
    sampling = SamplingConfig(max_new_tokens=4, temperature=1.0, top_k=2, continuous_batch_size=2)

    torch.manual_seed(0)
    output = rollout.generate(request, sampling)

    generated = {token for sample in output.samples for token in sample.tokens[3:]}
    # Top-k 2 keeps only tokens 7 and 8, each with probability 1/2.
    assert generated == {7, 8}


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.time.perf_counter")
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_records_profile_when_enabled(mock_get_accelerator, mock_perf_counter):
    _mock_cpu_accelerator(mock_get_accelerator)
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(enable_profiling=True)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    rollout.engine.module.generate.return_value = torch.tensor([
        [0, 1, 2, 5, 6],
        [0, 1, 2, 7, 8],
        [0, 3, 4, 9, 10],
        [0, 3, 4, 11, 12],
    ])
    mock_perf_counter.side_effect = [1.0, 1.001, 1.011, 1.013]

    output = rollout.generate(_make_request(), _make_sampling(n_samples_per_prompt=2))

    profile = rollout.get_last_profile()
    assert profile["prompt_expansion_ms"] == pytest.approx(1.0)
    assert profile["generation_ms"] == pytest.approx(10.0)
    assert profile["prefill_forward_ms"] is None
    assert profile["decode_forward_ms"] is None
    assert profile["generation_overhead_ms"] == pytest.approx(10.0)
    assert profile["num_decode_forwards"] == 0
    assert profile["post_processing_ms"] == pytest.approx(2.0)
    assert profile["total_ms"] == pytest.approx(13.0)
    assert profile["num_generated_tokens"] == 8
    assert profile["tokens_per_second"] == pytest.approx(8 / 0.013)
    assert profile["batch_size"] == 2
    assert profile["num_samples_per_prompt"] == 2
    assert profile["prompt_length"] == 3
    assert profile["response_length"] == 2
    assert [sample.attention_mask[:3] for sample in output.samples] == [[0, 1, 1]] * 4
    assert mock_get_accelerator.return_value.synchronize.call_count == 4


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_does_not_profile_when_disabled(mock_get_accelerator):
    _mock_cpu_accelerator(mock_get_accelerator)
    engine = _make_engine()
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    engine.module.generate.return_value = torch.tensor([[0, 1, 2, 5, 6], [0, 3, 4, 7, 8]])

    rollout.generate(_make_request(), _make_sampling())

    assert rollout.get_last_profile() is None
    mock_get_accelerator.return_value.synchronize.assert_not_called()


def test_profiling_does_not_change_rollout_output():
    generated = torch.tensor([[0, 1, 2, 5, 6], [0, 3, 4, 7, 8]])
    engine_without_profiling = _make_engine()
    engine_without_profiling.module.generate.return_value = generated
    rollout_without_profiling = HybridEngineRollout(engine_without_profiling, _make_tokenizer())
    engine_with_profiling = _make_engine()
    engine_with_profiling.module.generate.return_value = generated
    rollout_with_profiling = HybridEngineRollout(
        engine_with_profiling,
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(enable_profiling=True),
    )

    output_without_profiling = rollout_without_profiling.generate(_make_request(), _make_sampling())
    with patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator") as mock_get_accelerator:
        _mock_cpu_accelerator(mock_get_accelerator)
        output_with_profiling = rollout_with_profiling.generate(_make_request(), _make_sampling())

    assert output_with_profiling.samples == output_without_profiling.samples


def test_generate_preserves_zero_pad_token_id():
    engine = _make_engine()
    engine.module.generate.return_value = torch.tensor([[0, 1, 2, 0], [0, 3, 4, 0]])
    rollout = HybridEngineRollout(engine, _make_tokenizer())

    output = rollout.generate(_make_request(), _make_sampling())

    # A generated pad id is padding, not a response token.
    assert [sample.response_length for sample in output.samples] == [0, 0]
    assert [sample.tokens for sample in output.samples] == [[0, 1, 2], [0, 3, 4]]


def test_native_repeat_kv_cache_fp16_reverse_copy():
    """Exercise the native reverse copy with multiple source cache rows."""
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("CUDA is required for the native inference kernel")

    from deepspeed.ops.op_builder import InferenceBuilder

    builder = InferenceBuilder()
    try:
        is_compatible = builder.is_compatible()
    except Exception as exc:
        pytest.skip(f"Unable to inspect native transformer inference compatibility: {exc}")
    if not is_compatible:
        pytest.skip("The native transformer inference extension is not compatible")
    try:
        inference_op = builder.load()
    except Exception as exc:
        pytest.skip(f"Unable to load the native transformer inference extension: {exc}")

    repeat_kv_cache = getattr(inference_op, "repeat_kv_cache_fp16", None)
    if repeat_kv_cache is None:
        pytest.skip("The native transformer inference extension lacks repeat_kv_cache_fp16")

    device = torch.device("cuda")
    source_batch_size = 2
    repeats = 2
    target_batch_size = source_batch_size * repeats
    prompt_length = 2
    hidden_dim = 8  # The FP16 transform kernel processes eight values per thread.
    num_heads = 1

    inference_op.allocate_workspace_fp16(
        hidden_dim,
        num_heads,
        prompt_length,
        target_batch_size,
        1,
        1,
        False,
        0,
        4,
        1,
    )
    try:
        query_key_value = torch.zeros((source_batch_size, prompt_length, hidden_dim * 3),
                                      dtype=torch.float16,
                                      device=device)
        query_key_value = query_key_value.view(source_batch_size, prompt_length, 3, num_heads, hidden_dim)
        query_key_value[0, :, 1, :, :] = 1
        query_key_value[0, :, 2, :, :] = 10
        query_key_value[1, :, 1, :, :] = 3
        query_key_value[1, :, 2, :, :] = 30
        query_key_value = query_key_value.reshape(source_batch_size, prompt_length, hidden_dim * 3)

        empty_mask = torch.empty(1, dtype=torch.float16, device=device)
        inference_op.softmax_context_fp16(
            query_key_value,
            empty_mask,
            0,
            False,
            False,
            num_heads,
            0,
            1.0,
            False,
            False,
            1,
            True,
            0,
            1,
            empty_mask,
            1.0,
            True,
            None,
            None,
        )
        torch.cuda.synchronize()  #ignore-cuda

        repeated_cache = repeat_kv_cache(source_batch_size, repeats)
        torch.cuda.synchronize()  #ignore-cuda

        assert len(repeated_cache) == 2
        expected_key = torch.empty((target_batch_size, 1, prompt_length, hidden_dim),
                                   dtype=torch.float16,
                                   device=device)
        expected_key[:source_batch_size] = 1
        expected_key[source_batch_size:] = 3
        expected_value = expected_key * 10
        assert torch.equal(repeated_cache[0], expected_key)
        assert torch.equal(repeated_cache[1], expected_value)
    finally:
        inference_op.release_workspace()


def test_shared_prefill_hooks_reduce_prompt_and_expand_output():

    class PromptModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.forward_batch_sizes = []

        def forward(self, input_ids, attention_mask=None):
            self.forward_batch_sizes.append(input_ids.shape[0])
            values = input_ids[:, None, :, None].float()
            return SimpleNamespace(logits=input_ids[:, :, None].float(), past_key_values=((values, values), ))

    engine = _make_engine()
    engine.repeat_shared_prefill_cache.return_value = ((torch.zeros(4, 1, 2, 1), torch.zeros(4, 1, 2, 1)), )
    module = PromptModule()
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    handles = rollout._register_shared_prefill_hooks(module, batch_size=2, repeats=2)
    prompt_ids = torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]])

    output = module(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids))
    decode_output = module(input_ids=torch.ones(4, 1, dtype=torch.long))
    for handle in handles:
        handle.remove()

    assert module.forward_batch_sizes == [2, 4]
    assert output.logits.shape[0] == 4
    assert output.past_key_values[0][0].shape[0] == 4
    assert decode_output.logits.shape[0] == 4
    engine.repeat_shared_prefill_cache.assert_called_once_with(2, 2)


def test_generate_uses_shared_prefill_for_multiple_samples():

    class GenerateModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.forward_batch_sizes = []

        def forward(self, input_ids, attention_mask=None):
            self.forward_batch_sizes.append(input_ids.shape[0])
            values = input_ids[:, None, :, None].float()
            return SimpleNamespace(logits=input_ids[:, :, None].float(), past_key_values=((values, values), ))

        def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **_kwargs):
            self(input_ids=input_ids, attention_mask=attention_mask)
            response = torch.ones(input_ids.shape[0], max_new_tokens, dtype=input_ids.dtype)
            return torch.cat((input_ids, response), dim=1)

    engine = _make_engine()
    engine.module = GenerateModule()
    config = HybridEngineRolloutConfig(use_shared_prefill=True)
    rollout = HybridEngineRollout(engine, _make_tokenizer(), config)

    output = rollout.generate(_make_request(), _make_sampling(n_samples_per_prompt=2))

    assert engine.module.forward_batch_sizes == [2]
    assert [sample.tokens[3:] for sample in output.samples] == [[1, 1]] * 4
    engine.prepare_shared_prefill.assert_called_once_with(2, 2, 3)
    engine.repeat_shared_prefill_cache.assert_called_once_with(2, 2)


def test_shared_prefill_rejects_graph_capture():
    engine = _make_engine()
    config = HybridEngineRolloutConfig(use_graph_capture=True, use_shared_prefill=True)
    rollout = HybridEngineRollout(engine, _make_tokenizer(), config)

    with pytest.raises(RuntimeError, match="does not support CUDA graph capture"):
        rollout.generate(_make_request(), _make_sampling(n_samples_per_prompt=2))


def test_shared_prefill_fallback_repeats_prompt_cache():
    key_cache = torch.zeros(4, 1, 3, 1)
    value_cache = torch.zeros_like(key_cache)
    key_cache[:2, :, :2, :] = torch.tensor([[[[1.0], [2.0]]], [[[3.0], [4.0]]]])
    value_cache[:2, :, :2, :] = key_cache[:2, :, :2, :] + 10
    workspace = WorkspaceOp.__new__(WorkspaceOp)
    workspace.inference_context = SimpleNamespace(
        kv_cache_size=key_cache.shape,
        kv_cache=[(key_cache, value_cache)],
        current_tokens=lambda: 3,
    )

    repeated_cache = workspace.repeat_kv_cache_fallback(source_batch_size=2, repeats=2)

    expected_key = torch.tensor([1.0, 1.0, 3.0, 3.0])
    assert torch.equal(key_cache[:, 0, 0, 0], expected_key)
    assert torch.equal(value_cache[:, 0, 0, 0], expected_key + 10)
    assert repeated_cache[0].shape == (4, 1, 2, 1)
    assert repeated_cache[1].shape == (4, 1, 2, 1)


def test_engine_pairs_shared_prefill_cache_tensors():
    key_cache = torch.zeros(4, 1, 2, 1)
    value_cache = torch.ones_like(key_cache)
    workspace = MagicMock()
    workspace.repeat_kv_cache.return_value = [key_cache, value_cache]
    engine = SimpleNamespace(_shared_prefill_workspace=workspace)

    repeated_cache = DeepSpeedHybridEngine.repeat_shared_prefill_cache(engine, 2, 2)

    assert repeated_cache[0][0] is key_cache
    assert repeated_cache[0][1] is value_cache
    workspace.repeat_kv_cache.assert_called_once_with(2, 2)


# -- sync_weights is no-op ---------------------------------------------


def test_sync_weights_is_noop():
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer())
    assert rollout.sync_weights(step=0) is None


# -- generate dispatches correctly -------------------------------------


@pytest.mark.parametrize("temperature", [0, 0.7])
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.time.perf_counter")
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_calls_graph_capture_when_enabled(mock_get_accelerator, mock_perf_counter, temperature):
    _mock_cpu_accelerator(mock_get_accelerator)
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(use_graph_capture=True, enable_profiling=True)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    rollout._generate_graph = MagicMock(return_value=(torch.tensor([[1, 3, 5, 2]]), torch.zeros(1, 2)))
    mock_perf_counter.side_effect = [1.0, 1.001, 1.011, 1.013]
    request = RolloutRequest(samples=[_prompt([1, 3])])

    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=temperature))

    rollout._generate_graph.assert_called_once()
    # Sampling must not fall back to Hugging Face generate().
    engine.module.generate.assert_not_called()
    assert output.samples[0].tokens == [1, 3, 5, 2]
    profile = rollout.get_last_profile()
    assert profile["prefill_forward_ms"] is None
    assert profile["decode_forward_ms"] is None
    assert profile["generation_overhead_ms"] == pytest.approx(10.0)
    assert profile["num_decode_forwards"] == 0


class _EchoTool(Tool):
    name = "echo"
    description = "Returns ok."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return "ok"


_TOOL_CALL_TOKEN = 7


def _multi_turn_rollout(responses_per_turn):
    """A multi-turn rollout whose model emits ``responses_per_turn`` in order and calls ``echo`` on token 7."""
    engine = _make_engine()
    responses = iter(responses_per_turn)

    def generate(prompt_ids, **_kwargs):
        response = torch.tensor([next(responses)])
        return torch.cat((prompt_ids, response), dim=1)

    engine.module.generate.side_effect = generate
    tokenizer = _make_tokenizer()
    call_text = '<tool_call>{"name": "echo", "arguments": {}}</tool_call>'
    tokenizer.decode.side_effect = lambda ids, **_kwargs: call_text if _TOOL_CALL_TOKEN in ids else ""
    cfg = HybridEngineRolloutConfig(multi_turn=True, max_turns=4)
    return engine, HybridEngineRollout(engine, tokenizer, cfg=cfg, tools=ToolSet([_EchoTool()]))


@pytest.mark.parametrize("max_new_tokens_per_turn, turn_budgets", [(None, [3, 1]), (2, [2, 1])])
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=[9])
def test_multi_turn_caps_each_turn_and_the_whole_response(_tokenize_tool_responses, max_new_tokens_per_turn,
                                                          turn_budgets):
    engine, rollout = _multi_turn_rollout([[_TOOL_CALL_TOKEN, 2], [5, 6, 2][:turn_budgets[1]]])
    sampling = SamplingConfig(max_new_tokens=3, max_new_tokens_per_turn=max_new_tokens_per_turn, temperature=0)

    sample = rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]), sampling).samples[0]

    # The second turn gets what is left of max_new_tokens (3 - 2), whatever the per-turn cap.
    assert [call.kwargs["max_new_tokens"] for call in engine.module.generate.call_args_list] == turn_budgets
    assert sample.tokens == [10, 11, _TOOL_CALL_TOKEN, 2, 9, 5]
    assert sample.loss_mask == [1, 1, 0, 1]
    assert sample.status == Sample.Status.TRUNCATED


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=[9])
def test_multi_turn_per_turn_cap_resets_every_turn(_tokenize_tool_responses):
    # miles' rollout_max_response_len: every turn may generate up to the cap, beyond it in total.
    engine, rollout = _multi_turn_rollout([[_TOOL_CALL_TOKEN, 2], [_TOOL_CALL_TOKEN, 2], [5, 2]])
    sampling = SamplingConfig(max_new_tokens=100, max_new_tokens_per_turn=2, temperature=0)

    sample = rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]), sampling).samples[0]

    assert [call.kwargs["max_new_tokens"] for call in engine.module.generate.call_args_list] == [2, 2, 2]
    assert sample.response_length == 8
    assert sample.status == Sample.Status.COMPLETED


def test_single_turn_applies_the_per_turn_cap():
    engine = _make_engine()
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 6]])
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    sampling = SamplingConfig(max_new_tokens=8, max_new_tokens_per_turn=2, temperature=0)

    rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]), sampling)

    assert engine.module.generate.call_args.kwargs["max_new_tokens"] == 2


def test_rejects_non_positive_per_turn_cap():
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer())

    with pytest.raises(ValueError, match="max_new_tokens_per_turn"):
        rollout.generate(_make_request(), SamplingConfig(max_new_tokens=8, max_new_tokens_per_turn=0))


def test_generate_keeps_ranks_in_lockstep_and_pads_after_eos():
    engine = _make_engine()
    # ZeRO-3 gathers parameters in every decode step, so ranks must not stop at EOS.
    engine.zero_optimization_partition_weights.return_value = True
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 2, 7, 8]])
    request = RolloutRequest(samples=[_prompt([10, 11])])

    result = rollout.generate(request, SamplingConfig(max_new_tokens=4, temperature=0))

    assert engine.module.generate.call_args.kwargs['eos_token_id'] is None
    sample = result.samples[0]
    assert sample.tokens == [10, 11, 5, 2]
    assert sample.attention_mask == [1, 1, 1, 1]
    assert sample.loss_mask == [1, 1]
    assert sample.response_length == 2
    assert sample.status == Sample.Status.COMPLETED


def test_generate_forwards_top_k_for_sampling_with_multiple_samples_and_eos():
    engine = _make_engine()
    tok = _make_tokenizer()
    rollout = HybridEngineRollout(engine, tok)
    engine.module.generate.return_value = torch.tensor([
        [10, 11, 5, 2, 7, 8],
        [10, 11, 6, 7, 2, 8],
    ])
    request = RolloutRequest(samples=[_prompt([10, 11])])
    sampling = SamplingConfig(max_new_tokens=4, temperature=0.7, top_p=0.8, top_k=5, n_samples_per_prompt=2)

    result = rollout.generate(request, sampling)

    generate_call = engine.module.generate.call_args
    assert generate_call.args[0].tolist() == [[10, 11], [10, 11]]
    assert generate_call.kwargs["attention_mask"].tolist() == [[1, 1], [1, 1]]
    assert generate_call.kwargs["do_sample"] is True
    assert generate_call.kwargs["temperature"] == 0.7
    assert generate_call.kwargs["top_p"] == 0.8
    assert generate_call.kwargs["top_k"] == 5
    assert generate_call.kwargs["eos_token_id"] == tok.eos_token_id
    assert [sample.tokens for sample in result.samples] == [[10, 11, 5, 2], [10, 11, 6, 7, 2]]
    assert [sample.loss_mask for sample in result.samples] == [[1, 1], [1, 1, 1]]
    assert [sample.group_index for sample in result.samples] == [0, 0]
    assert [sample.index for sample in result.samples] == [0, 1]
    assert [sample.status for sample in result.samples] == [Sample.Status.COMPLETED] * 2


@pytest.mark.parametrize("top_k", [-1, 0])
def test_generate_disables_top_k_for_sampling(top_k):
    engine = _make_engine()
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    engine.module.generate.return_value = torch.tensor([[10, 11, 5]])
    request = RolloutRequest(samples=[_prompt([10, 11])])

    rollout.generate(request, SamplingConfig(max_new_tokens=1, temperature=0.7, top_k=top_k))

    assert engine.module.generate.call_args.kwargs["top_k"] == 0


def test_pad_after_eos_handles_different_lengths_and_missing_eos():
    output_ids = torch.tensor([
        [10, 11, 2, 7, 8, 9],
        [10, 11, 5, 6, 2, 9],
        [10, 11, 5, 6, 7, 8],
    ])

    padded, response_attn = HybridEngineRollout._pad_after_eos(output_ids, 2, eos_token_id=2, pad_token_id=0)

    assert padded.tolist() == [
        [10, 11, 2, 0, 0, 0],
        [10, 11, 5, 6, 2, 0],
        [10, 11, 5, 6, 7, 8],
    ]
    assert response_attn.tolist() == [
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]


def test_pad_after_eos_keeps_eos_attended_when_eos_is_pad():
    output_ids = torch.tensor([[10, 11, 5, 2, 7, 8]])

    padded, response_attn = HybridEngineRollout._pad_after_eos(output_ids, 2, eos_token_id=2, pad_token_id=2)

    assert padded.tolist() == [[10, 11, 5, 2, 2, 2]]
    assert response_attn.tolist() == [[1, 1, 0, 0]]


def test_pad_after_eos_supports_multiple_eos_ids():
    output_ids = torch.tensor([[10, 11, 5, 3, 7, 2]])

    padded, response_attn = HybridEngineRollout._pad_after_eos(output_ids, 2, eos_token_id=[2, 3], pad_token_id=0)

    assert padded.tolist() == [[10, 11, 5, 3, 0, 0]]
    assert response_attn.tolist() == [[1, 1, 0, 0]]


def test_generate_accepts_zero_pad_token_id():
    engine = _make_engine()
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 6]])
    request = RolloutRequest(samples=[_prompt([10, 11])])

    result = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))

    assert engine.module.generate.call_args.kwargs['pad_token_id'] == 0
    assert result.samples[0].tokens == [10, 11, 5, 6]
    # No EOS within max_new_tokens.
    assert result.samples[0].status == Sample.Status.TRUNCATED


def test_single_turn_records_log_probs_of_the_sampled_distribution():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64,
                         hidden_size=32,
                         intermediate_size=64,
                         num_hidden_layers=1,
                         num_attention_heads=2,
                         num_key_value_heads=1,
                         max_position_embeddings=64)
    model = LlamaForCausalLM(config).eval()
    engine = SimpleNamespace(module=model, zero_optimization_partition_weights=lambda: False)
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    request = RolloutRequest(samples=[_prompt([5, 6, 7]), _prompt([0, 8, 9], [0, 1, 1])])
    sampling = SamplingConfig(max_new_tokens=4, temperature=0.7, n_samples_per_prompt=2, return_log_probs=True)

    samples = rollout.generate(request, sampling).samples

    assert [sample.group_index for sample in samples] == [0, 0, 1, 1]
    for sample in samples:
        assert len(sample.rollout_log_probs) == len(sample.loss_mask) == sample.response_length > 0
        # Oracle: a full forward over the sample's real tokens, scored at the sampling temperature.
        ids = torch.tensor([token for token, keep in zip(sample.tokens, sample.attention_mask) if keep])
        with torch.no_grad():
            logits = model(ids[None]).logits[0].float() / sampling.temperature
        response = ids[-sample.response_length:]
        expected = torch.log_softmax(logits[-sample.response_length - 1:-1], dim=-1).gather(1, response[:, None])
        torch.testing.assert_close(torch.tensor(sample.rollout_log_probs), expected.squeeze(1), atol=1e-4, rtol=1e-4)


def _cache_config(head_dim=None):
    config = SimpleNamespace(num_hidden_layers=2, hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
    if head_dim is not None:
        config.head_dim = head_dim
    return config


def test_static_cache_preallocates_with_config_head_dim():
    cache = DeepSpeedStaticCache(_cache_config(head_dim=32),
                                 batch_size=1,
                                 max_cache_len=8,
                                 device="cpu",
                                 dtype=torch.float32)

    assert tuple(cache.layers[0].keys.shape) == (1, 2, 8, 32)
    assert tuple(cache.layers[0].values.shape) == (1, 2, 8, 32)


def test_static_cache_falls_back_to_derived_head_dim():
    cache = DeepSpeedStaticCache(_cache_config(), batch_size=1, max_cache_len=8, device="cpu", dtype=torch.float32)

    assert tuple(cache.layers[0].keys.shape) == (1, 2, 8, 16)

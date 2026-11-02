# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CPU unit tests for AutoTPRollout.

Two simulated TP ranks are built in one process: AutoTP layers read their rank from
``deepspeed.comm``, which is patched, so the layers partition and record their layout metadata
exactly as they do under ``init_inference``. The ZeRO gather is patched to return the full weight.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from deepspeed.module_inject import layers as autotp_layers
from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer
from deepspeed.runtime.rollout import build_rollout
from deepspeed.runtime.rollout import autotp_rollout
from deepspeed.runtime.rollout.autotp_rollout import AutoTPRollout, sync_autotp_weights
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutConfig, RolloutRecord, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig

TP_SIZE = 2


class _Block(torch.nn.Module):
    """Embedding, a column-parallel and a row-parallel projection, and a norm: one of each layout."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 8)
        self.up = torch.nn.Linear(8, 12, bias=True)
        self.down = torch.nn.Linear(12, 8, bias=True)
        self.norm = torch.nn.LayerNorm(8)


def _train_module():
    """The training engine's module, with weights unrelated to the AutoTP copy's initial ones."""
    torch.manual_seed(0)
    return _Block()


def _tp_copy(tp_rank, dtype):
    """Rank ``tp_rank``'s AutoTP copy, initialized with weights unrelated to the training module."""
    torch.manual_seed(1)
    block = _Block().to(dtype)
    with patch.object(autotp_layers.dist, "get_world_size", return_value=TP_SIZE), \
            patch.object(autotp_layers.dist, "get_rank", return_value=tp_rank):
        block.up = LinearLayer(block.up, mp_group=object(), name="up")
        block.down = LinearAllreduce(block.down, mp_group=object(), name="down")
    return block


def _sync(train_module, tp_module):
    # ZeRO-2 keeps a full low-precision copy on every rank, so the gather's result is its fp32 value.
    with patch.object(autotp_rollout, "safe_get_full_fp32_param", side_effect=lambda p: p.detach().float()):
        return sync_autotp_weights(train_module, tp_module)


@pytest.mark.parametrize("tp_rank", range(TP_SIZE))
def test_sync_copies_each_layout_into_its_shard(tp_rank):
    train = _train_module()
    tp_module = _tp_copy(tp_rank, torch.float32)
    full = dict(train.named_parameters())
    assert tp_module.up.weight.shape == (12 // TP_SIZE, 8)
    assert not torch.equal(tp_module.up.weight, full["up.weight"].chunk(TP_SIZE, dim=0)[tp_rank])

    assert _sync(train, tp_module) == len(full)

    tp = dict(tp_module.named_parameters())
    # Column-parallel: weight and bias split on the output dimension.
    assert torch.equal(tp["up.weight"], full["up.weight"].chunk(TP_SIZE, dim=0)[tp_rank])
    assert torch.equal(tp["up.bias"], full["up.bias"].chunk(TP_SIZE, dim=0)[tp_rank])
    # Row-parallel: weight split on the input dimension, bias replicated.
    assert torch.equal(tp["down.weight"], full["down.weight"].chunk(TP_SIZE, dim=1)[tp_rank])
    assert torch.equal(tp["down.bias"], full["down.bias"])
    # Modules AutoTP did not replace are copied whole.
    for name in ("embed.weight", "norm.weight", "norm.bias"):
        assert torch.equal(tp[name], full[name])


def test_sync_rounds_fp32_masters_to_the_inference_dtype():
    train = _train_module()
    tp_module = _tp_copy(0, torch.bfloat16)
    _sync(train, tp_module)
    full = dict(train.named_parameters())
    assert tp_module.down.weight.dtype == torch.bfloat16
    assert torch.equal(tp_module.down.weight, full["down.weight"].chunk(TP_SIZE, dim=1)[0].to(torch.bfloat16))


def test_sync_rejects_an_inference_copy_missing_a_parameter():
    train = _train_module()
    tp_module = _tp_copy(0, torch.float32)
    del tp_module.norm
    with pytest.raises(AssertionError, match="no parameter norm.weight"):
        _sync(train, tp_module)


def test_sync_rejects_a_training_model_missing_a_parameter():
    train = _train_module()
    del train.norm
    with pytest.raises(AssertionError, match="synced"):
        _sync(train, _tp_copy(0, torch.float32))


def test_constructor_rejects_multi_turn():
    engine = MagicMock()
    engine.zero_optimization_partition_weights.return_value = False
    with pytest.raises(NotImplementedError, match="single-turn"):
        AutoTPRollout(engine, MagicMock(), cfg=HybridEngineRolloutConfig(multi_turn=True, max_turns=2))


def test_engine_factory_autotp_requires_student_engine():
    with pytest.raises(ValueError, match="needs both"):
        build_rollout(RolloutConfig(engine="autotp"))


# -- generate: one TP group spans all ranks, so each rank generates the union of all prompts ------


def _rollout():
    rollout = AutoTPRollout.__new__(AutoTPRollout)
    rollout.tp_size = TP_SIZE
    rollout._seed = 1234
    rollout._generate_calls = 0
    return rollout


def _prompts(*first_tokens):
    return [RolloutRecord(tokens=[token, 0], attention_mask=[1, 1]) for token in first_tokens]


def _fake_union_generate(seeds):
    """Stand-in for HybridEngineRollout.generate: n copies per prompt, adjacent, in prompt order."""

    def generate(self, request, sampling):
        seeds.append(torch.initial_seed())
        samples = []
        for prompt in request.samples:
            for copy in range(sampling.n_samples_per_prompt):
                tokens = prompt.tokens + [copy]
                samples.append(
                    RolloutRecord(group_index=prompt.group_index,
                                  tokens=tokens,
                                  attention_mask=[1] * len(tokens),
                                  index=len(samples)))
        return RolloutBatch(samples=samples)

    return generate


def _generate(rank, per_rank_prompts, n_samples_per_prompt, gathered_tokens=None, rollout=None):
    rollout = rollout or _rollout()
    seeds = []

    def all_gather_object(output, obj):
        if isinstance(obj[0], RolloutRecord):
            output[:] = per_rank_prompts
        else:  # the token check: every rank reports what this rank generated, unless overridden
            output[:] = gathered_tokens(obj) if gathered_tokens else [obj] * TP_SIZE

    original_fork_rng = torch.random.fork_rng
    with patch.object(autotp_rollout.dist, "get_rank", return_value=rank), \
            patch.object(autotp_rollout.dist, "all_gather_object", side_effect=all_gather_object), \
            patch.object(autotp_rollout.torch.random, "fork_rng",
                         side_effect=lambda devices: original_fork_rng(devices=[])), \
            patch.object(HybridEngineRollout, "generate", _fake_union_generate(seeds)):
        sampling = SamplingConfig(max_new_tokens=1, n_samples_per_prompt=n_samples_per_prompt)
        batch = rollout.generate(RolloutRequest(samples=per_rank_prompts[rank]), sampling)
    return rollout, batch, seeds


@pytest.mark.parametrize("rank, expected_prompts", [(0, [7, 8]), (1, [9])])
def test_generate_returns_only_this_ranks_samples(rank, expected_prompts):
    per_rank_prompts = [_prompts(7, 8), _prompts(9)]
    _, batch, _ = _generate(rank, per_rank_prompts, n_samples_per_prompt=2)
    expected = [[prompt, 0, copy] for prompt in expected_prompts for copy in range(2)]
    assert [sample.tokens for sample in batch.samples] == expected
    assert [sample.index for sample in batch.samples] == list(range(len(expected)))


def test_generate_keeps_each_ranks_own_group_numbering():
    # Prompts without a group_index are grouped by their position in this rank's request, as with
    # HybridEngineRollout, not by their position in the union of all ranks' prompts.
    per_rank_prompts = [_prompts(7, 8), _prompts(9)]
    _, batch, _ = _generate(1, per_rank_prompts, n_samples_per_prompt=2)
    assert [sample.group_index for sample in batch.samples] == [0, 0]


def test_generate_seeds_every_rank_alike_and_each_call_differently():
    per_rank_prompts = [_prompts(7), _prompts(9)]
    seeds_per_rank = []
    for rank in range(TP_SIZE):
        rollout = _rollout()
        seeds = []
        for _ in range(2):
            _, _, call_seeds = _generate(rank, per_rank_prompts, n_samples_per_prompt=1, rollout=rollout)
            seeds += call_seeds
        seeds_per_rank.append(seeds)
    # Ranks that sample from different seeds pick different tokens and deadlock in AutoTP's all-reduce.
    assert seeds_per_rank[0] == seeds_per_rank[1]
    # Reusing one seed would make every rollout step draw the same samples for the same prompts.
    assert seeds_per_rank[0][0] != seeds_per_rank[0][1]


def test_generate_fails_loudly_when_ranks_disagree():
    per_rank_prompts = [_prompts(7), _prompts(9)]
    diverged = lambda tokens: [tokens, [token[:-1] + [99] for token in tokens]]
    with pytest.raises(RuntimeError, match="different tokens"):
        _generate(0, per_rank_prompts, n_samples_per_prompt=1, gathered_tokens=diverged)


def test_generate_rejects_continuous_batching():
    with pytest.raises(NotImplementedError, match="continuous batching"):
        _rollout().generate(RolloutRequest(samples=_prompts(7)),
                            SamplingConfig(max_new_tokens=1, continuous_batch_size=4))

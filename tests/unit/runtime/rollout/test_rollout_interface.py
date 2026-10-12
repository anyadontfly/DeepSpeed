# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Conformance tests for the RolloutEngine interface.

Validates the dataclass invariants and exercises the interface against a
``FakeRollout`` so the contract is testable without GPUs or a model. The real
backends are tested manually with a launched training script (see README).
"""

from dataclasses import replace

import pytest

import deepspeed.runtime.rollout as rollout_module
from deepspeed.runtime.rollout import (
    RolloutBatch,
    RolloutEngine,
    RolloutRequest,
    SamplingConfig,
    build_rollout,
)
from deepspeed.runtime.rollout.base import Sample

# --- dataclass invariants ---------------------------------------------------


def test_rollout_request_validates_shapes():
    RolloutRequest(samples=[Sample(prompt="not tokenized yet")])
    with pytest.raises(ValueError, match="does not match"):
        RolloutRequest(samples=[Sample(tokens=[1, 2, 3], attention_mask=[1, 1])])


def test_rollout_batch_validates_shapes():
    with pytest.raises(ValueError, match="does not match"):
        RolloutBatch(samples=[Sample(tokens=[1, 2], attention_mask=[1, 1]), Sample(tokens=[1, 2], attention_mask=[1])])


def test_rollout_batch_accessors():
    batch = RolloutBatch(samples=[Sample(tokens=[1] * length, attention_mask=[1] * length) for length in (4, 12, 7)])
    assert batch.batch_size == 3
    assert batch.seq_len == 12


def test_sampling_config_defaults():
    cfg = SamplingConfig(max_new_tokens=32)
    assert cfg.temperature == 1.0
    assert cfg.top_p == 1.0
    assert cfg.top_k == -1
    assert cfg.n_samples_per_prompt == 1
    assert cfg.continuous_batch_size is None


def test_continuous_batching_details_are_not_public_exports():
    assert "ContinuousBatchRequest" not in rollout_module.__all__
    assert "ContinuousBatchScheduler" not in rollout_module.__all__
    assert "ContinuousBatchUpdate" not in rollout_module.__all__


# --- interface conformance via FakeRollout ---------------------------------


class FakeRollout(RolloutEngine):
    """Deterministic stub: appends ``[42] * max_new_tokens`` to ``n_samples_per_prompt`` copies of each prompt."""

    name = "fake"

    def __init__(self, response_token: int = 42):
        self.response_token = response_token
        self.sync_calls: list = []

    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        response = [self.response_token] * sampling.max_new_tokens
        samples = []
        for group_index, prompt in enumerate(request.samples):
            for _ in range(sampling.n_samples_per_prompt):
                samples.append(
                    replace(prompt,
                            group_index=group_index,
                            index=len(samples),
                            tokens=prompt.tokens + response,
                            attention_mask=prompt.attention_mask + [1] * len(response),
                            loss_mask=[1] * len(response),
                            response_length=len(response),
                            status=Sample.Status.TRUNCATED))
        return RolloutBatch(samples=samples)

    def sync_weights(self, step: int) -> None:
        self.sync_calls.append(step)


def _prompt(tokens, attention_mask=None):
    return Sample(tokens=tokens, attention_mask=attention_mask or [1] * len(tokens))


def test_fake_rollout_shape_basic():
    fake = FakeRollout()
    req = RolloutRequest(samples=[_prompt([1, 2, 3]), _prompt([4, 5, 6])])
    out = fake.generate(req, SamplingConfig(max_new_tokens=4))
    assert out.batch_size == 2
    assert out.seq_len == 7
    assert [sample.tokens[3:] for sample in out.samples] == [[42] * 4] * 2
    assert [sample.response_length for sample in out.samples] == [4, 4]


def test_fake_rollout_with_n_samples():
    fake = FakeRollout()
    req = RolloutRequest(samples=[_prompt([1, 2]), _prompt([3, 4])])
    out = fake.generate(req, SamplingConfig(max_new_tokens=3, n_samples_per_prompt=4))
    assert out.batch_size == 8
    assert out.seq_len == 5
    assert [sample.group_index for sample in out.samples] == [0] * 4 + [1] * 4
    assert [sample.index for sample in out.samples] == list(range(8))


def test_fake_rollout_left_padded_prompts():
    fake = FakeRollout()
    # Prompt B is left-padded: its first two positions are padding, so the response
    # still follows its last real token and the padding stays masked out.
    req = RolloutRequest(samples=[_prompt([1, 2, 3, 4]), _prompt([0, 0, 5, 6], [0, 0, 1, 1])])
    out = fake.generate(req, SamplingConfig(max_new_tokens=2))
    assert out.samples[1].tokens == [0, 0, 5, 6, 42, 42]
    assert out.samples[1].attention_mask == [0, 0, 1, 1, 1, 1]


def test_sync_records_steps():
    fake = FakeRollout()
    fake.sync_weights(0)
    fake.sync_weights(5)
    assert fake.sync_calls == [0, 5]


def test_engine_factory_unknown_raises():
    from deepspeed.runtime.rollout.base import RolloutConfig

    with pytest.raises(ValueError, match="Unknown rollout engine"):
        build_rollout(RolloutConfig(engine="totally_made_up"))


def test_engine_factory_hybrid_requires_student_engine():
    from deepspeed.runtime.rollout.base import RolloutConfig

    with pytest.raises(ValueError, match="needs both"):
        build_rollout(RolloutConfig(engine="hybrid_engine"))

# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Conformance tests for the RolloutEngine interface.

Validates the dataclass invariants and exercises the interface against a
``FakeRollout`` so the contract is testable without GPUs or a model. The real
backends are tested manually with a launched training script (see README).
"""

from dataclasses import replace

import pytest
import torch

import deepspeed.runtime.rollout as rollout_module
from deepspeed.runtime.rollout import (
    JSDLossOutput,
    RolloutBatch,
    RolloutEngine,
    RolloutRequest,
    ResponseTokenBatch,
    SamplingConfig,
    build_rollout,
    generalized_jsd_loss,
)
from deepspeed.runtime.rollout.base import RolloutRecord

# --- dataclass invariants ---------------------------------------------------


def test_rollout_request_validates_shapes():
    RolloutRequest(samples=[RolloutRecord(prompt="not tokenized yet")])
    with pytest.raises(ValueError, match="does not match"):
        RolloutRequest(samples=[RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1])])


def test_rollout_batch_validates_shapes():
    with pytest.raises(ValueError, match="does not match"):
        RolloutBatch(samples=[
            RolloutRecord(tokens=[1, 2], attention_mask=[1, 1]),
            RolloutRecord(tokens=[1, 2], attention_mask=[1])
        ])


def test_rollout_batch_accessors():
    batch = RolloutBatch(
        samples=[RolloutRecord(tokens=[1] * length, attention_mask=[1] * length) for length in (4, 12, 7)])
    assert batch.batch_size == 3
    assert batch.seq_len == 12


def test_sampling_config_defaults():
    cfg = SamplingConfig(max_new_tokens=32)
    assert cfg.temperature == 1.0
    assert cfg.top_p == 1.0
    assert cfg.top_k == -1
    assert cfg.n_samples_per_prompt == 1
    assert cfg.continuous_batch_size is None


@pytest.mark.parametrize("unsupported", [{"top_k": 5}, {"top_p": 0.9}, {"repetition_penalty": 1.1}])
def test_sampling_config_rejects_log_probs_a_training_forward_cannot_reproduce(unsupported):
    # Rejected when the config is built, so no generation path can record log-probs of another distribution.
    with pytest.raises(ValueError, match="return_log_probs requires"):
        SamplingConfig(max_new_tokens=4, temperature=1.0, return_log_probs=True, **unsupported)


def test_sampling_config_allows_log_probs_of_greedy_and_plain_temperature_sampling():
    # Greedy decoding ignores top-k and top-p, so its log-probs are those of the raw logits either way.
    SamplingConfig(max_new_tokens=4, temperature=0, top_k=5, top_p=0.9, return_log_probs=True)
    SamplingConfig(max_new_tokens=4, temperature=0.7, return_log_probs=True)
    # Without log-probs, every sampling option is allowed.
    SamplingConfig(max_new_tokens=4, temperature=1.0, top_k=5, top_p=0.9, repetition_penalty=1.1)


def test_continuous_batching_details_are_not_public_exports():
    assert "ContinuousBatchRequest" not in rollout_module.__all__
    assert "ContinuousBatchScheduler" not in rollout_module.__all__
    assert "ContinuousBatchUpdate" not in rollout_module.__all__


def test_response_token_batch_masks_prompts_padding_and_tool_output():
    rollout = RolloutBatch(samples=[
        # Prompt [0, 11, 12] with its own left padding, response [21, 2].
        RolloutRecord(tokens=[0, 11, 12, 21, 2], attention_mask=[0, 1, 1, 1, 1], response_length=2),
        # Prompt [13, 14, 15], response [22, 23, 2].
        RolloutRecord(tokens=[13, 14, 15, 22, 23, 2], attention_mask=[1] * 6, response_length=3),
        # Multi-turn: generated 31, tool output 41 (loss mask 0), generated 32 and EOS.
        RolloutRecord(
            tokens=[16, 17, 31, 41, 32, 2], attention_mask=[1] * 6, response_length=4, loss_mask=[1, 0, 1, 1]),
    ])

    response_batch = ResponseTokenBatch.from_rollout(rollout, pad_token_id=0)

    assert response_batch.input_ids.tolist() == [[0, 0, 11, 12, 21, 2], [13, 14, 15, 22, 23, 2],
                                                 [16, 17, 31, 41, 32, 2]]
    assert response_batch.attention_mask.tolist() == [[0, 0, 1, 1, 1, 1], [1] * 6, [1] * 6]
    assert response_batch.target_ids.tolist() == [[0, 11, 12, 21, 2], [14, 15, 22, 23, 2], [17, 31, 41, 32, 2]]
    assert response_batch.response_mask.tolist() == [[False, False, False, True,
                                                      True], [False, False, True, True, True],
                                                     [False, True, False, True, True]]


@pytest.mark.parametrize(
    "sample",
    [
        RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1], response_length=3),
        RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1], response_length=4),
        RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1], response_length=-1),
        RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1], response_length=2, loss_mask=[1]),
    ],
)
def test_response_token_batch_rejects_invalid_response_boundaries(sample):
    with pytest.raises(ValueError):
        ResponseTokenBatch.from_rollout(RolloutBatch(samples=[sample]))


def test_response_token_batch_aligns_causal_logits():
    rollout = RolloutBatch(samples=[RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1], response_length=1)])
    logits = torch.randn(1, 3, 5)

    response_batch = ResponseTokenBatch.from_rollout(rollout)

    assert torch.equal(response_batch.select_causal_logits(logits), logits[:, :-1])
    with pytest.raises(ValueError, match="sequence"):
        response_batch.select_causal_logits(torch.randn(1, 2, 5))


@pytest.mark.parametrize(
    "beta, expected",
    [
        (
            0.0,
            0.5 * torch.log(torch.tensor(0.5 / 0.8)) + 0.5 * torch.log(torch.tensor(0.5 / 0.2)),
        ),
        (
            1.0,
            0.8 * torch.log(torch.tensor(0.8 / 0.5)) + 0.2 * torch.log(torch.tensor(0.2 / 0.5)),
        ),
        (
            0.5,
            0.5 * (0.5 * torch.log(torch.tensor(0.5 / 0.65)) + 0.5 * torch.log(torch.tensor(0.5 / 0.35))) + 0.5 *
            (0.8 * torch.log(torch.tensor(0.8 / 0.65)) + 0.2 * torch.log(torch.tensor(0.2 / 0.35))),
        ),
    ],
)
def test_generalized_jsd_loss_covers_beta_variants(beta, expected):
    student_logits = torch.log(torch.tensor([[[0.8, 0.2], [0.5, 0.5]]]))
    teacher_logits = torch.log(torch.tensor([[[0.5, 0.5], [0.9, 0.1]]]))
    response_mask = torch.tensor([[True, False]])

    output = generalized_jsd_loss(student_logits, teacher_logits, response_mask, beta=beta)

    assert isinstance(output, JSDLossOutput)
    assert output.valid_token_count.item() == 1
    assert torch.allclose(output.loss_sum, expected)
    assert torch.allclose(output.loss, expected)


def test_generalized_jsd_loss_supports_top_k_and_empty_response_masks():
    student_logits = torch.tensor([[[20.0, 0.0, 0.0]]])
    teacher_logits = torch.tensor([[[0.0, 10.0, 9.0]]])
    response_mask = torch.tensor([[True]])

    top_one = generalized_jsd_loss(student_logits, teacher_logits, response_mask, teacher_top_k=1)
    empty = generalized_jsd_loss(student_logits, teacher_logits, torch.tensor([[False]]))

    assert top_one.loss.item() == pytest.approx(0.0)
    assert empty.valid_token_count.item() == 0
    assert empty.loss.item() == pytest.approx(0.0)


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
                            status=RolloutRecord.Status.TRUNCATED))
        return RolloutBatch(samples=samples)

    def sync_weights(self, step: int) -> None:
        self.sync_calls.append(step)


def _prompt(tokens, attention_mask=None):
    return RolloutRecord(tokens=tokens, attention_mask=attention_mask or [1] * len(tokens))


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

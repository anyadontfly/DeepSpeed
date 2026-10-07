# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine interface.

The trainer talks to its rollout engine through three small dataclasses
(``RolloutRequest`` in / ``RolloutBatch`` out / ``SamplingConfig``) and one
ABC. This keeps engine-specific concerns out of the trainer loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


@dataclass
class RolloutConfig:
    """Configuration for the rollout engine."""
    engine: str = "hybrid_engine"

    # Use CUDA graph capture for decode acceleration.
    use_graph_capture: bool = False

    multi_turn: bool = False
    max_turns: Optional[int] = None
    max_context_len: Optional[int] = None
    # Extra arguments for ``tokenizer.apply_chat_template``, e.g. {"enable_thinking": False} for Qwen3.
    chat_template_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.multi_turn and not self.max_turns:
            raise ValueError("max_turns must be set when multi_turn is enabled")


@dataclass
class SamplingConfig:
    """Sampling knobs that the trainer passes to ``generate`` each step.

    ``top_k <= 0`` disables top-k filtering.
    """

    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    n_samples_per_prompt: int = 1
    continuous_batch_size: Optional[int] = None
    return_log_probs: bool = False
    # Passed explicitly so a checkpoint's generation_config (e.g. Qwen2.5-Instruct's 1.1)
    # never applies silently; 1.0 disables the penalty.
    repetition_penalty: float = 1.0


@dataclass
class Sample:
    """One generated sample."""

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    group_index: int | None = None
    index: int | None = None
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    status: Status = Status.PENDING
    # for OPSD
    teacher_log_probs: list[float] | None = None
    opd_reverse_kl: list[float] | None = None


@dataclass
class RolloutRequest:
    """Input to ``RolloutEngine.generate``."""

    samples: list[Sample]

    def __post_init__(self) -> None:
        for i, sample in enumerate(self.samples):
            if len(sample.attention_mask) != len(sample.tokens):
                raise ValueError(f"samples[{i}] attention_mask length {len(sample.attention_mask)} does not "
                                 f"match tokens length {len(sample.tokens)}")


@dataclass
class RolloutBatch:
    """Output of ``RolloutEngine.generate``."""

    samples: list[Sample]

    def __post_init__(self) -> None:
        for i, sample in enumerate(self.samples):
            if len(sample.attention_mask) != len(sample.tokens):
                raise ValueError(f"samples[{i}] attention_mask length {len(sample.attention_mask)} does not "
                                 f"match tokens length {len(sample.tokens)}")

    @property
    def batch_size(self) -> int:
        return len(self.samples)

    @property
    def seq_len(self) -> int:
        return max(len(sample.tokens) for sample in self.samples)


class RolloutEngine(ABC):
    """Abstract base for rollout engines."""

    name: str = "base"

    @abstractmethod
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        """Run generation, return prompt+response in one tensor."""

    @abstractmethod
    def sync_weights(self, step: int) -> None:
        """Push updated weights into the rollout backend.

        No-op when the rollout engine is co-located with the training engine
        (e.g. hybrid engine shares weights directly).
        """

    def shutdown(self) -> None:
        """Release any backend resources. Default no-op."""
        return None

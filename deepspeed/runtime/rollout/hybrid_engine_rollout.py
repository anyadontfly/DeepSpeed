# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine backed by DeepSpeed's hybrid engine.

Two generation paths:
  1. **model.generate()** (default): delegates to HuggingFace generate.
     Supports sampling (temperature, top_p) and greedy.
  2. **graph capture + DeepSpeedStaticCache**: greedy and sampling
     (temperature, top_k, top_p). Pre-allocates a StaticCache, captures the
     decode forward pass with a CUDA graph, and replays it for each decode
     step.  Eliminates kernel launch overhead.
  3. **continuous batching (experimental)**: an opt-in bounded path
     selected through ``SamplingConfig.continuous_batch_size``; greedy or
     sampled like the graph path, with or without graph capture.
"""

import asyncio
import time
from copy import copy
from dataclasses import dataclass, field, replace
from inspect import signature
from typing import Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRecord, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.continuous_batching import ContinuousBatchRequest, ContinuousBatchScheduler
from deepspeed.runtime.rollout.tool import TOOL_CALL_PARSERS, ToolSet
from deepspeed.runtime.rollout.utils import (
    expand_samples,
    left_pad_samples,
    masked_lists,
    render_prompts,
    tokenize_tool_responses,
    update_sample_from_response,
)


class _ForwardProfiler:
    """Measure top-level model forwards without synchronizing each call."""

    def __init__(self, accelerator):
        self.event_type = accelerator.Event
        self.use_host_timer = accelerator.use_host_timers()
        # Host intervals are only reliable when there is no asynchronous device
        # work, as on CPU. MPS exposes events but cannot time them, so its host
        # intervals would measure enqueue latency rather than model execution.
        self.is_available = not self.use_host_timer or self.event_type is None
        self.active_start = None
        self.measurements = []

    def register(self, module):
        if not self.is_available:
            return ()
        pre_handle = module.register_forward_pre_hook(self._start_forward)
        post_handle = module.register_forward_hook(self._end_forward)
        return pre_handle, post_handle

    def get_stage_times(self):
        if not self.measurements:
            return None, None, 0

        durations = [self._elapsed_time(start, end) for start, end in self.measurements]
        return durations[0], sum(durations[1:]), len(durations) - 1

    def _start_forward(self, _module, _args):
        self.active_start = self._record_time()

    def _end_forward(self, _module, _args, _output):
        end = self._record_time()
        self.measurements.append((self.active_start, end))
        self.active_start = None

    def _record_time(self):
        if self.use_host_timer:
            return time.perf_counter()
        event = self.event_type(enable_timing=True)
        event.record()
        return event

    def _elapsed_time(self, start, end):
        if self.use_host_timer:
            return (end - start) * 1000.0
        return start.elapsed_time(end)


def _sampled_log_prob_recorder(temperature: float = 1.0):
    """Return ``(processor, finish)`` that record each sampled token's log-probability."""
    steps = []
    previous = []

    def processor(input_ids, scores):
        if previous:
            steps.append(previous.pop().gather(1, input_ids[:, -1:]))
        previous.append(torch.log_softmax(scores.float() / temperature, dim=-1))
        return scores

    def finish(last_tokens):
        steps.append(previous.pop().gather(1, last_tokens))
        return torch.cat(steps, dim=1)

    return processor, finish


def _chosen_log_probs(logits, tokens, temperature):
    """Log-probability of ``tokens`` ``[B, 1]`` under ``softmax(logits / temperature)``; greedy uses the raw logits."""
    scores = logits.float()
    if temperature > 0.0:
        scores = scores / temperature
    return torch.log_softmax(scores, dim=-1).gather(1, tokens)


@dataclass
class _PromptRow:
    """One left-padded prompt."""
    prompt_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor


@dataclass
class HybridEngineRolloutConfig:
    """Configuration for HybridEngineRollout.

    ``align_decode_fronts`` is opt-in. When false, continuous batching keeps
    the existing equal-width padded layout. Requests may still have different
    effective lengths in their attention masks, but their physical decode
    fronts are not right-aligned or compacted. When true, the rollout uses
    effective prompt lengths to align physical decode fronts.

    Cache trimming is opt-in because it copies the live cache columns after
    each reclaim. Without it, the rollout trims only when the cache would
    otherwise exhaust and a dead prefix can be reclaimed.
    """
    use_graph_capture: bool = False
    enable_profiling: bool = False
    use_shared_prefill: bool = False
    align_decode_fronts: bool = False
    enable_cache_trimming: bool = False
    multi_turn: bool = False
    continuous_cache_capacity: Optional[int] = None
    max_turns: Optional[int] = None
    # Most prompt + response tokens a multi-turn trajectory may hold; None means no cap.
    max_context_len: Optional[int] = None
    # Extra arguments for ``tokenizer.apply_chat_template``, e.g. {"enable_thinking": False} for Qwen3.
    chat_template_kwargs: dict = field(default_factory=dict)
    # Format of the model's tool calls, a key of TOOL_CALL_PARSERS: "hermes" (Qwen2.5, Qwen3) or
    # "qwen3_coder" (Qwen3-Coder, Qwen3.5).
    tool_call_parser: str = "hermes"

    def __post_init__(self) -> None:
        if self.multi_turn and not self.max_turns:
            raise ValueError("Max turns not specified when using multi-turn rollout")
        if self.tool_call_parser not in TOOL_CALL_PARSERS:
            raise ValueError(f"unknown tool_call_parser {self.tool_call_parser!r}; "
                             f"choose from {', '.join(TOOL_CALL_PARSERS)}")


@dataclass
class _ContinuousGraphState:
    signature: tuple
    cache: object
    static_input: torch.Tensor
    static_attention: torch.Tensor
    static_causal_attention: torch.Tensor
    static_position_ids: torch.Tensor
    static_write_positions: torch.Tensor
    static_cache_position: torch.Tensor
    graph: object
    static_logits: torch.Tensor


class HybridEngineRollout(RolloutEngine):
    """Rollout engine using DeepSpeed hybrid engine.

    Args:
        engine: DeepSpeed engine wrapping the model.
        tokenizer: HuggingFace tokenizer (must have pad_token_id or eos_token_id).
        cfg: Optional HybridEngineRolloutConfig.
        tools: Optional ToolSet the model may call in multi-turn rollout.
    """

    def __init__(self, engine, tokenizer, cfg=None, tools: Optional[ToolSet] = None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.tools = tools if tools is not None else ToolSet([])
        self.use_graph_capture = getattr(cfg, 'use_graph_capture', False) if cfg else False
        self.enable_profiling = getattr(cfg, 'enable_profiling', False) if cfg else False
        self.use_shared_prefill = getattr(cfg, 'use_shared_prefill', False) if cfg else False
        self.align_decode_fronts = getattr(cfg, 'align_decode_fronts', False) if cfg else False
        self.enable_cache_trimming = getattr(cfg, 'enable_cache_trimming', False) if cfg else False
        self.continuous_cache_capacity = getattr(cfg, 'continuous_cache_capacity', None) if cfg else None
        self.multi_turn = getattr(cfg, 'multi_turn', False) if cfg else False
        self.max_turns = getattr(cfg, 'max_turns', None) if cfg else None
        self.max_context_len = getattr(cfg, 'max_context_len', None) if cfg else None
        self.chat_template_kwargs = (getattr(cfg, 'chat_template_kwargs', None) if cfg else None) or {}
        self.parse_tool_calls = TOOL_CALL_PARSERS[getattr(cfg, 'tool_call_parser', 'hermes') if cfg else 'hermes']
        self._last_profile = None
        self._continuous_graph_state = None
        self._last_continuous_stats = None

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        if self.multi_turn:
            return self._generate_multi_turn(request, sampling)
        return self._generate_single_turn(request, sampling)

    @torch.no_grad()
    def _generate_single_turn(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        samples = request.samples
        if not samples:
            return RolloutBatch(samples=[])
        self._check_sampling(sampling)
        if sampling.max_new_tokens_per_turn is not None:
            max_new_tokens = min(sampling.max_new_tokens, sampling.max_new_tokens_per_turn)
            sampling = replace(sampling, max_new_tokens=max_new_tokens, max_new_tokens_per_turn=None)
        if sampling.continuous_batch_size is not None:
            return self._generate_continuous(samples, sampling, sampling.continuous_batch_size)
        device = get_accelerator().current_device_name()
        n = sampling.n_samples_per_prompt
        pad_token_id = self.tokenizer.pad_token_id
        module = self.engine.module

        if self.enable_profiling:
            accelerator = get_accelerator()
            accelerator.synchronize()
            profile_start = time.perf_counter()

        samples = render_prompts(samples, self.tokenizer, self.tools.specs, pad_token_id, self.chat_template_kwargs)
        # Copies of a prompt are adjacent, the layout shared prefill expects.
        trajectories = expand_samples(samples, n)
        prompt_ids, prompt_attn = left_pad_samples(trajectories, pad_token_id)
        prompt_len = prompt_ids.shape[1]
        prompt_ids = prompt_ids.to(device)
        prompt_attn = prompt_attn.to(device)

        if self.enable_profiling:
            accelerator.synchronize()
            expansion_end = time.perf_counter()

        forward_profiler = None
        forward_profile_handles = []
        if self.enable_profiling and not self.use_graph_capture:
            forward_profiler = _ForwardProfiler(accelerator)
            forward_profile_handles = forward_profiler.register(module)

        shared_prefill_handles = []
        try:
            if self.use_shared_prefill and n > 1:
                if self.use_graph_capture:
                    raise RuntimeError("Shared prefill does not support CUDA graph capture")
                self.engine.prepare_shared_prefill(len(samples), n, prompt_len)
                shared_prefill_handles = self._register_shared_prefill_hooks(module, len(samples), n)
            output_ids, log_probs = self._generate_tokens(prompt_ids, prompt_attn, sampling.max_new_tokens, sampling,
                                                          pad_token_id)
        finally:
            for handle in shared_prefill_handles:
                handle.remove()
            for handle in forward_profile_handles:
                handle.remove()

        if self.enable_profiling:
            accelerator.synchronize()
            generation_end = time.perf_counter()

        # Keep each response up to its first EOS: under ZeRO-3, generation
        # ran past it, and rows that stopped early are padded either way.
        output_ids, response_attn = self._pad_after_eos(
            output_ids,
            response_start=prompt_len,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
        )
        responses = masked_lists(output_ids[:, prompt_len:], response_attn)
        response_log_probs = masked_lists(log_probs, response_attn) if sampling.return_log_probs else None

        for row, sample in enumerate(trajectories):
            response = responses[row]
            # A response that did not reach EOS was cut by max_new_tokens.
            if response and response[-1] == self.tokenizer.eos_token_id:
                sample.status = RolloutRecord.Status.COMPLETED
            else:
                sample.status = RolloutRecord.Status.TRUNCATED
            row_log_probs = response_log_probs[row] if response_log_probs is not None else None
            update_sample_from_response(sample, response, row_log_probs, [], self.tokenizer)

        if self.enable_profiling:
            accelerator.synchronize()
            post_processing_end = time.perf_counter()
            prompt_expansion_ms = (expansion_end - profile_start) * 1000.0
            generation_ms = (generation_end - expansion_end) * 1000.0
            post_processing_ms = (post_processing_end - generation_end) * 1000.0
            total_ms = (post_processing_end - profile_start) * 1000.0
            prefill_forward_ms = None
            decode_forward_ms = None
            num_decode_forwards = 0
            if forward_profiler is not None:
                prefill_forward_ms, decode_forward_ms, num_decode_forwards = forward_profiler.get_stage_times()
            generation_overhead_ms = generation_ms
            if prefill_forward_ms is not None:
                generation_overhead_ms -= prefill_forward_ms + decode_forward_ms
                generation_overhead_ms = max(0.0, generation_overhead_ms)
            response_length = int(output_ids.shape[1] - prompt_len)
            num_generated_tokens = int(output_ids.shape[0] * response_length)
            tokens_per_second = 0.0
            if total_ms > 0.0:
                tokens_per_second = num_generated_tokens / (total_ms / 1000.0)
            self._last_profile = {
                "prompt_expansion_ms": prompt_expansion_ms,
                "generation_ms": generation_ms,
                "prefill_forward_ms": prefill_forward_ms,
                "decode_forward_ms": decode_forward_ms,
                "generation_overhead_ms": generation_overhead_ms,
                "num_decode_forwards": num_decode_forwards,
                "post_processing_ms": post_processing_ms,
                "total_ms": total_ms,
                "num_generated_tokens": num_generated_tokens,
                "tokens_per_second": tokens_per_second,
                "batch_size": len(samples),
                "num_samples_per_prompt": n,
                "prompt_length": prompt_len,
                "response_length": response_length,
            }

        return RolloutBatch(samples=trajectories)

    @torch.no_grad()
    def _generate_multi_turn(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        if sampling.continuous_batch_size is not None:
            raise ValueError("Continuous batching is not supported")

        samples = request.samples
        if not samples:
            return RolloutBatch(samples=[])
        self._check_sampling(sampling)
        device = get_accelerator().current_device_name()
        n = sampling.n_samples_per_prompt
        max_new_tokens = sampling.max_new_tokens
        return_log_probs = sampling.return_log_probs
        pad_token_id = self.tokenizer.pad_token_id
        module = self.engine.module

        if self.enable_profiling:
            accelerator = get_accelerator()
            accelerator.synchronize()
            call_start = time.perf_counter()
            # Summed over turns, so the profile covers the whole multi-turn rollout.
            profile = {
                "prompt_expansion_ms": 0.0,
                "generation_ms": 0.0,
                "prefill_forward_ms": 0.0,
                "decode_forward_ms": 0.0,
                "num_decode_forwards": 0,
                "tool_ms": 0.0,
                "post_processing_ms": 0.0,
                "num_generated_tokens": 0,
                "num_turns": 0,
            }

        samples = render_prompts(samples, self.tokenizer, self.tools.specs, pad_token_id, self.chat_template_kwargs)
        trajectories = expand_samples(samples, n)
        generated = [0] * len(trajectories)
        active = list(range(len(trajectories)))

        for _turn in range(self.max_turns):
            # Each trajectory may generate up to its remaining token budget, the
            # per-turn cap and the room left in its context; one with no room
            # stops before generating.
            room = {}
            for i in active:
                room[i] = max_new_tokens - generated[i]
                if sampling.max_new_tokens_per_turn is not None:
                    room[i] = min(room[i], sampling.max_new_tokens_per_turn)
                if self.max_context_len is not None:
                    room[i] = min(room[i], self.max_context_len - sum(trajectories[i].attention_mask))
                if room[i] <= 0:
                    trajectories[i].status = RolloutRecord.Status.TRUNCATED
            active = [i for i in active if room[i] > 0]
            if not active:
                break
            turn_samples = [trajectories[i] for i in active]
            budgets = [room[i] for i in active]

            if self.enable_profiling:
                accelerator.synchronize()
                turn_start = time.perf_counter()

            prompt_ids, prompt_attn = left_pad_samples(turn_samples, pad_token_id)
            prompt_len = prompt_ids.shape[1]
            prompt_ids = prompt_ids.to(device)
            prompt_attn = prompt_attn.to(device)

            if self.enable_profiling:
                accelerator.synchronize()
                expansion_end = time.perf_counter()

            forward_profiler = None
            forward_profile_handles = []
            if self.enable_profiling and not self.use_graph_capture:
                forward_profiler = _ForwardProfiler(accelerator)
                forward_profile_handles = forward_profiler.register(module)

            # One generate call serves all active trajectories, so it runs to the
            # largest remaining budget; each row is cut back to its own below.
            turn_max_new_tokens = max(budgets)
            try:
                output_ids, log_probs = self._generate_tokens(prompt_ids, prompt_attn, turn_max_new_tokens, sampling,
                                                              pad_token_id)
            finally:
                for handle in forward_profile_handles:
                    handle.remove()

            if self.enable_profiling:
                accelerator.synchronize()
                generation_end = time.perf_counter()

            # Keep each response up to its first EOS: under ZeRO-3, generation
            # ran past it, and rows that stopped early are padded either way.
            output_ids, response_attn = self._pad_after_eos(
                output_ids,
                response_start=prompt_len,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )
            responses = masked_lists(output_ids[:, prompt_len:], response_attn)
            response_log_probs = masked_lists(log_probs, response_attn) if return_log_probs else None

            # A trajectory continues only if it has budget left and asked for a
            # tool. The last turn runs no tools: no later turn would read them.
            turn_responses = []
            turn_log_probs = []
            turn_calls = {}
            for row, i in enumerate(active):
                response = responses[row][:budgets[row]]
                turn_responses.append(response)
                turn_log_probs.append(response_log_probs[row][:budgets[row]] if return_log_probs else None)
                generated[i] += len(response)
                # A turn that did not reach EOS was cut by the token or context budget.
                if not response or response[-1] != self.tokenizer.eos_token_id:
                    trajectories[i].status = RolloutRecord.Status.TRUNCATED
                    continue
                # Keep special tokens when decoding: tool-call tags can be special tokens.
                calls = self.parse_tool_calls(self.tokenizer.decode(response), self.tools)
                if not calls:
                    trajectories[i].status = RolloutRecord.Status.COMPLETED
                    continue
                # It asked for a tool but has no budget or turn left to read the result.
                if generated[i] >= max_new_tokens or _turn == self.max_turns - 1:
                    trajectories[i].status = RolloutRecord.Status.TRUNCATED
                    continue
                turn_calls[row] = calls
            tool_start = time.perf_counter()
            tool_outputs = asyncio.run(self._run_tool_calls(turn_calls))
            tool_ms = (time.perf_counter() - tool_start) * 1000.0

            tool_active = []
            for row, i in enumerate(active):
                tool_response = []
                if row in turn_calls:
                    tool_messages = [{
                        "role": "tool",
                        "name": call.name or "unknown",
                        "content": output
                    } for call, output in zip(turn_calls[row], tool_outputs[row])]
                    tool_response = tokenize_tool_responses(tool_messages, self.tokenizer, self.chat_template_kwargs)
                    tool_active.append(i)
                update_sample_from_response(trajectories[i], turn_responses[row], turn_log_probs[row], tool_response,
                                            self.tokenizer)
            active = tool_active

            if self.enable_profiling:
                accelerator.synchronize()
                turn_end = time.perf_counter()
                profile["prompt_expansion_ms"] += (expansion_end - turn_start) * 1000.0
                profile["generation_ms"] += (generation_end - expansion_end) * 1000.0
                profile["tool_ms"] += tool_ms
                profile["post_processing_ms"] += (turn_end - generation_end) * 1000.0 - tool_ms
                profile["num_generated_tokens"] += int(output_ids.shape[0] * (output_ids.shape[1] - prompt_len))
                profile["num_turns"] += 1
                prefill_ms, decode_ms, num_decodes = (None, None, 0)
                if forward_profiler is not None:
                    prefill_ms, decode_ms, num_decodes = forward_profiler.get_stage_times()
                # Forward times are reported only when every turn measured them.
                if prefill_ms is None or profile["prefill_forward_ms"] is None:
                    profile["prefill_forward_ms"] = None
                    profile["decode_forward_ms"] = None
                else:
                    profile["prefill_forward_ms"] += prefill_ms
                    profile["decode_forward_ms"] += decode_ms
                profile["num_decode_forwards"] += num_decodes

        if self.enable_profiling:
            accelerator.synchronize()
            total_ms = (time.perf_counter() - call_start) * 1000.0
            generation_overhead_ms = profile["generation_ms"]
            if profile["prefill_forward_ms"] is not None:
                forward_ms = profile["prefill_forward_ms"] + profile["decode_forward_ms"]
                generation_overhead_ms = max(0.0, generation_overhead_ms - forward_ms)
            tokens_per_second = 0.0
            if total_ms > 0.0:
                tokens_per_second = profile["num_generated_tokens"] / (total_ms / 1000.0)
            profile.update({
                "generation_overhead_ms": generation_overhead_ms,
                "total_ms": total_ms,
                "tokens_per_second": tokens_per_second,
                "batch_size": len(trajectories),
                "num_samples_per_prompt": n,
                "prompt_length": len(samples[0].tokens),
                "response_length": max(sample.response_length for sample in trajectories),
            })
            self._last_profile = profile

        return RolloutBatch(samples=trajectories)

    def _check_sampling(self, sampling: SamplingConfig) -> None:
        if sampling.max_new_tokens_per_turn is not None and sampling.max_new_tokens_per_turn <= 0:
            raise ValueError("max_new_tokens_per_turn must be positive")
        return_log_probs = sampling.return_log_probs
        if return_log_probs and sampling.temperature > 0.0 and (sampling.top_p < 1.0 or sampling.top_k > 0):
            raise ValueError("return_log_probs requires top_p=1 and top_k<=0 when sampling")
        if return_log_probs and sampling.repetition_penalty != 1.0:
            # The penalty depends on the context and is applied before the
            # log-probs are recorded; a training forward pass cannot reproduce it.
            raise ValueError("return_log_probs requires repetition_penalty=1.0")

    def _generate_tokens(self, prompt_ids, prompt_attn, max_new_tokens, sampling, pad_token_id):
        module = self.engine.module
        if self.use_graph_capture:
            return self._generate_graph(prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module,
                                        prompt_ids.device, sampling)

        # ZeRO-3 gathers parameters during each decode forward, so every
        # data-parallel rank must run the same number of decode steps and
        # generation cannot stop at EOS. Other stages stop once every row ends.
        stop_at_eos = not self.engine.zero_optimization_partition_weights()
        generate_kwargs = {
            "attention_mask": prompt_attn,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id if stop_at_eos else None,
            "pad_token_id": pad_token_id,
            **self._hf_sampling_kwargs(sampling),
        }
        finish_log_probs = None
        if sampling.return_log_probs:
            from transformers import LogitsProcessorList
            # Greedy decoding picks from the raw logits, i.e. temperature 1.
            temperature = sampling.temperature if sampling.temperature > 0.0 else 1.0
            processor, finish_log_probs = _sampled_log_prob_recorder(temperature)
            generate_kwargs["logits_processor"] = LogitsProcessorList([processor])
        output_ids = module.generate(prompt_ids, **generate_kwargs)
        log_probs = finish_log_probs(output_ids[:, -1:]) if finish_log_probs is not None else None
        return output_ids, log_probs

    async def _run_tool_calls(self, calls_by_row):
        """Run every row's calls concurrently; returns each row's outputs in call order."""
        rows = list(calls_by_row)
        outputs = await asyncio.gather(*(asyncio.gather(*(self.tools.run(call) for call in calls_by_row[row]))
                                         for row in rows))
        return dict(zip(rows, outputs))

    @torch.no_grad()
    def _generate_continuous(self, samples, sampling, max_batch_size):
        """Generate independent responses for ``samples`` in an experimental continuous batch.

        Continuous batching currently requires one sample per prompt.
        Completed rows retire immediately and pending prompts prefill into the
        released rows before the next decode step.
        """
        self._validate_continuous_inputs(sampling, max_batch_size)
        pad_token_id = self.tokenizer.pad_token_id
        samples = render_prompts(samples, self.tokenizer, self.tools.specs, pad_token_id, self.chat_template_kwargs)
        trajectories = expand_samples(samples, 1)
        if not all(any(sample.attention_mask) for sample in trajectories):
            raise ValueError("continuous batching requires at least one prompt token per request")
        prompt_ids, prompt_attention_mask = left_pad_samples(trajectories, pad_token_id)
        module = self.engine.module
        device = next(module.parameters()).device
        prompt_ids = prompt_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)
        requests = tuple(
            _PromptRow(prompt_ids[index:index + 1], prompt_attention_mask[index:index + 1])
            for index in range(len(trajectories)))
        if self.use_graph_capture:
            return self._generate_continuous_graph(trajectories, requests, sampling, max_batch_size)
        select_next_tokens = self._token_selector(sampling)

        profile = self._start_continuous_profile() if self.enable_profiling else None
        profile_accelerator = profile["accelerator"] if profile is not None else None
        profile_start = profile["start"] if profile is not None else None
        prompt_lengths = {
            request_id: (int(request.prompt_attention_mask.sum().item())
                         if self.align_decode_fronts else request.prompt_ids.shape[1])
            for request_id, request in enumerate(requests)
        }
        prompt_len = max(prompt_lengths.values())
        max_positions = getattr(module.config, "max_position_embeddings", None)
        if max_positions is not None:
            if any(length + sampling.max_new_tokens > max_positions for length in prompt_lengths.values()):
                raise ValueError("continuous batching request exceeds the model maximum position embeddings")
        if self.continuous_cache_capacity is not None:
            if self.continuous_cache_capacity < prompt_len:
                raise ValueError("continuous_cache_capacity must be at least the padded prompt length")
            max_cache_len = self.continuous_cache_capacity
        else:
            max_cache_len = (prompt_len + sampling.max_new_tokens if self.align_decode_fronts else
                             self._estimate_continuous_cache_len(prompt_len, [sampling.max_new_tokens] *
                                                                 len(requests), max_batch_size))
        if max_positions is not None and max_cache_len > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")
        if getattr(module, "_supports_cache_class", None) is False:
            raise ValueError("continuous batching requires a model with cache-class support; use the default "
                             "generate() path or upgrade transformers")

        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        model_dtype = next(module.parameters()).dtype

        scheduler_start = self._profile_start(profile)
        scheduler = ContinuousBatchScheduler(max_batch_size, sampling.max_new_tokens)
        request_by_id = {}
        responses = {}
        # Each response token's log-prob under softmax(logits / temperature); greedy uses the raw logits.
        response_log_probs = {}
        request_order = (sorted(range(len(requests)), key=lambda request_id: -prompt_lengths[request_id])
                         if self.align_decode_fronts else range(len(requests)))
        for request_id in request_order:
            request = requests[request_id]
            scheduler.submit(ContinuousBatchRequest(request_id))
            request_by_id[request_id] = request
            responses[request_id] = []
            response_log_probs[request_id] = []
        self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        cache_start = self._profile_start(profile)
        cache = DeepSpeedStaticCache(
            module.config,
            batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=model_dtype,
        )
        write_positions = torch.full((max_batch_size, ), -1, dtype=torch.long, device=device)
        cache.set_write_position(write_positions)
        attention_mask = torch.zeros((max_batch_size, max_cache_len), dtype=torch.long, device=device)
        stats = {
            "cache_capacity":
            max_cache_len,
            "peak_cache_length":
            prompt_len,
            "cache_memory_bytes":
            sum(layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
                for layer in cache.layers) + attention_mask.numel() * attention_mask.element_size() +
            write_positions.numel() * write_positions.element_size(),
            "trim_count":
            0,
            "trimmed_columns":
            0,
            "trim_latency_ms":
            0.0,
            "decode_steps":
            0,
        }
        next_tokens = {}
        next_log_probs = {}
        logical_positions = dict(prompt_lengths)
        cache_position = prompt_len
        span_starts = [0] * max_batch_size

        self._profile_end(profile, "cache_management_overhead_ms", cache_start)
        scheduler_start = self._profile_start(profile)
        update = scheduler.schedule()
        self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        while update.active:
            if profile is not None:
                profile["active_batch_sizes"].append(len(update.active))
            cache_start = self._profile_start(profile)
            keep_slots = torch.tensor(update.keep_slots, dtype=torch.long, device=device)
            survivor_count = keep_slots.numel()
            if survivor_count:
                if update.retired:
                    cache.compact(keep_slots)
                    identity = torch.arange(survivor_count, dtype=torch.long, device=device)
                    if not torch.equal(keep_slots, identity):
                        survivor_attention = attention_mask.index_select(0, keep_slots).clone()
                        attention_mask[:survivor_count].copy_(survivor_attention)
                    attention_mask[survivor_count:].zero_()
                    if self.align_decode_fronts:
                        span_starts[:survivor_count] = [span_starts[index] for index in update.keep_slots]
                        span_starts[survivor_count:] = [0] * (max_batch_size - survivor_count)

                dead_prefix = 0
                if self.enable_cache_trimming:
                    if self.align_decode_fronts:
                        dead_prefix = min(span_starts[:survivor_count])
                    else:
                        trim_threshold = max(1, prompt_len)
                        dead_prefix = self._continuous_dead_prefix(attention_mask, survivor_count)
                        if dead_prefix < trim_threshold and cache_position < max_cache_len - trim_threshold:
                            dead_prefix = 0
                elif cache_position >= max_cache_len:
                    if self.align_decode_fronts:
                        dead_prefix = min(span_starts[:survivor_count])
                    else:
                        dead_prefix = self._continuous_dead_prefix(attention_mask, survivor_count)
                if dead_prefix:
                    if update.admitted:
                        longest_admitted = max(prompt_lengths[request.request_id] for request in update.admitted)
                        dead_prefix = min(dead_prefix, max(0, cache_position - longest_admitted))
                    trim_start = None
                    if profile_accelerator is not None:
                        profile_accelerator.synchronize()
                        trim_start = time.perf_counter()
                    moved_bytes = sum((layer.keys[:, :, dead_prefix:, :].numel() * layer.keys.element_size() +
                                       layer.values[:, :, dead_prefix:, :].numel() * layer.values.element_size())
                                      for layer in cache.layers)
                    moved_bytes += attention_mask[:, dead_prefix:].numel() * attention_mask.element_size()
                    cache.trim_left(dead_prefix)
                    remaining_columns = max_cache_len - dead_prefix
                    if remaining_columns:
                        attention_mask[:, :remaining_columns].copy_(attention_mask[:, dead_prefix:].clone())
                    attention_mask[:, remaining_columns:].zero_()
                    write_positions[:survivor_count].sub_(dead_prefix)
                    cache_position -= dead_prefix
                    stats["trim_count"] += 1
                    stats["trimmed_columns"] += dead_prefix
                    stats["trim_bytes_moved"] = stats.get("trim_bytes_moved", 0) + moved_bytes
                    if profile_accelerator is not None:
                        profile_accelerator.synchronize()
                        stats["trim_latency_ms"] += (time.perf_counter() - trim_start) * 1000.0
                    if self.align_decode_fronts:
                        span_starts[:survivor_count] = [start - dead_prefix for start in span_starts[:survivor_count]]
            else:
                cache.reset()
                write_positions.fill_(-1)
                attention_mask.zero_()
                cache_position = prompt_len
                span_starts = [0] * max_batch_size

            self._profile_end(profile, "cache_management_overhead_ms", cache_start)
            if survivor_count and cache_position >= max_cache_len:
                raise self._continuous_cache_exhaustion_error(max_cache_len)
            admitted_tokens, admitted_log_probs = self._continuous_prefill(
                module,
                StaticCache,
                cache,
                update,
                request_by_id,
                attention_mask,
                write_positions,
                cache_position,
                prompt_lengths,
                model_dtype,
                device,
                sampling,
                self.align_decode_fronts,
                profile,
            )
            if self.align_decode_fronts:
                for admitted, target_row in zip(update.admitted, update.admitted_slots):
                    span_starts[target_row] = cache_position - prompt_lengths[admitted.request_id]

            decoded_tokens = {}
            decoded_log_probs = {}
            if survivor_count:
                survivor_ids = update.active_ids[:survivor_count]
                decode_input = torch.cat([next_tokens[request_id] for request_id in survivor_ids], dim=0)
                if cache_position >= max_cache_len:
                    raise self._continuous_cache_exhaustion_error(max_cache_len)
                write_positions[:survivor_count].fill_(cache_position)
                if self.align_decode_fronts:
                    position_ids = torch.tensor([logical_positions[request_id] for request_id in survivor_ids],
                                                dtype=torch.long,
                                                device=device).unsqueeze(1)
                else:
                    position_ids = attention_mask[:survivor_count, :cache_position].sum(dim=1, keepdim=True)
                attention_mask[:survivor_count, cache_position] = 1
                decode_start = self._profile_start(profile)
                output = self._call_model(
                    module,
                    decode_input,
                    attention_mask=attention_mask[:survivor_count],
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([cache_position], dtype=torch.long, device=device),
                    position_ids=position_ids,
                )
                self._profile_end(profile, "decode_forward_ms", decode_start, count="num_decode_forwards")
                decoded = self._continuous_next_tokens(
                    output.logits[:, -1, :],
                    survivor_ids,
                    request_by_id,
                    responses,
                    sampling.repetition_penalty,
                    select_next_tokens,
                )
                stats["decode_steps"] += 1
                decoded_tokens = dict(zip(survivor_ids, decoded.split(1, dim=0)))
                decoded_log_prob_rows = _chosen_log_probs(output.logits[:, -1, :], decoded,
                                                          sampling.temperature).split(1, dim=0)
                decoded_log_probs = dict(zip(survivor_ids, decoded_log_prob_rows))
                for request_id in survivor_ids:
                    logical_positions[request_id] += 1

            next_tokens = decoded_tokens | admitted_tokens
            next_log_probs = decoded_log_probs | admitted_log_probs
            finished_ids = []
            for request_id in update.active_ids:
                token = next_tokens[request_id]
                responses[request_id].append(token)
                response_log_probs[request_id].append(next_log_probs[request_id])
                if self._is_eos(token):
                    finished_ids.append(request_id)

            if survivor_count:
                cache_position += 1
            stats["peak_cache_length"] = max(stats["peak_cache_length"], cache_position)
            scheduler_start = self._profile_start(profile)
            update = scheduler.advance(finished_ids)
            self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        generation_end = self._profile_start(profile)
        post_processing_start = generation_end
        output = self._build_continuous_batch(trajectories, responses, response_log_probs, sampling.return_log_probs)
        post_processing_end = self._profile_end(profile, None, post_processing_start)
        if profile_accelerator is not None:
            profile_accelerator.synchronize()
            total_ms = (time.perf_counter() - profile_start) * 1000.0
            generated_tokens = sum(len(response) for response in responses.values())
            stats["end_to_end_ms"] = total_ms
            stats["tokens_per_second"] = generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0
        else:
            stats["end_to_end_ms"] = None
            stats["tokens_per_second"] = None
        stats.setdefault("trim_bytes_moved", 0)
        stats["trim_frequency"] = (stats["trim_count"] / stats["decode_steps"] if stats["decode_steps"] else 0.0)
        self._last_continuous_stats = stats
        if profile is not None:
            self._finish_continuous_profile(profile, len(trajectories), responses, max_batch_size, prompt_len,
                                            generation_end, post_processing_end)
        return output

    @torch.no_grad()
    def _generate_continuous_graph(self, trajectories, requests, sampling, max_batch_size):
        """Run continuous decoding with a fixed-capacity CUDA graph.

        Scheduler slots stay attached to physical cache rows for their whole
        lifetime. Retired rows are masked until a new prompt is prefetched into
        them, so the captured decode always receives the same batch shape and
        static tensor addresses.
        """
        module = self.engine.module
        prompt_len = requests[0].prompt_ids.shape[1]
        self._validate_continuous_graph_options(prompt_len, sampling)
        if getattr(module, "_supports_cache_class", None) is False:
            raise ValueError("continuous batching requires a model with cache-class support; use the default "
                             "generate() path or upgrade transformers")
        if getattr(module.config, "_attn_implementation", None) == "flash_attention_2":
            raise ValueError("continuous CUDA graph capture does not support flash_attention_2")
        device = requests[0].prompt_ids.device
        if device.type != "cuda":
            raise ValueError("continuous CUDA graph capture requires CUDA rollout inputs")

        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        profile = self._start_continuous_profile() if self.enable_profiling else None
        profile_accelerator = profile["accelerator"] if profile is not None else None
        profile_start = profile["start"] if profile is not None else None
        minimum_cache_len = prompt_len + sampling.max_new_tokens
        max_cache_len = (self.continuous_cache_capacity
                         if self.continuous_cache_capacity is not None else minimum_cache_len)
        if max_cache_len < minimum_cache_len:
            raise ValueError("continuous_cache_capacity must fit the prompt and all generated tokens")
        max_positions = getattr(module.config, "max_position_embeddings", None)
        if max_positions is not None and max_cache_len > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")
        model_dtype = next(module.parameters()).dtype
        select_next_tokens = self._token_selector(sampling)
        scheduler = ContinuousBatchScheduler(max_batch_size, sampling.max_new_tokens)
        prompt_lengths = {request_id: prompt_len for request_id in range(len(requests))}
        request_by_id = {}
        responses = {}
        # Each response token's log-prob under softmax(logits / temperature); greedy uses the raw logits.
        response_log_probs = {}
        for request_id, request in enumerate(requests):
            scheduler.submit(ContinuousBatchRequest(request_id))
            request_by_id[request_id] = request
            responses[request_id] = []
            response_log_probs[request_id] = []

        state = self._get_continuous_graph_state(module, max_batch_size, prompt_len, max_cache_len, device,
                                                 model_dtype, DeepSpeedStaticCache)
        cache = state.cache
        static_input = state.static_input
        static_attention = state.static_attention
        static_causal_attention = state.static_causal_attention
        static_position_ids = state.static_position_ids
        static_write_positions = state.static_write_positions
        graph = state.graph
        static_logits = state.static_logits
        stats = {
            "cache_capacity":
            max_cache_len,
            "peak_cache_length":
            prompt_len,
            "cache_memory_bytes":
            sum(layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
                for layer in cache.layers) + static_input.numel() * static_input.element_size() +
            static_attention.numel() * static_attention.element_size() +
            static_causal_attention.numel() * static_causal_attention.element_size() +
            static_position_ids.numel() * static_position_ids.element_size() +
            static_write_positions.numel() * static_write_positions.element_size(),
            "trim_count":
            0,
            "trimmed_columns":
            0,
            "trim_latency_ms":
            0.0,
            "trim_bytes_moved":
            0,
            "decode_steps":
            0,
        }

        slot_by_request = {}
        write_positions = {}
        next_tokens = {}
        next_log_probs = {}
        update = scheduler.schedule()
        while update.active:
            if profile is not None:
                profile["active_batch_sizes"].append(len(update.active))

            for request_id in update.retired:
                static_attention[slot_by_request[request_id]].zero_()
                slot_by_request.pop(request_id, None)
                write_positions.pop(request_id, None)
                next_tokens.pop(request_id, None)
                next_log_probs.pop(request_id, None)

            free_slots = [slot for slot in range(max_batch_size) if slot not in slot_by_request.values()]
            admitted_ids = tuple(request.request_id for request in update.admitted)
            admitted_slots = []
            for request_id in admitted_ids:
                slot = free_slots.pop(0)
                slot_by_request[request_id] = slot
                write_positions[request_id] = prompt_len
                admitted_slots.append(slot)

            cache_start = self._profile_start(profile)
            if admitted_ids:
                admitted_tokens, admitted_log_probs = self._continuous_prefill(
                    module,
                    StaticCache,
                    cache,
                    update,
                    request_by_id,
                    static_attention,
                    static_write_positions,
                    prompt_len,
                    prompt_lengths,
                    model_dtype,
                    device,
                    sampling,
                    profile=profile,
                    target_slots=tuple(admitted_slots),
                )
                next_tokens.update(admitted_tokens)
                next_log_probs.update(admitted_log_probs)
            self._profile_end(profile, "cache_management_overhead_ms", cache_start)

            decode_ids = tuple(request_id for request_id in update.active_ids if request_id not in admitted_ids)
            static_input.zero_()
            static_position_ids.zero_()
            # Captured replay always executes every physical row. Keep non-decoding
            # rows away from their prefetched prompt KV until they become active.
            static_write_positions.fill_(max_cache_len - 1)
            for request_id in decode_ids:
                slot = slot_by_request[request_id]
                position = write_positions[request_id]
                static_input[slot:slot + 1].copy_(next_tokens[request_id])
                static_position_ids[slot, 0] = static_attention[slot, :position].sum()
                static_attention[slot, position] = 1
                static_write_positions[slot] = position

            decoded_tokens = {}
            if decode_ids:
                self._update_static_causal_attention(static_attention, static_causal_attention)
                decode_start = self._profile_start(profile)
                get_accelerator().replay_graph(graph)
                self._profile_end(profile, "decode_forward_ms", decode_start, count="num_decode_forwards")
                decode_logits = torch.cat(
                    [static_logits[slot_by_request[request_id], -1, :].unsqueeze(0) for request_id in decode_ids],
                    dim=0)
                decoded = self._continuous_next_tokens(
                    decode_logits,
                    decode_ids,
                    request_by_id,
                    responses,
                    sampling.repetition_penalty,
                    select_next_tokens,
                )
                decoded_log_prob_rows = _chosen_log_probs(decode_logits, decoded, sampling.temperature).split(1, dim=0)
                for request_id, token, log_prob in zip(decode_ids, decoded.split(1, dim=0), decoded_log_prob_rows):
                    decoded_tokens[request_id] = token
                    next_log_probs[request_id] = log_prob
                    write_positions[request_id] += 1
                stats["decode_steps"] += 1
                next_tokens.update(decoded_tokens)

            finished_ids = []
            for request_id in update.active_ids:
                token = next_tokens[request_id]
                responses[request_id].append(token)
                response_log_probs[request_id].append(next_log_probs[request_id])
                if self._is_eos(token):
                    finished_ids.append(request_id)

            scheduler_start = self._profile_start(profile)
            update = scheduler.advance(finished_ids)
            self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)
            if write_positions:
                stats["peak_cache_length"] = max(stats["peak_cache_length"], max(write_positions.values()))

        generation_end = self._profile_start(profile)
        output = self._build_continuous_batch(trajectories, responses, response_log_probs, sampling.return_log_probs)
        post_processing_end = self._profile_end(profile, None, generation_end)
        if profile_accelerator is not None:
            profile_accelerator.synchronize()
            total_ms = (time.perf_counter() - profile_start) * 1000.0
            generated_tokens = sum(len(response) for response in responses.values())
            stats["end_to_end_ms"] = total_ms
            stats["tokens_per_second"] = generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0
        else:
            stats["end_to_end_ms"] = None
            stats["tokens_per_second"] = None
        stats["trim_frequency"] = 0.0
        self._last_continuous_stats = stats
        if profile is not None:
            self._finish_continuous_profile(profile, len(trajectories), responses, max_batch_size, prompt_len,
                                            generation_end, post_processing_end)
        return output

    def _validate_continuous_graph_options(self, prompt_len, sampling):
        if self.align_decode_fronts:
            raise ValueError("continuous CUDA graph capture does not support aligned decode fronts")
        if self.enable_cache_trimming:
            raise ValueError("continuous CUDA graph capture does not support cache trimming")
        if (self.continuous_cache_capacity is not None
                and self.continuous_cache_capacity < prompt_len + sampling.max_new_tokens):
            raise ValueError("continuous_cache_capacity must fit the prompt and all generated tokens")

    def _get_continuous_graph_state(self, module, max_batch_size, prompt_len, max_cache_len, device, model_dtype,
                                    cache_type):
        signature = self._continuous_graph_signature(module, max_batch_size, prompt_len, max_cache_len, device,
                                                     model_dtype)
        state = self._continuous_graph_state
        if state is None or state.signature != signature:
            cache = cache_type(
                module.config,
                batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                device=device,
                dtype=model_dtype,
            )
            static_input = torch.zeros((max_batch_size, 1), dtype=torch.long, device=device)
            static_attention = torch.zeros((max_batch_size, max_cache_len), dtype=torch.long, device=device)
            static_causal_attention = torch.zeros((max_batch_size, 1, 1, max_cache_len),
                                                  dtype=model_dtype,
                                                  device=device)
            static_position_ids = torch.zeros((max_batch_size, 1), dtype=torch.long, device=device)
            static_write_positions = torch.zeros(max_batch_size, dtype=torch.long, device=device)
            cache.set_write_position(static_write_positions)
            # Transformers uses cache_position to build the causal mask. Per-row
            # validity is represented by static_attention, so a fixed upper bound
            # lets one captured graph serve requests at different decode lengths.
            static_cache_position = torch.tensor([max_cache_len - 1], dtype=torch.long, device=device)
            graph, static_logits = self._capture_continuous_graph(
                module,
                cache,
                static_input,
                static_causal_attention,
                static_write_positions,
                static_cache_position,
                static_position_ids,
            )
            state = _ContinuousGraphState(
                signature,
                cache,
                static_input,
                static_attention,
                static_causal_attention,
                static_position_ids,
                static_write_positions,
                static_cache_position,
                graph,
                static_logits,
            )
            self._continuous_graph_state = state

        state.cache.reset()
        state.static_input.zero_()
        state.static_attention.zero_()
        state.static_causal_attention.zero_()
        state.static_position_ids.zero_()
        state.static_write_positions.zero_()
        return state

    @staticmethod
    def _continuous_graph_signature(module, max_batch_size, prompt_len, max_cache_len, device, model_dtype):
        parameter_signature = tuple((parameter.data_ptr(), parameter._version) for parameter in module.parameters())
        return (id(module), device.type, device.index, model_dtype, max_batch_size, prompt_len, max_cache_len,
                parameter_signature)

    @staticmethod
    def _update_static_causal_attention(static_attention, static_causal_attention):
        static_causal_attention.copy_(static_attention[:, None, None, :])
        static_causal_attention.neg_().add_(1).mul_(torch.finfo(static_causal_attention.dtype).min)

    @staticmethod
    def _capture_continuous_graph(module, cache, static_input, static_attention, static_write_positions,
                                  static_cache_position, static_position_ids):
        """Capture one fixed-capacity decode forward and return its static logits."""
        accelerator = get_accelerator()
        saved_pre = dict(module._forward_pre_hooks)
        saved_post = dict(module._forward_hooks)
        module._forward_pre_hooks.clear()
        module._forward_hooks.clear()
        try:
            warmup_stream = accelerator.Stream()
            warmup_stream.wait_stream(accelerator.current_stream())
            with accelerator.stream(warmup_stream):
                for _ in range(3):
                    module(
                        static_input,
                        attention_mask=static_attention,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=static_cache_position,
                        position_ids=static_position_ids,
                    )
            accelerator.current_stream().wait_stream(warmup_stream)
            graph = accelerator.create_graph()
            with accelerator.capture_to_graph(graph):
                output = module(
                    static_input,
                    attention_mask=static_attention,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=static_cache_position,
                    position_ids=static_position_ids,
                )
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)
        return graph, output.logits

    @staticmethod
    def _continuous_cache_exhaustion_error(max_cache_len):
        return ValueError(
            f"continuous batching cache capacity ({max_cache_len}) exhausted before active requests retired; "
            "increase HybridEngineRolloutConfig.continuous_cache_capacity or enable cache trimming with "
            "HybridEngineRolloutConfig(enable_cache_trimming=True)")

    @staticmethod
    def _estimate_continuous_cache_len(prompt_len, max_new_tokens, max_batch_size):
        """Estimate the largest cache span before the scheduler becomes empty."""
        pending = list(max_new_tokens)
        active = []
        max_cache_len = prompt_len
        busy_span = 0

        while active or pending:
            if not active:
                active = pending[:max_batch_size]
                pending = pending[max_batch_size:]
                busy_span = 0

            busy_span += 1
            max_cache_len = max(max_cache_len, prompt_len + busy_span)
            survivors = [remaining - 1 for remaining in active if remaining > 1]
            free_slots = max_batch_size - len(survivors)
            admitted = pending[:free_slots]
            pending = pending[free_slots:]

            if survivors:
                active = survivors + admitted
                continue

            if admitted:
                active = admitted
                # All previous rows retired, so the implementation resets the cache
                # before prefilling the newly admitted requests.
                busy_span = 0
            else:
                active = []

        return max_cache_len

    def _validate_continuous_inputs(self, sampling, max_batch_size):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.use_shared_prefill:
            raise ValueError("continuous batching does not support shared prompt prefill")

        if sampling.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if sampling.n_samples_per_prompt != 1:
            raise ValueError("continuous batching currently supports one sample per prompt")

    @staticmethod
    def _continuous_dead_prefix(attention_mask, active_count):
        if active_count == 0:
            return 0
        occupied = attention_mask[:active_count].any(dim=0)
        if not occupied.any():
            return 0
        return int(occupied.to(dtype=torch.int32).argmax().item())

    def _continuous_prefill(self,
                            module,
                            static_cache_type,
                            cache,
                            update,
                            request_by_id,
                            attention_mask,
                            write_positions,
                            cache_position,
                            prompt_lengths,
                            model_dtype,
                            device,
                            sampling,
                            align_decode_fronts=False,
                            profile=None,
                            target_slots=None):
        if not update.admitted:
            return {}, {}

        if not align_decode_fronts:
            admitted_ids = tuple(request.request_id for request in update.admitted)
            if target_slots is None:
                target_slots = update.admitted_slots
            if len(target_slots) != len(admitted_ids):
                raise ValueError("continuous prefill needs one target slot per admitted request")
            prompt_ids = torch.cat([request_by_id[request_id].prompt_ids for request_id in admitted_ids], dim=0)
            prompt_attention = torch.cat(
                [request_by_id[request_id].prompt_attention_mask for request_id in admitted_ids], dim=0)
            prompt_len = prompt_ids.shape[1]
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_log_probs, prefill_cache = self._continuous_prefill_cache(
                module,
                static_cache_type,
                prompt_ids,
                prompt_attention,
                model_dtype,
                device,
                sampling,
            )
            self._profile_end(profile, "prefill_forward_ms", prefill_start, count="num_prefill_forwards")
            cache_copy_start = self._profile_start(profile)
            cache_start = cache_position - prompt_len
            if cache_start < 0:
                raise ValueError("continuous batching prompt does not fit behind the decode front")
            for layer_idx in range(self._cache_layer_count(prefill_cache)):
                prefill_keys, prefill_values = self._cache_layer_tensors(prefill_cache, layer_idx)
                target_layer = cache.layers[layer_idx]
                for source_row, target_row in enumerate(target_slots):
                    target_layer.keys[target_row, :, cache_start:cache_position].copy_(prefill_keys[source_row])
                    target_layer.values[target_row, :, cache_start:cache_position].copy_(prefill_values[source_row])
            for source_row, target_row in enumerate(target_slots):
                attention_mask[target_row, cache_start:cache_position].copy_(prompt_attention[source_row])
                write_positions[target_row] = cache_position
            self._profile_end(profile, "cache_management_overhead_ms", cache_copy_start)
            admitted_tokens = dict(zip(admitted_ids, prefill_tokens.split(1, dim=0)))
            admitted_log_probs = dict(zip(admitted_ids, prefill_log_probs.split(1, dim=0)))
            return admitted_tokens, admitted_log_probs

        admitted_tokens = {}
        admitted_log_probs = {}
        for admitted, target_row in zip(update.admitted, update.admitted_slots):
            request_id = admitted.request_id
            request = request_by_id[request_id]
            prompt_len = prompt_lengths[request_id]
            valid = request.prompt_attention_mask[0].bool()
            prompt_ids = request.prompt_ids[:, valid]
            prompt_attention = torch.ones_like(prompt_ids)
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_log_probs, prefill_cache = self._continuous_prefill_cache(
                module,
                static_cache_type,
                prompt_ids,
                prompt_attention,
                model_dtype,
                device,
                sampling,
            )
            self._profile_end(profile, "prefill_forward_ms", prefill_start, count="num_prefill_forwards")
            cache_copy_start = self._profile_start(profile)
            admitted_tokens[request_id] = prefill_tokens
            admitted_log_probs[request_id] = prefill_log_probs
            cache_start = cache_position - prompt_len
            if cache_start < 0:
                raise ValueError("continuous batching prompt does not fit behind the decode front")
            for layer_idx in range(self._cache_layer_count(prefill_cache)):
                prefill_keys, prefill_values = self._cache_layer_tensors(prefill_cache, layer_idx)
                target_layer = cache.layers[layer_idx]
                target_layer.keys[target_row, :, cache_start:cache_position].copy_(prefill_keys[0])
                target_layer.values[target_row, :, cache_start:cache_position].copy_(prefill_values[0])
            attention_mask[target_row, cache_start:cache_position].fill_(1)
            write_positions[target_row] = cache_position
            self._profile_end(profile, "cache_management_overhead_ms", cache_copy_start)
        return admitted_tokens, admitted_log_probs

    def _start_continuous_profile(self):
        accelerator = get_accelerator()
        accelerator.synchronize()
        return {
            "accelerator": accelerator,
            "start": time.perf_counter(),
            "prefill_forward_ms": 0.0,
            "decode_forward_ms": 0.0,
            "scheduler_overhead_ms": 0.0,
            "cache_management_overhead_ms": 0.0,
            "num_prefill_forwards": 0,
            "num_decode_forwards": 0,
            "active_batch_sizes": [],
        }

    @staticmethod
    def _profile_start(profile):
        if profile is None:
            return None
        profile["accelerator"].synchronize()
        return time.perf_counter()

    @staticmethod
    def _profile_end(profile, field, start, count=None):
        if profile is None or start is None:
            return None
        profile["accelerator"].synchronize()
        end = time.perf_counter()
        if field is not None:
            profile[field] += (end - start) * 1000.0
        if count is not None:
            profile[count] += 1
        return end

    def _finish_continuous_profile(self, profile, batch_size, responses, max_batch_size, prompt_len, generation_end,
                                   post_processing_end):
        total_ms = (post_processing_end - profile["start"]) * 1000.0
        generation_ms = (generation_end - profile["start"]) * 1000.0
        prefill_ms = profile["prefill_forward_ms"]
        decode_ms = profile["decode_forward_ms"]
        response_lengths = [len(response) for response in responses.values()]
        num_generated_tokens = sum(response_lengths)
        self._last_profile = {
            "prompt_expansion_ms": 0.0,
            "generation_ms": generation_ms,
            "prefill_forward_ms": prefill_ms,
            "decode_forward_ms": decode_ms,
            "generation_overhead_ms": max(0.0, generation_ms - prefill_ms - decode_ms),
            "num_prefill_forwards": profile["num_prefill_forwards"],
            "num_decode_forwards": profile["num_decode_forwards"],
            "scheduler_overhead_ms": profile["scheduler_overhead_ms"],
            "cache_management_overhead_ms": profile["cache_management_overhead_ms"],
            "post_processing_ms": (post_processing_end - generation_end) * 1000.0,
            "total_ms": total_ms,
            "num_generated_tokens": num_generated_tokens,
            "tokens_per_second": num_generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0,
            "batch_size": batch_size,
            "num_samples_per_prompt": 1,
            "prompt_length": prompt_len,
            "response_length": max(response_lengths, default=0),
            "active_batch_size": max(profile["active_batch_sizes"], default=0),
            "continuous_batch_size": max_batch_size,
        }

    def _continuous_prefill_cache(self, module, static_cache_type, prompt_ids, prompt_attention, model_dtype, device,
                                  sampling):
        position_ids = self._prefill_position_ids(prompt_attention)
        if getattr(module, "_supports_cache_class", None) is None:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            prefill_output = module.generate(
                prompt_ids,
                attention_mask=prompt_attention,
                position_ids=position_ids,
                max_new_tokens=1,
                eos_token_id=None,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
                output_logits=True,
                **self._hf_sampling_kwargs(sampling),
            )
            tokens = prefill_output.sequences[:, -1:]
            log_probs = _chosen_log_probs(prefill_output.logits[-1], tokens, sampling.temperature)
            return tokens, log_probs, prefill_output.past_key_values

        prompt_len = prompt_ids.shape[1]
        prefill_cache = self._create_static_cache(static_cache_type, module.config, prompt_ids.shape[0], prompt_len,
                                                  device, model_dtype)
        prefill_output = self._call_model(
            module,
            prompt_ids,
            attention_mask=prompt_attention,
            past_key_values=prefill_cache,
            use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
            position_ids=position_ids,
            **self._last_logits_kwargs(module),
        )
        last_logits = prefill_output.logits[:, -1, :]
        tokens = self._token_selector(sampling)(last_logits)
        return tokens, _chosen_log_probs(last_logits, tokens, sampling.temperature), prefill_cache

    @staticmethod
    def _continuous_next_tokens(logits, request_ids, request_by_id, responses, repetition_penalty, select_next_tokens):
        if repetition_penalty != 1.0:
            # generate() applies the penalty before the sampling warpers, so it shapes the sampled distribution.
            logits = logits.clone()
            for row, request_id in enumerate(request_ids):
                input_ids = torch.cat((request_by_id[request_id].prompt_ids, *responses[request_id]), dim=1)
                scores = logits[row:row + 1].gather(1, input_ids)
                scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
                logits[row:row + 1].scatter_(1, input_ids, scores)
        return select_next_tokens(logits)

    @staticmethod
    def _hf_sampling_kwargs(sampling):
        """Hugging Face ``generate()`` arguments for ``sampling``; ``temperature <= 0`` is greedy."""
        if sampling.temperature <= 0.0:
            return {
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "repetition_penalty": sampling.repetition_penalty,
            }
        return {
            "do_sample": True,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": max(sampling.top_k, 0),
            "repetition_penalty": sampling.repetition_penalty,
        }

    @staticmethod
    def _token_selector(sampling):
        """Return a function that picks the next tokens ``[B, 1]`` from last-position logits ``[B, V]``.

        Tokens are picked after each forward or graph replay, outside any
        captured graph. Sampling applies Hugging Face's temperature, top-k and
        top-p warpers in the order ``generate()`` applies them, so every path
        draws from the same distribution as ``generate()``.
        """
        if sampling.temperature <= 0.0:
            return lambda logits: logits.argmax(dim=-1, keepdim=True)

        from transformers.generation.logits_process import (TemperatureLogitsWarper, TopKLogitsWarper,
                                                            TopPLogitsWarper)

        warpers = []
        if sampling.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(sampling.temperature))
        if sampling.top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=sampling.top_k))
        if sampling.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=sampling.top_p))

        def sample(logits):
            # generate() also warps and samples in float32.
            scores = logits.float()
            for warper in warpers:
                # The warpers read only the scores, not the generated ids.
                scores = warper(None, scores)
            return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)

        return sample

    @staticmethod
    def _prefill_position_ids(attention_mask):
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 1)

    def _is_eos(self, token):
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            return False
        eos_ids = torch.as_tensor(eos_token_id, dtype=token.dtype, device=token.device).flatten()
        return bool((token == eos_ids).any().item())

    @staticmethod
    def _last_logits_kwargs(module):
        """Ask models that support it for last-position logits only, since prefill samples just one token."""
        # Without this the prefill materializes [batch, prompt_len, vocab] logits, the largest activation of the
        # rollout. Models that do not declare the argument may forward unknown kwargs to submodules, so skip them.
        if "logits_to_keep" in signature(module.forward).parameters:
            return {"logits_to_keep": 1}
        return {}

    @staticmethod
    def _call_model(module, input_ids, **kwargs):
        """Call models with only the cache arguments their HF version accepts."""
        parameters = signature(module.forward).parameters
        accepts_kwargs = any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values())
        if not accepts_kwargs:
            kwargs = {name: value for name, value in kwargs.items() if name in parameters}
        return module(input_ids, **kwargs)

    @staticmethod
    def _create_static_cache(static_cache_type, config, batch_size, max_cache_len, device, dtype):
        """Construct StaticCache across Transformers' batch-size API variants."""
        cache_config = config
        if not hasattr(config, "num_key_value_heads") or config.num_key_value_heads is None:
            cache_config = copy(config)
            cache_config.num_key_value_heads = config.num_attention_heads
        common_kwargs = {
            "config": cache_config,
            "max_cache_len": max_cache_len,
            "device": device,
            "dtype": dtype,
        }
        parameters = signature(static_cache_type).parameters
        if "batch_size" in parameters:
            common_kwargs["batch_size"] = batch_size
        elif "max_batch_size" in parameters:
            common_kwargs["max_batch_size"] = batch_size
        return static_cache_type(**common_kwargs)

    @staticmethod
    def _cache_layer_count(cache):
        if hasattr(cache, "layers"):
            return len(cache.layers)
        return len(cache.key_cache)

    @staticmethod
    def _cache_layer_tensors(cache, layer_idx):
        if hasattr(cache, "layers"):
            layer = cache.layers[layer_idx]
            return layer.keys, layer.values
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    def _build_continuous_batch(self, trajectories, responses, response_log_probs, return_log_probs):
        """Append each request's response, which stops at its EOS or ``max_new_tokens``, to its sample."""
        for request_id, sample in enumerate(trajectories):
            response = torch.cat(responses[request_id], dim=1)
            if self._is_eos(response[:, -1]):
                sample.status = RolloutRecord.Status.COMPLETED
            else:
                sample.status = RolloutRecord.Status.TRUNCATED
            log_probs = None
            if return_log_probs:
                log_probs = torch.cat(response_log_probs[request_id], dim=1)[0].tolist()
            update_sample_from_response(sample, response[0].tolist(), log_probs, [], self.tokenizer)
        return RolloutBatch(samples=trajectories)

    def get_last_profile(self):
        """Return the most recent profiling snapshot for this rollout instance."""
        return self._last_profile

    def get_last_continuous_stats(self):
        """Return cache and throughput statistics from the most recent continuous rollout."""
        return self._last_continuous_stats

    def _register_shared_prefill_hooks(self, module, batch_size, repeats):
        state = {"pending": True, "reduced": False}

        def reduce_prompt_batch(_module, args, kwargs):
            input_ids = kwargs.get("input_ids")
            if not state["pending"]:
                return args, kwargs
            if input_ids is None:
                raise RuntimeError("Shared prefill requires input_ids as a keyword argument")
            expected_batch_size = batch_size * repeats
            if input_ids.shape[0] != expected_batch_size:
                raise RuntimeError("Shared prefill input batch does not match the expanded rollout batch")
            if input_ids.shape[1] <= 1:
                raise RuntimeError("Shared prefill requires a prompt with more than one token")
            kwargs = dict(kwargs)
            kwargs["input_ids"] = input_ids[::repeats]
            for name in ("attention_mask", "position_ids", "token_type_ids"):
                value = kwargs.get(name)
                if isinstance(value, torch.Tensor) and value.shape[0] == expected_batch_size:
                    kwargs[name] = value[::repeats]
            state["reduced"] = True
            return args, kwargs

        def expand_prompt_output(_module, _args, _kwargs, output):
            if not state["pending"]:
                return output
            if not state["reduced"]:
                raise RuntimeError("Shared prefill did not reduce the prompt batch")
            state["pending"] = False
            output.past_key_values = self.engine.repeat_shared_prefill_cache(batch_size, repeats)
            output.logits = output.logits.repeat_interleave(repeats, dim=0)
            return output

        pre_handle = module.register_forward_pre_hook(reduce_prompt_batch, with_kwargs=True)
        post_handle = module.register_forward_hook(expand_prompt_output, with_kwargs=True)
        return pre_handle, post_handle

    @staticmethod
    def _pad_after_eos(output_ids, response_start, eos_token_id, pad_token_id):
        """Retain the first response EOS and pad every subsequent position."""
        response_ids = output_ids[:, response_start:]
        response_attn = (response_ids != pad_token_id)

        if eos_token_id is None or response_ids.shape[1] == 0:
            return output_ids, response_attn.long()

        eos_ids = torch.as_tensor(eos_token_id, device=response_ids.device, dtype=response_ids.dtype).flatten()
        is_eos = (response_ids.unsqueeze(-1) == eos_ids).any(dim=-1)
        has_eos = is_eos.any(dim=-1)
        first_eos_idx = is_eos.long().argmax(dim=-1)
        positions = torch.arange(response_ids.shape[1], device=response_ids.device).unsqueeze(0)
        after_first_eos = has_eos.unsqueeze(1) & (positions > first_eos_idx.unsqueeze(1))
        first_eos = has_eos.unsqueeze(1) & (positions == first_eos_idx.unsqueeze(1))

        output_ids = output_ids.clone()
        output_ids[:, response_start:].masked_fill_(after_first_eos, pad_token_id)
        # EOS is a valid generated token even when pad_token_id == eos_token_id.
        response_attn = ((response_ids != pad_token_id) | first_eos) & ~after_first_eos
        return output_ids, response_attn.long()

    # ------------------------------------------------------------------
    # Graph capture decode loop
    # ------------------------------------------------------------------

    def _generate_graph(self, prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module, device, sampling):
        """Decode with DeepSpeedStaticCache + CUDA graph capture, greedy or sampled per ``sampling``.

        Only the decode forward is captured. Tokens are chosen eagerly from the
        graph's output logits after each replay, so sampling draws fresh random
        numbers every step instead of replaying the captured ones.
        """
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        max_len = prompt_len + max_new_tokens
        eos_token_id = self.tokenizer.eos_token_id
        model_dtype = next(module.parameters()).dtype
        select_next_tokens = self._token_selector(sampling)

        # --- Prefill straight into the cache the graph decodes from ---
        # A second full-length cache for the prefill alone would double the KV memory of the call.
        ds_cache = DeepSpeedStaticCache(
            module.config,
            batch_size=batch_size,
            max_cache_len=max_len,
            device=device,
            dtype=model_dtype,
        )
        # The prompt fills slots 0..prompt_len-1; decoding then writes one slot per step.
        write_pos = torch.tensor(0, dtype=torch.long, device=device)
        ds_cache.set_write_position(write_pos)
        prefill_kwargs = {
            "attention_mask": prompt_attn,
            "past_key_values": ds_cache,
            "use_cache": True,
            "cache_position": torch.arange(prompt_len, device=device),
            # The cache's get_seq_length also counts the slot being written, so a model that derives
            # positions from it would shift the prompt by one.
            "position_ids": torch.arange(prompt_len, device=device).expand(batch_size, -1),
            **self._last_logits_kwargs(module),
        }
        prefill_out = module(prompt_ids, **prefill_kwargs)
        next_token = select_next_tokens(prefill_out.logits[:, -1, :])
        log_probs = [_chosen_log_probs(prefill_out.logits[:, -1, :], next_token, sampling.temperature)]

        output_ids = [prompt_ids, next_token]

        # --- Static buffers for graph capture ---
        static_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        static_attn = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        static_attn[:, :prompt_len] = prompt_attn
        static_attn[:, prompt_len] = 1  # first decode position
        static_pos = torch.tensor(prompt_len, dtype=torch.long, device=device)
        static_cache_pos = static_pos.unsqueeze(0)  # [1] for cache_position
        static_pos_ids = static_pos.reshape(1, 1).expand(batch_size, 1)  # [batch, 1]

        write_pos.fill_(prompt_len)

        # Remove forward hooks (they synchronize — illegal during graph capture)
        saved_pre = dict(module._forward_pre_hooks)
        saved_post = dict(module._forward_hooks)
        module._forward_pre_hooks.clear()
        module._forward_hooks.clear()

        try:
            # Warmup on side stream
            static_token.copy_(next_token)
            s = get_accelerator().Stream()
            s.wait_stream(get_accelerator().current_stream())
            with get_accelerator().stream(s):
                for _ in range(3):
                    out = module(
                        static_token,
                        attention_mask=static_attn,
                        past_key_values=ds_cache,
                        use_cache=True,
                        cache_position=static_cache_pos,
                        position_ids=static_pos_ids,
                    )
            get_accelerator().current_stream().wait_stream(s)

            # Capture
            graph = get_accelerator().create_graph()
            with get_accelerator().capture_to_graph(graph):
                out = module(
                    static_token,
                    attention_mask=static_attn,
                    past_key_values=ds_cache,
                    use_cache=True,
                    cache_position=static_cache_pos,
                    position_ids=static_pos_ids,
                )
            static_logits = out.logits
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)

        # --- Decode loop ---
        eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_new_tokens - 1):
            if eos_mask.all():
                output_ids.append(torch.full((batch_size, 1), pad_token_id, dtype=torch.long, device=device))
                log_probs.append(torch.zeros((batch_size, 1), device=device))
                continue

            # Update static inputs
            static_token.copy_(next_token)
            pos = prompt_len + step
            write_pos.fill_(pos)
            static_cache_pos.fill_(pos)
            static_pos_ids.fill_(pos)
            static_attn[:, pos] = 1

            # Replay
            get_accelerator().replay_graph(graph)
            next_token = select_next_tokens(static_logits[:, -1, :])
            output_ids.append(next_token)
            log_probs.append(_chosen_log_probs(static_logits[:, -1, :], next_token, sampling.temperature))
            eos_mask |= (next_token.squeeze(1) == eos_token_id)

        return torch.cat(output_ids, dim=1), torch.cat(log_probs, dim=1)

    def sync_weights(self, step: int) -> None:  # noqa: ARG002
        """No-op: hybrid engine reads model weights live."""
        return None

# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine that generates with an AutoTP tensor-parallel copy of the training model.

The training engine keeps the model in its ZeRO layout. This engine keeps a second, inference
copy of the same model sharded with AutoTP (``init_inference`` without kernel injection), and
``sync_weights`` refreshes that copy in memory from the training engine's fp32 masters:

  * gather: ``safe_get_full_fp32_param`` reassembles each ZeRO-partitioned master weight.
  * slice:  ``_resolve_autotp_partition`` cuts it into this rank's shard, using the layout
    metadata every AutoTP layer records on its weight (the same function that restores a
    universal checkpoint into AutoTP, so the two cannot disagree on a layout).

One tensor-parallel group spans all ranks, so the ranks generate together: each rank's prompts
are gathered, the union is generated once, and every rank keeps its own samples.

Supported: single-turn ``model.generate()`` rollout. Not supported yet: multi-turn, CUDA graph
capture, shared prefill, and continuous batching.
"""

import time
from copy import deepcopy
from dataclasses import replace

import torch

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.universal_checkpoint import _resolve_autotp_partition
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutRequest
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout
from deepspeed.utils import safe_get_full_fp32_param


def _params_with_owner(module):
    """Map each parameter name to (parameter, the module that owns it)."""
    owned = {}
    for module_name, owner in module.named_modules():
        for param_name, param in owner.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            owned[full_name] = (param, owner)
    return owned


@torch.no_grad()
def sync_autotp_weights(train_module, tp_module):
    """Overwrite every parameter of ``tp_module`` with the current fp32 masters of ``train_module``.

    Every rank must call this: the gather is a collective, and ``named_parameters`` yields the
    same order on every rank, so the collectives pair up the same tensor everywhere.
    Returns the number of tensors copied.
    """
    tp_params = _params_with_owner(tp_module)
    copied = 0
    for tp_name, train_param in train_module.named_parameters():
        full = safe_get_full_fp32_param(train_param)
        assert full is not None, f"{tp_name} has no ZeRO fp32 master; is the optimizer a ZeRO optimizer?"
        assert tp_name in tp_params, f"inference model has no parameter {tp_name}"
        tp_param, owner = tp_params[tp_name]

        # Only AutoTP layers carry tp_index; other modules (norms, embeddings) are replicated.
        tp_rank = getattr(owner, "tp_index", 0)
        tp_world_size = getattr(owner, "tp_world_size", 1)
        shard = _resolve_autotp_partition(tp_param, {}, full, tp_rank, tp_world_size)
        if shard is None:
            assert full.shape == tp_param.shape, (
                f"{tp_name}: no AutoTP layout recorded, but full shape {tuple(full.shape)} != "
                f"inference shape {tuple(tp_param.shape)}")
            shard = full
        # copy_ rounds fp32 to the inference dtype the same way ZeRO rounds its low-precision params.
        tp_param.copy_(shard.view(tp_param.shape))
        copied += 1

    assert copied == len(tp_params), f"synced {copied} tensors, but the inference model has {len(tp_params)}"
    return copied


class _TPGenerationEngine:
    """The part of the DeepSpeed engine interface that single-turn HybridEngineRollout reads."""

    def __init__(self, module):
        self.module = module

    def zero_optimization_partition_weights(self):
        # HybridEngineRollout turns off EOS early-stop when this is True, because ZeRO-3 runs a
        # collective in every decode forward. AutoTP all-reduces in every decode forward too, so
        # every rank must run the same number of decode steps for the same reason.
        return True


class AutoTPRollout(HybridEngineRollout):
    """Single-turn rollout with an AutoTP copy of the training model, refreshed by ``sync_weights``.

    Args:
        engine: DeepSpeed ZeRO training engine, or a view of it, whose ``module`` is the Hugging Face causal LM.
        tokenizer: HuggingFace tokenizer (must have pad_token_id or eos_token_id).
        cfg: Rollout config; ``multi_turn``, ``use_graph_capture`` and ``use_shared_prefill`` must be off.
        tools: Optional ToolSet; its specs are rendered into prompts as in HybridEngineRollout.
    """

    name = "autotp"

    def __init__(self, engine, tokenizer, cfg=None, tools=None):
        super().__init__(engine=engine, tokenizer=tokenizer, cfg=cfg, tools=tools)
        if self.multi_turn or self.use_graph_capture or self.use_shared_prefill:
            raise NotImplementedError("AutoTP rollout supports single-turn generate() only; turn off "
                                      "multi_turn, use_graph_capture and use_shared_prefill")
        self.train_engine = engine
        self.tp_size = dist.get_world_size()
        self.engine = _TPGenerationEngine(self._build_tp_copy(engine.module))

        # Every rank must sample the same tokens, so all ranks draw from one seed per generate call.
        seed = torch.randint(0, 2**31 - 1, (1, ), device=get_accelerator().current_device_name())
        dist.broadcast(seed, src=0)
        self._seed = seed.item()
        self._generate_calls = 0
        self.last_sync = None
        # The copy starts with random weights; it must never generate before its first sync.
        self.sync_weights(step=0)

    def _build_tp_copy(self, train_model):
        # Imported here: deepspeed/__init__.py imports the runtime, so a top-level import would be circular.
        import deepspeed
        from transformers import AutoModelForCausalLM

        dtype = next(train_model.parameters()).dtype
        # Built on the accelerator with random weights: sync_weights overwrites every tensor, so
        # loading weights from disk here would only cost time and host memory.
        with torch.device(get_accelerator().current_device_name()):
            model = AutoModelForCausalLM.from_config(train_model.config,
                                                     dtype=dtype,
                                                     attn_implementation=train_model.config._attn_implementation)
        model.generation_config = deepcopy(train_model.generation_config)
        inference_engine = deepspeed.init_inference(model.eval(),
                                                    config={
                                                        "dtype": dtype,
                                                        "tensor_parallel": {
                                                            "tp_size": self.tp_size
                                                        },
                                                        "replace_with_kernel_inject": False,
                                                        "enable_cuda_graph": False,
                                                    })
        return inference_engine.module

    def sync_weights(self, step: int) -> None:
        """Copy the training engine's current weights into the AutoTP copy (collective)."""
        accelerator = get_accelerator()
        accelerator.synchronize()
        start = time.perf_counter()
        copied = sync_autotp_weights(self.train_engine.module, self.engine.module)
        accelerator.synchronize()
        self.last_sync = {"step": step, "tensors": copied, "seconds": time.perf_counter() - start}

    @torch.no_grad()
    def generate(self, request, sampling):
        if sampling.continuous_batch_size is not None:
            raise NotImplementedError("AutoTP rollout does not support continuous batching")
        rank = dist.get_rank()
        gathered = [None] * self.tp_size
        dist.all_gather_object(gathered, request.samples)
        counts = [len(samples) for samples in gathered]
        union = []
        for samples in gathered:
            for position, sample in enumerate(samples):
                # Generation numbers an unset group by its position in the request, which in the union
                # would no longer be its position in its own rank's request.
                if sample.group_index is None:
                    sample = replace(sample, group_index=position)
                union.append(sample)

        # fork_rng restores the caller's RNG state, so training randomness is unaffected.
        with torch.random.fork_rng(devices=[get_accelerator().current_device()]):
            torch.manual_seed(self._seed + self._generate_calls)
            self._generate_calls += 1
            batch = super().generate(RolloutRequest(samples=union), sampling)
        self._check_same_tokens_on_all_ranks(batch)

        # expand_samples keeps each prompt's n copies adjacent and in prompt order.
        n = sampling.n_samples_per_prompt
        start = sum(counts[:rank]) * n
        mine = batch.samples[start:start + counts[rank] * n]
        for index, sample in enumerate(mine):
            sample.index = index
        return RolloutBatch(samples=mine)

    def _check_same_tokens_on_all_ranks(self, batch):
        tokens = [sample.tokens for sample in batch.samples]
        gathered = [None] * self.tp_size
        dist.all_gather_object(gathered, tokens)
        if any(other != gathered[0] for other in gathered[1:]):
            raise RuntimeError("AutoTP ranks generated different tokens; the tensor-parallel copy is out of sync")

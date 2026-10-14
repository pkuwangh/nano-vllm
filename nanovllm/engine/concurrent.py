"""Experimental TP1 prefill/decode loop with independent stream completion.

Greedy, fixed-length generation; no prefix reuse or preemption. Each admitted
request reserves enough cache blocks to finish, including future decode growth.
"""

from collections import deque
from dataclasses import dataclass
from math import ceil
from time import perf_counter, sleep

import torch
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context


@dataclass
class PendingBatch:
    """Keep inputs and outputs alive until the stream's completion event fires."""

    seqs: list[Sequence]
    tokens: torch.Tensor
    begin: torch.cuda.Event
    done: torch.cuda.Event
    keepalive: tuple
    submit_ms: float


class ConcurrentExecutor:
    """Share a model/cache while keeping one outstanding batch per execution path."""

    def __init__(self, runner):
        if runner.world_size != 1:
            raise ValueError("ConcurrentExecutor requires TP1")
        self.runner = runner
        self.config = runner.config
        self.streams = {
            "prefill": runner.prefill_stream or torch.cuda.current_stream(),
            "decode": runner.execution_stream or torch.cuda.current_stream(),
        }

    @torch.inference_mode()
    def submit(self, seqs, phase):
        runner = self.runner
        begin = torch.cuda.Event(enable_timing=True)
        done = torch.cuda.Event(enable_timing=True)
        host_tokens = torch.empty(len(seqs), dtype=torch.int64, pin_memory=True)
        start = perf_counter()
        with torch.cuda.stream(self.streams[phase]), torch.cuda.nvtx.range(f"submit_{phase}"):
            begin.record()
            try:
                prepare = runner.prepare_prefill if phase == "prefill" else runner.prepare_decode
                inputs, positions = prepare(seqs)
                context = get_context()
                logits = runner.run_model(inputs, positions, phase == "prefill")
                tokens = logits.argmax(dim=-1)
                host_tokens.copy_(tokens, non_blocking=True)
                done.record()
            finally:
                reset_context()
        return PendingBatch(
            seqs,
            host_tokens,
            begin,
            done,
            (inputs, positions, context, logits, tokens),
            (perf_counter() - start) * 1000,
        )

    def generate(self, prompts, max_tokens, concurrent=True, active_requests=None):
        """Refill active slots as requests finish; None admits up to cache capacity."""
        config = self.config
        if max_tokens < 1 or any(not p or len(p) + max_tokens > config.max_model_len for p in prompts):
            raise ValueError("Prompts must be nonempty and input + output must fit max_model_len")
        if concurrent and self.runner.partitions is None:
            raise ValueError("Concurrent execution requires prefill_sm_fraction")
        if active_requests is not None and active_requests < 1:
            raise ValueError("active_requests must be positive")
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        sequences = [Sequence(p, params) for p in prompts]
        waiting, partial, ready = deque(sequences), deque(), deque()
        blocks = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        reservations = {}
        reserved = 0
        pending = {"prefill": None, "decode": None}
        steps = []
        ttft_ms = []
        admitted_ms = {}
        completed_ms = []
        epoch = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        epoch.record()
        for stream in self.streams.values():
            stream.wait_event(epoch)
        start = perf_counter()

        def prefill_batch():
            nonlocal reserved
            batch = []
            budget = config.max_num_batched_tokens
            while budget and len(batch) < config.max_num_seqs and (partial or waiting):
                if partial:
                    seq = partial.popleft()
                else:
                    if active_requests is not None and len(reservations) >= active_requests:
                        break
                    seq = waiting[0]
                    # Reserve decode growth so admitted requests can finish without preemption.
                    needed = ceil((seq.num_prompt_tokens + max_tokens) / blocks.block_size)
                    if len(blocks.free_block_ids) - reserved < needed:
                        break
                    waiting.popleft()
                    admitted_ms[seq.seq_id] = (perf_counter() - start) * 1000
                    blocks.allocate(seq, 0)
                    reservations[seq.seq_id] = needed - len(seq.block_table)
                    reserved += reservations[seq.seq_id]
                seq.num_scheduled_tokens = min(budget, seq.num_tokens - seq.num_cached_tokens)
                budget -= seq.num_scheduled_tokens
                batch.append(seq)
            return batch

        def decode_batch():
            nonlocal reserved
            batch = []
            while ready and len(batch) < config.max_num_seqs:
                seq = ready.popleft()
                before = len(seq.block_table)
                blocks.may_append(seq)
                added = len(seq.block_table) - before
                reservations[seq.seq_id] -= added
                reserved -= added
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                batch.append(seq)
            return batch

        def collect(phase, work):
            nonlocal reserved
            tokens = work.tokens.tolist()  # The completion event already covers D2H.
            collected_ms = (perf_counter() - start) * 1000
            step = {
                "phase": phase,
                "start_ms": epoch.elapsed_time(work.begin),
                "end_ms": epoch.elapsed_time(work.done),
                "submit_ms": work.submit_ms,
                "tokens": sum(seq.num_scheduled_tokens for seq in work.seqs),
                "batch_size": len(work.seqs),
                "collected_ms": collected_ms,
                "output_tokens": 0,
                "completed_requests": 0,
                "active_requests": len(reservations),
            }
            steps.append(step)
            for seq, token in zip(work.seqs, tokens):
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if phase == "prefill" and seq.num_cached_tokens < seq.num_prompt_tokens:
                    partial.append(seq)
                    continue
                seq.append_token(token)
                step["output_tokens"] += 1
                if phase == "prefill":
                    ttft_ms.append(collected_ms - admitted_ms.pop(seq.seq_id))
                if seq.num_completion_tokens == max_tokens:
                    completed_ms.append(collected_ms)
                    step["completed_requests"] += 1
                    reserved -= reservations.pop(seq.seq_id)
                    blocks.deallocate(seq)
                else:
                    ready.append(seq)

        order = ("decode", "prefill") if concurrent else ("prefill", "decode")
        try:
            while waiting or partial or ready or any(pending.values()):
                progressed = False
                for phase in ("decode", "prefill"):
                    work = pending[phase]
                    if work is not None and work.done.query():
                        collect(phase, work)
                        pending[phase] = None
                        progressed = True
                # Decode can take many steps while an unrelated prefill is in flight.
                for phase in order:
                    if pending[phase] is not None or (not concurrent and any(pending.values())):
                        continue
                    batch = prefill_batch() if phase == "prefill" else decode_batch()
                    if batch:
                        pending[phase] = self.submit(batch, phase)
                        progressed = True
                if not progressed:
                    if not any(pending.values()):
                        raise RuntimeError("KV cache cannot fit the next request's input/output reservation")
                    sleep(0.00005)
        finally:
            # Also protect in-flight buffers on exceptions.
            for stream in self.streams.values():
                stream.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000
        assert not reservations and reserved == 0 and len(blocks.free_block_ids) == config.num_kvcache_blocks
        return [s.completion_token_ids for s in sequences], {
            "elapsed_ms": elapsed_ms,
            "output_tok_s": len(prompts) * max_tokens / (elapsed_ms / 1000),
            "ttft_ms": ttft_ms,
            "completed_ms": completed_ms,
            "steps": steps,
        }

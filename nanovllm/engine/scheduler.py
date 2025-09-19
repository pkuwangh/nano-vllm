from collections import deque
from loguru import logger

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        logger.info(f"Scheduler: {config.max_num_seqs=} {config.max_num_batched_tokens=}")
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                # exceed max batched tokens or cannot allocate kvcache blocks
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            # logger.debug(
            #     f"Schedule {len(scheduled_seqs)} seqs for prefill "
            #     f"(waiting={len(self.waiting)} running={len(self.running)}): "
            #     f"{[x.seq_id for x in scheduled_seqs]}"
            # )
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            # try to schedule each seq currently in running deque
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    # there are other running seqs, preempt the most recently enqueued one
                    # until we can append the current seq
                    self.preempt(self.running.pop())
                else:
                    # no other running seqs, have to preempt the current one
                    # i.e. other scheduled_seqs in this round has taken KV cache blocks
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # push the scheduled seqs to the left of running deque, in the same order they were scheduled
        self.running.extendleft(reversed(scheduled_seqs))
        # logger.debug(
        #     f"Schedule {len(scheduled_seqs)} seqs for decode: "
        #     f"{[x.seq_id for x in scheduled_seqs]}"
        # )
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # logger.debug(f"Sequence {seq.seq_id} finished: {seq.num_completion_tokens}/{seq.max_tokens} tokens, "
                #              f"last token={token_id}/{self.eos}")
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

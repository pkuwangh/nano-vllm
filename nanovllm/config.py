import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    num_sms: int | None = None
    prefill_sm_fraction: float | None = None
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.num_sms is not None:
            if self.num_sms <= 0:
                raise ValueError("num_sms must be positive")
            if self.tensor_parallel_size != 1:
                raise ValueError("Green-context execution currently requires TP1")
        if self.prefill_sm_fraction is not None:
            if not 0 < self.prefill_sm_fraction < 1:
                raise ValueError("prefill_sm_fraction must be between 0 and 1")
            if self.tensor_parallel_size != 1 or self.num_sms is not None:
                raise ValueError("Concurrent partitions require TP1 and cannot be combined with num_sms")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

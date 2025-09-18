import torch
from torch import nn
from loguru import logger


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self._logged = True

    def reset_logging(self):
        self._logged = False

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        if not self._logged:
            temperatures_unsqueezed = temperatures.unsqueeze(dim=1)
            logger.info(
                f"Sampler: {logits.shape=} {probs.shape=} {sample_tokens.shape=} "
                f"{temperatures_unsqueezed.shape=}"
            )
            self._logged = True
        return sample_tokens

"""Two disjoint SM partitions, with streams usable by PyTorch."""

import math

import torch
from cuda.bindings import driver as cu


def _checked(result):
    status, *values = result
    if status != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA green context: {status}")
    return values[0] if len(values) == 1 else values


class GreenContextPair:
    """Own a fixed SM split and its prefill/decode streams, in that order."""

    def __init__(self, prefill_fraction: float, device: int):
        torch.empty(0, device=f"cuda:{device}")  # Initialize the primary context.
        resource = _checked(cu.cuDeviceGetDevResource(device, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        self.total_sms = resource.sm.smCount
        groups, count, remainder = _checked(
            cu.cuDevSmResourceSplitByCount(1, resource, 0, math.ceil(self.total_sms * prefill_fraction))
        )
        if count != 1 or remainder.sm.smCount == 0:
            raise ValueError("The requested split leaves no SMs for decode")
        self.sm_counts = (groups[0].sm.smCount, remainder.sm.smCount)
        self.contexts = []
        self.handles = []
        self.streams = []
        try:
            for sm_resource in (groups[0], remainder):
                workqueues = _checked(
                    cu.cuDeviceGetDevResource(device, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG)
                )
                workqueues.wqConfig.sharingScope = cu.CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED
                workqueues.wqConfig.wqConcurrencyLimit = 1
                desc = _checked(cu.cuDevResourceGenerateDesc([sm_resource, workqueues], 2))
                context = _checked(
                    cu.cuGreenCtxCreate(desc, device, cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM)
                )
                self.contexts.append(context)
                handle = _checked(cu.cuGreenCtxStreamCreate(context, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, 0))
                self.handles.append(handle)
                self.streams.append(torch.cuda.ExternalStream(int(handle), device=device))
        except Exception:
            self.close()
            raise

    def close(self):
        """Release resources after pending work and captured graphs have been released."""
        for handle in reversed(self.handles):
            _checked(cu.cuStreamDestroy(handle))
        self.handles.clear()
        self.streams.clear()
        for context in reversed(self.contexts):
            _checked(cu.cuGreenCtxDestroy(context))
        self.contexts.clear()

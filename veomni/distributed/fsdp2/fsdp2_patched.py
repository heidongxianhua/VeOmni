import logging
from typing import Any

import torch
from torch.distributed._composable.fsdp._fully_shard._fsdp_collectives import foreach_reduce
from torch.distributed._composable.fsdp._fully_shard._fsdp_common import TrainingState, compiled_autograd_enabled
from torch.distributed._composable.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed._composable.fsdp._fully_shard._fsdp_param_group import (
    AllReduceState,
    FSDPParamGroup,
    ReduceScatterState,
)
from torch.profiler import record_function

from veomni.utils.import_utils import is_torch_npu_available


logger = logging.getLogger("torch.distributed.fsdp.fully_shard_patched")


# only for pytorch2.7.1, when torch version changed, this fun should be changed too.
def patched_post_backward(self, *unused: Any):
    # This method should be idempotent and safe to call even when this
    # FSDP parameter group was not used in backward (should be a no-op)
    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_backward"))
    self._training_state = TrainingState.POST_BACKWARD
    with record_function(self._with_fqn("FSDP::post_backward_accumulate")):
        for fsdp_param in self.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()
    with record_function(self._with_fqn("FSDP::post_backward_reshard")):
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.reshard()
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_accumulated_grad_if_needed()
            return
        # Save the autograd-computed gradients before resharding to only
        # access the unsharded parameters when their data is present
        fsdp_params_with_grad: list[FSDPParam] = []
        unsharded_grads: list[torch.Tensor] = []
        for fsdp_param in self.fsdp_params:
            if not hasattr(fsdp_param, "_unsharded_param"):
                continue
            # May have an accumulated gradient of the reduce dtype if the
            # previous backward did not reduce-scatter
            if fsdp_param.unsharded_accumulated_grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_accumulated_grad_data)
                fsdp_param.unsharded_accumulated_grad = None
            elif fsdp_param.unsharded_param.grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_grad_data)
                fsdp_param.unsharded_param.grad = None
        if self.reshard_after_backward:
            self.reshard()
    if len(fsdp_params_with_grad) == 0:
        return
    with record_function(self._with_fqn("FSDP::post_backward_reduce")):
        if self.comm_ctx.reduce_scatter_state is not None:
            ### we only change here
            # self.device_handle.current_stream().wait_event(
            #     self.comm_ctx.reduce_scatter_state.event
            # )
            self.comm_ctx.reduce_scatter_state = None
        all_reduce_pg = self._all_reduce_process_group if self._is_hsdp else None
        all_reduce_stream: torch.cuda.Stream
        if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
            # this means the native HSDP is not enabled,
            # but user may want to have a custom HSDP setup
            assert self._all_reduce_hook is not None, "all reduce hook stream is specified but hook itself is missing."
            all_reduce_stream = self._all_reduce_hook_stream
        else:
            all_reduce_stream = self.comm_ctx.all_reduce_stream

        self._wait_for_post_backward()
        (
            reduce_scatter_input,
            reduce_scatter_event,
            self._post_reduce_event,
            all_reduce_input,
            all_reduce_event,
            self._partial_reduce_output,
        ) = foreach_reduce(
            fsdp_params_with_grad,
            unsharded_grads,
            self._reduce_scatter_process_group,
            self.comm_ctx.reduce_scatter_stream,
            self._orig_dtype,
            self._reduce_dtype,
            self.device,
            self.reduce_scatter_reduce_op,
            self._all_reduce_process_group if self._is_hsdp else None,
            all_reduce_stream,
            self.all_reduce_grads,
            self._partial_reduce_output,
            self._all_reduce_hook,
        )
        self.comm_ctx.reduce_scatter_state = ReduceScatterState(reduce_scatter_input, reduce_scatter_event)
        if all_reduce_input is not None:
            assert all_reduce_event is not None
            self._all_reduce_state = AllReduceState(all_reduce_input, all_reduce_event)


def patched_fsdp2_methods():
    if is_torch_npu_available():
        logger.info_rank0("veomni fsdp2 methods patch applied")
        FSDPParamGroup.post_backward = patched_post_backward

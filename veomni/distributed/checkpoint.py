import contextlib

import torch
import weakref
from weakref import ReferenceType
import uuid
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state_if_fully_sharded_module, _module_handle
from torch.distributed.fsdp._runtime_utils import (
    _post_backward_hook,
    _pre_backward_hook,
)
from torch.utils.checkpoint import (
    _get_autocast_kwargs,
    _get_device_module,
    _infer_device_type,
    check_backward_validity,
    detach_variable,
    get_device_states,
    set_device_states,
    _Holder,
    _recomputation_hook,
    _StopRecomputationError,
    _internal_assert,
    CheckpointError,
)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(ctx.device)
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)

        # patch code, remove the extra allgather with use_reentrant + ckpt
        if not isinstance(ctx.run_function, torch.nn.Module):
            ctx.patch_module = ctx.run_function.__self__
        else:
            ctx.patch_module = ctx.run_function
        state = _get_module_fsdp_state_if_fully_sharded_module(ctx.patch_module)
        if state:
            handle = _module_handle(state, ctx.patch_module)
            if handle:
                handle._needs_pre_backward_unshard = True
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
            )
        # patch code, remove the extra allgather with use_reentrant + ckpt
        handle = None
        state = _get_module_fsdp_state_if_fully_sharded_module(ctx.patch_module)
        if state:
            handle = _module_handle(state, ctx.patch_module)
            if handle:
                _pre_backward_hook(state, ctx.patch_module, handle, None)

        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
            detached_inputs = detach_variable(tuple(inputs))

            device_autocast_ctx = (
                torch.amp.autocast(device_type=ctx.device, **ctx.device_autocast_kwargs)
                if torch.amp.is_autocast_available(ctx.device)
                else contextlib.nullcontext()
            )
            with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("none of output has requires_grad=True, this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)

        # patch code, remove the extra allgather with use_reentrant + ckpt
        if handle:
            _post_backward_hook(state, handle, None)

        return (None, None) + grads


class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame):
        def pack_hook(x):
            # See Rule 4 above
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            # Save metadata to detect non-determinism
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            
            gid = torch._C._current_graph_task_id()
            if gid == -1:
                # generate a temporary id if we trigger unpack outside of a backward call
                gid = int(uuid.uuid4())

            def fake_post_forward(self, *args, **kwargs):
                print("run fake post_forward")
                pass

            if not frame.is_recomputed[gid]:
                ctx = frame.input_saver.grad_fn
                args = ctx.get_args(ctx.saved_tensors)
                from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
                try:
                    with _recomputation_hook(
                        weakref.ref(frame), gid
                    ), torch.autograd.enable_grad():
                        # print(f"_checkpoint_hook unpack_hook, recompute_fn, fn= {frame.fn}, type={type(frame.fn.__self__)}, type={type(frame.fn.__self__)}")
                        # fsdp_group = frame.fn.__self__._get_fsdp_state()._fsdp_param_group
                        # print(f"fsdp_group info in ck {id(fsdp_group)},{type(fsdp_group)}")
                        origin_post_forward = FSDPParamGroup.post_forward
                        FSDPParamGroup.post_forward = fake_post_forward
                        frame.recompute_fn(*args)
                        # print("_checkpoint_hook unpack_hook, recompute_fn ===== done")
                except _StopRecomputationError:
                    # print("_checkpoint_hook unpack_hook, recompute_fn ===== stop")
                    pass
                FSDPParamGroup.post_forward = origin_post_forward
                frame.is_recomputed[gid] = True
                frame.check_recomputed_tensors_match(gid)

            _internal_assert(gid in holder.handles)

            if holder.handles[gid] is None:
                raise CheckpointError(
                    "torch.utils.checkpoint: Unpack is being triggered for a tensor that was already "
                    "unpacked once. If you are calling ctx.saved_tensors in backward, make sure to do "
                    "so only once. Otherwise please open an issue with details on your use case."
                )
            _internal_assert(holder.handles[gid] in frame.recomputed[gid])
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret

        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except CheckpointError as e:
                    frame.unpack_error_cb(e)
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            super().__init__(pack_hook, unpack_hook)

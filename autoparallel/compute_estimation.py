import torch
from torch.utils._pytree import tree_map_only
from torch.utils.flop_counter import flop_registry, FlopCounterMode


def _get_device_tflops(dtype):
    # for some reason the function from PyTorch is giving
    # wildly different TFlops compared to the specs. I'm
    # using had-coded values for now that I pulled from xFormers
    # https://github.com/fairinternal/xformers/blob/main/xformers/profiler/device_limits.py
    # TODO: fix PyTorch's implementation
    # from torch._inductor.utils import get_device_tflops

    device = None
    device_name = torch.cuda.get_device_name(device)
    assert "H100" in device_name, f"Only H100 supported from now, got {device_name}"

    return {
        torch.float64: 67,
        # NOTE: NVIDIA gives all numbers "with 2:4 sparsity"
        # but we want the full GEMM numbers
        torch.float32: 989 // 2,
        torch.float16: 1979 // 2,
        torch.bfloat16: 1979 // 2,
        torch.int8: 3958 // 2,
    }[dtype]


def _get_sharded_shape(spec):
    mesh = spec.mesh
    tensor_shape = spec.tensor_meta.shape
    # TODO: take dtype into account as well
    # tensor_dtype = spec.tensor_meta.dtype
    placements = spec.placements
    # TODO: find a better heuristic other than
    # running DTensor
    new_tensor_shape = list(tensor_shape)
    for mesh_size, placement in zip(mesh.shape, placements):
        if placement.is_shard():
            dim = placement.dim
            new_tensor_shape[dim] = (
                new_tensor_shape[dim] + mesh_size - 1
            ) // mesh_size
    return new_tensor_shape


def estimate_strategy_runtime_cost(node, strategy):
    if node.op != "call_function":
        return 0
    # suppose only matmul-like ops
    if node.target not in {torch.ops.aten.mm.default}:
        return 0

    if not isinstance(node.target, torch._ops.OpOverload):
        return 0

    if node.target.overloadpacket not in flop_registry:
        return 0

    args = tree_map_only(torch.fx.Node, lambda x: x.meta["val"], node.args)
    kwargs = tree_map_only(torch.fx.Node, lambda x: x.meta["val"], node.kwargs)
    fake_mode = next(arg.fake_mode for arg in args if isinstance(arg, torch._subclasses.fake_tensor.FakeTensor))
    assert len(kwargs) == 0
    args_shapes = tuple(_get_sharded_shape(spec) for spec in strategy.input_specs)

    counter = 0
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            with fake_mode:
                args[i] = torch.empty(args_shapes[counter], device=arg.device, dtype=arg.dtype)
            counter += 1

    with FlopCounterMode(display=False) as flop_counter:
        out = node.target(*args, **kwargs)

    flops = flop_counter.get_total_flops()

    # TODO: fix this
    dtype = strategy.input_specs[0].tensor_meta.dtype

    gpu_flops = _get_device_tflops(dtype) * 10 ** 12

    # suppose 50% efficiency
    factor = 1 / 0.5
    compute_time = factor * flops / gpu_flops * 1e6  # us

    return compute_time



def _get_mm_time(strategy):
    dtype = strategy.input_specs[0].tensor_meta.dtype
    a_shape = _get_sharded_shape(strategy.input_specs[0])
    b_shape = _get_sharded_shape(strategy.input_specs[1])

    m, k = a_shape
    k2, n = b_shape
    assert k == k2
    flops = m * n * 2 * k

    gpu_flops = _get_device_tflops(dtype) * 10 ** 12

    factor = 1 / 0.5
    compute_time = factor * flops / gpu_flops * 1e6  # us

    return compute_time


def legacy():

    cost_op = 0

    if node.op != "call_function":
        return cost_op

    func = node.op
    if func not in flop_registry:
        return cost_op

    args = node.args
    flops = flop_registry[func._overloadpacket]()

    if node.op == "call_function" and node.target == torch.ops.aten.mm.default:
        cost_op =  _get_mm_time(strategy)

    return cost_op

"""Structural-op role classification — shared, framework-agnostic.

A "role" tags an op by what it does to dataflow *topology* (merge / split /
activation), independent of the backend that produced it. The torch tracer
(``hooks.py``) and the numpy interceptor (``backends/numpy_trace.py``) both
classify their ops through ``_op_role`` so the GUI styles them identically. This
module imports nothing backend-specific (no torch / numpy), so any interceptor
can depend on it.

Op names are the canonical (torch-aligned) vocabulary; a backend whose native
names differ (numpy's ``subtract`` / ``multiply`` / ``true_divide`` / ...) maps
them onto these before calling ``_op_role``.
"""
from __future__ import annotations

# These ops change dataflow *topology* and are the skeleton of an architecture
# diagram (the SSMDP ⊕ ⊙ ◇). They become first-class nodes at OPLEVEL_STRUCT;
# everything else (view/transpose/getitem/...) is a transparent pass-through.
_BINARY_ROLE = {
    "add": "add", "__add__": "add", "__radd__": "add", "__iadd__": "add",
    "sub": "sub", "__sub__": "sub", "__rsub__": "sub",
    "mul": "mul", "__mul__": "mul", "__rmul__": "mul", "__imul__": "mul",
    "div": "div", "divide": "div", "true_divide": "div",
    "__truediv__": "div", "__div__": "div",
}
_MATMUL_OPS = frozenset({"matmul", "__matmul__", "bmm", "mm"})
_CONCAT_OPS = frozenset({"cat", "concat", "stack", "hstack", "vstack",
                         "dstack", "column_stack"})
_SPLIT_OPS = frozenset({"chunk", "split", "tensor_split", "unbind"})
# Nonlinearities are semantic, first-class diagram nodes (the SSMDP t/σ/S/R).
# As 1-in-1-out nodes they also keep parallel branches distinct, so a gate like
# ``sigmoid(a) * tanh(b)`` is still recognized as a merge downstream.
_ACT_ROLE = {
    "sigmoid": "sigmoid", "tanh": "tanh", "relu": "relu", "gelu": "gelu",
    "silu": "silu", "swish": "silu", "softplus": "softplus", "exp": "exp",
    "softmax": "softmax", "elu": "elu", "leaky_relu": "relu", "hardtanh": "tanh",
}


def _op_role(op_name: str, n_producers: int, n_outputs: int):
    """Role string for a structural op, or None for a noise/pass-through op.

    ``n_producers`` = how many distinct already-traced tensors this op consumes
    (so ``x + eps`` with one tensor input is *not* a merge), ``n_outputs`` = how
    many tensors it returns (a split must actually fan out).
    """
    if op_name in _ACT_ROLE:
        return _ACT_ROLE[op_name]
    if op_name in _SPLIT_OPS and n_outputs >= 2:
        return "split"
    if op_name in _CONCAT_OPS and n_producers >= 2:
        return "concat"
    if op_name in _BINARY_ROLE and n_producers >= 2:
        return _BINARY_ROLE[op_name]
    if op_name in _MATMUL_OPS and n_producers >= 2:
        return "matmul"
    return None

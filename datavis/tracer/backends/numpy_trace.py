"""numpy interceptor: record a numpy preprocessing chain with no per-step probes.

numpy exposes the dual of torch's ``__torch_function__``: **NEP-13**
(``__array_ufunc__``, for ufuncs / arithmetic operators) and **NEP-18**
(``__array_function__``, for the high-level API: ``concatenate``, ``mean``,
``fft``, ...). ``TracedArray`` implements both, plus the reduction *methods*
(``mean``/``std``/...) that bypass those protocols. The producer node is carried
on the array instance, so wrapping once at the source with ``viz.track`` lets the
node/edge structure propagate through the whole chain automatically — the numpy
answer to "from the raw input, the entire data evolution, without inserting
probes".

The nodes land in the same ``ExecutionTree`` + ``TracerServer`` as the torch
tracer, so one DAG can span ``numpy -> sklearn -> torch``; when this session has
an ``owner`` torch tracer, the chain's tail is handed to ``owner._last_probe`` so
the model's forward wires it into the model root (the same boundary handoff
``viz.probe`` uses).

This module imports no torch — only ``protocol`` / ``graph`` / ``roles`` /
``transport`` — so numpy tracing works with torch absent.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from datavis.common import protocol
from datavis.tracer.graph import ExecutionTree
from datavis.tracer.roles import _op_role
from datavis.tracer.transport import TracerServer

# numpy's native op names -> the canonical (torch-aligned) role vocabulary that
# ``_op_role`` understands. Names already shared (add / matmul / stack / hstack /
# vstack / split / exp / tanh / ...) need no entry.
_CANON = {
    "subtract": "sub", "multiply": "mul",
    "true_divide": "div", "divide": "div", "floor_divide": "div",
    "concatenate": "concat", "dot": "matmul", "array_split": "split",
}


def _strip(obj):
    """Replace every ``TracedArray`` with its plain ndarray view (recursively
    through tuples/lists), so the real computation never re-enters our hooks."""
    if isinstance(obj, TracedArray):
        return obj.view(np.ndarray)
    if isinstance(obj, tuple):
        return tuple(_strip(o) for o in obj)
    if isinstance(obj, list):
        return [_strip(o) for o in obj]
    return obj


def _flatten_traced(obj, acc: list) -> None:
    """Collect every ``TracedArray`` nested in ``obj`` (args / tuples / lists)."""
    if isinstance(obj, TracedArray):
        acc.append(obj)
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _flatten_traced(o, acc)


class TracedArray(np.ndarray):
    """An ndarray that records the ops it flows through into a ``NumpyTracer``.

    Created only via ``NumpyTracer.track`` / ``record_op`` — never instantiate
    directly. Carries ``_viz_session`` (the tracer) and ``_viz_node`` (the id of
    the node that produced this array). A plain view (slice/reshape) propagates
    both unchanged, so noise views are transparent pass-throughs, exactly like
    the torch tracer's.
    """

    _viz_session = None
    _viz_node = None

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self._viz_session = getattr(obj, "_viz_session", None)
        self._viz_node = getattr(obj, "_viz_node", None)

    # NEP-13: ufuncs (np.add/exp/...) and the arithmetic/comparison operators.
    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        session = self._viz_session
        raw = tuple(_strip(x) for x in inputs)
        if out is not None:
            kwargs["out"] = tuple(_strip(o) for o in out)
        result = getattr(ufunc, method)(*raw, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        # only record a plain elementwise/binary call producing an array; reduce/
        # accumulate go through the method overrides, in-place (out=) is skipped
        if (session is None or method != "__call__" or out is not None
                or not isinstance(result, np.ndarray)):
            return result
        traced: list = []
        _flatten_traced(inputs, traced)
        return session.record_op(ufunc.__name__, traced, result)

    # NEP-18: the high-level API (concatenate / mean / split / fft / ...).
    def __array_function__(self, func, types, args, kwargs):
        session = self._viz_session
        result = func(*_strip(tuple(args)), **{k: _strip(v) for k, v in kwargs.items()})
        if session is None:
            return result
        traced: list = []
        _flatten_traced(args, traced)
        op_name = getattr(func, "__name__", None) or str(func)
        if isinstance(result, np.ndarray):
            return session.record_op(op_name, traced, result)
        if (isinstance(result, (list, tuple)) and result
                and all(isinstance(r, np.ndarray) for r in result)):
            return session.record_op(op_name, traced, list(result))
        return result

    # Reduction / manipulation *methods*: these don't dispatch through the two
    # protocols above, so override the common preprocessing ones explicitly.
    def _viz_method(self, op_name, func, args, kwargs, extra=()):
        session = self._viz_session
        base = self.view(np.ndarray)
        result = func(base, *_strip(tuple(args)),
                      **{k: _strip(v) for k, v in kwargs.items()})
        if session is None or not isinstance(result, np.ndarray):
            return result
        return session.record_op(op_name, [self, *extra], result)

    def mean(self, *a, **k):
        return self._viz_method("mean", np.ndarray.mean, a, k)

    def std(self, *a, **k):
        return self._viz_method("std", np.ndarray.std, a, k)

    def var(self, *a, **k):
        return self._viz_method("var", np.ndarray.var, a, k)

    def sum(self, *a, **k):
        return self._viz_method("sum", np.ndarray.sum, a, k)

    def min(self, *a, **k):
        return self._viz_method("min", np.ndarray.min, a, k)

    def max(self, *a, **k):
        return self._viz_method("max", np.ndarray.max, a, k)

    def clip(self, *a, **k):
        return self._viz_method("clip", np.ndarray.clip, a, k)

    def dot(self, b, *a, **k):
        return self._viz_method("dot", np.ndarray.dot, (b, *a), k, extra=(b,))


class NumpyTracer:
    """A numpy preprocessing session writing into a shared ``ExecutionTree``.

    Bind to a torch ``ModuleTracer`` via ``owner`` to share its tree + server
    (one unified DAG, and the chain tail feeds the model). Standalone, it owns
    its own tree and you pass a ``TracerServer``.
    """

    def __init__(self, server: TracerServer, tree: Optional[ExecutionTree] = None,
                 *, owner=None) -> None:
        self.server = server
        self.tree = tree if tree is not None else ExecutionTree()
        self._owner = owner
        self._op_counters: dict[str, int] = {}
        self._published = -1
        self._published_edges = -1

    # -- public: wrap the raw input once at the source ---------------------
    def track(self, array, name: str = "input", *, kind: Optional[str] = None):
        """Begin a preprocessing pass: wrap ``array`` as a tracked source node."""
        self._op_counters.clear()   # a track() marks the start of a fresh pass
        base = np.asarray(array)
        node_id = f"np.{name}"
        self.tree.observe(node_id, name=name, path=f"input.{name}", parent=None,
                          depth=0, shape=tuple(base.shape), role="stage")
        ta = base.view(TracedArray)
        ta._viz_session = self
        ta._viz_node = node_id
        # cross-backend stitch: if an upstream backend (sklearn / probe) handed
        # off a tail, wire it into this source so the whole pipeline stays one
        # chain even though the array object lost its tag at the boundary.
        if self._owner is not None:
            prev = self._owner._last_probe
            if prev is not None and prev != node_id:
                self.tree.link(prev, node_id)
            self._owner._last_probe = node_id
        if self.server.is_subscribed(node_id):
            self._send(node_id, base, kind=kind)
        self._maybe_publish()
        return ta

    # -- record one traced op (called from TracedArray) --------------------
    def record_op(self, op_name: str, inputs: list, output):
        multi = isinstance(output, (list, tuple))
        outs = list(output) if multi else [output]

        srcs = [nid for nid in (getattr(a, "_viz_node", None) for a in inputs)
                if nid is not None]
        role = _op_role(_CANON.get(op_name, op_name), len(set(srcs)), len(outs))

        idx = self._op_counters.get(op_name, 0)
        self._op_counters[op_name] = idx + 1
        node_id = f"np > {op_name}#{idx}"
        first = np.asarray(outs[0])
        self.tree.observe(node_id, name=op_name, path=f"np/{op_name}#{idx}",
                          parent=None, depth=0, shape=tuple(first.shape), role=role)
        for s in dict.fromkeys(srcs):      # dedup, order-stable
            if s != node_id:
                self.tree.link(s, node_id)

        wrapped = []
        for o in outs:
            w = np.asarray(o).view(TracedArray)
            w._viz_session = self
            w._viz_node = node_id
            wrapped.append(w)

        if self._owner is not None:        # hand the chain tail to the model root
            self._owner._last_probe = node_id
        if self.server.is_subscribed(node_id):
            self._send(node_id, first)
        self._maybe_publish()
        return wrapped if multi else wrapped[0]

    # -- helpers -----------------------------------------------------------
    def _step(self) -> int:
        return max(self._owner._step, 0) if self._owner is not None else 0

    def _send(self, node_id: str, arr, *, kind: Optional[str] = None) -> None:
        a = np.asarray(arr, dtype=np.float32)
        self.server.send_frame(node_id, self._step(),
                               protocol.auto_payload(a, kind=kind),
                               protocol.compute_stats(a))

    def _maybe_publish(self) -> None:
        if (len(self.tree) != self._published
                or self.tree.n_edges() != self._published_edges):
            self.server.set_structure(self.tree.nodes(), self.tree.edges())
            self._published = len(self.tree)
            self._published_edges = self.tree.n_edges()

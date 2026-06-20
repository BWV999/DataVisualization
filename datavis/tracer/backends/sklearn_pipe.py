"""sklearn interceptor: a ``Pipeline``'s steps become a traced dataflow chain.

sklearn already exposes a *uniform interface* — every step is an estimator with
``.transform`` / ``.fit_transform`` — which is the tabular dual of
``nn.Module.forward``. So instead of tracing arbitrary code we wrap that one
interface: ``attach_pipeline(pipe)`` shadows each step's transform on the
instance, and a wrapper around the pipeline's own ``fit`` / ``transform`` /
``fit_transform`` marks each pass and seeds an input source node. Calling
``pipe.fit_transform(X)`` then records one node per named step
(``scaler -> pca -> ...``), chained by producer->consumer edges, into the same
``ExecutionTree`` + ``TracerServer`` as the torch tracer — and the chain's tail
is handed to the model root (via ``owner._last_probe``), so a sklearn
preprocessing pipeline and the downstream torch model render as one DAG.

Scope: a flat sequential ``Pipeline`` of transformers. Nested meta-
estimators (``ColumnTransformer``, ``FeatureUnion``) are not unwrapped — their
output still appears as a single step node. Imports no torch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from datavis.common import protocol
from datavis.tracer.graph import ExecutionTree
from datavis.tracer.transport import TracerServer

# pipeline-level methods that delimit one pass over the data
_PASS_METHODS = ("fit", "fit_transform", "transform")
# per-step methods that actually transform the data
_STEP_METHODS = ("transform", "fit_transform")


class SklearnTracer:
    """Wraps one sklearn ``Pipeline`` so its steps record into a shared tree.

    Bind to a torch ``ModuleTracer`` via ``owner`` to share its tree + server
    (one unified DAG, chain tail feeds the model). Standalone, it owns its tree
    and you pass a ``TracerServer``.
    """

    def __init__(self, server: TracerServer, tree: Optional[ExecutionTree] = None,
                 *, owner=None) -> None:
        self.server = server
        self.tree = tree if tree is not None else ExecutionTree()
        self._owner = owner
        self._chain_prev: Optional[str] = None
        self._busy = False                    # re-entrancy guard for nested steps
        self._restore: list = []              # (obj, attr) instance shadows to drop
        self._published = -1
        self._published_edges = -1

    # -- attach / detach ---------------------------------------------------
    def attach(self, pipe):
        for attr in _PASS_METHODS:
            orig = _get_method(pipe, attr)
            if orig is not None:
                self._shadow(pipe, attr, self._make_pass(orig))
        for name, est in pipe.steps:
            for attr in _STEP_METHODS:
                orig = _get_method(est, attr)
                if orig is not None:
                    self._shadow(est, attr, self._make_step(orig, name, est))
        return self

    def detach(self) -> None:
        for obj, attr in self._restore:
            try:
                delattr(obj, attr)            # drop the instance shadow -> class method
            except AttributeError:
                pass
        self._restore.clear()

    def _shadow(self, obj, attr: str, wrapped) -> None:
        object.__setattr__(obj, attr, wrapped)
        self._restore.append((obj, attr))

    # -- wrappers ----------------------------------------------------------
    def _make_pass(self, orig):
        def pass_wrapper(X, *args, **kwargs):
            self._begin_pass(X)
            result = orig(X, *args, **kwargs)
            self._end_pass()
            return result
        return pass_wrapper

    def _make_step(self, orig, name: str, est):
        def step_wrapper(X, *args, **kwargs):
            if self._busy:                    # nested call (fit_transform -> transform)
                return orig(X, *args, **kwargs)
            self._busy = True
            try:
                result = orig(X, *args, **kwargs)
            finally:
                self._busy = False
            self._record_step(name, est, result)
            return result
        return step_wrapper

    # -- recording ---------------------------------------------------------
    def _begin_pass(self, X) -> None:
        node_id = "sk.input"
        shape = _shape(X)
        self.tree.observe(node_id, name="input", path="pipeline.input",
                          parent=None, depth=0, shape=shape, role="stage")
        # cross-backend stitch: an upstream backend (numpy / probe) may have
        # handed off a tail via the shared cursor; wire it into this pipeline's
        # input so numpy -> sklearn stays one chain across the boundary (where
        # sklearn's check_array strips any TracedArray tag).
        if self._owner is not None and self._owner._last_probe not in (None, node_id):
            self.tree.link(self._owner._last_probe, node_id)
        self._chain_prev = node_id
        self._capture(node_id, X)
        self._maybe_publish()

    def _record_step(self, name: str, est, output) -> None:
        node_id = f"sk.{name}"
        self.tree.observe(node_id, name=type(est).__name__,
                          path=f"pipeline.{name}", parent=None, depth=0,
                          shape=_shape(output), role=None)
        if self._chain_prev is not None and self._chain_prev != node_id:
            self.tree.link(self._chain_prev, node_id)
        self._chain_prev = node_id
        self._capture(node_id, output)
        self._maybe_publish()

    def _end_pass(self) -> None:
        if self._owner is not None and self._chain_prev is not None:
            self._owner._last_probe = self._chain_prev   # tail feeds the model root

    # -- helpers -----------------------------------------------------------
    def _step(self) -> int:
        return max(self._owner._step, 0) if self._owner is not None else 0

    def _capture(self, node_id: str, data) -> None:
        if not self.server.is_subscribed(node_id):
            return
        arr = _as_float_array(data)
        if arr is None:
            return
        self.server.send_frame(node_id, self._step(),
                               protocol.auto_payload(arr), protocol.compute_stats(arr))

    def _maybe_publish(self) -> None:
        if (len(self.tree) != self._published
                or self.tree.n_edges() != self._published_edges):
            self.server.set_structure(self.tree.nodes(), self.tree.edges())
            self._published = len(self.tree)
            self._published_edges = self.tree.n_edges()


# ---- module helpers ------------------------------------------------------
def _get_method(obj, attr: str):
    """Bound method ``obj.attr`` if it exists, else None.

    sklearn gates ``transform`` / ``fit_transform`` behind ``available_if``
    descriptors that *raise* when unavailable, so ``hasattr`` is the right probe.
    """
    if not hasattr(obj, attr):
        return None
    method = getattr(obj, attr)
    return method if callable(method) else None


def _shape(X) -> tuple:
    try:
        return tuple(np.asarray(X).shape)
    except (TypeError, ValueError):
        return ()


def _as_float_array(data):
    try:
        return np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError):
        return None

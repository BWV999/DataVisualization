"""Tracer library: runs inside the training process.

Public API:
    import datavis.tracer as viz
    viz.attach(model)                 # persistent zero-intrusion hooks
    with viz.trace(model): train(...)  # scoped variant

``TracerServer`` is the lower-level IPC server (used directly by the no-torch
``examples/demo_signal.py``). ``attach`` / ``trace`` import torch lazily, so the
package stays importable without torch installed.
"""
from contextlib import contextmanager

from datavis.tracer.transport import TracerServer

__all__ = ["TracerServer", "attach", "trace", "probe", "region", "track",
           "attach_pipeline"]

_DEFAULT_CTRL = "tcp://127.0.0.1:5750"
_DEFAULT_DATA = "tcp://127.0.0.1:5751"

# the most recently attached tracer; ``probe`` routes preprocessing stages to it
_active = None
# standalone backend sessions, used when no model is attached
_np_standalone = None
_sk_standalone = None
# the one process-shared server on the default ports (see _get_default_server)
_default_server = None


def _get_default_server() -> TracerServer:
    """The single process-shared server bound to the default ports.

    ``attach`` / ``track`` / ``attach_pipeline`` all adopt it, so calling
    ``track`` or ``attach_pipeline`` *before* ``attach`` doesn't bind the default
    ports out from under the later ``attach`` (which would self-conflict with an
    "Address already in use" inside one process).
    """
    global _default_server
    if _default_server is None:
        _default_server = TracerServer(_DEFAULT_CTRL, _DEFAULT_DATA)
    return _default_server


def attach(
    model,
    ctrl_addr: str = _DEFAULT_CTRL,
    data_addr: str = _DEFAULT_DATA,
    server: TracerServer | None = None,
    op_max_repeat: int | None = None,
):
    """Register auto-trace hooks on ``model``. Returns a ModuleTracer.

    With the default ports, the one process-shared server is used (so it is safe
    to ``track`` / ``attach_pipeline`` before ``attach``); with custom ports a
    dedicated server is created and owned by the returned tracer (closed by
    ``trace`` on exit).

    ``op_max_repeat`` bounds loop unrolling at op level: a repeated op gets at
    most this many distinct nodes before further repeats roll into one recurrent
    node (default 1 = full roll; ``None`` honours ``$DATAVIS_OP_MAX_REPEAT``).
    """
    from datavis.tracer.hooks import ModuleTracer

    global _active
    if server is not None:
        owns_server = False
    elif ctrl_addr == _DEFAULT_CTRL and data_addr == _DEFAULT_DATA:
        server = _get_default_server()        # shared; may already exist
        owns_server = False
    else:
        server = TracerServer(ctrl_addr, data_addr)
        owns_server = True
    tracer = ModuleTracer(model, server, op_max_repeat=op_max_repeat)
    tracer._owns_server = owns_server
    _active = tracer
    return tracer


def probe(name: str, data, *, kind: str | None = None) -> None:
    """Visualize an out-of-model preprocessing stage (raw input, scaling,
    windowing, ...) as an upstream node feeding the model.

        x = load_batch()
        viz.probe("raw", x)
        x = scaler.transform(x); viz.probe("standardized", x)
        out = model(x)            # the graph shows raw -> standardized -> model

    ``data`` may be a torch tensor, numpy array, pandas object, or list. A no-op
    if no model has been attached yet, so probes left in data code cost nothing.
    """
    if _active is not None:
        _active.probe(name, data, kind=kind)


@contextmanager
def region(name: str = "preprocess", *inputs, level: int | None = None):
    """Auto-trace torch ops run OUTSIDE the model forward (preprocessing the
    forward-only op-mode never sees) as nodes under a ``pre.{name}`` stage,
    stitched into the unified DAG feeding the model.

        x = load_batch()
        with viz.region("standardize", x):     # x sourced from the stage node
            x = (x - x.mean(0)) / (x.std(0) + 1e-5)   # auto add/sub/div nodes
        out = model(x)                          # stage tail -> model root

    A no-op (just yields) until a model is attached, so it's free to leave in
    data code. Mix ``viz.probe(...)`` inside for non-torch steps (pandas etc.) —
    both advance the same chain cursor. ``level`` overrides the op detail
    (default ALL: a region's steps are the content to show).
    """
    if _active is None:
        yield
        return
    kw = {} if level is None else {"level": level}
    with _active.region(name, *inputs, **kw):
        yield


def track(array, name: str = "input", *, kind: str | None = None):
    """Wrap a raw numpy array so the whole numpy preprocessing chain downstream
    is auto-recorded — no per-stage ``probe`` needed.

        x = viz.track(load_raw())          # wrap once at the source
        x = (x - x.mean(0)) / (x.std(0) + 1e-5)   # auto nodes/edges
        x = np.concatenate([x, feats], axis=-1)
        out = model(torch.as_tensor(x))    # chain feeds the model root

    If a model is attached (``viz.attach``), the numpy nodes share its DAG and
    the chain's tail is wired into the model root; otherwise a standalone
    session (own server) is used. Returns a ``TracedArray`` (a numpy ndarray
    subclass) — drop-in for the original array.
    """
    from datavis.tracer.backends.numpy_trace import NumpyTracer

    if _active is not None:
        np_tr = getattr(_active, "_np", None)
        if np_tr is None or np_tr._owner is not _active:
            np_tr = _active._np = NumpyTracer(_active.server, _active.tree,
                                              owner=_active)
        return np_tr.track(array, name, kind=kind)

    global _np_standalone
    if _np_standalone is None:
        _np_standalone = NumpyTracer(_get_default_server())
    return _np_standalone.track(array, name, kind=kind)


def attach_pipeline(pipe):
    """Trace an sklearn ``Pipeline``: each step becomes an upstream node.

        pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(8))])
        viz.attach_pipeline(pipe)
        Xt = pipe.fit_transform(X)         # graph: input -> scaler -> pca
        out = model(torch.as_tensor(Xt))   # ... -> pca -> model

    Wraps each step's ``transform`` / ``fit_transform`` (restored by
    ``detach()`` on the returned tracer). If a model is attached, the pipeline
    shares its DAG and the chain tail is wired into the model root; otherwise a
    standalone session (own server) is used. Returns the ``SklearnTracer``.
    """
    from datavis.tracer.backends.sklearn_pipe import SklearnTracer

    if _active is not None:
        sk = getattr(_active, "_sk", None)
        if sk is None or sk._owner is not _active:
            sk = _active._sk = SklearnTracer(_active.server, _active.tree,
                                             owner=_active)
        return sk.attach(pipe)

    global _sk_standalone
    if _sk_standalone is None:
        _sk_standalone = SklearnTracer(_get_default_server())
    return _sk_standalone.attach(pipe)


@contextmanager
def trace(model, **kwargs):
    global _active
    tracer = attach(model, **kwargs)
    try:
        yield tracer
    finally:
        tracer.remove()
        if getattr(tracer, "_owns_server", False):
            tracer.server.close()
        if _active is tracer:
            _active = None

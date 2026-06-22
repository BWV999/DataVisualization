"""Tracer library: runs inside the training process.

Public API:
    import datavis.tracer as viz
    viz.attach(model)                  # persistent zero-intrusion hooks
    viz.attach(model, bind="0.0.0.0")  # reachable from a remote GUI / SSH tunnel
    viz.reattach(next_model)           # k-fold: swap the hooked model, keep ports
    viz.serve(model, batches)          # keep alive so a GUI can join after training
    with viz.trace(model): train(...)  # scoped variant

``TracerServer`` is the lower-level IPC server (used directly by the no-torch
``examples/demo_signal.py``). ``attach`` / ``trace`` import torch lazily, so the
package stays importable without torch installed.
"""
import os
import socket
import sys
import time
from contextlib import contextmanager, nullcontext

from datavis.tracer.transport import TracerServer

__all__ = ["TracerServer", "attach", "reattach", "detach", "trace", "serve",
           "probe", "region", "track", "attach_pipeline"]

# Default bind host / ports. The host is configurable so the "remote GPU box +
# local GUI over an SSH tunnel" workflow works without hand-writing addresses:
# pass ``attach(bind="0.0.0.0")`` or export ``DATAVIS_BIND=0.0.0.0``.
_DEFAULT_BIND = "127.0.0.1"
_DEFAULT_CTRL_PORT = "5750"
_DEFAULT_DATA_PORT = "5751"

# the most recently attached tracer; ``probe`` routes preprocessing stages to it
_active = None
# standalone backend sessions, used when no model is attached
_np_standalone = None
_sk_standalone = None
# Servers cached by (ctrl_addr, data_addr). Repeat ``attach`` on the same ports
# adopts the running server and rebinds the hooks to the new model instead of
# re-binding the socket ("Address already in use") — the k-fold case. This also
# subsumes the old default-port singleton: calling ``track`` / ``attach_pipeline``
# before ``attach`` no longer binds the ports out from under the later ``attach``.
_servers: dict[tuple[str, str], TracerServer] = {}
# the tracer currently hooked onto each server, so a re-attach can unhook the old
_server_tracer: dict[tuple[str, str], object] = {}


def _resolve_addrs(ctrl_addr=None, data_addr=None, bind=None) -> tuple[str, str]:
    """Build the ctrl/data addresses from an explicit value, a ``bind`` host, or
    the ``DATAVIS_BIND`` / ``DATAVIS_CTRL_PORT`` / ``DATAVIS_DATA_PORT`` env."""
    host = bind or os.environ.get("DATAVIS_BIND") or _DEFAULT_BIND
    cport = os.environ.get("DATAVIS_CTRL_PORT", _DEFAULT_CTRL_PORT)
    dport = os.environ.get("DATAVIS_DATA_PORT", _DEFAULT_DATA_PORT)
    if ctrl_addr is None:
        ctrl_addr = f"tcp://{host}:{cport}"
    if data_addr is None:
        data_addr = f"tcp://{host}:{dport}"
    return ctrl_addr, data_addr


def _get_server(ctrl_addr: str, data_addr: str) -> tuple[TracerServer, bool]:
    """The cached server for these addresses (binding it on first use), and
    whether it was freshly created. A cached server that was closed out-of-band
    (e.g. ``tracer.server.close()``) is replaced rather than handed back dead."""
    key = (ctrl_addr, data_addr)
    srv = _servers.get(key)
    if srv is None or srv.closed:
        srv = _servers[key] = TracerServer(ctrl_addr, data_addr)
        _server_tracer.pop(key, None)
        return srv, True
    return srv, False


def _announce(server: TracerServer, *, adopted: bool, quiet: bool) -> None:
    """Tell the user where the tracer is listening, so they know what to tunnel
    to. Suppress with ``quiet=True`` or ``DATAVIS_QUIET=1``."""
    if quiet or os.environ.get("DATAVIS_QUIET"):
        return
    verb = "re-bound on" if adopted else "listening on"
    print(f"[datavis] tracer {verb} {socket.gethostname()} "
          f"{server.ctrl_addr} (ctrl) / {server.data_addr} (data)",
          file=sys.stderr)
    if "127.0.0.1" in server.ctrl_addr or "localhost" in server.ctrl_addr:
        print("[datavis]   (local-only; for a remote GUI use bind=\"0.0.0.0\" "
              "or DATAVIS_BIND=0.0.0.0)", file=sys.stderr)


def attach(
    model,
    ctrl_addr: str | None = None,
    data_addr: str | None = None,
    *,
    bind: str | None = None,
    server: TracerServer | None = None,
    op_max_repeat: int | None = None,
    quiet: bool = False,
):
    """Register auto-trace hooks on ``model``. Returns a ModuleTracer.

    Addresses resolve from ``ctrl_addr``/``data_addr`` if given, else from
    ``bind`` (host) / ``DATAVIS_BIND`` / the default ports. The server for those
    addresses is cached: a second ``attach`` on the same ports adopts it and
    rebinds the hooks to the new model (see ``reattach``) — so a k-fold loop that
    builds a fresh model per fold no longer hits "Address already in use".

    ``op_max_repeat`` bounds loop unrolling at op level: a repeated op gets at
    most this many distinct nodes before further repeats roll into one recurrent
    node (default 1 = full roll; ``None`` honours ``$DATAVIS_OP_MAX_REPEAT``).
    """
    from datavis.tracer.hooks import ModuleTracer

    global _active
    if server is not None:
        key = None
        adopted = False
        created = False
    else:
        ctrl_addr, data_addr = _resolve_addrs(ctrl_addr, data_addr, bind)
        key = (ctrl_addr, data_addr)
        server, created = _get_server(ctrl_addr, data_addr)
        adopted = not created              # reused a live server on these ports
        if adopted:
            prev = _server_tracer.get(key)
            if prev is not None:           # rebind: unhook the previous model
                prev.remove()
    tracer = ModuleTracer(model, server, op_max_repeat=op_max_repeat)
    tracer._server_key = key
    # ``trace`` tears down only a server it created on a *custom* address — the
    # default-port server is shared and left alive, as before.
    tracer._close_on_trace = created and key != _resolve_addrs()
    if key is not None:
        _server_tracer[key] = tracer
    _active = tracer
    _announce(server, adopted=adopted, quiet=quiet)
    return tracer


def reattach(model, *, quiet: bool = False, **kwargs):
    """Swap the hooked model on the *current* tracer's server, keeping it bound.

    The natural k-fold / multi-model primitive: ``viz.attach`` once for the first
    fold, then ``viz.reattach(next_model)`` for each subsequent fold. Equivalent
    to ``attach`` on the same ports (it unhooks the previous model first).
    """
    if _active is not None and getattr(_active, "_server_key", None) is not None:
        ctrl_addr, data_addr = _active._server_key
        return attach(model, ctrl_addr=ctrl_addr, data_addr=data_addr,
                      quiet=quiet, **kwargs)
    return attach(model, quiet=quiet, **kwargs)


def detach(tracer=None, *, close_server: bool = False) -> None:
    """Remove a tracer's hooks (default: the active one), keeping its server bound
    so a later ``attach``/``reattach`` reuses it. Pass ``close_server=True`` to
    also release the ports."""
    global _active
    tracer = tracer if tracer is not None else _active
    if tracer is None:
        return
    tracer.remove()
    key = getattr(tracer, "_server_key", None)
    if key is not None and _server_tracer.get(key) is tracer:
        _server_tracer.pop(key, None)
    if close_server and key is not None:
        srv = _servers.pop(key, None)
        if srv is not None:
            srv.close()
    if _active is tracer:
        _active = None


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
        _np_standalone = NumpyTracer(_get_server(*_resolve_addrs())[0])
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
        _sk_standalone = SklearnTracer(_get_server(*_resolve_addrs())[0])
    return _sk_standalone.attach(pipe)


def _run_forward(model, batch, forward) -> None:
    """Drive one forward for ``serve`` (the only framework-specific guesswork)."""
    if forward is not None:
        forward(model, batch)
    elif isinstance(batch, dict):
        model(**batch)
    elif isinstance(batch, (list, tuple)):
        model(*batch)
    else:
        model(batch)


def serve(
    model=None,
    batches=None,
    *,
    forward=None,
    interval: float = 0.5,
    quiet: bool = False,
    **attach_kwargs,
):
    """Keep the tracer alive so a GUI can connect *after* the training loop ends,
    and inspect a finished run — blocks until Ctrl-C.

        viz.serve(model, val_batches, bind="0.0.0.0")

    The tracer's server lives inside this process; once the process exits the
    ports close and a late GUI gets "connection refused". ``serve`` is the
    first-class replacement for a hand-rolled keep-alive script:

    - With ``batches`` it re-forwards them in a loop so subscribed nodes keep
      streaming live (capture is on-demand, so an idle GUI still costs nothing).
      ``batches`` may be a list / DataLoader (re-iterated each cycle) or a
      zero-arg factory returning a fresh iterable; a one-shot generator runs once
      then the call simply lingers. Each batch is fed as ``model(batch)``, or
      ``model(*batch)`` / ``model(**batch)`` for a tuple/dict — override with a
      ``forward(model, batch)`` callable.
    - Without ``batches`` it just holds the process (and its last-published
      structure) open.

    Any ``attach_kwargs`` (``bind=``, ``op_max_repeat=``, ...) are forwarded to
    ``attach`` when ``model`` is given; pass an already-attached run with
    ``serve()`` / ``serve(batches=...)``.
    """
    if model is not None:
        attach(model, quiet=quiet, **attach_kwargs)
    elif attach_kwargs:
        raise TypeError("serve(**attach_kwargs) requires a model to attach")
    if _active is None:
        raise RuntimeError("serve() needs an attached model (call attach first)")
    target = _active.model

    server = _active.server
    if not quiet and not os.environ.get("DATAVIS_QUIET"):
        print(f"[datavis] serving on {server.ctrl_addr} (ctrl) / "
              f"{server.data_addr} (data) — Ctrl-C to stop", file=sys.stderr)

    # torch, if loaded, must not build autograd graphs while we re-forward.
    torch = sys.modules.get("torch")
    cm = torch.inference_mode() if torch is not None else nullcontext()

    try:
        with cm:
            if batches is None:
                while True:
                    time.sleep(interval)
            else:
                src = batches() if callable(batches) else batches
                repeatable = callable(batches) or iter(src) is not src
                while True:
                    produced = False
                    for batch in (batches() if callable(batches) else src):
                        produced = True
                        try:
                            _run_forward(target, batch, forward)
                        except Exception as exc:  # keep serving past a bad batch
                            print(f"[datavis] forward failed: {exc!r}",
                                  file=sys.stderr)
                        time.sleep(interval)
                    if not (repeatable and produced):
                        while True:           # exhausted / non-repeatable: hold
                            time.sleep(interval)
    except KeyboardInterrupt:
        if not quiet:
            print("\n[datavis] stopped serving", file=sys.stderr)


@contextmanager
def trace(model, **kwargs):
    tracer = attach(model, **kwargs)
    try:
        yield tracer
    finally:
        detach(tracer, close_server=getattr(tracer, "_close_on_trace", False))

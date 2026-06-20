"""Wire protocol for DataVisualization IPC.

Every message is a msgpack-encoded dict carrying a ``type`` field. numpy arrays
are encoded inline as ``{"__nd__": True, "dtype", "shape", "data"}`` so that both
control metadata and tensor payloads share one transport.

Payload helpers (``auto_payload``, ``compute_stats``) live here so the tracer and
the GUI agree on exactly how a tensor is reduced for the wire.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import msgpack
import numpy as np

# ---- message types -------------------------------------------------------
MSG_HELLO = "hello"            # GUI -> tracer (control), reply is a structure msg
MSG_STRUCTURE = "structure"    # tracer -> GUI: the execution tree
MSG_SUBSCRIBE = "subscribe"    # GUI -> tracer: start capturing a node
MSG_UNSUBSCRIBE = "unsubscribe"
MSG_SET_OPLEVEL = "set_oplevel"  # GUI -> tracer: enable/disable op-level tracing
MSG_ACK = "ack"
MSG_FRAME = "frame"            # tracer -> GUI (data): one captured tensor

# ---- op-level detail tiers ----------------------------------------------
# How much of the op graph (below modules) the tracer materializes:
OPLEVEL_OFF = 0     # modules only (default; the op-mode is never entered)
OPLEVEL_STRUCT = 1  # + structural ops only: merges (add/mul/cat/matmul) & splits
OPLEVEL_ALL = 2     # + every op (view/transpose/getitem/...): the deep LOD


def normalize_oplevel(value) -> int:
    """Accept bool (legacy) or int; clamp to a known tier."""
    if isinstance(value, bool):
        return OPLEVEL_ALL if value else OPLEVEL_OFF
    try:
        v = int(value)
    except (TypeError, ValueError):
        return OPLEVEL_OFF
    return max(OPLEVEL_OFF, min(OPLEVEL_ALL, v))


# ---- payload kinds -------------------------------------------------------
KIND_CURVE = "curve"      # y: (channels, T) for y-t plots
KIND_HEATMAP = "heatmap"  # z: (H, W)
KIND_NONE = "none"        # scalar / unsupported rank; rely on stats only

# Above this many leading-dim "channels" a rank-2 tensor is shown as a heatmap
# instead of overlaid y-t curves.
MAX_CURVE_CHANNELS = 16


# ---- numpy <-> msgpack ---------------------------------------------------
def _pack_ndarray(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "__nd__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


def _default(obj: Any):
    if isinstance(obj, np.ndarray):
        return _pack_ndarray(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Cannot serialize object of type {type(obj)!r}")


def _object_hook(obj: dict):
    if obj.get("__nd__"):
        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


def encode(msg: dict) -> bytes:
    return msgpack.packb(msg, default=_default, use_bin_type=True)


def decode(buf: bytes) -> dict:
    return msgpack.unpackb(buf, object_hook=_object_hook, raw=False)


# ---- tensor reduction for the wire --------------------------------------
def compute_stats(arr: np.ndarray) -> dict:
    """Cheap summary stats sent for every frame regardless of payload size."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "norm": 0.0}
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "norm": float(np.linalg.norm(a)),
    }


def _decimate_1d(y: np.ndarray, max_points: int) -> np.ndarray:
    n = y.shape[-1]
    if n <= max_points:
        return y
    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    return y[..., idx]


def _downscale_2d(z: np.ndarray, max_side: int) -> np.ndarray:
    h, w = z.shape[:2]
    sh = max(1, int(np.ceil(h / max_side)))
    sw = max(1, int(np.ceil(w / max_side)))
    return z[::sh, ::sw]


# How an ND tensor (rank >= 3) is collapsed to a renderable 2D slice:
REDUCE_SLICE = "slice"  # keep a real representative sample (index 0 of each
#                         leading axis) -> temporal/feature structure survives
REDUCE_MEAN = "mean"    # average leading axes -> smoother, but blends distinct
#                         samples/series (can hide per-series time semantics)


def _reduce_leading(a: np.ndarray, how: str) -> tuple[np.ndarray, list[int]]:
    """Collapse leading axes of an ND array down to its trailing 2D, returning
    the 2D array and, for ``slice``, the index taken on each collapsed axis."""
    idx: list[int] = []
    while a.ndim > 2:
        if how == REDUCE_MEAN:
            a = a.mean(axis=0)
        else:
            idx.append(0)
            a = a[0]
    return a, idx


def auto_payload(
    arr: np.ndarray,
    *,
    max_points: int = 2000,
    max_side: int = 256,
    kind: Optional[str] = None,
    reduce: str = REDUCE_SLICE,
) -> dict:
    """Reduce an arbitrary-rank tensor to a compact, renderable payload.

    Rank heuristics (overridable via ``kind``):
      0D            -> none (stats only)
      1D            -> curve, single channel
      2D <=16 chans -> curve, multi-channel; otherwise heatmap
      >=3D          -> collapse leading axes to 2D (``reduce``: a real ``slice``
                       by default, or ``mean``), then heatmap

    For rank >= 3 the payload carries ``src_shape`` + ``reduced`` so the GUI can
    label *what* it's showing (e.g. one sample of a batch) instead of presenting
    a silent, lossy projection as if it were the whole tensor. ND always yields a
    heatmap (kind defaults stay consistent with ``gui.panels.make_panel``).
    """
    a = np.asarray(arr)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    nd = a.ndim

    if kind == KIND_NONE or nd == 0:
        return {"kind": KIND_NONE}

    src_shape = list(a.shape)
    reduced_idx: Optional[list[int]] = None
    if nd > 2:
        a, reduced_idx = _reduce_leading(a, reduce)

    if kind == KIND_CURVE or (kind is None and a.ndim == 1):
        y = a.reshape(1, -1) if a.ndim == 1 else a
        payload = {"kind": KIND_CURVE, "y": _decimate_1d(y, max_points)}
    elif kind is None and nd == 2 and a.shape[0] <= MAX_CURVE_CHANNELS:
        payload = {"kind": KIND_CURVE, "y": _decimate_1d(a, max_points)}
    else:
        z = a if a.ndim == 2 else a.reshape(1, -1)
        payload = {"kind": KIND_HEATMAP, "z": _downscale_2d(z, max_side)}

    if nd > 2:                      # tell the GUI this is a projection, and how
        payload["src_shape"] = src_shape
        payload["reduced"] = reduce
        if reduced_idx is not None:
            payload["reduced_idx"] = reduced_idx
    return payload

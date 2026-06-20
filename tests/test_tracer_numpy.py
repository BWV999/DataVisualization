"""numpy interceptor (NEP-13 / NEP-18) — torch-free.

Verifies that wrapping a raw array once with the session's ``track`` records the
whole downstream numpy preprocessing chain into the ExecutionTree: nodes per op,
producer->consumer edges, structural roles, transparent views, multi-output
splits, and on-demand capture.
"""
import time

import numpy as np

from datavis.common import protocol
from datavis.gui.transport import GuiClient
from datavis.tracer.backends.numpy_trace import NumpyTracer, TracedArray
from datavis.tracer.transport import TracerServer


def _server(base):
    return TracerServer(f"tcp://127.0.0.1:{base}", f"tcp://127.0.0.1:{base + 1}")


def _nodes(tr):
    return {n["id"]: n for n in tr.tree.nodes()}


def _edges(tr):
    return set(map(tuple, tr.tree.edges()))


def test_track_source_is_upstream_stage():
    srv = _server(5820)
    tr = NumpyTracer(srv)
    try:
        x = tr.track(np.random.randn(4, 8).astype(np.float32), "raw")
        assert isinstance(x, TracedArray)
        n = _nodes(tr)["np.raw"]
        assert n["parent"] is None and n["role"] == "stage" and n["depth"] == 0
        assert n["shape"] == [4, 8]
    finally:
        srv.close()


def test_standardization_chain_nodes_and_edges():
    srv = _server(5822)
    tr = NumpyTracer(srv)
    try:
        x = tr.track(np.random.randn(6, 4).astype(np.float32), "raw")
        x = (x - x.mean(0)) / (x.std(0) + 1e-5)
        assert isinstance(x, TracedArray)

        nodes = _nodes(tr)
        edges = _edges(tr)
        # every transform step is its own node (the whole evolution, not folded).
        # ('/' is the np.divide ufunc; its __name__ is "divide" on numpy >= 2.)
        for nid in ("np > mean#0", "np > std#0", "np > subtract#0",
                    "np > add#0", "np > divide#0"):
            assert nid in nodes, nid
        # structural roles classified through the shared _op_role
        assert nodes["np > subtract#0"]["role"] == "sub"   # x and mean both traced
        assert nodes["np > divide#0"]["role"] == "div"
        assert nodes["np > mean#0"]["role"] is None         # reduction -> plain box

        # producer->consumer edges form the chain from the raw source
        assert ("np.raw", "np > mean#0") in edges
        assert ("np.raw", "np > std#0") in edges
        assert ("np.raw", "np > subtract#0") in edges
        assert ("np > mean#0", "np > subtract#0") in edges
        assert ("np > std#0", "np > add#0") in edges
        assert ("np > subtract#0", "np > divide#0") in edges
        assert ("np > add#0", "np > divide#0") in edges

        # source is upstream of every op it feeds
        assert nodes["np.raw"]["order"] < nodes["np > divide#0"]["order"]
    finally:
        srv.close()


def test_concatenate_is_a_merge_via_array_function():
    srv = _server(5824)
    tr = NumpyTracer(srv)
    try:
        a = tr.track(np.ones((3, 2), np.float32), "a")
        b = tr.track(np.zeros((3, 5), np.float32), "b")
        c = np.concatenate([a, b], axis=1)     # NEP-18 dispatch
        assert isinstance(c, TracedArray) and c.shape == (3, 7)

        nodes, edges = _nodes(tr), _edges(tr)
        assert nodes["np > concatenate#0"]["role"] == "concat"  # 2 producers
        assert ("np.a", "np > concatenate#0") in edges
        assert ("np.b", "np > concatenate#0") in edges
    finally:
        srv.close()


def test_split_is_multi_output():
    srv = _server(5826)
    tr = NumpyTracer(srv)
    try:
        x = tr.track(np.arange(12, dtype=np.float32).reshape(2, 6), "raw")
        parts = np.split(x, 3, axis=1)         # one node, fanning out
        assert len(parts) == 3 and all(isinstance(p, TracedArray) for p in parts)

        nodes, edges = _nodes(tr), _edges(tr)
        assert nodes["np > split#0"]["role"] == "split"
        assert ("np.raw", "np > split#0") in edges
        # each output carries the split as its producer
        assert all(p._viz_node == "np > split#0" for p in parts)
    finally:
        srv.close()


def test_view_is_transparent_passthrough():
    srv = _server(5828)
    tr = NumpyTracer(srv)
    try:
        x = tr.track(np.random.randn(4, 8).astype(np.float32), "raw")
        n0 = len(tr.tree)
        sl = x[:, :3]                          # a slice/view adds no node
        assert isinstance(sl, TracedArray)
        assert sl._viz_node == "np.raw"        # inherits the source producer
        assert len(tr.tree) == n0
    finally:
        srv.close()


def test_op_ids_stable_across_passes():
    srv = _server(5830)
    tr = NumpyTracer(srv)
    try:
        for _ in range(3):                     # a loop re-tracks each pass
            x = tr.track(np.random.randn(5, 4).astype(np.float32), "raw")
            _ = (x - x.mean(0)) / (x.std(0) + 1e-5)
        # node + edge sets must not grow per pass (stable ids, deduped edges)
        nodes = _nodes(tr)
        assert sum(1 for k in nodes if k.startswith("np > mean")) == 1
        assert sum(1 for k in nodes if k.startswith("np > subtract")) == 1
    finally:
        srv.close()


def test_capture_subscribed_numpy_node():
    srv = _server(5832)
    tr = NumpyTracer(srv)
    received: list[dict] = []
    client = GuiClient("tcp://127.0.0.1:5832", "tcp://127.0.0.1:5833")
    client.start(received.append)
    try:
        tr.track(np.random.randn(4, 8).astype(np.float32), "raw")  # publish struct
        client.hello()
        client.subscribe("np.raw", rate=0)

        deadline = time.time() + 3.0
        while (not any(m.get("node_id") == "np.raw" for m in received
                       if m.get("type") == protocol.MSG_FRAME)
               and time.time() < deadline):
            tr.track(np.random.randn(4, 8).astype(np.float32), "raw")
            time.sleep(0.02)

        frames = [m for m in received
                  if m.get("type") == protocol.MSG_FRAME and m["node_id"] == "np.raw"]
        assert frames, "no frame captured for subscribed numpy source"
    finally:
        client.stop()
        srv.close()

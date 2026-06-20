"""sklearn Pipeline interceptor — torch-free.

Verifies that ``attach_pipeline`` turns a Pipeline's steps into a traced chain
(one node per named step, chained by edges) by wrapping the uniform
``transform`` / ``fit_transform`` interface, with no per-step probes.
"""
import time

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: E402

from datavis.common import protocol  # noqa: E402
from datavis.gui.transport import GuiClient  # noqa: E402
from datavis.tracer.backends.sklearn_pipe import SklearnTracer  # noqa: E402
from datavis.tracer.transport import TracerServer  # noqa: E402


def _server(base):
    return TracerServer(f"tcp://127.0.0.1:{base}", f"tcp://127.0.0.1:{base + 1}")


def _make_pipe():
    return Pipeline([("scaler", StandardScaler()),
                     ("pca", PCA(n_components=4))])


def _nodes(tr):
    return {n["id"]: n for n in tr.tree.nodes()}


def _edges(tr):
    return set(map(tuple, tr.tree.edges()))


def test_pipeline_steps_become_a_chain():
    srv = _server(5840)
    tr = SklearnTracer(srv)
    pipe = _make_pipe()
    try:
        tr.attach(pipe)
        X = np.random.randn(20, 8).astype(np.float32)
        Xt = pipe.fit_transform(X)
        assert Xt.shape == (20, 4)

        nodes, edges = _nodes(tr), _edges(tr)
        # an input source plus one node per named step
        assert nodes["sk.input"]["role"] == "stage" and nodes["sk.input"]["parent"] is None
        assert nodes["sk.scaler"]["name"] == "StandardScaler"
        assert nodes["sk.pca"]["name"] == "PCA"
        assert nodes["sk.pca"]["shape"] == [20, 4]      # output shape captured
        # chained in pipeline order
        assert ("sk.input", "sk.scaler") in edges
        assert ("sk.scaler", "sk.pca") in edges
        assert (nodes["sk.input"]["order"] < nodes["sk.scaler"]["order"]
                < nodes["sk.pca"]["order"])
    finally:
        tr.detach()
        srv.close()


def test_transform_pass_is_stable():
    srv = _server(5842)
    tr = SklearnTracer(srv)
    pipe = _make_pipe()
    try:
        tr.attach(pipe)
        X = np.random.randn(16, 8).astype(np.float32)
        pipe.fit_transform(X)               # builds the structure
        n_nodes, n_edges = len(tr.tree), tr.tree.n_edges()

        for _ in range(3):                  # re-running must not grow it
            pipe.transform(np.random.randn(16, 8).astype(np.float32))
        assert len(tr.tree) == n_nodes and tr.tree.n_edges() == n_edges
    finally:
        tr.detach()
        srv.close()


def test_detach_restores_original_methods():
    srv = _server(5844)
    tr = SklearnTracer(srv)
    pipe = _make_pipe()
    scaler = pipe.steps[0][1]
    try:
        tr.attach(pipe)
        # the wrappers are instance-level shadows
        assert "transform" in scaler.__dict__ and "transform" in pipe.__dict__
        tr.detach()
        # ... dropped on detach, falling back to the class methods
        assert "transform" not in scaler.__dict__ and "transform" not in pipe.__dict__
        # still functional after restore
        out = pipe.fit_transform(np.random.randn(12, 8).astype(np.float32))
        assert out.shape == (12, 4)
    finally:
        srv.close()


def test_three_step_pipeline_orders_all_steps():
    srv = _server(5848)
    tr = SklearnTracer(srv)
    pipe = Pipeline([("minmax", MinMaxScaler()),
                     ("scaler", StandardScaler()),
                     ("pca", PCA(n_components=3))])
    try:
        tr.attach(pipe)
        pipe.fit_transform(np.random.randn(24, 8).astype(np.float32))
        edges = _edges(tr)
        assert ("sk.input", "sk.minmax") in edges
        assert ("sk.minmax", "sk.scaler") in edges
        assert ("sk.scaler", "sk.pca") in edges
    finally:
        tr.detach()
        srv.close()


def test_capture_subscribed_step():
    srv = _server(5846)
    tr = SklearnTracer(srv)
    pipe = _make_pipe()
    received: list[dict] = []
    client = GuiClient("tcp://127.0.0.1:5846", "tcp://127.0.0.1:5847")
    client.start(received.append)
    try:
        tr.attach(pipe)
        pipe.fit_transform(np.random.randn(16, 8).astype(np.float32))  # publish
        client.hello()
        client.subscribe("sk.pca", rate=0)

        deadline = time.time() + 3.0
        while (not any(m.get("node_id") == "sk.pca" for m in received
                       if m.get("type") == protocol.MSG_FRAME)
               and time.time() < deadline):
            pipe.transform(np.random.randn(16, 8).astype(np.float32))
            time.sleep(0.02)

        frames = [m for m in received
                  if m.get("type") == protocol.MSG_FRAME and m["node_id"] == "sk.pca"]
        assert frames, "no frame captured for subscribed pipeline step"
    finally:
        client.stop()
        tr.detach()
        srv.close()

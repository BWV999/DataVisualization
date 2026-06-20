import time

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import datavis.tracer as viz  # noqa: E402
from datavis.common import protocol  # noqa: E402
from datavis.gui.transport import GuiClient  # noqa: E402


def make_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_build_tree_from_module_hooks():
    model = make_model()
    tracer = viz.attach(model, "tcp://127.0.0.1:5792", "tcp://127.0.0.1:5793")
    try:
        model(torch.randn(2, 8))  # one forward -> structure built/published
        nodes = {n["id"]: n for n in tracer.tree.nodes()}
        assert "model" in nodes and {"0", "1", "2"} <= set(nodes)

        root = nodes["model"]
        assert root["parent"] is None and root["depth"] == 0

        child = nodes["0"]
        assert child["parent"] == "model" and child["depth"] == 1
        assert child["name"] == "Linear"
        assert child["shape"] == [2, 16]

        # children execute before the root's forward hook fires
        assert nodes["0"]["order"] < nodes["model"]["order"]
    finally:
        tracer.remove()
        tracer.server.close()


def test_capture_subscribed_module():
    model = make_model()
    tracer = viz.attach(model, "tcp://127.0.0.1:5794", "tcp://127.0.0.1:5795")
    received: list[dict] = []
    client = GuiClient("tcp://127.0.0.1:5794", "tcp://127.0.0.1:5795")
    client.start(received.append)
    try:
        model(torch.randn(2, 8))  # publish structure
        client.hello()
        client.subscribe("0", rate=0)

        deadline = time.time() + 3.0
        while (not any(m.get("type") == protocol.MSG_FRAME for m in received)
               and time.time() < deadline):
            model(torch.randn(2, 8))
            time.sleep(0.02)

        frames = [m for m in received if m.get("type") == protocol.MSG_FRAME]
        assert frames, "no frame captured for subscribed module"
        assert frames[0]["node_id"] == "0"
        assert frames[0]["payload"]["kind"] == protocol.KIND_CURVE  # (B, 16), B<=16
    finally:
        client.stop()
        tracer.remove()
        tracer.server.close()


def test_raw_forward_input_is_captured_as_its_own_source_node():
    import numpy as np
    model = nn.Linear(8, 4)            # input (B,8) != output (B,4): unambiguous
    tracer = viz.attach(model, "tcp://127.0.0.1:5824", "tcp://127.0.0.1:5825")
    received: list[dict] = []
    client = GuiClient("tcp://127.0.0.1:5824", "tcp://127.0.0.1:5825")
    client.start(received.append)
    try:
        x = torch.arange(16.0).reshape(2, 8)
        model(x)
        client.hello()
        nodes = {n["id"]: n for n in tracer.tree.nodes()}
        # the raw input is now a real source node, with the INPUT shape
        assert "model.input" in nodes and nodes["model.input"]["shape"] == [2, 8]
        # it feeds the model and is the producer the first layer consumes from
        assert ("model.input", "model") in set(map(tuple, tracer.tree.edges()))

        client.subscribe("model.input", rate=0)
        deadline = time.time() + 3.0
        while (not any(m.get("node_id") == "model.input" for m in received
                       if m.get("type") == protocol.MSG_FRAME)
               and time.time() < deadline):
            model(x)
            time.sleep(0.02)
        fr = next(m for m in received if m.get("type") == protocol.MSG_FRAME
                  and m["node_id"] == "model.input")
        y = np.asarray(fr["payload"]["y"])
        # the captured frame is the INPUT (2,8), not the (2,4) output
        assert y.shape == (2, 8) and np.allclose(y, x.numpy())
    finally:
        client.stop()
        tracer.remove()
        tracer.server.close()


def test_unsubscribed_module_adds_no_capture():
    model = make_model()
    tracer = viz.attach(model, "tcp://127.0.0.1:5796", "tcp://127.0.0.1:5797")
    try:
        # nothing subscribed -> hooks observe structure but never serialize tensors
        model(torch.randn(2, 8))
        assert len(tracer.tree) >= 4
        assert tracer.server.is_subscribed("0") is False
    finally:
        tracer.remove()
        tracer.server.close()


def test_oplevel_adds_op_leaves_under_modules():
    model = make_model()
    tracer = viz.attach(model, "tcp://127.0.0.1:5798", "tcp://127.0.0.1:5799")
    try:
        model(torch.randn(2, 8))  # module level only
        n_modules = len(tracer.tree)
        assert not any(" > " in n["id"] for n in tracer.tree.nodes())

        tracer.server.set_oplevel(True)  # what MSG_SET_OPLEVEL does
        model(torch.randn(2, 8))         # now op-level mode is entered

        op_nodes = [n for n in tracer.tree.nodes() if " > " in n["id"]]
        assert op_nodes, "no op-level nodes discovered"
        assert len(tracer.tree) > n_modules

        ids = {n["id"] for n in tracer.tree.nodes()}
        op = op_nodes[0]
        assert op["parent"] in ids                       # attributed to a module
        assert op["depth"] == tracer._depth_of[op["parent"]] + 1  # one level deeper

        # turning it back off stops growing the tree
        tracer.server.set_oplevel(False)
        size = len(tracer.tree)
        model(torch.randn(2, 8))
        assert len(tracer.tree) == size
    finally:
        tracer.remove()
        tracer.server.close()


class _Gated(nn.Module):
    """split -> two branches -> elementwise mult -> residual add."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 16)
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        g = torch.sigmoid(a) * torch.tanh(b)
        return self.proj(g) + x


def test_dataflow_edges_capture_branch_and_merge():
    model = _Gated()
    tracer = viz.attach(model, "tcp://127.0.0.1:5802", "tcp://127.0.0.1:5803")
    try:
        tracer.server.set_oplevel(True)
        model(torch.randn(4, 8))
        model(torch.randn(4, 8))  # edge set must not double on a second pass

        edges = set(map(tuple, tracer.tree.edges()))
        ids = {n["id"]: n for n in tracer.tree.nodes()}

        def op(name):  # resolve an op node id by its op name
            return next(i for i in ids if i.endswith(f"> {name}#0"))

        chunk, sig, tanh, mul, add = (op(n) for n in
                                      ("chunk", "sigmoid", "tanh", "mul", "add"))
        # fan-out: chunk feeds both activation branches
        assert (chunk, sig) in edges and (chunk, tanh) in edges
        # merge: both branches feed the elementwise multiply
        assert (sig, mul) in edges and (tanh, mul) in edges
        # leaf module participates: fc produces what chunk consumes
        assert ("fc", chunk) in edges
        # residual merge: proj output flows into the add
        assert ("proj", add) in edges

        # edges are deduped/stable across forwards (no per-step growth)
        n_edges = tracer.tree.n_edges()
        model(torch.randn(4, 8))
        assert tracer.tree.n_edges() == n_edges
    finally:
        tracer.remove()
        tracer.server.close()


class _Struct(nn.Module):
    """Exercises every structural role: split, activations, mul- & add-merges."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(8, 16)
        self.left = nn.Linear(16, 16)
        self.right = nn.Linear(16, 16)

    def forward(self, x):
        h = self.inp(x)
        a, b = h.chunk(2, dim=-1)               # split
        _g = torch.sigmoid(a) * torch.tanh(b)   # acts + elementwise mul-merge
        return self.left(h) + self.right(h)     # add-merge of two leaf outputs


def test_structural_tier_elevates_ops_and_folds_noise():
    model = _Struct()
    tracer = viz.attach(model, "tcp://127.0.0.1:5804", "tcp://127.0.0.1:5805")
    try:
        tracer.server.set_oplevel(protocol.OPLEVEL_STRUCT)
        model(torch.randn(4, 8))
        model(torch.randn(4, 8))

        nodes = {n["id"]: n for n in tracer.tree.nodes()}
        op_nodes = {n["name"]: n for nid, n in nodes.items() if " > " in nid}

        # structural ops are first-class, each tagged with its role
        assert op_nodes["chunk"]["role"] == "split"
        assert op_nodes["sigmoid"]["role"] == "sigmoid"
        assert op_nodes["tanh"]["role"] == "tanh"
        assert op_nodes["mul"]["role"] == "mul"     # sigmoid(a) * tanh(b)
        assert op_nodes["add"]["role"] == "add"     # left(h) + right(h)

        # noise ops (the Linear's internal `linear`, transpose, ...) stay folded
        assert "linear" not in op_nodes
        assert all(n.get("role") for n in op_nodes.values()), "only structural ops"

        # edges route *through* the (transparent) nothing and connect real nodes
        edges = set(map(tuple, tracer.tree.edges()))

        def oid(name):
            return next(i for i in nodes if i.endswith(f"> {name}#0"))

        assert (oid("chunk"), oid("sigmoid")) in edges
        assert (oid("sigmoid"), oid("mul")) in edges and (oid("tanh"), oid("mul")) in edges
        assert ("left", oid("add")) in edges and ("right", oid("add")) in edges
    finally:
        tracer.remove()
        tracer.server.close()


class _Scan(nn.Module):
    """An SSM-style sequential recurrence: ``h = tanh(h + proj(x)[:, t])`` over a
    time loop — the case that unrolls into hundreds of op leaves without rolling."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):                       # x: (B, L, 4)
        u = self.proj(x)
        h = torch.zeros_like(u[:, 0])
        ys = []
        for t in range(u.shape[1]):
            h = torch.tanh(h + u[:, t])         # add-merge + activation per step
            ys.append(h)
        return torch.stack(ys, dim=1)


def _scan_op_nodes(L, *, port, op_max_repeat=None):
    model = _Scan()
    tracer = viz.attach(model, f"tcp://127.0.0.1:{port}", f"tcp://127.0.0.1:{port + 1}",
                        op_max_repeat=op_max_repeat)
    try:
        tracer.server.set_oplevel(protocol.OPLEVEL_STRUCT)
        model(torch.randn(2, L, 4))
        return {nid: n for nid, n in
                ((n["id"], n) for n in tracer.tree.nodes()) if " > " in nid}
    finally:
        tracer.remove()
        tracer.server.close()


def test_loop_unrolling_rolls_into_bounded_recurrent_nodes():
    # Same model, very different loop lengths -> identical op-node set (bounded),
    # not O(L) leaves. The rolled node's ``count`` reports the recurrence depth.
    small = _scan_op_nodes(8, port=5810)
    large = _scan_op_nodes(64, port=5812)

    assert set(small) == set(large), "node set must not grow with loop length"
    # exactly one tanh and one add survive, each rolled
    tanh = next(n for n in small.values() if n["name"] == "tanh")
    add = next(n for n in small.values() if n["name"] == "add")
    assert tanh["id"].endswith("#0") and add["id"].endswith("#0")
    # tanh fires every step; add fires once h has a distinct producer (steps >= 1)
    assert _scan_op_nodes(8, port=5814)[tanh["id"]]["count"] == 8
    assert _scan_op_nodes(64, port=5816)[tanh["id"]]["count"] == 64


def test_op_max_repeat_widens_the_unroll_window():
    # op_max_repeat=3 keeps the first 3 tanh occurrences as distinct nodes.
    nodes = _scan_op_nodes(8, port=5818, op_max_repeat=3)
    tanh_ids = sorted(nid for nid, n in nodes.items() if n["name"] == "tanh")
    assert [i.rsplit("#", 1)[1] for i in tanh_ids] == ["0", "1", "2"]
    # the last (#2) absorbs the remaining steps (3..7) -> count 6
    assert nodes[tanh_ids[-1]]["count"] == 6


def test_region_traces_torch_preprocessing_outside_forward():
    # forward-external torch ops are invisible to the module tracer; a region
    # makes them auto-trace as nodes under a stage, stitched into the model DAG.
    model = nn.Linear(4, 2)
    tracer = viz.attach(model, "tcp://127.0.0.1:5820", "tcp://127.0.0.1:5821")
    try:
        x = torch.randn(2, 4)
        with viz.region("standardize", x):
            mean = x.mean(0, keepdim=True)
            std = x.std(0, keepdim=True) + 1e-5
            x = (x - mean) / std
        out = model(x)
        assert out.shape == (2, 2)

        nodes = {n["id"]: n for n in tracer.tree.nodes()}
        edges = set(map(tuple, tracer.tree.edges()))

        assert "pre.standardize" in nodes and nodes["pre.standardize"]["role"] == "stage"

        def op(nm):
            return next(i for i in nodes if i.startswith(f"pre.standardize > {nm}"))

        # the normalization steps are real nodes under the stage
        assert nodes[op("sub")]["role"] == "sub" and nodes[op("div")]["role"] == "div"
        # x (the stage's input) feeds the subtract; the divide is the chain tail
        assert ("pre.standardize", op("sub")) in edges
        assert (op("sub"), op("div")) in edges
        # the region tail is wired into the model root (preprocessing -> model)
        assert (op("div"), "model.input") in edges

        # the mode is balanced: a plain forward after the region adds no stray
        # op nodes (it stays at the default module level)
        n = len(tracer.tree)
        model(torch.randn(2, 2) @ torch.zeros(2, 4))  # any (2,4) input
        assert len(tracer.tree) == n
    finally:
        tracer.remove()
        tracer.server.close()


def test_region_composes_with_probe_for_non_torch_stages():
    # non-torch preprocessing (no interceptor) drops to viz.probe; it shares the
    # region's chain cursor so torch ops + manual probes form one connected chain.
    import numpy as np
    model = nn.Linear(3, 2)
    tracer = viz.attach(model, "tcp://127.0.0.1:5822", "tcp://127.0.0.1:5823")
    try:
        x = torch.randn(2, 3)
        with viz.region("prep", x):
            x = x * 2.0                       # torch op -> node
            viz.probe("from_pandas", np.asarray(x))  # non-torch stage (manual)
        model(x)

        nodes = {n["id"]: n for n in tracer.tree.nodes()}
        edges = set(map(tuple, tracer.tree.edges()))
        assert "pre.prep" in nodes and "pre.from_pandas" in nodes
        mul = next(i for i in nodes if i.startswith("pre.prep > mul"))
        # chain: stage -> mul -> probe -> model, all connected
        assert ("pre.prep", mul) in edges
        assert (mul, "pre.from_pandas") in edges
        assert ("pre.from_pandas", "model.input") in edges
    finally:
        tracer.remove()
        tracer.server.close()


def test_probe_adds_upstream_preprocessing_chain():
    model = make_model()
    tracer = viz.attach(model, "tcp://127.0.0.1:5806", "tcp://127.0.0.1:5807")
    try:
        # preprocessing stages, declared before the forward (like a data pipeline)
        x = torch.randn(2, 8)
        viz.probe("raw", x)                 # any array-like; torch here
        viz.probe("standardized", (x - x.mean()) / (x.std() + 1e-5))
        model(x)                            # forward wires the chain into the root

        nodes = {n["id"]: n for n in tracer.tree.nodes()}
        # stage nodes exist, are top-level (upstream), and tagged role "stage"
        assert nodes["pre.raw"]["parent"] is None
        assert nodes["pre.raw"]["role"] == "stage"
        assert nodes["pre.standardized"]["role"] == "stage"
        # they execute before the model
        assert nodes["pre.raw"]["order"] < nodes["model"]["order"]

        edges = set(map(tuple, tracer.tree.edges()))
        assert ("pre.raw", "pre.standardized") in edges     # chained in call order
        assert ("pre.standardized", "model.input") in edges       # tail feeds the model

        # stable across iterations: stage ids + chain don't grow per forward
        viz.probe("raw", x)
        viz.probe("standardized", x)
        model(x)
        assert tracer.tree.n_edges() == len(edges)
    finally:
        tracer.remove()
        tracer.server.close()


def test_probe_captures_subscribed_stage():
    model = make_model()
    tracer = viz.attach(model, "tcp://127.0.0.1:5808", "tcp://127.0.0.1:5809")
    received: list[dict] = []
    client = GuiClient("tcp://127.0.0.1:5808", "tcp://127.0.0.1:5809")
    client.start(received.append)
    try:
        viz.probe("raw", torch.randn(2, 8))   # publish structure incl. stage node
        client.hello()
        client.subscribe("pre.raw", rate=0)

        deadline = time.time() + 3.0
        while (not any(m.get("node_id") == "pre.raw" for m in received
                       if m.get("type") == protocol.MSG_FRAME)
               and time.time() < deadline):
            viz.probe("raw", torch.randn(2, 8))
            time.sleep(0.02)

        frames = [m for m in received
                  if m.get("type") == protocol.MSG_FRAME and m["node_id"] == "pre.raw"]
        assert frames, "no frame captured for subscribed preprocessing stage"
    finally:
        client.stop()
        tracer.remove()
        tracer.server.close()


def test_numpy_track_chain_feeds_model_root():
    import numpy as np

    model = make_model()
    tracer = viz.attach(model, "tcp://127.0.0.1:5810", "tcp://127.0.0.1:5811")
    try:
        for _ in range(2):
            # numpy preprocessing, auto-traced from the source (no per-stage probe)
            x = viz.track(np.random.randn(2, 8).astype(np.float32), "raw")
            x = (x - x.mean(0)) / (x.std(0) + 1e-5)
            model(torch.as_tensor(np.asarray(x)))  # forward wires the tail in

        nodes = {n["id"]: n for n in tracer.tree.nodes()}
        edges = set(map(tuple, tracer.tree.edges()))

        # numpy nodes live in the SAME tree as the torch model
        assert "np.raw" in nodes and "model" in nodes
        assert nodes["np.raw"]["role"] == "stage"
        # the numpy chain's tail (the divide) is handed to the model root
        assert ("np > divide#0", "model.input") in edges
        # and the whole chain is upstream of the model
        assert nodes["np.raw"]["order"] < nodes["model"]["order"]

        # stable across iterations: shared structure doesn't grow per pass
        n_nodes, n_edges = len(nodes), len(edges)
        x = viz.track(np.random.randn(2, 8).astype(np.float32), "raw")
        x = (x - x.mean(0)) / (x.std(0) + 1e-5)
        model(torch.as_tensor(np.asarray(x)))
        assert len(tracer.tree) == n_nodes and tracer.tree.n_edges() == n_edges
    finally:
        tracer.remove()
        tracer.server.close()


def test_sklearn_pipeline_chain_feeds_model_root():
    import numpy as np
    sklearn = pytest.importorskip("sklearn")
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    tracer = viz.attach(model, "tcp://127.0.0.1:5812", "tcp://127.0.0.1:5813")
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=4))])
    sk = viz.attach_pipeline(pipe)
    try:
        Xt = pipe.fit_transform(np.random.randn(16, 8).astype(np.float32))
        model(torch.as_tensor(np.asarray(Xt), dtype=torch.float32))

        nodes = {n["id"]: n for n in tracer.tree.nodes()}
        edges = set(map(tuple, tracer.tree.edges()))
        # pipeline nodes live in the SAME tree as the torch model
        assert "sk.input" in nodes and "sk.scaler" in nodes and "sk.pca" in nodes
        # the pipeline tail (last step) is handed to the model root
        assert ("sk.pca", "model.input") in edges
        assert nodes["sk.input"]["order"] < nodes["model"]["order"]
    finally:
        sk.detach()
        tracer.remove()
        tracer.server.close()


def test_numpy_then_sklearn_then_torch_is_one_chain():
    import numpy as np
    pytest.importorskip("sklearn")
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    tracer = viz.attach(model, "tcp://127.0.0.1:5814", "tcp://127.0.0.1:5815")
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=4))])
    sk = viz.attach_pipeline(pipe)
    try:
        for i in range(2):
            # numpy preprocessing -> sklearn pipeline -> torch model, all in one pass
            x = viz.track(np.random.randn(16, 8).astype(np.float32), "raw")
            x = (x - x.mean(0)) / (x.std(0) + 1e-5)
            Xt = (pipe.fit_transform if i == 0 else pipe.transform)(np.asarray(x))
            model(torch.as_tensor(np.asarray(Xt), dtype=torch.float32))

        edges = set(map(tuple, tracer.tree.edges()))
        # the two cross-backend stitches + the model handoff
        assert ("np > divide#0", "sk.input") in edges   # numpy tail -> sklearn input
        assert ("sk.pca", "model.input") in edges             # sklearn tail -> model root

        # the whole thing is ONE connected chain: raw numpy source reaches the model
        adj: dict[str, list[str]] = {}
        for s, d in edges:
            adj.setdefault(s, []).append(d)
        seen, stack = set(), ["np.raw"]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj.get(n, ()))
        assert "model.input" in seen, "numpy source does not reach the model"
        for nid in ("np > divide#0", "sk.input", "sk.scaler", "sk.pca"):
            assert nid in seen

        # stable across iterations (no per-pass growth)
        n_nodes, n_edges = len(tracer.tree), tracer.tree.n_edges()
        x = viz.track(np.random.randn(16, 8).astype(np.float32), "raw")
        x = (x - x.mean(0)) / (x.std(0) + 1e-5)
        Xt = pipe.transform(np.asarray(x))
        model(torch.as_tensor(np.asarray(Xt), dtype=torch.float32))
        assert len(tracer.tree) == n_nodes and tracer.tree.n_edges() == n_edges
    finally:
        sk.detach()
        tracer.remove()
        tracer.server.close()


def test_capture_op_node_is_not_self_polluted():
    model = make_model()
    tracer = viz.attach(model, "tcp://127.0.0.1:5800", "tcp://127.0.0.1:5801")
    try:
        tracer.server.set_oplevel(True)
        model(torch.randn(2, 8))
        # our own .cpu()/.detach() capture ops must not appear as nodes
        names = {n["name"] for n in tracer.tree.nodes() if " > " in n["id"]}
        assert "numpy" not in names and "_to_copy" not in names
    finally:
        tracer.remove()
        tracer.server.close()

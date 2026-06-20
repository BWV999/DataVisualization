# DataVisualization

A **zero-intrusion dataflow visualizer / visual debugger** for ML pipelines. It
turns a training run into a live, explorable **dataflow DAG** — every transform a
node, every producer→consumer hand-off an edge — so you can watch how a
multi-channel tensor evolves stage by stage, from the raw input through
preprocessing and into the model, while it trains (locally or on a remote GPU).

You wrap the run; you do **not** restructure model code:

```python
import datavis.tracer as viz
viz.attach(model)        # registers forward hooks — that's the whole change
# ... train as usual; open the GUI and click nodes to stream their tensors
```

## Design goals

- **Zero code intrusion** — the target model is auto-traced through `nn.Module`
  forward hooks; you only wrap the run, then choose how deep to look (level of
  detail) in the GUI.
- **Complete dataflow** — not just the model's input/output, but *every* stage's
  actual data: raw batch → each preprocessing op → the model input → internal ops
  → output, each a subscribable node that streams real tensor values.
- **Real-time, without slowing training** — capture is on-demand and rate-limited;
  a tensor is materialized and sent **only** for nodes you've subscribed to in the
  GUI, so an untouched GUI adds negligible overhead.
- **Two processes** — a tracer library inside the training process and a separate
  PyQtGraph GUI, connected over ZeroMQ (works locally or across an SSH tunnel to a
  remote GPU box).

## Architecture

```
[training process]                          [GUI process]
  datavis.tracer  --- ZeroMQ (msgpack) --->  datavis.gui (PyQtGraph)
  REP ctrl :5750  <---- subscribe -----       REQ ctrl
  PUB data :5751  ----- frames ------->       SUB data
```

- **`datavis/common/protocol.py`** — the wire contract. Every message is a msgpack
  dict with a `type`; numpy arrays are packed inline. It also owns the tensor
  *reduction* logic (`auto_payload`, `compute_stats`) so both processes agree on
  how a tensor becomes a renderable payload. Rank heuristics: 1D / small-2D →
  y-t curves; large-2D / ND → heatmap.
- **`datavis/tracer/`** — runs inside training:
  - `graph.py` (`ExecutionTree`) — the framework-agnostic sink: nodes + deduped
    producer→consumer edges, accumulated in first-seen order.
  - `transport.py` (`TracerServer`) — the IPC server (subscriptions, rate limit);
    a daemon control thread keeps the training hot path to cheap `is_subscribed` /
    `send_frame` calls.
  - `hooks.py` (`ModuleTracer`) — the PyTorch auto-trace.
  - `roles.py` — framework-free structural-op classification (shared by the torch
    and numpy backends).
  - `backends/` — the non-torch interceptors (`numpy_trace.py`, `sklearn_pipe.py`).
- **`datavis/gui/`** — the viewer: `app.py` (window wiring), `tree_view.py`
  (containment tree + LOD slider + start-point), `graph_view.py` (the dataflow
  DAG), `panels.py` (curve / heatmap), `transport.py` (the GUI-side client).

## What it captures

**The model, automatically.** `viz.attach(model)` hooks every submodule, giving
the module tree and execution order at zero op-level cost. The raw forward input
is captured as its own `model.input` source node (the data's first shape), kept
distinct from the model's output.

**Op-level detail, on demand.** A `TorchFunctionMode` turns each `torch.*` / tensor
op into a leaf of its enclosing module. The **Ops** tier selector chooses how much
to materialize:

| tier | shows | cost |
|------|-------|------|
| **off** (default) | modules only | zero — the op-mode is never entered |
| **structural** | + topology-defining ops as icon nodes: merges `+ × − ÷ ‖ @`, splits `◇`, activations `σ t R G S …` (noise ops like `view`/`transpose` stay transparent pass-throughs) | one Python callback per op while enabled |
| **all** | + every op | as above |

**Loops are rolled, not unrolled.** A recurrence (e.g. an SSM's `for t in range(L)`
selective scan) would otherwise emit *L* copies of each op. Instead repeated ops
fold into a single recurrent node carrying a `×N` multiplicity badge, so the node
count stays O(distinct ops), independent of sequence length. Widen the window with
`attach(op_max_repeat=k)` or `$DATAVIS_OP_MAX_REPEAT`.

**Preprocessing, through whatever library produced it.** Out-of-model preprocessing
isn't a module and often isn't even torch, so it can't be hooked the same way. A
coverage ladder captures it into the *same* DAG, upstream of the model — each
backend traced through its own *uniform interface*:

| your preprocessing | one-liner | how |
|--------------------|-----------|-----|
| numpy | `x = viz.track(raw)` | `TracedArray` rides NEP-13 (`__array_ufunc__`) + NEP-18 (`__array_function__`); wrap once, the whole chain records itself |
| sklearn `Pipeline` | `viz.attach_pipeline(pipe)` | shadows each step's `transform` — the tabular dual of `forward` |
| forward-external torch | `with viz.region("name", x): ...` | enters the same op-mode for ops the forward-only mode never sees |
| anything else (pandas, pure Python) | `viz.probe("name", x)` | manual fallback; records any array-like as a stage node |

These compose: a `numpy → sklearn → torch` pipeline stitches into **one connected
chain** even though the array loses its tag at each boundary (sklearn's
`check_array` strips the `TracedArray`; torch sees a plain tensor) — each backend
links from a shared "current tail" cursor and writes its own tail back.

**ND tensors, honestly.** A rank-≥3 activation (e.g. `(batch, seq, dim)`) is
collapsed to a 2D heatmap by **slicing a real representative sample** (index 0 of
each leading axis) rather than averaging — so per-series temporal/feature structure
survives instead of being blended away. The panel title labels the projection
(`8×96×16 slice[0]→2D`) so it's never a silent, lossy reduction. Opt into averaging
with `reduce="mean"`.

## Install & run

Dependencies are managed with [**uv**](https://docs.astral.sh/uv/). The tracer and
the GUI install separately, so instrumenting your program never drags in the Qt
viewer stack — only `pyzmq` / `msgpack` / `numpy`.

To add just the tracer to an existing project (e.g. on a headless remote GPU box),
install it straight from GitHub — no clone needed:

```bash
uv pip install "git+https://github.com/BWV999/DataVisualization"               # tracer only
uv pip install "datavis[gui] @ git+https://github.com/BWV999/DataVisualization"  # + viewer
```

To work on the repo (run the demos / GUI / tests), clone it and install editable:

```bash
git clone https://github.com/BWV999/DataVisualization
cd DataVisualization
uv venv
uv pip install -e .              # tracer only — for the process being visualized
uv pip install -e ".[gui]"       # + the PyQtGraph viewer (the machine you watch on)
uv pip install -e ".[dev]"       # + the test suite (includes the gui stack)
```

The demos use frameworks the visualizer itself does not depend on — install per
demo:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu  # torch demos
uv pip install scikit-learn                                            # sklearn demos
```

Start the GUI in one terminal and a demo in another (either order — the structure
arrives whichever starts first):

```bash
# terminal A — the GUI
uv run python -m datavis.gui.app

# terminal B — pick a demo
uv run python examples/demo_signal.py      # synthetic signal pipeline, no torch
uv run python examples/demo_torch.py       # zero-intrusion auto-trace of a CNN
uv run python examples/demo_torch.py --bench   # trace overhead with 0 subscribers
uv run python examples/demo_gated.py       # gated-residual net: split/merge/act icons
uv run python examples/demo_mamba.py       # Bi-Mamba+ SSM (rolled recurrence + region)
uv run python examples/demo_preprocess.py  # numpy preprocessing -> torch, one DAG
uv run python examples/demo_sklearn.py     # sklearn Pipeline (scale -> PCA) -> torch
uv run python examples/demo_unified.py     # numpy -> sklearn -> torch, all stitched
```

`demo_signal` runs a finite sequence then holds the final state, so a panel opened
afterward still shows it. `demo_mamba` needs the repo cloned to `/tmp/Bi-Mamba4TS`
plus `einops` (see its module docstring); it runs on CUDA automatically when a GPU
is available.

**In the GUI:** the **Dataflow** dock shows the run as a DAG (split / branch / merge
topology), framed at the source — pan to follow the flow, right-click → *View All*
to fit. Set **Ops** to *structural* to reveal the merge/split/activation operators
as icon nodes; drag the **LOD** slider to collapse/expand depth; pick a node in
**Start from** to begin partway down the pipeline; and **click / check** any node
(in the graph or the tree) to open a live panel (1D/small → curves, 2D/ND →
heatmap). Hidden nodes are *contracted* — bypassed by reachability — so the DAG
stays connected at any level of detail.

## Test

```bash
uv run pytest -q
QT_QPA_PLATFORM=offscreen uv run pytest -q   # headless (WSL2 has no display)
```

Tests that need torch / sklearn `pytest.importorskip` them, so the suite still runs
(those tests just skip) when they aren't installed.

## Status

Stage 1 (PyTorch auto-trace + dataflow DAG + structural ops + preprocessing
probes) and Stage 1.5 (multi-backend: numpy, sklearn, and the unified
`numpy → sklearn → torch` DAG) are complete. The next stage is presentation polish
(transition animation); a pandas-specific interceptor remains a low-priority
fallback covered today by `viz.probe`.

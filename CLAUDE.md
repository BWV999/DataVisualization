# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **zero-intrusion dataflow visualizer / visual debugger** for ML pipelines. Two
processes connected over ZeroMQ: a **tracer** library imported into the training
process, and a separate **PyQtGraph GUI**. You wrap the run (`viz.attach(model)`)
without restructuring model code; the GUI shows the execution as a live DAG and
you click nodes to stream their tensors. Capture is on-demand + rate-limited so an
untouched GUI adds negligible overhead.

## Commands

Dependencies are managed with **uv** (see the global policy; never use bare `pip`).

```bash
uv venv && uv pip install -e ".[dev]"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu  # torch demos/tests
uv pip install scikit-learn                                            # sklearn interceptor

# run the GUI (one process) and a demo (another); either order is fine
uv run python -m datavis.gui.app
uv run python examples/demo_torch.py        # also: demo_mamba / demo_preprocess / demo_sklearn / demo_unified / demo_signal

# tests
uv run pytest -q
QT_QPA_PLATFORM=offscreen uv run pytest -q                    # headless (WSL2 has no display by default)
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_tracer_numpy.py::test_split_is_multi_output -q  # single test
uv run python examples/demo_torch.py --bench                  # trace overhead with 0 subscribers
```

Tests that touch the GUI need `QT_QPA_PLATFORM=offscreen` on headless hosts. Tests
that import torch / sklearn `pytest.importorskip` them, so the suite still runs
without those installed (those tests just skip).

## Architecture: the big picture

The system is built around **one framework-agnostic sink** that many
framework-specific **interceptors** write into, and a **wire protocol** that both
processes share.

- **`datavis/common/protocol.py`** is the contract. Every IPC message is a
  msgpack dict with a `type`; numpy arrays are packed inline. It also owns the
  tensor *reduction* logic (`auto_payload`, `compute_stats`) so the tracer and GUI
  agree on exactly how a tensor becomes a renderable payload (rank heuristics:
  1D/small-2D → curves, large-2D/ND → heatmap). Change reduction or message shape
  here and both sides stay in sync. **ND (rank ≥ 3) is collapsed to 2D by
  *slicing* a real representative sample (`reduce=REDUCE_SLICE`, index 0 of each
  leading axis) — NOT averaging — so time-series/feature structure survives and
  distinct samples/series aren't blended; `reduce=REDUCE_MEAN` is opt-in. ND
  payloads carry `src_shape`/`reduced`/`reduced_idx`; `gui.panels._proj_note`
  surfaces them in the panel title so the projection is never silent. Keep ND →
  heatmap consistent with `gui.panels.make_panel` (rank ≥ 3 → HeatmapPanel).**

- **`datavis/tracer/graph.py` — `ExecutionTree`** is the shared sink: nodes
  (id/name/path/parent/depth/shape/order/role) accumulated in first-seen order,
  plus deduped producer→consumer `edges`. It knows nothing about torch/numpy. Every
  interceptor calls `observe(...)` / `link(...)` on the same tree, which is how a
  single DAG can span multiple libraries.

- **`datavis/tracer/transport.py` — `TracerServer`** is the IPC server (binds REP
  ctrl :5750 + PUB data :5751). The control loop runs on a daemon thread; the
  training thread only calls cheap `is_subscribed` (subscribed AND due under the
  per-node rate limit) and `send_frame`. Payloads are built **only** for subscribed
  nodes — that is the "doesn't slow training" guarantee.

- **Interceptors** (each adapts one library's *uniform interface* into the tree):
  - **torch — `datavis/tracer/hooks.py` (`ModuleTracer`)**: `nn.Module` forward
    pre/post hooks give the module tree + execution order; a `TorchFunctionMode`
    (`_OpMode`) entered only during forward (and only when op-level is on) gives
    op-level nodes. Producer→consumer edges are tracked by `id(tensor)`, reset each
    forward. This is the primary path; the others mirror it.
  - **torch preprocessing OUTSIDE forward — `with viz.region(name, *inputs)`**:
    forward-external tensor ops (standardization, windowing in the train loop) are
    invisible to the forward-only op-mode. A `region` enters the *same* `_OpMode`
    and attributes its ops to a synthetic `pre.{name}` stage (op-level ALL by
    default — a region's steps *are* the content), seeding `inputs` as the stage's
    source and handing the block's tail to the model root via `_last_probe`. The
    op-mode is now ref-counted (`_mode_depth`) so a region and a forward can nest.
  - **numpy — `datavis/tracer/backends/numpy_trace.py` (`TracedArray`)**: the dual
    of `__torch_function__` — `__array_ufunc__` (NEP-13) + `__array_function__`
    (NEP-18) + overridden reduction methods. `viz.track(raw)` wraps once at the
    source; the chain records itself. Imports no torch.
  - **sklearn — `datavis/tracer/backends/sklearn_pipe.py` (`SklearnTracer`)**:
    `viz.attach_pipeline(pipe)` instance-shadows each step's `transform`/
    `fit_transform` (the tabular dual of `forward`). Imports no torch.
  - **`viz.probe(name, data)`** is the manual fallback for anything no interceptor
    covers (pandas, pure-Python, non-array). It chains via `_last_probe` like the
    others, so it composes *inside* a `region` (auto torch ops + manual probes for
    non-torch steps → one connected chain). Its array conversion is `_capturing`-
    guarded so probing a tensor inside an active region doesn't trace its own
    detach/cpu ops. Coverage ladder: numpy→`track`, sklearn→`attach_pipeline`,
    forward-external torch→`region`, everything else→`probe`.

- **`datavis/tracer/roles.py`** holds `_op_role` and the role tables (merge / split
  / activation classification) — extracted here precisely so it is **torch-free**
  and shared by both the torch and numpy interceptors. The numpy backend maps
  numpy's native op names onto this canonical vocabulary before calling `_op_role`.

- **GUI — `datavis/gui/`**: `app.py` wires `tree_view.py` (containment tree, LOD
  slider, start-point), `graph_view.py` (the dataflow DAG), and `panels.py`
  (curve/heatmap). `transport.py` is the GUI-side client (connect REQ/SUB, frames
  marshalled onto Qt signals). The graph **contracts** hidden nodes (bypasses them
  by reachability) so the DAG stays connected at any level of detail.

## Critical invariants (these will bite you)

- **`datavis.tracer` (and `datavis.common`) must stay Qt-free.** The dependency
  split rests on it: core install (`pip install -e .`) is only `pyzmq`/`msgpack`/
  `numpy` so the tracer embeds into the user's training process (incl. a headless
  remote GPU box) without the GUI stack; `pyqtgraph`/`PyQt6` are the `[gui]` extra.
  Importing pyqtgraph/PyQt anywhere outside `datavis/gui/` silently re-couples the
  two and breaks the lightweight tracer install. Verify with: `import datavis.tracer`
  must load zero `PyQt*`/`pyqtgraph` modules.

- **Cross-backend stitching uses the shared `owner._last_probe` cursor, not object
  identity.** When data crosses a boundary the tag is lost (sklearn's `check_array`
  strips `TracedArray`; torch sees a plain tensor), so each backend instead links
  from `_last_probe` (the upstream tail) at the start of its pass and writes its own
  tail back; the torch root pre-hook's `_last_probe→model.input` + reset closes it.
  This is what makes `numpy → sklearn → torch` one connected chain. Requires a
  shared owner — i.e. a model attached, so the backends bind to the same `_active`.

- **The raw forward input is its own source node `{root}.input`** (role `stage`),
  created in the root pre-hook from `_first_tensor(args)`: it is the data's first
  shape, captured on subscribe, is the producer the first layers/residuals consume
  from, and is where preprocessing chains feed (`_last_probe → model.input`). The
  root container node (`model`) is kept distinct — its own frame is the model
  OUTPUT (post-hook) — so no single node is overloaded with both. A pure-container
  root then carries no flow edges and naturally drops from the DAG; subscribe it in
  the *tree* to see the output. (Leaf-root models, e.g. a bare `nn.Linear`, link
  `model.input → model`, so both input and output show in the DAG.)

- **One process-shared server on the default ports.** `viz.attach` (default-port
  path), `viz.track`, and `viz.attach_pipeline` all adopt `_get_default_server()`
  in `tracer/__init__.py`, so calling `track`/`attach_pipeline` before `attach`
  does not bind 5750 out from under `attach` (an intra-process self-conflict that
  *looks* like a stale process but is not — check `/proc/net/tcp` before blaming a
  ghost). Custom ports still get a dedicated, owned server.

- **`app._load_structure` must call `graph.set_structure` BEFORE
  `pipeline.set_structure`** — the tree's filter emits `filterChanged → set_visible`,
  which drops node ids the graph hasn't learned yet, so newly-traced nodes silently
  vanish if the order is reversed.

- **The capture path is already GPU-safe** (`_to_numpy` does
  `.detach().to("cpu", float32).numpy()`); no CUDA-specific code is needed.

- **Op detail is a 3-tier `oplevel`** (`protocol.OPLEVEL_OFF/STRUCT/ALL`). At STRUCT
  only topology-defining ops become nodes (merges/splits/activations, arity-aware);
  noise ops (`view/transpose/getitem/...`) are transparent pass-throughs (output
  inherits its input's producer). ALL = every op a node. OFF = modules only.

- **Repeated ops are rolled, not unrolled** (`hooks.py` `_op_max_repeat`, default 1;
  override via `attach(op_max_repeat=)` or `$DATAVIS_OP_MAX_REPEAT`). A `for t in
  range(L)` scan would otherwise emit L copies of each op (an SSM hits hundreds per
  forward); instead the first `_op_max_repeat` occurrences of a `(module, op_name)`
  get distinct `#i` nodes and further repeats fold into the last, which carries a
  `count` (multiplicity, shown as `×N` in tree label / graph badge / tooltip). The
  loop-carried self-edge a recurrence implies is dropped by `_link_inputs`
  (`src != dst`); only the slot-owning occurrence (`count == 1`) is captured, so a
  subscribed rolled node streams one frame/forward, not one/iteration. Node count is
  thus O(distinct ops), independent of L.

- **numpy ≥ 2 op-name quirks**: the `/` ufunc `__name__` is `divide` (not
  `true_divide`) and `x**2` dispatches as `square`; `_CANON` in `numpy_trace.py`
  normalizes these to the role vocabulary.

## Status / direction

Stage 1 (M0–M3.2): torch auto-trace + dataflow DAG + structural ops + probes —
done. Stage 1.5 multi-backend: numpy (M4.1), sklearn (M4.2), unified
`numpy→sklearn→torch` DAG (M4.3) — done; pandas (M4.4) is a low-priority probe
fallback. The living plan with full per-milestone notes is at
`~/.claude/plans/y-t-gpu-enchanted-lerdorf.md`; project memory is under
`~/.claude/projects/-home-claude-projects-DataVisualization/memory/`.

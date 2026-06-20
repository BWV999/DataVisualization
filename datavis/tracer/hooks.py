"""Zero-intrusion PyTorch auto-trace via forward hooks + an op-level mode.

Module level (always on, cheap):
  A forward hook on every submodule. The dotted ``named_modules`` paths give the
  tree's parent links; hook firing order gives execution order.

Op level (on demand — the deep LOD):
  A ``TorchFunctionMode`` intercepts every ``torch.*`` / tensor-method call. Each
  op becomes a leaf node of the module currently executing (tracked with a module
  stack maintained by forward pre/post hooks). The mode is entered only for the
  forward pass and only while the GUI has requested op-level tracing, so the
  default (module-level) path pays nothing.

Capture cost (the ``.cpu()`` copy) is paid only for subscribed nodes, and our own
capture ops are skipped via the ``_capturing`` guard so they never appear as
spurious op nodes.

The structure is (re)published whenever the node set grows — once for the initial
module discovery, and again when op-level tracing reveals new leaves.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

from datavis.common import protocol
from datavis.tracer.graph import ExecutionTree
from datavis.tracer.roles import _op_role
from datavis.tracer.transport import TracerServer

# Ops that return tensors but carry no useful dataflow to visualize.
_SKIP_OPS = frozenset({"__get__"})


def _first_tensor(obj) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for o in obj:
            t = _first_tensor(o)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for o in obj.values():
            t = _first_tensor(o)
            if t is not None:
                return t
    return None


def _iter_tensors(obj):
    """Yield every tensor nested in args/outputs (lists/tuples/dicts)."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _iter_tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _iter_tensors(o)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu", dtype=torch.float32).numpy()


def _as_array(data) -> np.ndarray:
    """Any array-like (torch / numpy / pandas / list) -> numpy, for probes."""
    if isinstance(data, torch.Tensor):
        return _to_numpy(data)
    try:
        return np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError):
        return np.asarray(data)


class _OpMode(TorchFunctionMode):
    def __init__(self, tracer: "ModuleTracer") -> None:
        super().__init__()
        self._tracer = tracer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if not self._tracer._capturing:
            self._tracer._on_op(func, args, kwargs, out)
        return out


class ModuleTracer:
    def __init__(self, model: torch.nn.Module, server: TracerServer,
                 root_name: str = "model", op_max_repeat: Optional[int] = None) -> None:
        self.model = model
        self.server = server
        self.tree = ExecutionTree()

        # How many distinct nodes a repeated op gets before further repeats roll
        # into the last one (loop unrolling -> a single recurrent node). 1 = full
        # roll. Env override lets the user widen it without touching code.
        if op_max_repeat is None:
            op_max_repeat = int(os.environ.get("DATAVIS_OP_MAX_REPEAT", "1") or "1")
        self._op_max_repeat = max(1, op_max_repeat)

        self._meta: dict[torch.nn.Module, tuple[str, str, str, Optional[str], int]] = {}
        self._depth_of: dict[str, int] = {}
        self._path_of: dict[str, str] = {}
        self._is_leaf: dict[str, bool] = {}
        self._handles: list = []

        self._step = -1
        self._published = 0
        self._published_edges = 0
        self._module_stack: list[str] = []
        self._op_counters: dict[tuple[str, str], int] = {}
        # id(tensor) -> node_id that produced it; reset each forward
        self._producer: dict[int, str] = {}
        # upstream preprocessing chain (viz.probe): the last stage node, linked
        # into the model root on the next forward so pre-processing feeds it
        self._last_probe: Optional[str] = None

        self._mode = _OpMode(self)
        # The op-mode may be entered by a forward AND by a ``region`` at once, so
        # ref-count entries instead of a bool (balance enter/exit at depth 0).
        self._mode_depth = 0
        self._fwd_mode = False
        self._capturing = False
        # active ``region`` scopes (synthetic preprocessing stages outside any
        # forward): a stack of (stage_node_id, op-detail level).
        self._region_stack: list[tuple[str, int]] = []
        # the raw forward input gets its own source node (the data's first shape)
        self._root_name = root_name
        self._input_id = f"{root_name}.input"

        self._build_index(root_name)
        self._register()

    # -- op-mode ref counting ---------------------------------------------
    def _enter_mode(self) -> None:
        self._mode_depth += 1
        if self._mode_depth == 1:
            self._mode.__enter__()

    def _exit_mode(self) -> None:
        if self._mode_depth <= 0:
            return
        self._mode_depth -= 1
        if self._mode_depth == 0:
            self._mode.__exit__(None, None, None)

    # -- index / registration ---------------------------------------------
    def _build_index(self, root_name: str) -> None:
        for path, module in self.model.named_modules():
            cls = type(module).__name__
            if path == "":
                meta = (root_name, cls, root_name, None, 0)
            else:
                disp = f"{root_name}.{path}"
                parent = path.rsplit(".", 1)[0] if "." in path else root_name
                meta = (path, cls, disp, parent, path.count(".") + 1)
            self._meta[module] = meta
            self._depth_of[meta[0]] = meta[4]
            self._path_of[meta[0]] = meta[2]
            self._is_leaf[meta[0]] = next(module.children(), None) is None

    def _register(self) -> None:
        for module in self._meta:
            self._handles.append(module.register_forward_pre_hook(self._make_pre(module)))
            self._handles.append(module.register_forward_hook(self._make_post(module)))

    # -- capture helper ----------------------------------------------------
    def _capture(self, node_id: str, tensor: torch.Tensor) -> None:
        self._capturing = True
        try:
            arr = _to_numpy(tensor)
        finally:
            self._capturing = False
        self.server.send_frame(
            node_id, max(self._step, 0),
            protocol.auto_payload(arr), protocol.compute_stats(arr)
        )

    # -- dataflow edges ----------------------------------------------------
    def _link_inputs(self, dst: str, *containers) -> None:
        """Edge from each input tensor's producer to ``dst`` (skip self-loops)."""
        for c in containers:
            for t in _iter_tensors(c):
                src = self._producer.get(id(t))
                if src is not None and src != dst:
                    self.tree.link(src, dst)

    def _register_outputs(self, node_id: str, output) -> None:
        for t in _iter_tensors(output):
            self._producer[id(t)] = node_id

    def _maybe_publish(self) -> None:
        """Republish the structure if the node or edge set has grown."""
        if (len(self.tree) != self._published
                or self.tree.n_edges() != self._published_edges):
            self.server.set_structure(self.tree.nodes(), self.tree.edges())
            self._published = len(self.tree)
            self._published_edges = self.tree.n_edges()

    # -- preprocessing probes (upstream of the model) ----------------------
    def probe(self, name: str, data, *, kind: Optional[str] = None,
              role: str = "stage") -> None:
        """Capture an out-of-model preprocessing stage as an upstream node.

        ``data`` is any array-like (torch / numpy / pandas / list). Probes
        called in sequence before a forward form a chain (``raw -> scaled ->
        windowed``) whose tail is wired into the model root, so the whole
        pipeline — not just the model — appears in one dataflow graph. It is the
        manual fallback for stages no interceptor covers (pandas, arbitrary
        code), and composes inside a ``region`` (both advance ``_last_probe``).
        """
        # guard the array conversion: if a ``region`` has the op-mode active,
        # probing a torch tensor would otherwise trace its detach/cpu ops.
        self._capturing = True
        try:
            arr = _as_array(data)
        finally:
            self._capturing = False
        node_id = f"pre.{name}"
        self.tree.observe(node_id, name=name, path=f"input.{name}", parent=None,
                          depth=0, shape=tuple(arr.shape), role=role)
        if self._last_probe is not None and self._last_probe != node_id:
            self.tree.link(self._last_probe, node_id)
        self._last_probe = node_id
        if self.server.is_subscribed(node_id):
            self.server.send_frame(node_id, max(self._step, 0),
                                   protocol.auto_payload(arr, kind=kind),
                                   protocol.compute_stats(arr))
        self._maybe_publish()

    # -- region: auto-trace torch ops done OUTSIDE the model forward --------
    @contextmanager
    def region(self, name: str = "preprocess", *inputs,
               level: int = protocol.OPLEVEL_ALL):
        """Auto-trace torch ops run inside the ``with`` block (preprocessing the
        module tracer's forward-only op-mode never sees) as op nodes under a
        synthetic ``pre.{name}`` stage, wired into the unified DAG.

            with viz.region("standardize", x):     # x sourced from the stage
                x = (x - x.mean(0)) / (x.std(0) + 1e-5)
            out = model(x)                          # stage tail -> model root

        Any ``inputs`` are captured at the stage and become the source the block's
        first ops link from; the block's last op is handed to the model root via
        ``_last_probe``. Defaults to op-level ALL (a region's steps *are* the
        content to show, even though they are structurally "noise" ops).
        """
        stage_id = f"pre.{name}"
        first = _first_tensor(inputs)
        shape = tuple(first.shape) if first is not None else ()
        self.tree.observe(stage_id, name=name, path=f"input.{name}", parent=None,
                          depth=0, shape=shape, role="stage")
        if self._last_probe is not None and self._last_probe != stage_id:
            self.tree.link(self._last_probe, stage_id)
        self._last_probe = stage_id
        # the block's external inputs flow from the stage node
        for t in _iter_tensors(inputs):
            self._producer[id(t)] = stage_id
        if first is not None and self.server.is_subscribed(stage_id):
            self._capture(stage_id, first)
        self._region_stack.append((stage_id, protocol.normalize_oplevel(level)))
        self._enter_mode()
        self._maybe_publish()
        try:
            yield
        finally:
            self._exit_mode()
            if self._region_stack and self._region_stack[-1][0] == stage_id:
                self._region_stack.pop()
            self._maybe_publish()

    # -- module hooks ------------------------------------------------------
    def _make_pre(self, module: torch.nn.Module):
        node_id = self._meta[module][0]
        is_root = module is self.model
        is_leaf = self._is_leaf.get(node_id, True)

        def pre(mod, args) -> None:
            if is_root:
                self._step += 1
                self._op_counters.clear()
                self._producer.clear()
                # Capture the raw forward input as its own source node — the
                # data's first shape. It is the genuine source feeding the model
                # (residuals on it merge here), kept distinct from the root
                # container, whose own frame is the model OUTPUT (post-hook), so
                # no single node is overloaded with both input and output.
                in_tensor = _first_tensor(args)
                in_id = self._input_id
                if in_tensor is not None:
                    self.tree.observe(in_id, name="input", path=in_id,
                                      parent=node_id, depth=1,
                                      shape=tuple(in_tensor.shape), role="stage")
                    self._register_outputs(in_id, args)
                    if self.server.is_subscribed(in_id):
                        self._capture(in_id, in_tensor)
                # wire any upstream preprocessing chain into the model input, then
                # reset so the next iteration's probes start a fresh (stable) chain
                entry = in_id if in_tensor is not None else node_id
                if self._last_probe is not None:
                    self.tree.link(self._last_probe, entry)
                    self._last_probe = None
                self._fwd_mode = self.server.oplevel() >= protocol.OPLEVEL_STRUCT
                if self._fwd_mode:
                    self._enter_mode()
            if is_leaf:
                self._link_inputs(node_id, args)
            self._module_stack.append(node_id)

        return pre

    def _make_post(self, module: torch.nn.Module):
        node_id, name, path, parent, depth = self._meta[module]
        is_root = module is self.model
        is_leaf = self._is_leaf.get(node_id, True)

        def post(mod, inputs, output) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                self.tree.observe(node_id, name=name, path=path, parent=parent,
                                  depth=depth, shape=tuple(tensor.shape))
                if is_leaf:
                    self._register_outputs(node_id, output)
                if self.server.is_subscribed(node_id):
                    self._capture(node_id, tensor)

            if self._module_stack and self._module_stack[-1] == node_id:
                self._module_stack.pop()

            if is_root:
                if self._fwd_mode:
                    self._exit_mode()
                    self._fwd_mode = False
                self._maybe_publish()

        return post

    # -- op-level callback (runs inside the TorchFunctionMode) -------------
    def _on_op(self, func, args, kwargs, output) -> None:
        # attribute the op to the current scope: a ``region`` stage (preprocessing
        # outside any forward) takes precedence over the executing module.
        if self._region_stack:
            mod_id, level = self._region_stack[-1]
            in_region = True
        elif self._module_stack:
            mod_id, level, in_region = self._module_stack[-1], self.server.oplevel(), False
        else:
            return
        tensor = _first_tensor(output)
        if tensor is None:
            return
        op_name = getattr(func, "__name__", None) or str(func)
        if op_name in _SKIP_OPS:
            return

        # in a region, an untraced external input is sourced from the stage node
        # so the block's first ops link to a visible source rather than dangling.
        if in_region:
            for t in _iter_tensors((args, kwargs)):
                self._producer.setdefault(id(t), mod_id)

        # resolve which already-traced tensors this op consumes (for role + edges)
        srcs = [s for s in (self._producer.get(id(t))
                            for t in _iter_tensors((args, kwargs))) if s is not None]
        n_outputs = sum(1 for _ in _iter_tensors(output))
        role = _op_role(op_name, len(set(srcs)), n_outputs)

        make_node = level >= protocol.OPLEVEL_ALL or (
            level >= protocol.OPLEVEL_STRUCT and role is not None)

        if not make_node:
            # transparent pass-through: outputs inherit the input's producer, so
            # the dataflow connects real nodes straight through this noise op.
            if srcs:
                for t in _iter_tensors(output):
                    self._producer[id(t)] = srcs[0]
            return

        key = (mod_id, op_name)
        idx = self._op_counters.get(key, 0)
        self._op_counters[key] = idx + 1

        # Roll repeated occurrences (e.g. an SSM scan's per-step ops) into a
        # bounded node set: the first ``_op_max_repeat`` get their own ``#i``
        # node, further repeats fold into the last one. ``count`` is that node's
        # multiplicity this forward; edges/producers are still tracked every
        # occurrence so the loop-carried output reaches downstream consumers (the
        # self-edge a recurrence implies is dropped by ``_link_inputs``).
        node_idx = idx if idx < self._op_max_repeat else self._op_max_repeat - 1
        count = idx - node_idx + 1
        node_id = f"{mod_id} > {op_name}#{node_idx}"
        depth = self._depth_of.get(mod_id, 0) + 1
        op_path = f"{self._path_of.get(mod_id, mod_id)}/{op_name}#{node_idx}"
        self.tree.observe(node_id, name=op_name, path=op_path, parent=mod_id,
                          depth=depth, shape=tuple(tensor.shape), role=role,
                          count=count)
        self._link_inputs(node_id, args, kwargs)
        self._register_outputs(node_id, output)
        # a region op is the chain's current tail: hand it to the model root (the
        # forward's pre-hook links _last_probe -> model), so preprocessing feeds in.
        if in_region:
            self._last_probe = node_id
        # Capture only the slot-owning occurrence (count == 1) so a rolled
        # recurrence streams one representative frame per forward, not one per
        # loop iteration (the rate limiter would drop the rest anyway).
        if count == 1 and self.server.is_subscribed(node_id):
            self._capture(node_id, tensor)

    # -- teardown ----------------------------------------------------------
    def remove(self) -> None:
        while self._mode_depth > 0:
            self._exit_mode()
        self._fwd_mode = False
        self._region_stack.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._module_stack.clear()

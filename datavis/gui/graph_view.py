"""Dataflow graph view: the execution as a DAG of producer -> consumer edges.

Where the tree shows *containment* (which module holds which), this shows *flow*:
tensor splits, parallel branches, and elementwise/residual merges — the topology
of a figure like the SSMDP block diagram. Nodes are laid out in execution layers
(longest-path from the sources) and wired with arrows. Clicking a node toggles a
subscription, exactly like checking it in the tree.

The view always renders only the *visible* node set (driven by the tree's LOD /
start-from filters). Edges whose endpoints are hidden are **contracted**: a
hidden node is bypassed by wiring its visible producers straight to its visible
consumers, so the graph stays connected at every level of detail — e.g. at module
LOD the functional ops between two layers collapse into a single arrow.
"""
from __future__ import annotations

from collections import defaultdict

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from datavis.gui.labels import display_label

_X_GAP = 240.0
_Y_GAP = 86.0
_NODE_W = 184.0
_NODE_H = 46.0
_GLYPH_R = 26.0   # half-size of a structural-op icon node

_COL_IDLE = pg.mkBrush(54, 60, 74)
_COL_SUB = pg.mkBrush(36, 110, 96)
_COL_MERGE = pg.mkBrush(96, 64, 40)   # in-degree >= 2  (Add / Mult)
_COL_SPLIT = pg.mkBrush(58, 52, 96)   # out-degree >= 2 (Tensor Split / fan-out)
_COL_OP_MERGE = pg.mkBrush(124, 84, 38)  # structural merge op (add/mul/cat/matmul)
_COL_OP_SPLIT = pg.mkBrush(72, 62, 124)  # structural split op (chunk/split)
_COL_OP_ACT = pg.mkBrush(40, 84, 96)     # nonlinearity (sigmoid/tanh/relu/...)
_COL_STAGE = pg.mkBrush(58, 76, 60)      # preprocessing stage (viz.probe, upstream)
_PEN_NODE = pg.mkPen(150, 158, 176, width=1)
_PEN_SUB = pg.mkPen(120, 240, 210, width=2)
_PEN_EDGE = pg.mkPen(150, 158, 176, width=1.4)

# structural-op roles -> (icon glyph, node shape). These are the first-class
# dataflow operators a block diagram is built from (SSMDP ⊕ ⊙ ◇ and t/σ/S/R).
_ROLE_GLYPH = {"add": "+", "sub": "−", "mul": "×", "div": "÷",
               "matmul": "@", "concat": "‖", "split": "◇",
               "sigmoid": "σ", "tanh": "t", "relu": "R", "gelu": "G",
               "silu": "S", "softplus": "ζ", "exp": "eˣ", "softmax": "sm",
               "elu": "elu"}
_ROLE_SHAPE = {"add": "ellipse", "sub": "ellipse", "mul": "ellipse",
               "div": "ellipse", "matmul": "square", "concat": "square",
               "split": "diamond"}
# activations are round icons too
for _r in ("sigmoid", "tanh", "relu", "gelu", "silu", "softplus", "exp",
           "softmax", "elu"):
    _ROLE_SHAPE[_r] = "ellipse"

_MERGE_ROLES = frozenset({"add", "sub", "mul", "div", "matmul", "concat"})
_ACT_ROLES = frozenset({"sigmoid", "tanh", "relu", "gelu", "silu", "softplus",
                        "exp", "softmax", "elu"})


def _contract(visible: set[str], edges: list) -> set[tuple[str, str]]:
    """Visible->visible edges, bypassing hidden nodes by reachability."""
    out = defaultdict(list)
    for s, d in edges:
        out[s].append(d)
    result: set[tuple[str, str]] = set()
    for s in visible:
        seen: set[str] = set()
        stack = list(out.get(s, ()))
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in visible:
                if n != s:
                    result.add((s, n))
            else:
                stack.extend(out.get(n, ()))
    return result


def _layers(visible: list[str], cedges: set[tuple[str, str]]) -> dict[str, int]:
    """Longest-path layer per node (sources at layer 0); cycle-safe."""
    layer = {n: 0 for n in visible}
    for _ in range(len(visible)):
        changed = False
        for s, d in cedges:
            if layer[d] < layer[s] + 1:
                layer[d] = layer[s] + 1
                changed = True
        if not changed:
            break
    return layer


class _ClickShape:
    """Mixin: a left click toggles this node's subscription."""

    def init_click(self, node_id: str, on_click) -> None:
        self._id = node_id
        self._on_click = on_click
        self.setZValue(10)

    def mousePressEvent(self, ev) -> None:  # noqa: N802 (Qt naming)
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._on_click(self._id)
            ev.accept()
        else:
            super().mousePressEvent(ev)


class _RectNode(_ClickShape, QtWidgets.QGraphicsRectItem):
    pass


class _EllipseNode(_ClickShape, QtWidgets.QGraphicsEllipseItem):
    pass


class _PolyNode(_ClickShape, QtWidgets.QGraphicsPolygonItem):
    pass


def _make_node_item(role, node_id, on_click):
    """A clickable shape: a wide box for modules/generic ops, or a compact
    icon (circle / square / diamond) for a structural operator."""
    shape = _ROLE_SHAPE.get(role)
    if shape == "ellipse":
        item = _EllipseNode(-_GLYPH_R, -_GLYPH_R, 2 * _GLYPH_R, 2 * _GLYPH_R)
    elif shape == "square":
        item = _RectNode(-_GLYPH_R, -_GLYPH_R, 2 * _GLYPH_R, 2 * _GLYPH_R)
    elif shape == "diamond":
        poly = QtGui.QPolygonF([
            QtCore.QPointF(0, -_GLYPH_R), QtCore.QPointF(_GLYPH_R, 0),
            QtCore.QPointF(0, _GLYPH_R), QtCore.QPointF(-_GLYPH_R, 0),
        ])
        item = _PolyNode(poly)
    else:
        item = _RectNode(-_NODE_W / 2, -_NODE_H / 2, _NODE_W, _NODE_H)
    item.init_click(node_id, on_click)
    return item


class GraphView(QtWidgets.QWidget):
    subscribeChanged = QtCore.pyqtSignal(str, bool)

    def __init__(self) -> None:
        super().__init__()
        self._nodes: dict[str, dict] = {}
        self._edges: list = []
        self._visible: set[str] = set()
        self._subscribed: set[str] = set()
        self._items: dict[str, object] = {}
        self._framed = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._gv = pg.GraphicsView()
        self._vb = pg.ViewBox()
        self._vb.setAspectLocked(True)  # keep node boxes rectangular
        self._vb.invertY(True)  # execution flows top-to-bottom within a layer
        self._gv.setCentralItem(self._vb)
        layout.addWidget(self._gv)

    # -- inputs from the app ----------------------------------------------
    def set_structure(self, nodes: list[dict], edges: list) -> None:
        self._nodes = {n["id"]: n for n in nodes}
        self._edges = [tuple(e) for e in edges]
        if not self._visible:
            self._visible = set(self._nodes)
        self._relayout()

    def set_visible(self, ids) -> None:
        self._visible = {i for i in ids if i in self._nodes}
        self._relayout()

    def set_subscribed(self, node_id: str, on: bool) -> None:
        if on:
            self._subscribed.add(node_id)
        else:
            self._subscribed.discard(node_id)
        item = self._items.get(node_id)
        if item is not None:
            self._style(item, node_id)

    # -- click -> toggle (mirrors a tree checkbox) ------------------------
    def _on_node_click(self, node_id: str) -> None:
        on = node_id not in self._subscribed
        self.subscribeChanged.emit(node_id, on)

    # -- rendering ---------------------------------------------------------
    def _indeg_outdeg(self, cedges):
        indeg = defaultdict(int)
        outdeg = defaultdict(int)
        for s, d in cedges:
            outdeg[s] += 1
            indeg[d] += 1
        return indeg, outdeg

    def _halfw(self, node_id: str) -> float:
        """Half-width along x, for stopping edges at the node boundary."""
        return _GLYPH_R if self._nodes[node_id].get("role") in _ROLE_SHAPE else _NODE_W / 2

    def _halfh(self, node_id: str) -> float:
        """Half-height along y, for placing a badge below a node."""
        return _GLYPH_R if self._nodes[node_id].get("role") in _ROLE_SHAPE else _NODE_H / 2

    def _style(self, item, node_id: str) -> None:
        if node_id in self._subscribed:
            item.setBrush(_COL_SUB)
            item.setPen(_PEN_SUB)
            return
        item.setPen(_PEN_NODE)
        role = self._nodes[node_id].get("role")
        if role == "stage":
            item.setBrush(_COL_STAGE)
        elif role == "split":
            item.setBrush(_COL_OP_SPLIT)
        elif role in _MERGE_ROLES:                # add/sub/mul/div/cat/matmul
            item.setBrush(_COL_OP_MERGE)
        elif role in _ACT_ROLES:                  # sigmoid/tanh/relu/...
            item.setBrush(_COL_OP_ACT)
        elif self._indeg.get(node_id, 0) >= 2:
            item.setBrush(_COL_MERGE)
        elif self._outdeg.get(node_id, 0) >= 2:
            item.setBrush(_COL_SPLIT)
        else:
            item.setBrush(_COL_IDLE)

    def _relayout(self) -> None:
        self._vb.clear()
        self._items.clear()
        visible = [n for n in self._nodes if n in self._visible]
        if not visible:
            return

        cedges = _contract(self._visible, self._edges)
        self._indeg, self._outdeg = self._indeg_outdeg(cedges)

        # a flow diagram only shows nodes that carry flow: drop ones left with no
        # contracted edge (e.g. pure container modules). Keep all if there are no
        # edges at all, so a not-yet-wired structure still renders.
        if cedges:
            connected = {n for e in cedges for n in e}
            visible = [n for n in visible if n in connected]
            if not visible:
                return
        layer = _layers(visible, cedges)

        by_layer: dict[int, list[str]] = defaultdict(list)
        for n in sorted(visible, key=lambda i: self._nodes[i].get("order", 0)):
            by_layer[layer[n]].append(n)

        pos: dict[str, tuple[float, float]] = {}
        for lyr, members in by_layer.items():
            y0 = -(_Y_GAP * (len(members) - 1)) / 2.0
            for j, n in enumerate(members):
                pos[n] = (lyr * _X_GAP, y0 + j * _Y_GAP)

        # edges first (under nodes), stopping at each node's boundary
        for s, d in cedges:
            if s in pos and d in pos:
                x1, y1 = pos[s]
                x2, y2 = pos[d]
                xe, xt = x1 + self._halfw(s), x2 - self._halfw(d)
                line = QtWidgets.QGraphicsLineItem(xe, y1, xt, y2)
                line.setPen(_PEN_EDGE)
                line.setZValue(1)
                self._vb.addItem(line)
                ang = QtCore.QLineF(x1, y1, x2, y2).angle()
                arrow = pg.ArrowItem(
                    angle=ang, headLen=12, tipAngle=32, pen=None,
                    brush=pg.mkBrush(150, 158, 176),
                )
                arrow.setPos(xt, y2)
                arrow.setZValue(2)
                self._vb.addItem(arrow)

        for n in visible:
            node = self._nodes[n]
            role = node.get("role")
            x, y = pos[n]
            item = _make_node_item(role, n, self._on_node_click)
            shp = "x".join(map(str, node.get("shape", [])))
            count = node.get("count", 1)
            mult = f"  ×{count}" if isinstance(count, int) and count > 1 else ""
            item.setToolTip(f"{node['path']}  [{shp}]{mult}")
            item.setPos(x, y)
            self._vb.addItem(item)
            self._items[n] = item
            self._style(item, n)
            # non-scaling label: stays legible at any zoom (boxes scale, text doesn't).
            # structural ops show their operator glyph (bigger); others their name.
            glyph = _ROLE_GLYPH.get(role)
            text = pg.TextItem(glyph or display_label(node),
                               color=(238, 240, 248), anchor=(0.5, 0.5))
            if glyph is not None:
                font = text.textItem.font()
                font.setPointSizeF(max(font.pointSizeF(), 1.0) * 1.7)
                font.setBold(True)
                text.textItem.setFont(font)
            text.setPos(x, y)
            text.setZValue(20)
            self._vb.addItem(text)
            # rolled-up loop body: a small ×N badge below the node marks how many
            # times this op ran (its recurrence multiplicity).
            if isinstance(count, int) and count > 1:
                badge = pg.TextItem(f"×{count}", color=(222, 192, 150),
                                    anchor=(0.5, 0.0))
                badge.setPos(x, y + self._halfh(n))
                badge.setZValue(20)
                self._vb.addItem(badge)

        # Frame once at a readable natural zoom anchored on the source layer; the
        # user pans/scrolls to follow the flow (right-click -> View All to fit).
        # Later relayouts (LOD changes) keep the current view.
        if not self._framed and pos:
            x0 = min(x for x, _ in pos.values())
            ys = [pos[n][1] for n in visible if pos[n][0] == x0]
            yc = sum(ys) / len(ys) if ys else 0.0
            span = 6.0 * _X_GAP
            self._vb.setRange(
                xRange=(x0 - _NODE_W, x0 - _NODE_W + span),
                yRange=(yc - span * 0.3, yc + span * 0.3),
                padding=0,
            )
            self._framed = True

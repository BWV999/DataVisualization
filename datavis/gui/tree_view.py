"""Pipeline tree with a level-of-detail (LOD) slider and start-point selector.

The execution tree is shown as a checkable QTreeWidget (check = subscribe). Three
controls shape what is visible / how deep the trace goes:
  - "Trace ops" checkbox: ask the tracer to enable op-level tracing (the deep LOD
    leaves: view/transpose/relu/...). Off by default so training pays nothing.
  - LOD slider: hide nodes deeper than the chosen depth.
  - Start-from selector: hide nodes that execute before the chosen node.

``set_structure`` is incremental: op-level nodes that appear later are added
without disturbing existing checks/subscriptions.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from datavis.common import protocol
from datavis.gui.labels import tree_label


class PipelineTree(QtWidgets.QWidget):
    subscribeChanged = QtCore.pyqtSignal(str, bool)
    opLevelChanged = QtCore.pyqtSignal(int)    # OPLEVEL_OFF / _STRUCT / _ALL
    filterChanged = QtCore.pyqtSignal(object)  # set[str] of currently visible ids

    def __init__(self) -> None:
        super().__init__()
        self._nodes: dict[str, dict] = {}
        self._items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._initialized = False

        layout = QtWidgets.QVBoxLayout(self)

        op_row = QtWidgets.QHBoxLayout()
        op_row.addWidget(QtWidgets.QLabel("Ops"))
        self.op_level = QtWidgets.QComboBox()
        self.op_level.addItem("off (modules)", protocol.OPLEVEL_OFF)
        self.op_level.addItem("structural (+ - x / cat @ split)", protocol.OPLEVEL_STRUCT)
        self.op_level.addItem("all (deep LOD)", protocol.OPLEVEL_ALL)
        self.op_level.setToolTip(
            "off: modules only · structural: also show merge/split ops as nodes "
            "· all: every op")
        self.op_level.currentIndexChanged.connect(
            lambda _i: self.opLevelChanged.emit(int(self.op_level.currentData())))
        op_row.addWidget(self.op_level, 1)
        layout.addLayout(op_row)

        lod_row = QtWidgets.QHBoxLayout()
        lod_row.addWidget(QtWidgets.QLabel("LOD"))
        self.lod = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.lod.setMinimum(0)
        self.lod.setMaximum(0)
        self.lod.valueChanged.connect(self._apply_filter)
        self.lod_label = QtWidgets.QLabel("0")
        lod_row.addWidget(self.lod, 1)
        lod_row.addWidget(self.lod_label)
        layout.addLayout(lod_row)

        start_row = QtWidgets.QHBoxLayout()
        start_row.addWidget(QtWidgets.QLabel("Start from"))
        self.start = QtWidgets.QComboBox()
        self.start.currentIndexChanged.connect(self._apply_filter)
        start_row.addWidget(self.start, 1)
        layout.addLayout(start_row)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["module", "shape"])
        self.tree.header().setStretchLastSection(False)
        self.tree.setColumnWidth(0, 175)
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree, 1)

    # -- structure (incremental) ------------------------------------------
    def set_structure(self, nodes: list[dict]) -> None:
        self.tree.blockSignals(True)
        self._nodes = {n["id"]: n for n in nodes}
        new = [n for n in nodes if n["id"] not in self._items]

        # create new items shallow-first so parents exist before children attach
        for node in sorted(new, key=lambda n: n.get("depth", 0)):
            shape = "x".join(str(s) for s in node.get("shape", []))
            item = QtWidgets.QTreeWidgetItem([tree_label(node), shape])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, node["id"])
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            self._items[node["id"]] = item
        for node in sorted(new, key=lambda n: n.get("depth", 0)):
            item = self._items[node["id"]]
            parent = node.get("parent")
            if parent and parent in self._items:
                self._items[parent].addChild(item)
            else:
                self.tree.addTopLevelItem(item)

        self._refresh_controls()
        self.tree.expandAll()
        self.tree.resizeColumnToContents(1)
        self.tree.blockSignals(False)
        self._apply_filter()

    def _refresh_controls(self) -> None:
        max_depth = max((n.get("depth", 0) for n in self._nodes.values()), default=0)
        self.lod.setMaximum(max_depth)
        if not self._initialized:
            self.lod.setValue(max_depth)  # first load: show everything
            self._initialized = True
        elif int(self.op_level.currentData()) >= protocol.OPLEVEL_STRUCT:
            self.lod.setValue(max_depth)  # reveal newly traced op leaves

        current = self.start.currentData()
        self.start.blockSignals(True)
        self.start.clear()
        for node in sorted(self._nodes.values(), key=lambda n: n.get("order", 0)):
            self.start.addItem(node["path"], node["id"])
        idx = self.start.findData(current)
        self.start.setCurrentIndex(idx if idx >= 0 else 0)
        self.start.blockSignals(False)

    # -- filters -----------------------------------------------------------
    def _apply_filter(self) -> None:
        lod = self.lod.value()
        self.lod_label.setText(str(lod))
        start_id = self.start.currentData()
        start_order = self._nodes.get(start_id, {}).get("order", 0) if start_id else 0
        visible_ids = set()
        for node_id, item in self._items.items():
            node = self._nodes[node_id]
            visible = node.get("depth", 0) <= lod and node.get("order", 0) >= start_order
            item.setHidden(not visible)
            if visible:
                visible_ids.add(node_id)
        self.filterChanged.emit(visible_ids)

    def _on_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        node_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        checked = item.checkState(0) == QtCore.Qt.CheckState.Checked
        self.subscribeChanged.emit(node_id, checked)

    def set_checked(self, node_id: str, checked: bool) -> None:
        """Reflect a subscription toggled elsewhere (e.g. the graph) silently."""
        item = self._items.get(node_id)
        if item is None:
            return
        state = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
        if item.checkState(0) != state:
            self.tree.blockSignals(True)
            item.setCheckState(0, state)
            self.tree.blockSignals(False)

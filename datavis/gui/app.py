"""PyQtGraph application (separate process from training).

Left: the pipeline tree (PipelineTree) with LOD slider + start-point selector;
checking a node subscribes to it and opens a live panel in the dock area.

The structure can arrive two ways: as the reply to the initial ``hello`` (GUI
started after training) or live on the data channel after the first forward (GUI
started first). Both are routed through one Qt signal so all UI work stays on the
GUI thread.
"""
from __future__ import annotations

import argparse

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.dockarea import Dock, DockArea

from datavis.common import protocol
from datavis.gui.graph_view import GraphView
from datavis.gui.panels import make_panel
from datavis.gui.transport import GuiClient
from datavis.gui.tree_view import PipelineTree


class MessageBridge(QtCore.QObject):
    """Carries a decoded message from the network thread to the GUI thread."""

    message = QtCore.pyqtSignal(object)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ctrl_addr: str, data_addr: str, rate: float) -> None:
        super().__init__()
        self.setWindowTitle("DataVisualization")
        self.rate = rate
        self.client = GuiClient(ctrl_addr, data_addr)
        self.nodes: dict[str, dict] = {}
        self.panels: dict[str, tuple[Dock, object]] = {}
        self._loaded_sig: tuple = ()

        # the data area (dataflow DAG + panels) is the whole central space
        self.dock_area = DockArea()
        self.setCentralWidget(self.dock_area)

        self.pipeline = PipelineTree()
        self.pipeline.subscribeChanged.connect(self._on_subscribe_changed)
        self.pipeline.opLevelChanged.connect(self.client.set_oplevel)

        # the module selector lives in a collapsible left dock, so it never eats
        # into the data-display area: drag its edge to resize, or toggle it off
        # from the toolbar and the central area reclaims the full width.
        self.side = QtWidgets.QDockWidget("Pipeline", self)
        self.side.setObjectName("pipeline_dock")
        self.side.setWidget(self.pipeline)
        self.side.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.side)
        self.pipeline.setMinimumWidth(240)
        self.resizeDocks([self.side], [330], QtCore.Qt.Orientation.Horizontal)

        toolbar = self.addToolBar("View")
        toolbar.setMovable(False)
        toggle = self.side.toggleViewAction()
        toggle.setText("Pipeline")
        toggle.setToolTip("Show/hide the module selector")
        toolbar.addAction(toggle)

        # the dataflow DAG occupies the main area; data panels dock below it
        self.graph = GraphView()
        self.graph.subscribeChanged.connect(self._on_subscribe_changed)
        self.pipeline.filterChanged.connect(self.graph.set_visible)
        self.graph_dock = Dock("Dataflow", widget=self.graph, closable=False)
        self.dock_area.addDock(self.graph_dock)

        self.statusBar().showMessage("waiting for structure (start training)…")

        self.bridge = MessageBridge()
        self.bridge.message.connect(self._on_message)
        self.client.start(self.bridge.message.emit)

        # late-join case: structure may already be available
        self._on_message(self.client.hello())

    # -- message routing ---------------------------------------------------
    def _on_message(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == protocol.MSG_STRUCTURE:
            self._load_structure(msg.get("nodes", []), msg.get("edges", []))
        elif mtype == protocol.MSG_FRAME:
            entry = self.panels.get(msg.get("node_id"))
            if entry is not None:
                entry[1].update_frame(
                    msg.get("payload", {}), msg.get("stats", {}), msg.get("step", 0)
                )

    def _load_structure(self, nodes: list[dict], edges: list) -> None:
        sig = (tuple(n["id"] for n in nodes), len(edges))
        if not nodes or sig == self._loaded_sig:
            return
        self._loaded_sig = sig
        self.nodes = {n["id"]: n for n in nodes}
        # graph first: the tree's set_structure emits filterChanged -> set_visible,
        # which must see the new nodes (else fresh op nodes get filtered out)
        self.graph.set_structure(nodes, edges)
        self.pipeline.set_structure(nodes)  # incremental: keeps existing checks
        self.statusBar().showMessage(
            f"{len(nodes)} nodes, {len(edges)} dataflow edges — click a node to visualize"
        )

    # -- subscription / panels ---------------------------------------------
    def _on_subscribe_changed(self, node_id: str, checked: bool) -> None:
        if checked:
            self._add_panel(node_id)
            self.client.subscribe(node_id, self.rate)
        else:
            self.client.unsubscribe(node_id)
            self._remove_panel(node_id)
        # keep tree checkbox and graph highlight in sync regardless of origin
        self.pipeline.set_checked(node_id, checked)
        self.graph.set_subscribed(node_id, checked)

    def _add_panel(self, node_id: str) -> None:
        if node_id in self.panels:
            return
        node = self.nodes[node_id]
        panel = make_panel(node)
        dock = Dock(node["path"], widget=panel, closable=True)
        self.dock_area.addDock(dock, "bottom", self.graph_dock)
        self.panels[node_id] = (dock, panel)

    def _remove_panel(self, node_id: str) -> None:
        entry = self.panels.pop(node_id, None)
        if entry is not None:
            entry[0].close()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.client.stop()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="DataVisualization GUI")
    parser.add_argument("--ctrl", default="tcp://127.0.0.1:5750")
    parser.add_argument("--data", default="tcp://127.0.0.1:5751")
    parser.add_argument("--rate", type=float, default=15.0,
                        help="per-node capture rate (frames/sec)")
    args = parser.parse_args()

    pg.mkQApp("DataVisualization")
    window = MainWindow(args.ctrl, args.data, args.rate)
    window.resize(1280, 800)
    window.show()
    pg.exec()


if __name__ == "__main__":
    main()

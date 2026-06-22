"""Tracer-side IPC: binds the control (REP) and data (PUB) sockets.

Lives in the training process. The control loop runs on a daemon thread so the
training thread only ever calls cheap, non-blocking ``is_subscribed`` /
``maybe_send`` helpers. Payloads are built *only* for subscribed nodes that are
also due under their per-node rate limit, keeping the training hot path light.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np
import zmq

from datavis.common import protocol


class TracerServer:
    def __init__(
        self,
        ctrl_addr: str = "tcp://127.0.0.1:5750",
        data_addr: str = "tcp://127.0.0.1:5751",
    ) -> None:
        self.ctrl_addr = ctrl_addr
        self.data_addr = data_addr
        self._ctx = zmq.Context.instance()
        self._ctrl = self._ctx.socket(zmq.REP)
        self._ctrl.bind(ctrl_addr)
        self._data = self._ctx.socket(zmq.PUB)
        self._data.bind(data_addr)

        self._subs: dict[str, float] = {}       # node_id -> rate (frames/sec, 0 = unlimited)
        self._last_sent: dict[str, float] = {}   # node_id -> monotonic time
        self._structure: Optional[dict] = None
        self._oplevel = protocol.OPLEVEL_OFF      # op detail tier requested by GUI
        self._lock = threading.Lock()
        self.closed = False

        self._running = True
        self._thread = threading.Thread(target=self._ctrl_loop, daemon=True)
        self._thread.start()

    # -- structure ---------------------------------------------------------
    def set_structure(self, nodes: list[dict], edges: Optional[list] = None) -> None:
        """Store the structure (for late-joining GUIs via hello) and publish it
        live on the data channel (for GUIs already connected). Called from the
        same thread as ``send_frame``, so the PUB socket stays single-threaded.
        """
        msg = {"type": protocol.MSG_STRUCTURE, "nodes": nodes, "edges": edges or []}
        with self._lock:
            self._structure = msg
        self._data.send(protocol.encode(msg))

    # -- control loop ------------------------------------------------------
    def _ctrl_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._ctrl, zmq.POLLIN)
        while self._running:
            if dict(poller.poll(timeout=200)).get(self._ctrl) != zmq.POLLIN:
                continue
            msg = protocol.decode(self._ctrl.recv())
            self._ctrl.send(protocol.encode(self._handle_ctrl(msg)))

    def _handle_ctrl(self, msg: dict) -> dict:
        mtype = msg.get("type")
        if mtype == protocol.MSG_HELLO:
            with self._lock:
                return self._structure or {"type": protocol.MSG_STRUCTURE, "nodes": [], "edges": []}
        if mtype == protocol.MSG_SUBSCRIBE:
            with self._lock:
                self._subs[msg["node_id"]] = float(msg.get("rate", 10.0))
            return {"type": protocol.MSG_ACK}
        if mtype == protocol.MSG_UNSUBSCRIBE:
            with self._lock:
                self._subs.pop(msg["node_id"], None)
            return {"type": protocol.MSG_ACK}
        if mtype == protocol.MSG_SET_OPLEVEL:
            # accept new "level" tier; fall back to legacy "enabled" bool
            level = msg["level"] if "level" in msg else msg.get("enabled", False)
            self.set_oplevel(level)
            return {"type": protocol.MSG_ACK}
        return {"type": protocol.MSG_ACK}

    # -- op detail tier (read on the training thread) ---------------------
    def set_oplevel(self, level) -> None:
        with self._lock:
            self._oplevel = protocol.normalize_oplevel(level)

    def oplevel(self) -> int:
        with self._lock:
            return self._oplevel

    # -- data path (called from the training thread) -----------------------
    def is_subscribed(self, node_id: str) -> bool:
        """True if the node is subscribed AND due under its rate limit."""
        with self._lock:
            rate = self._subs.get(node_id)
            if rate is None:
                return False
            if rate > 0:
                last = self._last_sent.get(node_id, 0.0)
                if (time.monotonic() - last) < (1.0 / rate):
                    return False
            return True

    def send_frame(self, node_id: str, step: int, payload: dict, stats: dict) -> None:
        with self._lock:
            self._last_sent[node_id] = time.monotonic()
        self._data.send(
            protocol.encode(
                {
                    "type": protocol.MSG_FRAME,
                    "node_id": node_id,
                    "step": step,
                    "t": time.time(),
                    "payload": payload,
                    "stats": stats,
                }
            )
        )

    def maybe_send(
        self,
        node_id: str,
        step: int,
        get_array: Callable[[], np.ndarray],
        kind: Optional[str] = None,
    ) -> bool:
        """Build + send a frame only if subscribed and due. Returns True if sent.

        ``get_array`` is a thunk so the (potentially expensive) tensor->numpy
        conversion never runs for unsubscribed nodes.
        """
        if not self.is_subscribed(node_id):
            return False
        arr = np.asarray(get_array())
        self.send_frame(
            node_id,
            step,
            protocol.auto_payload(arr, kind=kind),
            protocol.compute_stats(arr),
        )
        return True

    # -- teardown ----------------------------------------------------------
    def close(self) -> None:
        self.closed = True
        self._running = False
        self._thread.join(timeout=1.0)
        self._ctrl.close(0)
        self._data.close(0)

"""GUI-side IPC: connects the control (REQ) and data (SUB) sockets.

Kept free of any Qt dependency so it can be unit-tested headless. The frame
receive loop runs on a daemon thread and hands each decoded message to a
callback; the Qt app wraps that callback to marshal onto the GUI thread.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import zmq

from datavis.common import protocol


class GuiClient:
    def __init__(
        self,
        ctrl_addr: str = "tcp://127.0.0.1:5750",
        data_addr: str = "tcp://127.0.0.1:5751",
    ) -> None:
        self._ctx = zmq.Context.instance()
        self._ctrl = self._ctx.socket(zmq.REQ)
        self._ctrl.connect(ctrl_addr)
        self._data = self._ctx.socket(zmq.SUB)
        self._data.connect(data_addr)
        self._data.setsockopt(zmq.SUBSCRIBE, b"")

        self._on_frame: Optional[Callable[[dict], None]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # -- control (called from the GUI thread) ------------------------------
    def _request(self, msg: dict) -> dict:
        self._ctrl.send(protocol.encode(msg))
        return protocol.decode(self._ctrl.recv())

    def hello(self) -> dict:
        return self._request({"type": protocol.MSG_HELLO})

    def subscribe(self, node_id: str, rate: float = 10.0) -> dict:
        return self._request(
            {"type": protocol.MSG_SUBSCRIBE, "node_id": node_id, "rate": rate}
        )

    def unsubscribe(self, node_id: str) -> dict:
        return self._request({"type": protocol.MSG_UNSUBSCRIBE, "node_id": node_id})

    def set_oplevel(self, level: int) -> dict:
        return self._request({"type": protocol.MSG_SET_OPLEVEL, "level": int(level)})

    # -- data --------------------------------------------------------------
    def start(self, on_frame: Callable[[dict], None]) -> None:
        self._on_frame = on_frame
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._data, zmq.POLLIN)
        while self._running:
            if dict(poller.poll(timeout=200)).get(self._data) == zmq.POLLIN:
                msg = protocol.decode(self._data.recv())
                if self._on_frame is not None:
                    self._on_frame(msg)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._ctrl.close(0)
        self._data.close(0)

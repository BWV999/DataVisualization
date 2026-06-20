import threading
import time

import numpy as np
import pytest

from datavis.common import protocol
from datavis.gui.transport import GuiClient
from datavis.tracer.transport import TracerServer

CTRL = "tcp://127.0.0.1:5790"
DATA = "tcp://127.0.0.1:5791"


@pytest.fixture
def server():
    srv = TracerServer(CTRL, DATA)
    yield srv
    srv.close()


def test_hello_returns_structure(server):
    nodes = [{"id": "raw", "name": "raw", "path": "signal.raw",
              "parent": None, "rank": 2, "shape": [4, 128], "depth": 0}]
    server.set_structure(nodes)
    client = GuiClient(CTRL, DATA)
    try:
        structure = client.hello()
        assert structure["type"] == protocol.MSG_STRUCTURE
        assert structure["nodes"][0]["id"] == "raw"
    finally:
        client.stop()


def test_subscribe_then_receive_frame(server):
    server.set_structure([{"id": "raw", "path": "signal.raw", "rank": 2,
                           "shape": [4, 128]}])
    received: list[dict] = []
    client = GuiClient(CTRL, DATA)
    client.start(received.append)
    try:
        client.hello()
        client.subscribe("raw", rate=0)  # unlimited

        arr = np.random.randn(4, 128).astype(np.float32)
        deadline = time.time() + 3.0
        while not received and time.time() < deadline:
            server.send_frame("raw", 0, protocol.auto_payload(arr),
                              protocol.compute_stats(arr))
            time.sleep(0.02)

        assert received, "no frame received within timeout"
        frame = received[0]
        assert frame["node_id"] == "raw"
        assert frame["payload"]["kind"] == protocol.KIND_CURVE
        assert frame["payload"]["y"].shape[0] == 4
    finally:
        client.stop()


def test_unsubscribed_node_is_not_sent(server):
    server.set_structure([{"id": "raw", "path": "signal.raw"}])
    assert server.is_subscribed("raw") is False
    sent = server.maybe_send("raw", 0, lambda: np.zeros((4, 8), np.float32))
    assert sent is False


def test_rate_limit_gates_second_call(server):
    server.set_structure([{"id": "raw", "path": "signal.raw"}])
    # subscribe directly via control to avoid PUB/SUB timing
    client = GuiClient(CTRL, DATA)
    try:
        client.hello()
        client.subscribe("raw", rate=5.0)  # >=200ms between frames
        first = server.maybe_send("raw", 0, lambda: np.zeros((4, 8), np.float32))
        second = server.maybe_send("raw", 1, lambda: np.zeros((4, 8), np.float32))
        assert first is True
        assert second is False
    finally:
        client.stop()

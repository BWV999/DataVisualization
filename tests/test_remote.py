"""Remote-workflow features: bind resolution, server re-attach, serve, probe.

Covers the field-report fixes — the GUI-over-SSH-tunnel scenario where a fresh
model is built per k-fold and the tracer must stay reachable after training.
"""
import threading
import time

import pytest

import datavis.tracer as viz
from datavis.common import protocol
from datavis.probe import probe
from datavis.tracer.transport import TracerServer


# -- address resolution (Issue 3: bind shorthand + env) --------------------
def test_resolve_addrs_default():
    assert viz._resolve_addrs() == ("tcp://127.0.0.1:5750", "tcp://127.0.0.1:5751")


def test_resolve_addrs_bind_arg():
    assert viz._resolve_addrs(bind="0.0.0.0") == (
        "tcp://0.0.0.0:5750", "tcp://0.0.0.0:5751")


def test_resolve_addrs_env(monkeypatch):
    monkeypatch.setenv("DATAVIS_BIND", "0.0.0.0")
    monkeypatch.setenv("DATAVIS_CTRL_PORT", "6000")
    assert viz._resolve_addrs() == ("tcp://0.0.0.0:6000", "tcp://0.0.0.0:5751")


def test_explicit_addr_overrides_bind():
    assert viz._resolve_addrs("tcp://a:1", "tcp://b:2", bind="0.0.0.0") == (
        "tcp://a:1", "tcp://b:2")


# -- probe (Issue 4) -------------------------------------------------------
def test_probe_reachable():
    srv = TracerServer("tcp://127.0.0.1:5792", "tcp://127.0.0.1:5793")
    srv.set_structure([{"id": "raw", "path": "signal.raw", "shape": [4, 8]}])
    try:
        reply = probe("tcp://127.0.0.1:5792", timeout=2.0)
        assert reply is not None
        assert reply["type"] == protocol.MSG_STRUCTURE
        assert reply["nodes"][0]["id"] == "raw"
    finally:
        srv.close()


def test_probe_dead_port_returns_none():
    assert probe("tcp://127.0.0.1:5999", timeout=0.3) is None


# -- server re-attach (Issue 2) --------------------------------------------
def test_reattach_same_custom_ports_does_not_raise():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    ctrl, data = "tcp://127.0.0.1:5794", "tcp://127.0.0.1:5795"
    m1, m2 = nn.Linear(4, 4), nn.Linear(4, 4)
    t1 = viz.attach(m1, ctrl_addr=ctrl, data_addr=data, quiet=True)
    try:
        # the field-report crash: a second attach on the same ports used to raise
        # "Address already in use". It must now adopt the server and rebind hooks.
        t2 = viz.reattach(m2, quiet=True)
        assert t2 is not t1
        assert viz._active is t2
        assert t1._handles == []          # previous model unhooked
        assert t2.server is t1.server     # same bound server adopted
    finally:
        viz.detach(close_server=True)


def test_attach_twice_same_ports_via_attach():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    ctrl, data = "tcp://127.0.0.1:5796", "tcp://127.0.0.1:5797"
    viz.attach(nn.Linear(2, 2), ctrl_addr=ctrl, data_addr=data, quiet=True)
    try:
        viz.attach(nn.Linear(2, 2), ctrl_addr=ctrl, data_addr=data, quiet=True)
    finally:
        viz.detach(close_server=True)


# -- serve keep-alive (Issue 1) --------------------------------------------
def test_serve_reforwards_to_late_subscriber():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from datavis.gui.transport import GuiClient

    ctrl, data = "tcp://127.0.0.1:5798", "tcp://127.0.0.1:5799"
    model = nn.Linear(4, 4)
    batch = [torch.randn(2, 4)]

    server_thread = threading.Thread(
        target=lambda: viz.serve(model, batch, ctrl_addr=ctrl, data_addr=data,
                                 interval=0.02, quiet=True),
        daemon=True,
    )
    server_thread.start()
    try:
        client = GuiClient(ctrl, data)
        # subscribe AFTER serving started — the late-join case the report wants
        deadline = time.time() + 3.0
        while not client.hello().get("nodes") and time.time() < deadline:
            time.sleep(0.05)
        received: list[dict] = []
        client.start(received.append)
        client.subscribe("model", rate=0)
        while not received and time.time() < deadline:
            time.sleep(0.05)
        assert received, "serve did not re-forward to a late subscriber"
        client.stop()
    finally:
        # daemon thread dies with the process; release the cached server
        viz.detach(close_server=True)

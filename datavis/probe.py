"""Connectivity probe — is a tracer reachable at this address?

Does the control handshake (the same ``hello`` the GUI sends) and prints OK plus
the available node list, *without* launching the Qt GUI. Ideal for confirming an
SSH tunnel actually reaches the tracer before opening the viewer:

    python -m datavis.probe tcp://127.0.0.1:5750

Qt-free and depends only on ``pyzmq`` / ``msgpack`` (the core tracer install), so
it runs on a headless remote box too.
"""
from __future__ import annotations

import argparse
import sys

import zmq

from datavis.common import protocol


def probe(ctrl_addr: str, timeout: float = 2.0) -> dict | None:
    """Send a ``hello`` to ``ctrl_addr`` and return the structure reply, or
    ``None`` if nothing answers within ``timeout`` seconds."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(ctrl_addr)
    try:
        sock.send(protocol.encode({"type": protocol.MSG_HELLO}))
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        if dict(poller.poll(timeout=int(timeout * 1000))).get(sock) != zmq.POLLIN:
            return None
        return protocol.decode(sock.recv())
    finally:
        sock.close(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m datavis.probe",
        description="Check that a datavis tracer is reachable (e.g. over an SSH tunnel).",
    )
    parser.add_argument("ctrl", nargs="?", default="tcp://127.0.0.1:5750",
                        help="tracer control address (default: tcp://127.0.0.1:5750)")
    parser.add_argument("--timeout", type=float, default=2.0,
                        help="seconds to wait for a reply (default: 2.0)")
    args = parser.parse_args(argv)

    reply = probe(args.ctrl, args.timeout)
    if reply is None:
        print(f"[datavis] NO tracer at {args.ctrl} "
              f"(no reply within {args.timeout:g}s)")
        print("  - is training running with viz.attach(...)?")
        print("  - remote box: bind 0.0.0.0 (attach(bind=\"0.0.0.0\")) and check "
              "your SSH -L tunnel")
        return 1

    nodes = reply.get("nodes", [])
    print(f"[datavis] OK — tracer reachable at {args.ctrl}")
    print(f"  {len(nodes)} node(s):")
    for n in nodes[:50]:
        print(f"    - {n.get('id')}  ({n.get('path', '')})  shape={n.get('shape')}")
    if len(nodes) > 50:
        print(f"    ... (+{len(nodes) - 50} more)")
    if not nodes:
        print("  (no nodes yet — run a forward pass so the structure is published)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

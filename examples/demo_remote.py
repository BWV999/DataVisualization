"""Remote k-fold workflow: bind for a tunnel, re-attach per fold, serve at the end.

Mirrors the "remote GPU box + local GUI over an SSH tunnel" scenario with a
k-fold loop that builds a fresh model per fold. On the training box run:

    python examples/demo_remote.py            # binds 0.0.0.0 so a tunnel reaches it

On your laptop, tunnel both ports and (optionally) probe before opening the GUI:

    ssh -N -L 5750:<node>:5750 -L 5751:<node>:5751 user@login-node
    python -m datavis.probe tcp://127.0.0.1:5750
    python -m datavis.gui.app

Key points this demonstrates:
  * ``bind="0.0.0.0"`` makes the tracer reachable from another machine.
  * ``viz.reattach`` swaps the hooked model each fold without re-binding ports
    (a plain second ``attach`` on the same ports would work too — no crash).
  * ``viz.serve`` keeps the process alive after the last fold and re-forwards a
    batch, so a GUI that connects *late* can still inspect the run.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import datavis.tracer as viz

N_FOLDS = 3


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


def train_fold(model: nn.Module, steps: int = 40) -> None:
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        x = torch.randn(32, 16)
        y = torch.randint(0, 4, (32,))
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()


def main() -> None:
    model = make_model()
    # bind on all interfaces so the laptop's tunnel can reach this box; attach
    # prints the listening host/ports so you know what to forward.
    viz.attach(model, bind="0.0.0.0")

    for fold in range(N_FOLDS):
        if fold > 0:
            model = make_model()
            viz.reattach(model)      # fresh model, same ports — no "address in use"
        print(f"[fold {fold}] training…")
        train_fold(model)

    # the loop is done; keep the tracer alive and re-forward a batch so a GUI
    # connecting now still streams data instead of hitting "connection refused".
    print("training complete — serving for inspection (Ctrl-C to exit)")
    viz.serve(batches=[torch.randn(32, 16)], interval=0.1)


if __name__ == "__main__":
    main()

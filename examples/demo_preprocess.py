"""A numpy preprocessing chain auto-traced into the model's DAG.

The point of this demo is the *left* of the graph. Earlier demos only showed the
torch model; here the out-of-model numpy preprocessing — standardization, a
derived feature, concatenation, clipping — is captured automatically by wrapping
the raw array *once* with ``viz.track``. No per-stage ``viz.probe`` calls: the
``TracedArray`` propagates through every numpy op (NEP-13 ufuncs + NEP-18
functions), and its tail is handed to the model root, so the GUI's Dataflow view
shows one connected DAG::

    raw_features -> mean -> subtract -> divide -> power -> mean -> sqrt
                 -> std  ->         concatenate -> clip -> model -> ...

Run the GUI first (``python -m datavis.gui.app``), then this script; check nodes
(in either the graph or the tree) to watch each preprocessing stage update live.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

import datavis.tracer as viz

N_FEATURES = 16


def make_model() -> nn.Module:
    # input is N_FEATURES standardized channels + 1 derived "energy" feature
    return nn.Sequential(
        nn.Linear(N_FEATURES + 1, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 4),
    )


def raw_batch(batch: int = 64) -> np.ndarray:
    """A raw, unnormalized multi-channel batch (varied per-channel scale)."""
    scale = np.linspace(0.5, 8.0, N_FEATURES).astype(np.float32)
    return (np.random.randn(batch, N_FEATURES).astype(np.float32) * scale
            + scale * 0.3)


def preprocess(raw: np.ndarray):
    """Pure numpy preprocessing — every step is auto-recorded once ``raw`` is
    a tracked array. Mirrors a typical tabular/feature pipeline."""
    x = viz.track(raw, "raw_features")                  # wrap once at the source
    x = (x - x.mean(0)) / (x.std(0) + 1e-5)             # per-channel standardize
    energy = np.sqrt((x ** 2).mean(1, keepdims=True))   # derived feature (B, 1)
    x = np.concatenate([x, energy], axis=1)             # append it -> (B, 17)
    x = np.clip(x, -4.0, 4.0)                           # guard outliers
    return x


def serve() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    viz.attach(model)
    print("attached; preprocessing + training. Open the GUI: the numpy "
          "preprocessing appears upstream of the model in the Dataflow view.")

    step = 0
    try:
        while True:
            x = preprocess(raw_batch())                 # numpy, auto-traced
            xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            y = torch.randint(0, 4, (xt.shape[0],))
            opt.zero_grad()
            loss = loss_fn(model(xt), y)
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}  loss {loss.item():.3f}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    serve()

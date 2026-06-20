"""An sklearn Pipeline auto-traced into the model's DAG.

A classic tabular setup — an sklearn preprocessing ``Pipeline`` (scale -> PCA)
feeding a small torch classifier. ``attach_pipeline`` wraps the pipeline's
uniform ``transform`` interface, so each named step becomes an upstream node with
no per-step probes; the chain's tail is handed to the model root. The GUI's
Dataflow view shows one connected DAG::

    input -> scaler (StandardScaler) -> pca (PCA) -> model -> Linear -> ...

Run the GUI first (``python -m datavis.gui.app``), then this script; check the
``scaler`` / ``pca`` nodes to watch the transformed features update live as the
loop streams batches through the fitted pipeline.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import datavis.tracer as viz

N_RAW = 20
N_COMPONENTS = 8


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(N_COMPONENTS, 64), nn.ReLU(),
        nn.Linear(64, 16), nn.ReLU(),
        nn.Linear(16, 3),
    )


def raw_batch(batch: int = 64) -> np.ndarray:
    scale = np.linspace(0.5, 10.0, N_RAW).astype(np.float32)
    return np.random.randn(batch, N_RAW).astype(np.float32) * scale + scale * 0.4


def serve() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("pca", PCA(n_components=N_COMPONENTS))])
    pipe.fit(raw_batch(512))                     # fit the preprocessing once

    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    viz.attach(model)
    viz.attach_pipeline(pipe)                    # trace the pipeline's steps
    print("attached; streaming batches through the fitted pipeline + model. "
          "Open the GUI: scaler -> pca appear upstream of the model.")

    step = 0
    try:
        while True:
            Xt = pipe.transform(raw_batch())     # each step is auto-traced
            xt = torch.as_tensor(np.asarray(Xt), dtype=torch.float32)
            y = torch.randint(0, 3, (xt.shape[0],))
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

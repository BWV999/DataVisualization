"""numpy -> sklearn -> torch captured as ONE connected DAG.

This is the multi-backend payoff. Three different libraries handle three stages
of the pipeline, and none of them knows about the others — yet the whole data
evolution shows up as a single chain in the GUI:

    raw -> clip -> subtract (numpy)  ->  input -> scaler -> pca (sklearn)  ->  model -> ... (torch)

Each backend is traced through its own uniform interface (numpy's NEP-13/18
protocols, sklearn's ``transform``, torch's module/`__torch_function__` hooks),
and the stages are stitched at the boundaries through a shared "current tail"
cursor — so even though sklearn strips numpy's ``TracedArray`` tag and torch
sees only a plain tensor, the edges stay connected end to end. No per-stage
probes.

Run the GUI first (``python -m datavis.gui.app``), then this script.
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
CLIP = 6.0


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(N_COMPONENTS, 64), nn.ReLU(),
        nn.Linear(64, 16), nn.ReLU(),
        nn.Linear(16, 3),
    )


def raw_batch(batch: int = 64) -> np.ndarray:
    scale = np.linspace(0.5, 12.0, N_RAW).astype(np.float32)
    return np.random.randn(batch, N_RAW).astype(np.float32) * scale + scale * 0.4


def clean_numpy(raw: np.ndarray):
    """Stage 1 (numpy): wrap once, then clip outliers and center — auto-traced."""
    x = viz.track(raw, "raw")
    x = np.clip(x, -CLIP, CLIP)          # guard outliers
    x = x - x.mean(0)                    # center per channel
    return x


def serve() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    model = make_model()                 # stage 3 (torch)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # attach the model first, so the numpy / sklearn passes below share its
    # server + DAG (and don't bind the default ports before attach() does)
    viz.attach(model)

    # stage 2 (sklearn): fit the preprocessing pipeline once on cleaned data
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("pca", PCA(n_components=N_COMPONENTS))])
    pipe.fit(np.asarray(clean_numpy(raw_batch(512))))
    viz.attach_pipeline(pipe)
    print("attached; streaming numpy -> sklearn -> torch. Open the GUI: the "
          "Dataflow view is one chain spanning all three libraries.")

    step = 0
    try:
        while True:
            x = clean_numpy(raw_batch())                 # numpy stage
            Xt = pipe.transform(np.asarray(x))           # sklearn stage
            xt = torch.as_tensor(np.asarray(Xt), dtype=torch.float32)
            y = torch.randint(0, 3, (xt.shape[0],))
            opt.zero_grad()
            loss = loss_fn(model(xt), y)                 # torch stage
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

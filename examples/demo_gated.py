"""Gated-residual net — the model used to demo first-class structural ops.

Each block is a gated activation unit with a residual, i.e. the SSMDP-style
pattern: ``fc`` -> tensor **split** (◇) -> two activations (σ, t) -> elementwise
**mul** (⊙) -> ``proj`` -> **add** (⊕) residual -> ``norm``. Run the GUI, set
**Ops** to *structural*, and the Dataflow view draws exactly those icon nodes.

    # terminal A
    uv run python -m datavis.gui.app
    # terminal B
    uv run python examples/demo_gated.py

Then in the GUI: Ops -> "structural", and click any node to open a live panel.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

import datavis.tracer as viz


class GatedBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)       # ◇ tensor split
        g = torch.sigmoid(a) * torch.tanh(b)     # σ, t  -> ⊙ gate
        return self.norm(self.proj(g) + x)       # ⊕ residual


def main() -> None:
    model = nn.Sequential(GatedBlock(16), GatedBlock(16), GatedBlock(16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    viz.attach(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"gated-residual net attached ({n_params/1e3:.1f}k params). Open the GUI "
          f"and set Ops -> 'structural' to see ◇ σ t ⊙ ⊕.")

    step = 0
    try:
        while True:
            x = torch.randn(32, 16)
            target = torch.zeros(32, 16)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}  loss {loss.item():.4f}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()

"""Visualize Bi-Mamba+ (Bidirectional Mamba for Time Series Forecasting).

This runs the *actual* model code from the official repo
(https://github.com/Leopold2333/Bi-Mamba4TS) — its PatchEmbedding, bidirectional
Encoder/EncoderLayer, Conv1d FFN, Flatten head and RevIN — and only swaps the
CUDA-only ``mamba_plus.Mamba`` (whose selective-scan + Mamba+ forget gate live in
a compiled kernel) for a faithful pure-PyTorch core so it runs on CPU.

Setup:
    git clone --depth 1 https://github.com/Leopold2333/Bi-Mamba4TS /tmp/Bi-Mamba4TS
    uv pip install einops

Then run the GUI (``python -m datavis.gui.app``) and this script. Set **Ops**
to *structural* to unroll the selective-scan recurrence as ⊕ ⊙ ◇ icon nodes.
Runs on CUDA automatically when a GPU is available (the tracer moves captured
tensors to CPU for serialization).
"""
from __future__ import annotations

import math
import sys
import time
import types
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import datavis.tracer as viz

REPO = "/tmp/Bi-Mamba4TS"


class MambaPlus(nn.Module):
    """Pure-PyTorch Bi-Mamba+ core (CPU).

    Same components as the repo's CUDA Mamba (in_proj, causal conv1d, x_proj for
    dt/B/C, dt_proj, A from A_log, out_proj), with a sequential selective scan
    and the Mamba+ forget gate (``D = 0``: out = silu(z)*y + (1-silu(z))*x),
    which preserves historical information.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 conv_bias=True, bias=False, **_ignored):
        super().__init__()
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):                                   # x: (B, L, D)
        L = x.shape[1]
        x_inner, z = self.in_proj(x).chunk(2, dim=-1)       # (B, L, d_inner)
        xc = self.conv1d(x_inner.transpose(1, 2))[..., :L].transpose(1, 2)
        xc = self.act(xc)                                   # (B, L, d_inner)

        x_dbl = self.x_proj(xc)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                   # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())                  # (d_inner, d_state)

        y = self._selective_scan(xc, dt, A, B, C)           # (B, L, d_inner)
        gate = self.act(z)                                  # silu(z)
        y = y * gate + (1.0 - gate) * xc                    # Mamba+ forget gate
        return self.out_proj(y)

    def _selective_scan(self, u, dt, A, B, C):
        batch, L, d_inner = u.shape
        dA = torch.exp(dt.unsqueeze(-1) * A)                       # (B, L, d_inner, N)
        dBu = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, d_inner, N)
        h = u.new_zeros(batch, d_inner, A.shape[1])
        ys = []
        for t in range(L):                                  # the SSM recurrence
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))   # (B, d_inner)
        return torch.stack(ys, dim=1)                       # (B, L, d_inner)


def _install_repo() -> None:
    stub = types.ModuleType("mamba_plus")
    stub.Mamba = MambaPlus
    sys.modules["mamba_plus"] = stub
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


def build_config() -> SimpleNamespace:
    return SimpleNamespace(
        seq_len=96, pred_len=24, enc_in=7,
        revin=1, embed_type=0, SRA=0, ch_ind=1, threshold=0.6,
        patch_len=16, stride=8, padding_patch="end",
        d_model=32, dropout=0.1, pos_embed_type="sincos", pos_learnable=False,
        d_state=16, d_conv=4, e_fact=2, d_ff=64,
        activation="relu", bi_dir=1, residual=1, e_layers=1,
    )


def make_model():
    _install_repo()
    from models.BiMamba4TS import Model  # the repo's actual model
    return Model(build_config())


def _batch(cfg, bs=16):
    t = torch.linspace(0, 4 * math.pi, cfg.seq_len + cfg.pred_len)
    series = []
    for _ in range(bs):
        chans = [torch.sin(t * (1 + 0.5 * m) + torch.rand(1) * 6.28)
                 + 0.1 * torch.randn(cfg.seq_len + cfg.pred_len)
                 for m in range(cfg.enc_in)]
        series.append(torch.stack(chans, dim=-1))           # (T, M)
    data = torch.stack(series)                              # (bs, T, M)
    return data[:, :cfg.seq_len], data[:, cfg.seq_len:]


def main() -> None:
    cfg = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    viz.attach(model)
    n_params = sum(p.numel() for p in model.parameters())
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Bi-Mamba+ attached ({n_params/1e3:.0f}k params) on {dev_name}. Open "
          f"the GUI; set Ops -> 'structural' to unroll the selective scan.")

    step = 0
    try:
        while True:
            x, y = _batch(cfg)
            # --- forward-external preprocessing: the full data flow is recorded ---
            # The module op-mode only sees ops *inside* forward; a region also
            # intercepts these standardization ops. Passing the raw batch captures
            # it at the stage, so EVERY stage is subscribable — the data's whole
            # evolution: raw (B,L,M) series -> mean/std/sub/div -> the standardized
            # tensor (auto-captured at the `model.input` source node) -> the model.
            with viz.region("standardize", x):             # x: (B, L, M) raw series
                mean = x.mean(dim=1, keepdim=True)
                std = x.std(dim=1, keepdim=True) + 1e-5
                x = (x - mean) / std                       # per-series standardization
            # ---------------------------------------------------------------------
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out, _ = model(x, None, None, None)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            step += 1
            if step % 20 == 0:
                print(f"step {step}  mse {loss.item():.4f}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()

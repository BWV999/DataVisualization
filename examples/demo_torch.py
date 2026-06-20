"""Zero-intrusion auto-trace of a small CNN training loop.

Run the GUI first (``python -m datavis.gui.app``), then this script. The
execution tree appears in the GUI after the first forward; check modules to see
their activations update live as training proceeds. Use ``--bench`` to print a
throughput comparison (with vs without trace) — capture is on-demand, so an
untouched GUI adds negligible overhead.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

import datavis.tracer as viz
from datavis.common import protocol


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _train_steps(model, opt, loss_fn, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return n / (time.perf_counter() - t0)


def _overhead(base: float, val: float) -> str:
    return f"{val:6.1f} steps/s  ({base / val - 1:+.1%} time/step)"


def bench(n: int = 300) -> None:
    torch.manual_seed(0)
    model = SmallCNN()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    _train_steps(model, opt, loss_fn, 50)  # warmup (torch lazy init)
    base = _train_steps(model, opt, loss_fn, n)
    print(f"no trace:                    {base:6.1f} steps/s")

    tracer = viz.attach(model)
    print(f"module-level trace (0 subs): {_overhead(base, _train_steps(model, opt, loss_fn, n))}")

    tracer.server.set_oplevel(protocol.OPLEVEL_ALL)
    print(f"op-level trace (0 subs):     {_overhead(base, _train_steps(model, opt, loss_fn, n))}")
    print("(sandbox CPU is noisy; module-level does no .cpu() copy with 0 subscribers,"
          " op-level adds a per-op Python callback only while enabled.)")


def serve() -> None:
    torch.manual_seed(0)
    model = SmallCNN()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    viz.attach(model)
    print("attached; training. Open the GUI and check modules to visualize.")

    step = 0
    try:
        while True:
            x = torch.randn(32, 1, 28, 28)
            y = torch.randint(0, 10, (32,))
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}  loss {loss.item():.3f}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nstopping")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true",
                        help="print throughput with/without trace and exit")
    args = parser.parse_args()
    bench() if args.bench else serve()


if __name__ == "__main__":
    main()

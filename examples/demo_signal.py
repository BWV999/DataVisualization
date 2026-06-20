"""Synthetic multi-channel signal pipeline producer (no torch).

Defines a small ordered pipeline (raw -> normalized -> filtered -> fft, plus a
spectrogram heatmap), advertises it as a structure, and streams a **finite**
sequence of evolving frames. When the sequence ends the producer does not exit:
it holds the final frame so the visualization is *retained* (a frozen final
state), and a panel opened after completion still shows it. Run the GUI first
(``python -m datavis.gui.app``), then this script.
"""
from __future__ import annotations

import time

import numpy as np

from datavis.tracer import TracerServer

T = 512    # samples per channel window
STEPS = 240  # the sequence is finite: it evolves for STEPS frames, then holds


def build_nodes() -> list[dict]:
    return [
        {"id": "raw",  "name": "raw",         "path": "signal.raw",
         "parent": None,  "rank": 2, "shape": [4, T],      "depth": 0},
        {"id": "norm", "name": "normalized",  "path": "signal.normalized",
         "parent": "raw", "rank": 2, "shape": [4, T],      "depth": 1},
        {"id": "filt", "name": "filtered",    "path": "signal.filtered",
         "parent": "norm", "rank": 2, "shape": [4, T],     "depth": 2},
        {"id": "fft",  "name": "fft_mag",     "path": "signal.fft_mag",
         "parent": "filt", "rank": 2, "shape": [4, T // 2], "depth": 3},
        {"id": "spec", "name": "spectrogram", "path": "signal.spectrogram",
         "parent": "filt", "rank": 2, "shape": [64, 128],  "depth": 3},
    ]


def step_data(phase: float) -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 4.0 * np.pi, T)
    raw = np.stack([
        np.sin(t * (k + 1) + phase * (k + 1)) + 0.2 * np.random.randn(T)
        for k in range(4)
    ]).astype(np.float32)
    norm = (raw - raw.mean(1, keepdims=True)) / (raw.std(1, keepdims=True) + 1e-6)
    kernel = np.ones(8, dtype=np.float32) / 8.0
    filt = np.stack([np.convolve(norm[k], kernel, mode="same") for k in range(4)]
                    ).astype(np.float32)
    fft = np.abs(np.fft.rfft(filt, axis=1))[:, : T // 2].astype(np.float32)

    fy = np.arange(64)[:, None]
    fx = np.arange(128)[None, :]
    spec = (np.sin(fy * 0.2 + fx * 0.1 + phase) * np.exp(-fy / 40.0)).astype(np.float32)

    return {"raw": raw, "norm": norm, "filt": filt, "fft": fft, "spec": spec}


def main() -> None:
    server = TracerServer()
    server.set_structure(build_nodes())
    print("tracer serving on tcp://127.0.0.1:5750 (ctrl) / :5751 (data)")
    print("nodes:", [n["id"] for n in build_nodes()])

    data: dict[str, np.ndarray] = {}
    try:
        # stream the finite sequence: data evolves frame by frame, then ends
        for step in range(STEPS):
            data = step_data(step * 0.05)
            for node_id, arr in data.items():
                server.maybe_send(node_id, step, (lambda a=arr: a))
            time.sleep(0.02)

        # sequence complete -> hold the final state so the view is retained
        # (re-emitting the same last frame, so a late-opened panel still shows it)
        print(f"\nsequence complete ({STEPS} steps) — holding final state; Ctrl-C to exit")
        while data:
            for node_id, arr in data.items():
                server.maybe_send(node_id, STEPS - 1, (lambda a=arr: a))
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.close()


if __name__ == "__main__":
    main()

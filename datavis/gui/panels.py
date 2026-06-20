"""Render panels. A panel consumes ``(payload, stats, step)`` frames.

CurvePanel  -> y-t overlaid channels (payload kind "curve"), with a "sticky"
               y-range that expands instantly but contracts slowly to avoid
               per-frame axis jitter.
HeatmapPanel-> a clean image tile (payload kind "heatmap"): a bare ViewBox +
               ImageItem with a perceptual colormap, no histogram/ROI/menu chrome.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from datavis.common import protocol


def _get_colormap():
    for name in ("viridis", "CET-L9", "CET-L4", "inferno"):
        try:
            cmap = pg.colormap.get(name)
        except Exception:
            cmap = None
        if cmap is not None:
            return cmap
    return None


_COLORMAP = _get_colormap()


def _proj_note(payload: dict) -> str:
    """A short ' · 8×96×16 slice[0]→2D' tag when the payload is a projection of
    a higher-rank tensor, so the view never silently passes off a lossy
    reduction as the whole thing."""
    src = payload.get("src_shape")
    if not src:
        return ""
    how = payload.get("reduced", "slice")
    if how == protocol.REDUCE_SLICE:
        how = f"slice{payload.get('reduced_idx', [0])}"
    return f"  ·  {'×'.join(map(str, src))} {how}→2D"


class CurvePanel(pg.PlotWidget):
    _CONTRACT = 0.05  # how fast the y-range shrinks toward the current data

    def __init__(self, title: str) -> None:
        super().__init__(title=title)
        self.setLabel("bottom", "t")
        self.setLabel("left", "y")
        self.addLegend(offset=(10, 10))
        self.disableAutoRange()
        self._title = title
        self._curves: list[pg.PlotDataItem] = []
        self._ylo: float | None = None
        self._yhi: float | None = None

    def update_frame(self, payload: dict, stats: dict, step: int) -> None:
        if payload.get("kind") != protocol.KIND_CURVE:
            return
        y = np.asarray(payload["y"])
        if y.ndim == 1:
            y = y.reshape(1, -1)
        channels, n = y.shape
        x = np.arange(n)
        while len(self._curves) < channels:
            i = len(self._curves)
            pen = pg.mkPen(pg.intColor(i, hues=max(channels, 6)), width=1)
            self._curves.append(self.plot(pen=pen, name=f"ch{i}"))
        for i in range(channels):
            self._curves[i].setData(x, y[i])

        self._update_yrange(float(np.min(y)), float(np.max(y)))
        self.setXRange(0, max(n - 1, 1), padding=0.0)
        self.setTitle(
            f"{self._title}  ·  step {step}  μ={stats.get('mean', 0):.3g} "
            f"σ={stats.get('std', 0):.3g}{_proj_note(payload)}"
        )

    def _update_yrange(self, ymin: float, ymax: float) -> None:
        if self._ylo is None:
            self._ylo, self._yhi = ymin, ymax
        else:
            self._ylo = min(self._ylo, ymin)
            self._yhi = max(self._yhi, ymax)
            if ymin > self._ylo:
                self._ylo += (ymin - self._ylo) * self._CONTRACT
            if ymax < self._yhi:
                self._yhi += (ymax - self._yhi) * self._CONTRACT
        pad = (self._yhi - self._ylo) * 0.05 + 1e-6
        self.setYRange(self._ylo - pad, self._yhi + pad, padding=0.0)


class HeatmapPanel(pg.GraphicsLayoutWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._label = self.addLabel(title, row=0, col=0)
        self._view = self.addViewBox(row=1, col=0)
        self._view.setAspectLocked(False)
        self._view.invertY(True)
        self._view.setMouseEnabled(x=False, y=False)
        self._img = pg.ImageItem()
        if _COLORMAP is not None:
            self._img.setColorMap(_COLORMAP)
        self._view.addItem(self._img)
        self._title = title

    def update_frame(self, payload: dict, stats: dict, step: int) -> None:
        if payload.get("kind") != protocol.KIND_HEATMAP:
            return
        z = np.asarray(payload["z"])
        # pyqtgraph images are column-major (x, y); transpose so rows -> y axis.
        self._img.setImage(z.T, autoLevels=False)
        lo, hi = float(z.min()), float(z.max())
        if hi <= lo:
            hi = lo + 1e-6
        self._img.setLevels([lo, hi])
        self._view.autoRange(padding=0.0)
        self._label.setText(f"{self._title}  ·  step {step}{_proj_note(payload)}")


def make_panel(node: dict):
    """Pick a panel type from a structure node's shape/rank."""
    rank = node.get("rank", 1)
    shape = node.get("shape", [])
    if rank >= 3 or (rank == 2 and shape and shape[0] > protocol.MAX_CURVE_CHANNELS):
        return HeatmapPanel(node["path"])
    return CurvePanel(node["path"])

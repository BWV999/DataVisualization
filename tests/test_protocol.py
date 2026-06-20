import numpy as np

from datavis.common import protocol


def test_ndarray_roundtrip():
    for arr in [
        np.random.randn(512).astype(np.float32),
        np.random.randn(4, 256).astype(np.float32),
        np.random.randn(8, 16, 16).astype(np.float32),
        np.arange(10, dtype=np.int64),
    ]:
        out = protocol.decode(protocol.encode({"type": "frame", "a": arr}))
        assert out["a"].dtype == arr.dtype
        assert out["a"].shape == arr.shape
        assert np.array_equal(out["a"], arr)


def test_message_roundtrip_preserves_scalars():
    msg = {"type": protocol.MSG_FRAME, "node_id": "raw", "step": 7,
           "stats": {"mean": 1.5, "norm": 2.0}, "parent": None}
    out = protocol.decode(protocol.encode(msg))
    assert out["node_id"] == "raw"
    assert out["step"] == 7
    assert out["stats"]["mean"] == 1.5
    assert out["parent"] is None


def test_auto_payload_curve_decimates():
    arr = np.random.randn(4, 5000).astype(np.float32)
    payload = protocol.auto_payload(arr, max_points=1000)
    assert payload["kind"] == protocol.KIND_CURVE
    assert payload["y"].shape[0] == 4
    assert payload["y"].shape[1] <= 1000


def test_auto_payload_1d_is_single_channel_curve():
    payload = protocol.auto_payload(np.random.randn(300).astype(np.float32))
    assert payload["kind"] == protocol.KIND_CURVE
    assert payload["y"].shape[0] == 1


def test_auto_payload_heatmap_downscales():
    arr = np.random.randn(64, 128).astype(np.float32)
    payload = protocol.auto_payload(arr, max_side=32)
    assert payload["kind"] == protocol.KIND_HEATMAP
    assert max(payload["z"].shape) <= 32


def test_auto_payload_highdim_reduces_to_heatmap():
    arr = np.random.randn(8, 3, 20, 20).astype(np.float32)
    payload = protocol.auto_payload(arr, max_side=64)
    assert payload["kind"] == protocol.KIND_HEATMAP
    assert payload["z"].ndim == 2


def test_nd_default_slice_keeps_a_real_sample_not_a_blend():
    # leading axes are sliced (a real sample), not averaged: the shown 2D must
    # equal arr[0, 0], preserving temporal/feature structure (no cross-blend).
    arr = np.random.randn(8, 3, 20, 16).astype(np.float32)
    payload = protocol.auto_payload(arr, max_side=64)
    assert np.allclose(payload["z"], arr[0, 0])
    # and it self-documents the projection so the GUI can label it
    assert payload["src_shape"] == [8, 3, 20, 16]
    assert payload["reduced"] == protocol.REDUCE_SLICE
    assert payload["reduced_idx"] == [0, 0]


def test_nd_mean_reduction_is_opt_in():
    arr = np.random.randn(4, 20, 16).astype(np.float32)
    payload = protocol.auto_payload(arr, reduce=protocol.REDUCE_MEAN, max_side=64)
    assert np.allclose(payload["z"], arr.mean(axis=0))
    assert payload["reduced"] == protocol.REDUCE_MEAN


def test_nd_payload_stays_heatmap_consistent_with_panel():
    # even when the reduced 2D is small (<=16 leading), an originally-ND tensor
    # must yield a heatmap, matching make_panel (rank>=3 -> HeatmapPanel).
    from datavis.gui.panels import make_panel, HeatmapPanel
    arr = np.random.randn(8, 3, 16).astype(np.float32)   # slice -> (3,16), small
    payload = protocol.auto_payload(arr)
    assert payload["kind"] == protocol.KIND_HEATMAP
    node = {"rank": 3, "shape": [8, 3, 16], "path": "x"}
    assert isinstance(make_panel(node), HeatmapPanel)


def test_compute_stats():
    s = protocol.compute_stats(np.ones((3, 4), dtype=np.float32))
    assert abs(s["mean"] - 1.0) < 1e-6
    assert abs(s["std"]) < 1e-6
    assert abs(s["norm"] - np.sqrt(12)) < 1e-5


def test_compute_stats_empty():
    s = protocol.compute_stats(np.zeros((0,), dtype=np.float32))
    assert s == {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "norm": 0.0}

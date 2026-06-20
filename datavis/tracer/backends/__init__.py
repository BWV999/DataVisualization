"""Pluggable, per-library dataflow interceptors.

Each backend adapts one library's *uniform interface* into the same
framework-agnostic ``ExecutionTree`` + ``TracerServer`` the torch tracer writes
to, so a single dataflow DAG can span numpy preprocessing -> sklearn pipeline ->
torch model. The torch tracer (``datavis.tracer.hooks``) is one such interceptor
(via ``__torch_function__``); these are the non-torch ones.

  - ``numpy_trace`` — ``TracedArray`` over NEP-13 (``__array_ufunc__``) and
    NEP-18 (``__array_function__``): wrap once at the source with ``viz.track``,
    and the whole numpy preprocessing chain is recorded automatically.
"""

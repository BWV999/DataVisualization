"""DataVisualization: a dataflow visualization / visual debugger for PyTorch.

Two-process design:
- ``datavis.tracer`` runs inside the training process (zero-intrusion auto-trace).
- ``datavis.gui`` runs as a separate PyQtGraph application.
They communicate over ZeroMQ using the wire format in ``datavis.common.protocol``.
"""

__version__ = "0.1.0"

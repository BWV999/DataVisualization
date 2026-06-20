"""ExecutionTree: accumulates traced nodes in first-seen execution order.

Built incrementally from forward-hook callbacks. ``order`` is assigned the first
time a node is observed (its execution rank in the first forward); ``shape`` is
refreshed each pass so the structure reflects the latest tensor sizes.
"""
from __future__ import annotations


class ExecutionTree:
    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._counter = 0
        # dataflow edges: (producer_id, consumer_id) -> first-seen order
        self._edges: dict[tuple[str, str], int] = {}

    def observe(
        self,
        node_id: str,
        *,
        name: str,
        path: str,
        parent: str | None,
        depth: int,
        shape: tuple[int, ...],
        role: str | None = None,
        count: int = 1,
    ) -> None:
        """``count`` is the node's multiplicity: how many times this op fired in
        the latest forward. >1 marks a rolled-up loop body (a recurrence), so an
        SSM scan's per-step ops stay one node each instead of unrolling."""
        node = self._nodes.get(node_id)
        if node is None:
            self._nodes[node_id] = {
                "id": node_id,
                "name": name,
                "path": path,
                "parent": parent,
                "depth": depth,
                "rank": len(shape),
                "shape": list(shape),
                "order": self._counter,
                "role": role,
                "count": count,
            }
            self._counter += 1
        else:
            node["rank"] = len(shape)
            node["shape"] = list(shape)
            node["count"] = count

    def link(self, src: str, dst: str) -> None:
        """Record a dataflow edge producer ``src`` -> consumer ``dst``.

        Deduped and order-stable: the first time a pair is seen it is kept; the
        edge set only ever grows, mirroring how ``observe`` accumulates nodes.
        """
        key = (src, dst)
        if key not in self._edges:
            self._edges[key] = len(self._edges)

    def nodes(self) -> list[dict]:
        return list(self._nodes.values())

    def edges(self) -> list[list[str]]:
        return [[src, dst] for (src, dst) in self._edges]

    def n_edges(self) -> int:
        return len(self._edges)

    def __len__(self) -> int:
        return len(self._nodes)

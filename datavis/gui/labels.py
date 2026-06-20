"""Human-readable node labels shared by the tree and the dataflow graph.

A node's ``path`` tail is the most specific name, but for ``Sequential`` /
``ModuleList`` children that tail is a bare index (``0``, ``1``, …) which says
nothing about what the node *is*. In that case we fall back to the module's
class plus its index (``Block[0]``, ``Linear[2]``). Real attribute names
(``fc``, ``proj``, ``norm``) and op names (``mul#0``, ``chunk#0``) are kept.
"""
from __future__ import annotations


def _tail(node: dict) -> str:
    path = node.get("path") or node.get("id", "")
    return path.split(".")[-1].split("/")[-1]


def display_label(node: dict) -> str:
    """Compact, self-explanatory label (for graph nodes)."""
    tail = _tail(node)
    name = node.get("name", "")
    if "#" in tail:                 # op leaf, e.g. 'mul#0' — already descriptive
        count = node.get("count", 1)
        # a rolled-up loop body: show its multiplicity (executed N times)
        return f"{tail} ×{count}" if isinstance(count, int) and count > 1 else tail
    if tail.isdigit() and name:     # Sequential / ModuleList index -> class[index]
        return f"{name}[{tail}]"
    return tail or name


def tree_label(node: dict) -> str:
    """Label for the tree; appends the class when the name alone hides it."""
    tail = _tail(node)
    name = node.get("name", "")
    label = display_label(node)
    # an attribute-named module (fc/proj/...) doesn't reveal its type — show it
    if "#" not in tail and not tail.isdigit() and name and name != label:
        return f"{label}  ({name})"
    return label

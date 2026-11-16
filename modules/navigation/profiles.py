"""Route profiles for the simulated indoor map.

Profiles change which edges are usable and how they are weighted. They are
preferences on a FICTIONAL map, not real-world accessibility guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.navigation.graph_data import IndoorGraph, edge_key


@dataclass(frozen=True)
class RouteProfile:
    key: str
    label: str
    description: str


PROFILES = {
    "fastest": RouteProfile(
        "fastest",
        "Fastest",
        "Shortest walking distance; stairs and crowded corridors are allowed.",
    ),
    "step_free": RouteProfile(
        "step_free",
        "Step-free",
        "Excludes stairs and requires a working lift for floor changes.",
    ),
    "quiet": RouteProfile(
        "quiet",
        "Quiet",
        "Penalises crowded segments; stairs are still allowed.",
    ),
}

PROFILE_ORDER = ("fastest", "step_free", "quiet")

QUIET_CROWD_WEIGHT = 2.5


def normalize_blocked_edges(
    blocked_edges: object = (),
) -> frozenset[tuple[str, str]]:
    """Canonicalise an iterable of (a, b) pairs for stable comparisons."""
    pairs: set[tuple[str, str]] = set()
    for item in blocked_edges or ():
        try:
            a, b = item  # type: ignore[misc]
        except (TypeError, ValueError):
            continue
        if isinstance(a, str) and isinstance(b, str) and a != b:
            pairs.add(edge_key(a, b))
    return frozenset(pairs)


def make_edge_rules(
    graph: IndoorGraph,
    profile_key: str,
    blocked_edges: object = (),
    blocked_nodes: object = (),
):
    """Build (allowed, cost) callables for the search algorithms."""
    if profile_key not in PROFILES:
        raise ValueError(f"Unknown route profile '{profile_key}'.")
    blocked = normalize_blocked_edges(blocked_edges)
    blocked_node_set = set(blocked_nodes or ())

    def allowed(a: str, b: str) -> bool:
        if a in blocked_node_set or b in blocked_node_set:
            return False
        if edge_key(a, b) in blocked:
            return False
        attrs = graph.edge_attrs(a, b) or {}
        return not (profile_key == "step_free" and attrs.get("type") == "stairs")

    def cost(a: str, b: str, weight: float) -> float:
        attrs = graph.edge_attrs(a, b) or {}
        if profile_key == "quiet":
            crowd = float(attrs.get("crowd", 0.0))
            return weight * (1.0 + QUIET_CROWD_WEIGHT * crowd)
        return weight

    return allowed, cost


def blocked_summary(
    graph: IndoorGraph,
    blocked_edges: object = (),
    blocked_nodes: object = (),
) -> list[str]:
    """Human-readable labels for active simulated closures."""
    labels: list[str] = []
    for node in sorted(set(blocked_nodes or ())):
        if node in graph.nodes:
            labels.append(f"{graph.display_name(node)} closed")
    for a, b in sorted(normalize_blocked_edges(blocked_edges)):
        if a in graph.nodes and b in graph.nodes:
            labels.append(
                f"{graph.display_name(a)} ↔ {graph.display_name(b)} closed"
            )
    return labels

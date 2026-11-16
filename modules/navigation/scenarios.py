"""Predefined simulated closure scenarios for the navigation demo.

Each scenario blocks parts of the FICTIONAL graph so visitors can see the
planner adapt. Nothing here reflects real-time building conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modules.navigation.graph_data import IndoorGraph


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    description: str
    blocked_edges: frozenset[tuple[str, str]] = frozenset()
    blocked_nodes: frozenset[str] = field(default_factory=frozenset)


def _edges_of_type(graph: IndoorGraph, edge_type: str) -> frozenset[tuple[str, str]]:
    return frozenset(
        (a, b)
        for a, b, attrs in graph.edge_items()
        if attrs.get("type") == edge_type
    )


def all_scenarios(graph: IndoorGraph) -> dict[str, Scenario]:
    """Scenario key -> Scenario, always beginning with 'none'."""
    scenarios = {
        "none": Scenario(
            "none",
            "No closures",
            "The full fictional map is open.",
        ),
        "lift_out": Scenario(
            "lift_out",
            "Lift out of service",
            "Both lift stops are closed; step-free routes that need a "
            "floor change become unavailable.",
            blocked_edges=_edges_of_type(graph, "lift"),
        ),
        "stairs_closed": Scenario(
            "stairs_closed",
            "All stairs closed",
            "Every stair edge is closed; only lift-based floor changes work.",
            blocked_edges=_edges_of_type(graph, "stairs"),
        ),
    }

    if "corridor_c" in graph.nodes:
        scenarios["corridor_c_closed"] = Scenario(
            "corridor_c_closed",
            "Corridor C closed",
            "A whole corridor node is removed from the graph.",
            blocked_nodes=frozenset({"corridor_c"}),
        )
    if "main_hall" in graph.nodes:
        scenarios["main_hall_closed"] = Scenario(
            "main_hall_closed",
            "Main Hall closed",
            "The busy central hall is removed; routes must detour.",
            blocked_nodes=frozenset({"main_hall"}),
        )
    return scenarios

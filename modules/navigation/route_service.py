"""Route planning service: clean interface between the UI and algorithms.

Runs BFS and A* on the fictional sample graph, applies accessibility
profiles and simulated closures, and converts every outcome into an
explicit RouteResult the UI can render without special cases.

SIMULATION ONLY: results describe the fictional sample map. They say
nothing about any real building, stairs, doors, lifts, obstacles, or
accessible routes.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from modules.navigation import astar, bfs, instructions, profiles
from modules.navigation.graph_data import (
    DEFAULT_MAP_PATH,
    IndoorGraph,
    euclidean_heuristic,
    load_graph,
)

_GRAPH_CACHE: dict[str, IndoorGraph] = {}


def get_graph(path: str = DEFAULT_MAP_PATH) -> IndoorGraph:
    """Load and cache the sample map (parsed once per process)."""
    if path not in _GRAPH_CACHE:
        _GRAPH_CACHE[path] = load_graph(path)
    return _GRAPH_CACHE[path]


@dataclass
class RouteResult:
    algorithm: str = ""             # "BFS" | "A*"
    start: str = ""
    goal: str = ""
    path: list[str] = field(default_factory=list)   # empty when unreachable/invalid
    path_display: list[str] = field(default_factory=list)  # human names
    cost: float = 0.0               # metres (weighted) along the path
    hops: int = 0
    explored_count: int = 0
    explored_names: list[str] = field(default_factory=list)
    explored_ids: list[str] = field(default_factory=list)
    time_ms: int = 0
    error: str | None = None        # invalid node etc.
    unreachable: bool = False       # no route under the active constraints

    # Profile-aware fields (defaults keep the simple BFS/A* API working).
    profile_key: str = "fastest"
    profile_label: str = "Fastest"
    profile_cost: float = 0.0
    stairs_count: int = 0
    lift_count: int = 0
    min_width_m: float | None = None
    eta_s: int = 0
    detour_m: float = 0.0
    blocked_edges: list[tuple[str, str]] = field(default_factory=list)
    blocked_nodes: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    explanation: str = ""
    unreachable_reason: str | None = None

    @property
    def found(self) -> bool:
        return not self.error and not self.unreachable and bool(self.path)


def _edge_weight(graph: IndoorGraph, a: str, b: str) -> float:
    attrs = graph.edge_attrs(a, b) or {}
    weight = float(attrs.get("weight", 0.0))
    if not weight:
        for neighbour, neighbour_weight in graph.neighbours(a):
            if neighbour == b:
                return float(neighbour_weight)
    return weight


def _path_stats(
    graph: IndoorGraph, path: list[str], speed_mps: float
) -> tuple[int, int, float | None, int]:
    stairs = 0
    lifts = 0
    widths: list[float] = []
    for a, b in itertools.pairwise(path):
        attrs = graph.edge_attrs(a, b) or {}
        edge_type = attrs.get("type", "corridor")
        if edge_type == "stairs":
            stairs += 1
        elif edge_type == "lift":
            lifts += 1
        width = attrs.get("width_m")
        if width is not None:
            widths.append(float(width))
    eta = instructions.total_eta_seconds(graph, path, speed_mps)
    return stairs, lifts, (min(widths) if widths else None), eta


def _core_route(
    graph: IndoorGraph,
    start: str,
    goal: str,
    *,
    profile_key: str,
    blocked_edges: frozenset[tuple[str, str]],
    blocked_nodes: set[str],
    algorithm: str,
    speed_mps: float,
) -> RouteResult:
    profile = profiles.PROFILES.get(profile_key)
    result = RouteResult(
        algorithm=algorithm,
        start=start,
        goal=goal,
        profile_key=profile_key,
        profile_label=profile.label if profile else profile_key,
        blocked_edges=sorted(blocked_edges),
        blocked_nodes=sorted(blocked_nodes),
    )

    if algorithm not in {"BFS", "A*"}:
        result.error = f"Unknown algorithm '{algorithm}'."
        return result
    if start not in graph.nodes or goal not in graph.nodes:
        result.error = (
            "Unknown location selected. Please pick a start and destination "
            "from the lists."
        )
        return result
    if profile is None:
        result.error = f"Unknown route profile '{profile_key}'."
        return result
    if algorithm == "BFS" and profile_key != "fastest":
        result.error = (
            "BFS ignores edge weights, so it cannot apply this route profile. "
            "Use A* for step-free or quiet routing."
        )
        return result
    if start in blocked_nodes:
        result.error = (
            f"Start location ({graph.display_name(start)}) is closed in the "
            "selected scenario."
        )
        return result
    if goal in blocked_nodes:
        result.error = (
            f"Destination ({graph.display_name(goal)}) is closed in the "
            "selected scenario."
        )
        return result

    allowed, cost = profiles.make_edge_rules(
        graph, profile_key, blocked_edges, blocked_nodes
    )

    try:
        started = time.perf_counter()
        if algorithm == "BFS":
            search = bfs.bfs_shortest_path(graph, start, goal, allowed=allowed)
        else:
            search = astar.astar_shortest_path(
                graph,
                start,
                goal,
                euclidean_heuristic(graph, goal),
                cost=cost,
                allowed=allowed,
            )
        result.time_ms = int((time.perf_counter() - started) * 1000)
    except Exception as exc:  # algorithm failure on odd graphs
        result.error = (
            f"Route computation failed ({exc.__class__.__name__}: {exc})."
        )
        return result

    result.explored_count = len(search.explored_order)
    result.explored_ids = list(search.explored_order)
    result.explored_names = [
        graph.display_name(node) for node in search.explored_order
    ]

    if search.path is None:
        result.unreachable = True
        result.profile_cost = 0.0
        result.unreachable_reason = _unreachable_reason(
            graph, start, goal, profile_key, blocked_edges, blocked_nodes
        )
        result.explanation = result.unreachable_reason
        return result

    result.path = search.path
    result.path_display = [graph.display_name(node) for node in search.path]
    result.cost = search.cost
    result.profile_cost = float(search.search_cost)
    result.hops = search.hops
    stairs, lifts, min_width, eta = _path_stats(graph, search.path, speed_mps)
    result.stairs_count = stairs
    result.lift_count = lifts
    result.min_width_m = min_width
    result.eta_s = eta
    result.constraints = _constraints(
        profile_key, blocked_edges, blocked_nodes
    )
    return result


def _unreachable_reason(
    graph: IndoorGraph,
    start: str,
    goal: str,
    profile_key: str,
    blocked_edges: frozenset[tuple[str, str]],
    blocked_nodes: set[str],
) -> str:
    if profile_key == "step_free":
        lift_blocked = any(
            attrs.get("type") == "lift"
            and (
                (a, b) in blocked_edges
                or a in blocked_nodes
                or b in blocked_nodes
            )
            for a, b, attrs in graph.edge_items()
        )
        if lift_blocked and graph.floor(start) != graph.floor(goal):
            return (
                "Step-free route unavailable: the lift is out of service and "
                "this profile excludes stairs. Try the Fastest or Quiet "
                "profile, or reopen the lift."
            )
        return (
            "No step-free route exists on this fictional map with the "
            "current closures. The step-free profile excludes every stairs "
            "edge."
        )
    if blocked_edges or blocked_nodes:
        return (
            "No route exists with the current simulated closures. Reopen a "
            "segment or destination and try again."
        )
    return (
        "No route exists between these locations on the fictional sample "
        "map."
    )


def _constraints(
    profile_key: str,
    blocked_edges: frozenset[tuple[str, str]],
    blocked_nodes: set[str],
) -> list[str]:
    constraints: list[str] = []
    if profile_key == "step_free":
        constraints.append("Stairs excluded")
    elif profile_key == "quiet":
        constraints.append("Crowded segments penalised")
    closures = len(blocked_edges) + len(blocked_nodes)
    if closures:
        constraints.append(
            f"{closures} simulated closure{'s' if closures != 1 else ''} avoided"
        )
    return constraints


def _explanation(
    result: RouteResult,
    baseline: RouteResult | None,
    blocked_edges: frozenset[tuple[str, str]],
    blocked_nodes: set[str],
) -> str:
    parts: list[str] = []
    closures = len(blocked_edges) + len(blocked_nodes)
    if closures:
        parts.append(
            f"Avoids {closures} simulated closure{'s' if closures != 1 else ''}."
        )
    if baseline is not None and baseline.found:
        detour = result.cost - baseline.cost
        if result.profile_key == "step_free":
            if detour > 0.5:
                parts.append(
                    f"The step-free route is {detour:.0f} m longer than the "
                    "fastest route because stairs are excluded."
                )
            else:
                parts.append(
                    "The step-free route matches the fastest distance and "
                    "avoids every stairs edge."
                )
            if result.lift_count:
                parts.append("It uses the lift for the floor change.")
        elif result.profile_key == "quiet":
            if detour > 0.5:
                parts.append(
                    f"The quiet route adds {detour:.0f} m to avoid crowded "
                    "segments."
                )
            else:
                parts.append(
                    "The quiet route matches the fastest distance under the "
                    "simulated crowding weights."
                )
    elif result.profile_key == "step_free" and result.lift_count:
        parts.append("It uses the lift for the floor change.")
    return " ".join(parts)


def plan_route(
    graph: IndoorGraph,
    start: str,
    goal: str,
    *,
    profile_key: str = "fastest",
    blocked_edges: object = (),
    blocked_nodes: object = (),
    algorithm: str = "A*",
    speed_mps: float = instructions.DEFAULT_SPEED_MPS,
    include_baseline: bool = True,
) -> RouteResult:
    """Plan one route with a profile and optional simulated closures."""
    normalized_edges = profiles.normalize_blocked_edges(blocked_edges)
    blocked_node_set = set(blocked_nodes or ())

    result = _core_route(
        graph,
        start,
        goal,
        profile_key=profile_key,
        blocked_edges=normalized_edges,
        blocked_nodes=blocked_node_set,
        algorithm=algorithm,
        speed_mps=speed_mps,
    )

    baseline = None
    if (
        include_baseline
        and profile_key != "fastest"
        and result.found
    ):
        baseline = _core_route(
            graph,
            start,
            goal,
            profile_key="fastest",
            blocked_edges=normalized_edges,
            blocked_nodes=blocked_node_set,
            algorithm=algorithm,
            speed_mps=speed_mps,
        )
        if baseline.found:
            result.detour_m = max(0.0, result.cost - baseline.cost)

    if result.explanation:
        return result
    result.explanation = _explanation(
        result, baseline, normalized_edges, blocked_node_set
    )
    return result


def run_route(
    graph: IndoorGraph,
    start: str,
    goal: str,
    algorithm: str = "A*",
) -> RouteResult:
    """Backward-compatible fastest-profile route used by the comparison view."""
    return plan_route(
        graph,
        start,
        goal,
        profile_key="fastest",
        algorithm=algorithm,
        include_baseline=False,
    )


def compare_routes(
    graph: IndoorGraph, start: str, goal: str
) -> tuple[RouteResult, RouteResult]:
    """Run both algorithms on the fastest profile and return (bfs, astar)."""
    return (
        run_route(graph, start, goal, algorithm="BFS"),
        run_route(graph, start, goal, algorithm="A*"),
    )

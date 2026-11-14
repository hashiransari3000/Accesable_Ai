"""Route service: clean interface between the UI and the algorithms.

Runs BFS and/or A* on the sample graph and converts every outcome -
route found, unreachable, invalid node - into an explicit RouteResult
the UI can render without special cases. Also returns explored-node
counts and execution times for the BFS vs A* comparison view.

SIMULATION ONLY: results describe the fictional sample map. They say
nothing about any real building, stairs, doors, or obstacles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from modules.navigation import astar, bfs
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
    time_ms: int = 0
    error: str | None = None        # invalid node etc.
    unreachable: bool = False       # connected graph has no such route

    @property
    def found(self) -> bool:
        return not self.error and not self.unreachable and bool(self.path)


def run_route(
    graph: IndoorGraph,
    start: str,
    goal: str,
    algorithm: str = "A*",
) -> RouteResult:
    result = RouteResult(algorithm=algorithm, start=start, goal=goal)
    result.path_display = [graph.display_name(n) for n in result.path]

    if start not in graph.nodes or goal not in graph.nodes:
        result.error = (
            "Unknown location selected. Please pick a start and destination "
            "from the lists."
        )
        return result

    try:
        started = time.perf_counter()
        if algorithm == "BFS":
            search = bfs.bfs_shortest_path(graph, start, goal)
        elif algorithm == "A*":
            search = astar.astar_shortest_path(
                graph, start, goal, euclidean_heuristic(graph, goal)
            )
        else:
            result.error = f"Unknown algorithm '{algorithm}'."
            return result
        result.time_ms = int((time.perf_counter() - started) * 1000)
    except Exception as exc:  # algorithm failure on odd graphs
        result.error = f"Route computation failed ({exc.__class__.__name__}: {exc})."
        return result

    if search.path is None:
        result.unreachable = True
        result.explored_count = len(search.explored_order)
        result.explored_names = [graph.display_name(n) for n in search.explored_order]
        return result

    result.path = search.path
    result.path_display = [graph.display_name(n) for n in search.path]
    result.cost = search.cost
    result.hops = search.hops
    result.explored_count = len(search.explored_order)
    result.explored_names = [graph.display_name(n) for n in search.explored_order]
    return result


def compare_routes(
    graph: IndoorGraph, start: str, goal: str
) -> tuple[RouteResult, RouteResult]:
    """Run both algorithms and return (bfs_result, astar_result)."""
    return (
        run_route(graph, start, goal, algorithm="BFS"),
        run_route(graph, start, goal, algorithm="A*"),
    )

"""BFS - breadth-first search for the UNWEIGHTED shortest-hop route.

Teaching notes (kept explicit and readable):
  - Uses a FIFO queue (collections.deque). BFS explores nodes in rings
    of increasing distance-in-hops from the start, so the first time a
    node is reached is via a minimum number of hops.
  - A visited set prevents re-processing and infinite loops.
  - Parent pointers record where each node was first discovered from;
    the path is reconstructed backwards from the goal.
  - BFS ignores edge weights: it minimizes HOPS, not distance. A* in
    astar.py minimizes weighted cost; the demo compares both.

Result contract:
  - Known start/goal, route exists  -> path (list), explored_order, hops
  - Known start/goal, no route      -> path = None, explored_order set
  - Unknown node                    -> ValueError (converted to an
    explicit error state by route_service).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from modules.navigation.graph_data import IndoorGraph


@dataclass
class SearchResult:
    path: list[str] | None          # None when unreachable
    explored_order: list[str] = field(default_factory=list)
    hops: int = 0                   # edges in the path
    cost: float = 0.0               # actual walking distance of the path
    search_cost: float = 0.0        # cost the search minimised


def bfs_shortest_path(
    graph: IndoorGraph,
    start: str,
    goal: str,
    allowed=None,
) -> SearchResult:
    if start not in graph.nodes or goal not in graph.nodes:
        raise ValueError(f"Unknown node: {start if start not in graph.nodes else goal}")

    parent: dict[str, str | None] = {start: None}
    explored_order: list[str] = []
    queue: deque[str] = deque([start])

    while queue:
        current = queue.popleft()
        explored_order.append(current)
        if current == goal:
            path = _reconstruct(parent, goal)
            return SearchResult(
                path=path,
                explored_order=explored_order,
                hops=len(path) - 1,
                cost=_path_cost(graph, path),
                search_cost=_path_cost(graph, path),
            )
        for neighbour, _weight in graph.neighbours(current):
            if allowed is not None and not allowed(current, neighbour):
                continue
            if neighbour not in parent:
                parent[neighbour] = current
                queue.append(neighbour)

    return SearchResult(path=None, explored_order=explored_order)


def _reconstruct(parent: dict[str, str | None], goal: str) -> list[str]:
    path: list[str] = []
    node: str | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def _path_cost(graph: IndoorGraph, path: list[str]) -> float:
    """Sum of edge weights along the reconstructed path (0 if start==goal)."""
    total = 0.0
    for a, b in zip(path, path[1:]):  # explicit pairs; clear for teaching
        for neighbour, weight in graph.neighbours(a):
            if neighbour == b:
                total += weight
                break
    return total

"""A* search for the WEIGHTED shortest route on the sample indoor map.

Teaching notes (kept explicit and readable):
  - Uses a priority queue (heapq). The priority of a node is
    f(n) = g(n) + h(n), where g(n) is the cost from the start along the
    best path found so far, and h(n) is the heuristic estimate to the
    goal (straight-line distance from graph coordinates).
  - The heuristic is ADMISSIBLE for our sample map because every edge
    weight equals the straight-line distance between its endpoints, so
    h never overestimates the true remaining cost (triangle inequality).
    With an admissible heuristic and positive weights, A* returns an
    optimal route. A zero heuristic is also supported (plain Dijkstra).
  - Best-first ordering focuses exploration toward the goal, so A*
    typically expands fewer nodes than BFS/Dijkstra - the demo UI shows
    this comparison with real counts and timings.
  - Path reconstruction follows the same parent-pointer idea as BFS.
  - A 'settled' set marks nodes whose minimum cost is finalized.

Result contract matches bfs.SearchResult; unknown nodes raise ValueError
(converted to an explicit UI error by route_service).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from modules.navigation.bfs import SearchResult, _path_cost
from modules.navigation.graph_data import IndoorGraph


@dataclass
class _FrontierEntry:
    f: float
    counter: int          # tie-breaker so node ids never get compared
    node: str

    def __lt__(self, other: "_FrontierEntry") -> bool:
        return (self.f, self.counter) < (other.f, other.counter)


def astar_shortest_path(
    graph: IndoorGraph,
    start: str,
    goal: str,
    heuristic=None,
) -> SearchResult:
    if start not in graph.nodes or goal not in graph.nodes:
        raise ValueError(f"Unknown node: {start if start not in graph.nodes else goal}")

    if heuristic is None:
        heuristic = lambda _node: 0.0  # noqa: E731  (zero heuristic = Dijkstra)

    counter = 0  # insertion order; prevents heapq comparing dict values
    frontier: list[_FrontierEntry] = [_FrontierEntry(heuristic(start), counter, start)]
    g_cost: dict[str, float] = {start: 0.0}
    parent: dict[str, str | None] = {start: None}
    settled: set[str] = set()
    explored_order: list[str] = []

    while frontier:
        entry = heapq.heappop(frontier)
        current = entry.node
        if current in settled:
            continue  # stale frontier entry; a better route was found
        settled.add(current)
        explored_order.append(current)

        if current == goal:
            path = _reconstruct(parent, goal)
            return SearchResult(
                path=path,
                explored_order=explored_order,
                hops=len(path) - 1,
                cost=_path_cost(graph, path),
            )

        for neighbour, weight in graph.neighbours(current):
            if neighbour in settled:
                continue
            new_cost = g_cost[current] + weight
            if neighbour not in g_cost or new_cost < g_cost[neighbour]:
                g_cost[neighbour] = new_cost
                parent[neighbour] = current
                counter += 1
                heapq.heappush(
                    frontier,
                    _FrontierEntry(new_cost + heuristic(neighbour), counter, neighbour),
                )

    return SearchResult(path=None, explored_order=explored_order)


def _reconstruct(parent: dict[str, str | None], goal: str) -> list[str]:
    path: list[str] = []
    node: str | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path

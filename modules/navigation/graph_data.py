"""Graph data for the indoor navigation demo (fictional sample map).

Parses the plain-text map file under assets/sample_maps/ into an
IndoorGraph used by bfs.py, astar.py, and route_service.py.

Format (see the sample file for a full example):
    node <id> | <display name> | <x> | <y>
    edge <id_a> <id_b> | <weight>

Load-time validation rejects unknown endpoints in edges, non-positive
weights, and duplicate node ids, so algorithms can assume a clean graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAP_PATH = "assets/sample_maps/sample_building.txt"


@dataclass
class IndoorGraph:
    nodes: dict[str, dict] = field(default_factory=dict)      # id -> {name, x, y}
    adjacency: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def display_name(self, node_id: str) -> str:
        return self.nodes.get(node_id, {}).get("name", node_id)

    def ids(self) -> list[str]:
        return list(self.nodes.keys())

    def neighbours(self, node_id: str) -> list[tuple[str, float]]:
        return self.adjacency.get(node_id, [])

    def coords(self, node_id: str) -> tuple[float, float]:
        data = self.nodes[node_id]
        return (float(data["x"]), float(data["y"]))


def load_graph(path: str = DEFAULT_MAP_PATH) -> IndoorGraph:
    """Parse and validate the sample map file."""
    graph = IndoorGraph()
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("node "):
                _parse_node(line, line_no, graph)
            elif line.startswith("edge "):
                _parse_edge(line, line_no, graph)
            else:
                raise ValueError(
                    f"{path}:{line_no}: unrecognized line format: '{line}'"
                )
    if not graph.nodes:
        raise ValueError(f"{path}: no nodes found - map file is empty?")
    return graph


def _parse_node(line: str, line_no: int, graph: IndoorGraph) -> None:
    body = line[len("node "):]
    parts = [p.strip() for p in body.split("|")]
    if len(parts) != 4:
        raise ValueError(f"map line {line_no}: node needs 'id | name | x | y'")
    node_id, name, x, y = parts
    if node_id in graph.nodes:
        raise ValueError(f"map line {line_no}: duplicate node id '{node_id}'")
    try:
        coords = (float(x), float(y))
    except ValueError as exc:
        raise ValueError(f"map line {line_no}: x/y must be numbers") from exc
    graph.nodes[node_id] = {"name": name, "x": coords[0], "y": coords[1]}
    graph.adjacency.setdefault(node_id, [])


def _parse_edge(line: str, line_no: int, graph: IndoorGraph) -> None:
    body = line[len("edge "):]
    left, _, right = body.partition("|")
    ids = [p.strip() for p in left.split()]
    try:
        weight = float(right.strip())
    except ValueError as exc:
        raise ValueError(f"map line {line_no}: weight must be a number") from exc
    if len(ids) != 2:
        raise ValueError(f"map line {line_no}: edge needs exactly two node ids")
    a, b = ids
    for node_id in (a, b):
        if node_id not in graph.nodes:
            raise ValueError(
                f"map line {line_no}: edge references unknown node '{node_id}'"
            )
    if weight <= 0:
        raise ValueError(f"map line {line_no}: weight must be positive")
    graph.adjacency[a].append((b, weight))
    graph.adjacency[b].append((a, weight))


def euclidean_heuristic(graph: IndoorGraph, goal: str):
    """Admissible heuristic: straight-line distance to the goal.

    It is admissible because every edge weight in the sample map is the
    straight-line distance between its endpoints (rounded up), so no path
    can be shorter than the straight line to the goal (triangle
    inequality). If a future map violated this, a zero heuristic (plain
    Dijkstra behaviour) would be the safe fallback.
    """
    gx, gy = graph.coords(goal)

    def h(node_id: str) -> float:
        x, y = graph.coords(node_id)
        return ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5

    return h

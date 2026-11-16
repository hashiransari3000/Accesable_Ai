"""Graph data for the indoor navigation demo (fictional sample map).

Parses the plain-text map file under assets/sample_maps/ into an
IndoorGraph used by bfs.py, astar.py, profiles.py, and route_service.py.

Format (see the sample file for a full example):
    node <id> | <display name> | <x> | <y> [| <floor> | <kind> | <tags>]
    edge <id_a> <id_b> | <weight> [| <type> | <width_m> | <crowd>]

Floors are integers (0 = ground). Kinds: entrance, room, corridor, lift,
stairs, facility, exit. Edge types: corridor, stairs, lift, door, ramp.
Tags are comma-separated free labels such as "accessible_toilet,quiet".

Load-time validation rejects unknown endpoints, non-positive weights,
duplicate nodes/edges, malformed attributes, and floor changes on edges
that are not lift/stairs connectors, so algorithms can assume a clean
graph. The graph is never mutated after loading; route closures are
applied by the planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAP_PATH = "assets/sample_maps/sample_building.txt"

NODE_KINDS = {
    "entrance",
    "room",
    "corridor",
    "lift",
    "stairs",
    "facility",
    "exit",
}
EDGE_TYPES = {"corridor", "stairs", "lift", "door", "ramp"}

DEFAULT_FLOOR = 0
DEFAULT_KIND = "room"
DEFAULT_EDGE_TYPE = "corridor"
DEFAULT_WIDTH_M = 2.0
DEFAULT_CROWD = 0.0


def floor_label(floor: int) -> str:
    """Human label for a floor number."""
    return {
        0: "Ground floor",
        1: "Upper floor",
        -1: "Lower floor",
    }.get(floor, f"Floor {floor}")


@dataclass
class IndoorGraph:
    nodes: dict[str, dict] = field(default_factory=dict)
    adjacency: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    edges: dict[tuple[str, str], dict] = field(default_factory=dict)

    def display_name(self, node_id: str) -> str:
        return self.nodes.get(node_id, {}).get("name", node_id)

    def ids(self) -> list[str]:
        return list(self.nodes.keys())

    def neighbours(self, node_id: str) -> list[tuple[str, float]]:
        return self.adjacency.get(node_id, [])

    def coords(self, node_id: str) -> tuple[float, float]:
        data = self.nodes[node_id]
        return (float(data["x"]), float(data["y"]))

    def floor(self, node_id: str) -> int:
        return int(self.nodes.get(node_id, {}).get("floor", DEFAULT_FLOOR))

    def kind(self, node_id: str) -> str:
        return str(self.nodes.get(node_id, {}).get("kind", DEFAULT_KIND))

    def tags(self, node_id: str) -> tuple[str, ...]:
        return tuple(self.nodes.get(node_id, {}).get("tags", ()))

    def floors(self) -> list[int]:
        return sorted({self.floor(node) for node in self.nodes})

    def edge_attrs(self, a: str, b: str) -> dict | None:
        return self.edges.get(edge_key(a, b))

    def edge_items(self) -> list[tuple[str, str, dict]]:
        return [
            (a, b, attrs) for (a, b), attrs in sorted(self.edges.items())
        ]


def edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


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
    if not 4 <= len(parts) <= 7:
        raise ValueError(
            f"map line {line_no}: node needs "
            "'id | name | x | y [| floor | kind | tags]'"
        )
    node_id, name, x, y = parts[:4]
    if node_id in graph.nodes:
        raise ValueError(f"map line {line_no}: duplicate node id '{node_id}'")
    try:
        coords = (float(x), float(y))
    except ValueError as exc:
        raise ValueError(f"map line {line_no}: x/y must be numbers") from exc

    floor_text = parts[4] if len(parts) > 4 else ""
    try:
        floor_no = int(floor_text) if floor_text else DEFAULT_FLOOR
    except ValueError as exc:
        raise ValueError(f"map line {line_no}: floor must be an integer") from exc

    kind = (parts[5] if len(parts) > 5 and parts[5] else DEFAULT_KIND).lower()
    if kind not in NODE_KINDS:
        allowed = ", ".join(sorted(NODE_KINDS))
        raise ValueError(
            f"map line {line_no}: unknown node kind '{kind}' (use {allowed})"
        )

    tags_text = parts[6] if len(parts) > 6 else ""
    tags = tuple(
        tag.strip().lower() for tag in tags_text.split(",") if tag.strip()
    )

    graph.nodes[node_id] = {
        "name": name,
        "x": coords[0],
        "y": coords[1],
        "floor": floor_no,
        "kind": kind,
        "tags": tags,
    }
    graph.adjacency.setdefault(node_id, [])


def _parse_edge(line: str, line_no: int, graph: IndoorGraph) -> None:
    body = line[len("edge "):]
    left, _, right = body.partition("|")
    ids = [p.strip() for p in left.split()]
    if len(ids) != 2:
        raise ValueError(f"map line {line_no}: edge needs exactly two node ids")
    a, b = ids
    for node_id in (a, b):
        if node_id not in graph.nodes:
            raise ValueError(
                f"map line {line_no}: edge references unknown node '{node_id}'"
            )
    if a == b:
        raise ValueError(f"map line {line_no}: an edge cannot join a node to itself")

    attributes = [p.strip() for p in right.split("|")]
    try:
        weight = float(attributes[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"map line {line_no}: weight must be a number") from exc
    if weight <= 0:
        raise ValueError(f"map line {line_no}: weight must be positive")

    edge_type = (
        attributes[1] if len(attributes) > 1 and attributes[1]
        else DEFAULT_EDGE_TYPE
    ).lower()
    if edge_type not in EDGE_TYPES:
        allowed = ", ".join(sorted(EDGE_TYPES))
        raise ValueError(
            f"map line {line_no}: unknown edge type '{edge_type}' "
            f"(use {allowed})"
        )
    try:
        width = (
            float(attributes[2])
            if len(attributes) > 2 and attributes[2]
            else DEFAULT_WIDTH_M
        )
        crowd = (
            float(attributes[3])
            if len(attributes) > 3 and attributes[3]
            else DEFAULT_CROWD
        )
    except ValueError as exc:
        raise ValueError(
            f"map line {line_no}: width and crowd must be numbers"
        ) from exc
    if width <= 0:
        raise ValueError(f"map line {line_no}: width must be positive")
    if not 0.0 <= crowd <= 1.0:
        raise ValueError(f"map line {line_no}: crowd must be between 0 and 1")

    if graph.floor(a) != graph.floor(b) and edge_type not in {"lift", "stairs"}:
        raise ValueError(
            f"map line {line_no}: floor changes must use a lift or stairs edge"
        )

    key = edge_key(a, b)
    if key in graph.edges:
        raise ValueError(
            f"map line {line_no}: duplicate edge between '{a}' and '{b}'"
        )

    graph.adjacency[a].append((b, weight))
    graph.adjacency[b].append((a, weight))
    graph.edges[key] = {
        "weight": weight,
        "type": edge_type,
        "width_m": width,
        "crowd": crowd,
    }


def euclidean_heuristic(graph: IndoorGraph, goal: str):
    """Admissible heuristic: straight-line distance to the goal.

    It uses x/y only and ignores floor changes, so it never overestimates:
    same-floor edge weights are at least the straight-line distance between
    endpoints, and cross-floor edges add positive cost on top. A zero
    heuristic (plain Dijkstra behaviour) would also be safe.
    """
    gx, gy = graph.coords(goal)

    def h(node_id: str) -> float:
        x, y = graph.coords(node_id)
        return ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5

    return h

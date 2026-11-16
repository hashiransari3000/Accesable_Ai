"""Turn-by-turn text instructions for a simulated indoor route.

The instructions are derived only from the FICTIONAL sample graph: node
names, coordinates, edge types, and simulated distances. They are not live
navigation and must never be presented as real-world guidance.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from modules.navigation.graph_data import IndoorGraph, floor_label

DEFAULT_SPEED_MPS = 1.1
STAIR_PENALTY_S = 20
LIFT_WAIT_S = 30
DOOR_PENALTY_S = 5

_COMPASS = (
    (22.5, "north"),
    (67.5, "north-east"),
    (112.5, "east"),
    (157.5, "south-east"),
    (202.5, "south"),
    (247.5, "south-west"),
    (292.5, "west"),
    (337.5, "north-west"),
    (360.0, "north"),
)


@dataclass(frozen=True)
class RouteStep:
    index: int
    node_id: str
    name: str
    floor: int
    action: str
    instruction: str
    spoken: str
    distance_m: float
    cumulative_m: float
    eta_s: int
    remaining_m: float


def format_distance(distance_m: float) -> str:
    if distance_m >= 10:
        return f"{distance_m:.0f} m"
    return f"{distance_m:.1f} m"


def format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    if total < 60:
        return f"{total} s"
    minutes, remainder = divmod(total, 60)
    if remainder == 0:
        return f"{minutes} min"
    return f"{minutes} min {remainder} s"


def _bearing(graph: IndoorGraph, a: str, b: str) -> float:
    ax, ay = graph.coords(a)
    bx, by = graph.coords(b)
    return math.degrees(math.atan2(bx - ax, by - ay)) % 360.0


def _compass(bearing: float) -> str:
    for limit, label in _COMPASS:
        if bearing < limit:
            return label
    return "north"


def _turn_phrase(previous: float, following: float) -> tuple[str, str]:
    delta = (following - previous + 180.0) % 360.0 - 180.0
    magnitude = abs(delta)
    if magnitude < 25:
        return "continue straight", "straight"
    if magnitude > 155:
        return "turn sharply around", "turn"
    side = "right" if delta > 0 else "left"
    if magnitude < 60:
        return f"bear {side}", side
    return f"turn {side}", side


def _edge_seconds(weight: float, attrs: dict, speed_mps: float) -> float:
    seconds = weight / speed_mps
    edge_type = attrs.get("type", "corridor")
    if edge_type == "stairs":
        seconds += STAIR_PENALTY_S
    elif edge_type == "lift":
        seconds += LIFT_WAIT_S
    elif edge_type == "door":
        seconds += DOOR_PENALTY_S
    return seconds


def build_steps(
    graph: IndoorGraph,
    path: list[str],
    speed_mps: float = DEFAULT_SPEED_MPS,
) -> list[RouteStep]:
    """Create one walk-through step per node on the route."""
    if speed_mps <= 0:
        raise ValueError("Walking speed must be positive.")
    if not path:
        return []
    if path[0] not in graph.nodes or path[-1] not in graph.nodes:
        raise ValueError("Route contains an unknown node.")

    edge_data: list[tuple[str, float, dict]] = []
    for a, b in itertools.pairwise(path):
        attrs = graph.edge_attrs(a, b) or {}
        weight = float(attrs.get("weight", 0.0))
        if not weight:
            for neighbour, neighbour_weight in graph.neighbours(a):
                if neighbour == b:
                    weight = neighbour_weight
                    break
        edge_data.append((b, weight, attrs))

    total_distance = sum(weight for _, weight, _ in edge_data)
    cumulative = 0.0
    cumulative_seconds = 0.0
    previous_bearing: float | None = None
    graph.floor(path[0])
    steps: list[RouteStep] = []

    if len(path) == 1:
        name = graph.display_name(path[0])
        floor = graph.floor(path[0])
        return [
            RouteStep(
                index=0,
                node_id=path[0],
                name=name,
                floor=floor,
                action="arrive",
                instruction=f"You are already at {name} on the {floor_label(floor).lower()}.",
                spoken=f"You are already at {name}. This is the end of the simulated route.",
                distance_m=0.0,
                cumulative_m=0.0,
                eta_s=0,
                remaining_m=0.0,
            )
        ]

    for index, node_id in enumerate(path):
        name = graph.display_name(node_id)
        floor = graph.floor(node_id)

        if index == len(path) - 1:
            steps.append(
                RouteStep(
                    index=index,
                    node_id=node_id,
                    name=name,
                    floor=floor,
                    action="arrive",
                    instruction=f"Arrive at {name} on the {floor_label(floor).lower()}.",
                    spoken=(
                        f"You have arrived at {name}. This is the end of the "
                        "simulated route."
                    ),
                    distance_m=0.0,
                    cumulative_m=cumulative,
                    eta_s=round(cumulative_seconds),
                    remaining_m=0.0,
                )
            )
            break

        next_id, weight, attrs = edge_data[index]
        next_name = graph.display_name(next_id)
        next_floor = graph.floor(next_id)
        edge_type = attrs.get("type", "corridor")
        next_bearing = None if floor != next_floor else _bearing(graph, node_id, next_id)

        if edge_type == "lift":
            action = "lift"
            phrase = f"take the lift to the {floor_label(next_floor).lower()}"
        elif edge_type == "stairs":
            action = "stairs"
            phrase = f"take the stairs to the {floor_label(next_floor).lower()}"
        elif index == 0:
            action = "start"
            phrase = (
                f"head {_compass(next_bearing)}"
                if next_bearing is not None
                else "start"
            )
        elif previous_bearing is not None and next_bearing is not None:
            turn, action = _turn_phrase(previous_bearing, next_bearing)
            phrase = turn
        else:
            action = "straight"
            phrase = f"head {_compass(next_bearing)}" if next_bearing is not None else "continue"

        if index == 0:
            if edge_type in {"lift", "stairs"}:
                instruction = f"Start at {name}. {phrase.capitalize()}."
            else:
                instruction = (
                    f"Start at {name}, {phrase}, then walk "
                    f"{format_distance(weight)} to {next_name}."
                )
        elif edge_type in {"lift", "stairs"}:
            instruction = f"At {name}, {phrase}."
        else:
            instruction = (
                f"At {name}, {phrase} and walk {format_distance(weight)} "
                f"to {next_name}."
            )

        steps.append(
            RouteStep(
                index=index,
                node_id=node_id,
                name=name,
                floor=floor,
                action=action,
                instruction=instruction,
                spoken=instruction,
                distance_m=weight,
                cumulative_m=cumulative,
                eta_s=round(cumulative_seconds),
                remaining_m=max(0.0, total_distance - cumulative),
            )
        )

        cumulative += weight
        cumulative_seconds += _edge_seconds(weight, attrs, speed_mps)
        previous_bearing = next_bearing if next_bearing is not None else None

    return steps


def total_eta_seconds(
    graph: IndoorGraph,
    path: list[str],
    speed_mps: float = DEFAULT_SPEED_MPS,
) -> int:
    """Estimated simulated travel time including lift/stair/door penalties."""
    seconds = 0.0
    for a, b in itertools.pairwise(path):
        attrs = graph.edge_attrs(a, b) or {}
        weight = float(attrs.get("weight", 0.0))
        if not weight:
            for neighbour, neighbour_weight in graph.neighbours(a):
                if neighbour == b:
                    weight = neighbour_weight
                    break
        seconds += _edge_seconds(weight, attrs, speed_mps)
    return round(seconds)


def route_speech_summary(
    graph: IndoorGraph,
    path: list[str],
    *,
    stairs: int,
    lifts: int,
    speed_mps: float = DEFAULT_SPEED_MPS,
) -> str:
    """Short spoken overview; the per-step instructions stay available."""
    if not path:
        return "No route is available."
    start = graph.display_name(path[0])
    goal = graph.display_name(path[-1])
    distance = sum(
        float((graph.edge_attrs(a, b) or {}).get("weight", 0.0))
        for a, b in itertools.pairwise(path)
    )
    eta = total_eta_seconds(graph, path, speed_mps)
    parts = [
        f"Simulated route from {start} to {goal}.",
        f"{format_distance(distance)}, about {format_duration(eta)}.",
    ]
    if lifts:
        parts.append(f"Uses the lift {lifts} time{'s' if lifts != 1 else ''}.")
    if stairs:
        parts.append(f"Includes stairs {stairs} time{'s' if stairs != 1 else ''}.")
    else:
        parts.append("No stairs on this route.")
    return " ".join(parts)

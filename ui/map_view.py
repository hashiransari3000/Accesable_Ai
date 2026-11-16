"""Interactive route-map rendering for the Indoor Navigation Demo.

Draws the fictional two-floor sample graph with Plotly: walkable edges,
simulated closures, the computed route, floor transitions, and markers.
Clicking a node or an edge midpoint is handled by the navigation page.

This is a schematic of the sample graph only - never live guidance and not
to scale for any real building. The textual route remains the primary
channel; colour is never the only signal (markers are labelled and shaped
differently).
"""

from __future__ import annotations

import itertools

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.navigation.graph_data import IndoorGraph, edge_key, floor_label
from modules.navigation.profiles import normalize_blocked_edges

PAGE_BG = "#F8FAFC"
EDGE = "#CBD5E1"
BLOCKED = "#B91C1C"
ROUTE = "#0F766E"
NODE_FILL = "#FFFFFF"
NODE_EDGE = "#64748B"
LABEL = "#334155"
START = "#15803D"
GOAL = "#B45309"
CURRENT = "#1D4ED8"
EXPLORED = "#93C5FD"
TRANSITION = "#0E7490"

_SYMBOLS = {
    "entrance": "square",
    "room": "circle",
    "corridor": "circle",
    "lift": "diamond",
    "stairs": "triangle-up",
    "facility": "star",
    "exit": "x",
}


def _midpoint(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _segment_trace(
    xs: list[float | None],
    ys: list[float | None],
    *,
    color: str,
    width: float,
    dash: str = "solid",
    name: str,
    hover: str | None = None,
) -> go.Scatter:
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": color, "width": width, "dash": dash},
        hoverinfo="text" if hover else "skip",
        hovertext=hover,
        name=name,
        showlegend=False,
    )


def build_route_figure(
    graph: IndoorGraph,
    *,
    path: list[str] | tuple[str, ...] = (),
    start: str | None = None,
    goal: str | None = None,
    current: str | None = None,
    explored: list[str] | tuple[str, ...] = (),
    blocked_edges: object = (),
    blocked_nodes: object = (),
    show_closures: bool = False,
    label_scope: str = "all",
    route_color: str = ROUTE,
    route_name: str = "Route",
    height: int | None = None,
) -> go.Figure:
    """Build the two-floor schematic figure (pure; no Streamlit calls).

    label_scope: "all" labels every location, "route" labels only route
    nodes (used for the BFS/A* comparison maps), "none" hides labels.
    """
    if label_scope not in {"all", "route", "none"}:
        raise ValueError(f"Unknown label scope '{label_scope}'.")
    floors = graph.floors() or [0]
    rows = {floor: index + 1 for index, floor in enumerate(floors)}
    blocked = normalize_blocked_edges(blocked_edges)
    blocked_node_set = set(blocked_nodes or ())
    path = list(path)
    route_edges = {edge_key(a, b) for a, b in itertools.pairwise(path)}
    explored_set = set(explored)
    current_floor = graph.floor(current) if current in graph.nodes else None
    fig_height = height or (680 if len(floors) > 1 else 430)

    all_x = [graph.coords(node)[0] for node in graph.ids()]
    all_y = [graph.coords(node)[1] for node in graph.ids()]
    x_range = (min(all_x) - 4.0, max(all_x) + 4.0)
    y_range = (min(all_y) - 3.0, max(all_y) + 3.0)
    if label_scope == "all":
        labelled_nodes = set(graph.ids())
    elif label_scope == "route":
        labelled_nodes = set(path)
    else:
        labelled_nodes = set()
    plot_height = max(170.0, (fig_height - 110) / len(floors))

    fig = make_subplots(
        rows=len(floors),
        cols=1,
        subplot_titles=[floor_label(floor) for floor in floors],
        vertical_spacing=0.09 if len(floors) > 1 else 0.02,
    )

    for floor in floors:
        row = rows[floor]
        floor_nodes = [node for node in graph.ids() if graph.floor(node) == floor]

        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        closed_x: list[float | None] = []
        closed_y: list[float | None] = []
        route_x: list[float | None] = []
        route_y: list[float | None] = []
        transition_x: list[float] = []
        transition_y: list[float] = []
        transition_hover: list[str] = []
        closure_x: list[float] = []
        closure_y: list[float] = []
        closure_hover: list[str] = []
        closure_customdata: list[list[str]] = []

        for a, b, attrs in graph.edge_items():
            same_floor = graph.floor(a) == floor and graph.floor(b) == floor
            crosses_floor = graph.floor(a) == floor or graph.floor(b) == floor
            key = edge_key(a, b)
            is_closed = key in blocked or a in blocked_node_set or b in blocked_node_set

            if same_floor:
                xa, ya = graph.coords(a)
                xb, yb = graph.coords(b)
                target_x, target_y = (closed_x, closed_y) if is_closed else (edge_x, edge_y)
                target_x += [xa, xb, None]
                target_y += [ya, yb, None]
                if show_closures:
                    mx, my = _midpoint(xa, ya, xb, yb)
                    closure_x.append(mx)
                    closure_y.append(my)
                    closure_hover.append(
                        f"{graph.display_name(a)} ↔ {graph.display_name(b)}"
                        f"<br>{attrs.get('type', 'corridor')} · "
                        f"{float(attrs.get('width_m', 0)):.1f} m wide · "
                        f"crowd {float(attrs.get('crowd', 0)):.0%}"
                        f"<br><b>{'Reopen this segment' if is_closed else 'Close this segment'}</b>"
                    )
                    closure_customdata.append(["edge", a, b])

            if route_edges and key in route_edges:
                if same_floor:
                    xa, ya = graph.coords(a)
                    xb, yb = graph.coords(b)
                    route_x += [xa, xb, None]
                    route_y += [ya, yb, None]
                elif crosses_floor:
                    node = a if graph.floor(a) == floor else b
                    x, y = graph.coords(node)
                    transition_x.append(x)
                    transition_y.append(y)
                    transition_hover.append(
                        f"{attrs.get('type', 'lift').title()} floor change"
                        f"<br>{graph.display_name(a)} ↔ {graph.display_name(b)}"
                    )

        fig.add_trace(
            _segment_trace(
                edge_x, edge_y,
                color=EDGE, width=1.8, name="Walkable path",
            ),
            row=row, col=1,
        )
        fig.add_trace(
            _segment_trace(
                closed_x, closed_y,
                color=BLOCKED, width=2.0, dash="dash", name="Closed segment",
            ),
            row=row, col=1,
        )
        fig.add_trace(
            _segment_trace(
                route_x, route_y,
                color=route_color, width=4.2, name=route_name,
            ),
            row=row, col=1,
        )

        if transition_x:
            fig.add_trace(
                go.Scatter(
                    x=transition_x,
                    y=transition_y,
                    mode="markers",
                    marker={
                        "size": 26,
                        "symbol": "circle-open",
                        "color": TRANSITION,
                        "line": {"width": 4, "color": TRANSITION},
                    },
                    hovertext=transition_hover,
                    hoverinfo="text",
                    name="Route floor change",
                    showlegend=False,
                ),
                row=row, col=1,
            )

        if show_closures and closure_x:
            fig.add_trace(
                go.Scatter(
                    x=closure_x,
                    y=closure_y,
                    mode="markers",
                    marker={
                        "size": 9,
                        "symbol": "square",
                        "color": "#FFFFFF",
                        "line": {"width": 1.5, "color": BLOCKED},
                    },
                    customdata=closure_customdata,
                    hovertext=closure_hover,
                    hoverinfo="text",
                    name="Click to close / reopen",
                    showlegend=False,
                ),
                row=row, col=1,
            )

        if explored_set:
            ex = [graph.coords(n)[0] for n in explored_set if n in floor_nodes]
            ey = [graph.coords(n)[1] for n in explored_set if n in floor_nodes]
            if ex:
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        mode="markers",
                        marker={
                            "size": 17,
                            "color": "rgba(0,0,0,0)",
                            "line": {"width": 2, "color": EXPLORED},
                        },
                        hoverinfo="skip",
                        name="Explored node",
                        showlegend=False,
                    ),
                    row=row, col=1,
                )

        if floor_nodes:
            node_x = [graph.coords(n)[0] for n in floor_nodes]
            node_y = [graph.coords(n)[1] for n in floor_nodes]
            node_symbols = [
                _SYMBOLS.get(graph.kind(n), "circle") for n in floor_nodes
            ]
            node_hover = [
                _node_hover(graph, n) for n in floor_nodes
            ]
            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers",
                    marker={
                        "size": 13,
                        "symbol": node_symbols,
                        "color": NODE_FILL,
                        "line": {"width": 1.5, "color": NODE_EDGE},
                    },
                    customdata=[["node", n] for n in floor_nodes],
                    hovertext=node_hover,
                    hoverinfo="text",
                    name="Location",
                    showlegend=False,
                ),
                row=row, col=1,
            )

            label_positions = _place_labels(
                graph,
                floor_nodes,
                labelled_nodes,
                x_range,
                y_range,
                width_px=620.0,
                height_px=plot_height,
            )
            leader_x: list[float | None] = []
            leader_y: list[float | None] = []
            for node, (label_x, label_y, dx, dy) in label_positions.items():
                if abs(dx) > 1.0 or abs(dy) > 1.0:
                    x, y = graph.coords(node)
                    leader_x += [x, label_x, None]
                    leader_y += [y, label_y, None]
                fig.add_annotation(
                    x=label_x,
                    y=label_y,
                    text=graph.display_name(node),
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    bgcolor="rgba(248,250,252,0.88)",
                    borderpad=1,
                    font={"size": 9, "color": LABEL},
                    row=row,
                    col=1,
                )
            if leader_x:
                fig.add_trace(
                    _segment_trace(
                        leader_x,
                        leader_y,
                        color="#94A3B8",
                        width=0.8,
                        name="Label leader",
                    ),
                    row=row,
                    col=1,
                )

        _overlay_node(
            fig, graph, start, floor, START, "square", 18, "Start", row
        )
        _overlay_node(fig, graph, goal, floor, GOAL, "diamond", 18, "Destination", row)
        _overlay_node(
            fig, graph, current, floor, CURRENT, "circle", 23, "Current step", row
        )

        for node in sorted(blocked_node_set - {start, goal}):
            if node in graph.nodes and graph.floor(node) == floor:
                x, y = graph.coords(node)
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers",
                        marker={
                            "size": 20,
                            "symbol": "x",
                            "color": BLOCKED,
                            "line": {"width": 2, "color": BLOCKED},
                        },
                        hovertext=[f"{graph.display_name(node)} closed"],
                        hoverinfo="text",
                        name="Closed location",
                        showlegend=False,
                    ),
                    row=row, col=1,
                )

        fig.update_xaxes(
            visible=False,
            range=list(x_range),
            fixedrange=True,
            row=row,
            col=1,
        )
        fig.update_yaxes(
            visible=False,
            range=list(y_range),
            fixedrange=True,
            row=row,
            col=1,
        )

    for name, color, symbol, dash in (
        ("Start", START, "square", None),
        ("Destination", GOAL, "diamond", None),
        ("Current step", CURRENT, "circle", None),
        (route_name, route_color, None, "solid"),
        ("Walkable path", EDGE, None, "solid"),
        ("Closed segment", BLOCKED, None, "dash"),
        ("Lift / stairs link", TRANSITION, "circle-open", None),
    ):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers" if symbol else "lines",
                line={"color": color, "width": 3, "dash": dash or "solid"},
                marker={
                    "color": color,
                    "size": 11,
                    "symbol": symbol or "circle",
                    "line": {"width": 1, "color": color},
                },
                name=name,
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.update_layout(
        height=fig_height,
        margin={"l": 12, "r": 12, "t": 52, "b": 10},
        paper_bgcolor=PAGE_BG,
        plot_bgcolor=PAGE_BG,
        font={"family": "system-ui, sans-serif", "size": 11, "color": LABEL},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.05,
            "x": 0.0,
            "font": {"size": 10},
        },
        hovermode="closest",
        dragmode="pan",
        clickmode="event+select",
    )
    if current_floor is not None:
        fig.update_layout(
            annotations=[
                *fig.layout.annotations,
                {
                    "text": f"Current step: {floor_label(current_floor)}",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.0,
                    "y": 1.04,
                    "showarrow": False,
                    "font": {"size": 11, "color": CURRENT},
                },
            ]
        )
    return fig


def _box_overlap(a: tuple[float, float, float, float],
                 b: tuple[float, float, float, float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _place_labels(
    graph: IndoorGraph,
    floor_nodes: list[str],
    labelled_nodes: set[str],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    width_px: float,
    height_px: float,
) -> dict[str, tuple[float, float, float, float]]:
    """Greedy label placement that avoids overlaps in data coordinates.

    Labels are tried above, below, and beside each marker (then farther
    out) and the first non-overlapping box wins. Marker exclusion boxes
    stop labels covering other nodes. Returns node -> (x, y, dx, dy).
    """
    x0, x1 = x_range
    y0, y1 = y_range
    x_span = x1 - x0
    y_span = y1 - y0
    px_per_x = width_px / x_span if x_span else 1.0
    px_per_y = height_px / y_span if y_span else 1.0

    marker_boxes = {
        node: (
            graph.coords(node)[0] - 0.45,
            graph.coords(node)[1] - 0.55,
            graph.coords(node)[0] + 0.45,
            graph.coords(node)[1] + 0.55,
        )
        for node in floor_nodes
    }
    label_boxes: list[tuple[float, float, float, float]] = []

    positions: dict[str, tuple[float, float, float, float]] = {}
    ordered = sorted(
        labelled_nodes,
        key=lambda node: (
            -len(graph.display_name(node)),
            -graph.coords(node)[1],
            graph.coords(node)[0],
        ),
    )
    for node in ordered:
        if node not in floor_nodes:
            continue
        x, y = graph.coords(node)
        text = graph.display_name(node)
        label_w = min(
            x_span * 0.42,
            max(1.4, (len(text) * 5.1 + 8.0) / px_per_x),
        )
        label_h = max(0.75, 14.0 / px_per_y)
        gap = 0.65
        candidates = [
            (0.0, label_h / 2 + gap),
            (0.0, -label_h / 2 - gap),
            (label_w / 2 + gap, 0.0),
            (-label_w / 2 - gap, 0.0),
            (label_w / 2 + gap, label_h / 2 + gap),
            (-label_w / 2 - gap, label_h / 2 + gap),
            (label_w / 2 + gap, -label_h / 2 - gap),
            (-label_w / 2 - gap, -label_h / 2 - gap),
            (0.0, label_h * 1.6 + gap),
            (0.0, -label_h * 1.6 - gap),
            (label_w * 1.1 + gap, 0.0),
            (-label_w * 1.1 - gap, 0.0),
        ]

        existing = [
            box for other, box in marker_boxes.items() if other != node
        ] + label_boxes
        best: tuple[float, float, float, float, float, float] | None = None
        for dx, dy in candidates:
            lx, ly = x + dx, y + dy
            box = (
                lx - label_w / 2,
                ly - label_h / 2,
                lx + label_w / 2,
                ly + label_h / 2,
            )
            if not (x0 <= box[0] and box[2] <= x1 and y0 <= box[1] and box[3] <= y1):
                continue
            overlap = sum(_box_overlap(box, other) for other in existing)
            distance = abs(dx) + abs(dy)
            if overlap == 0:
                best = (0.0, dx, dy, lx, ly, distance)
                break
            if best is None or (overlap, distance) < (best[0], best[5]):
                best = (overlap, dx, dy, lx, ly, distance)

        if best is None:
            dx, dy = candidates[0]
            lx, ly = x + dx, y + dy
            best = (0.0, dx, dy, lx, ly, abs(dx) + abs(dy))

        _, dx, dy, lx, ly, _ = best
        positions[node] = (lx, ly, dx, dy)
        label_boxes.append(
            (
                lx - label_w / 2,
                ly - label_h / 2,
                lx + label_w / 2,
                ly + label_h / 2,
            )
        )
    return positions


def _node_hover(graph: IndoorGraph, node: str) -> str:
    tags = ", ".join(graph.tags(node)) or "none"
    return (
        f"<b>{graph.display_name(node)}</b>"
        f"<br>{floor_label(graph.floor(node))} · {graph.kind(node)}"
        f"<br>tags: {tags}"
        f"<br><b>Click to select this location</b>"
    )


def _overlay_node(
    fig: go.Figure,
    graph: IndoorGraph,
    node: str | None,
    floor: int,
    color: str,
    symbol: str,
    size: int,
    legend_name: str,
    row: int,
) -> None:
    if node is None or node not in graph.nodes or graph.floor(node) != floor:
        return
    x, y = graph.coords(node)
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker={
                "size": size,
                "symbol": symbol,
                "color": color,
                "line": {"width": 2, "color": "#FFFFFF"},
            },
            customdata=[["node", node]],
            hovertext=[legend_name],
            hoverinfo="text",
            name=legend_name,
            showlegend=False,
        ),
        row=row, col=1,
    )


def decode_selection(event) -> tuple[str, ...] | None:
    """Extract customdata from a Streamlit Plotly selection event."""
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None
    for point in points:
        if isinstance(point, dict):
            customdata = point.get("customdata")
        else:
            customdata = getattr(point, "customdata", None)
        if customdata is None:
            continue
        if isinstance(customdata, str):
            return (customdata,)
        try:
            parts = tuple(str(part) for part in customdata)
        except TypeError:
            continue
        if parts:
            return parts
    return None

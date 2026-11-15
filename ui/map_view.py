"""Static route-map rendering for the Indoor Navigation Demo.

Draws the fictional sample map with matplotlib: walkable edges, the
computed route, and start/destination markers. This is a schematic of
the sample graph only - it is never live guidance and not to scale for
any real building. The textual route remains the primary channel; the
map is a supplementary visual (colour is never the only signal - every
marker is also labelled in the legend and captions).
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAGE_BG = "#F8FAFC"
EDGE = "#CBD5E1"
ROUTE = "#0F766E"
NODE_FILL = "#FFFFFF"
NODE_EDGE = "#64748B"
LABEL = "#334155"
START = "#15803D"
GOAL = "#B45309"


def route_figure(graph, path: list[str]):
    """Build a matplotlib figure for a graph and a node path."""
    fig, ax = plt.subplots(figsize=(7.6, 4.9), dpi=120)
    fig.patch.set_facecolor(PAGE_BG)
    ax.set_facecolor(PAGE_BG)

    drawn: set[tuple[str, str]] = set()
    for node in graph.ids():
        x1, y1 = graph.coords(node)
        for neighbour, _weight in graph.neighbours(node):
            pair = tuple(sorted((node, neighbour)))
            if pair in drawn:
                continue
            drawn.add(pair)
            x2, y2 = graph.coords(neighbour)
            ax.plot(
                [x1, x2], [y1, y2],
                color=EDGE, lw=1.6, zorder=1, solid_capstyle="round",
            )

    for a, b in itertools.pairwise(path):
        xa, ya = graph.coords(a)
        xb, yb = graph.coords(b)
        ax.plot(
            [xa, xb], [ya, yb],
            color=ROUTE, lw=3.6, zorder=2, solid_capstyle="round",
        )

    placed: list[tuple[float, float, int]] = []
    for node in graph.ids():
        x, y = graph.coords(node)
        if path and node == path[0]:
            fill, edge, size = START, START, 190
        elif path and node == path[-1]:
            fill, edge, size = GOAL, GOAL, 190
        else:
            fill, edge, size = NODE_FILL, NODE_EDGE, 110
        ax.scatter(
            [x], [y], s=size, c=fill, edgecolors=edge, linewidths=1.4, zorder=3,
        )
        # Alternate crowded neighbouring labels above/below to avoid overlap.
        label_dy = 9
        for px, py, pdy in placed:
            if abs(py - y) < 2.5 and abs(px - x) < 7.5 and pdy == label_dy:
                label_dy = -17
                break
        placed.append((x, y, label_dy))
        ax.annotate(
            graph.display_name(node),
            (x, y),
            textcoords="offset points",
            xytext=(0, label_dy),
            ha="center",
            fontsize=7.6,
            color=LABEL,
            zorder=4,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": PAGE_BG,
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.16)
    ax.axis("off")

    legend = [
        Line2D([], [], marker="o", ls="", mfc=START, mec=START, label="Start"),
        Line2D([], [], marker="o", ls="", mfc=GOAL, mec=GOAL, label="Destination"),
        Line2D([], [], color=ROUTE, lw=3.4, label="Route"),
        Line2D([], [], color=EDGE, lw=1.6, label="Walkable path"),
    ]
    ax.legend(
        handles=legend,
        loc="upper right",
        frameon=True,
        fontsize=7.5,
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
    )
    fig.tight_layout(pad=0.4)
    return fig


def render_route_map(graph, path: list[str], caption: str | None = None) -> None:
    """Render the route map in Streamlit and release the figure."""
    import streamlit as st

    fig = route_figure(graph, path)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    if caption:
        st.caption(caption)

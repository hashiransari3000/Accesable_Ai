"""Navigation algorithm tests (TEST_PLAN N1-N8)."""

import pytest

from modules.navigation import astar, bfs, route_service
from modules.navigation.graph_data import (
    DEFAULT_MAP_PATH,
    euclidean_heuristic,
    load_graph,
)

MAP = "assets/sample_maps/sample_building.txt"


@pytest.fixture(scope="module")
def graph():
    return load_graph(MAP)


# ---- N1: route exists -> valid contiguous path ---------------------------------

def test_bfs_finds_valid_route(graph):
    result = bfs.bfs_shortest_path(graph, "entrance", "library")
    assert result.path is not None
    assert result.path[0] == "entrance" and result.path[-1] == "library"
    # contiguous: every consecutive pair is connected
    for a, b in zip(result.path, result.path[1:]):
        assert any(n == b for n, _ in graph.neighbours(a))


def test_astar_finds_valid_route(graph):
    result = astar.astar_shortest_path(
        graph, "entrance", "library", euclidean_heuristic(graph, "library")
    )
    assert result.path is not None
    assert result.path[0] == "entrance" and result.path[-1] == "library"
    for a, b in zip(result.path, result.path[1:]):
        assert any(n == b for n, _ in graph.neighbours(a))


# ---- N2: unreachable destination -------------------------------------------------

def test_unreachable_goal_explicit(graph):
    result = bfs.bfs_shortest_path(graph, "entrance", "annex_office")
    assert result.path is None
    assert result.explored_order  # it did explore the reachable part

    result2 = astar.astar_shortest_path(
        graph, "entrance", "annex_lobby", euclidean_heuristic(graph, "annex_lobby")
    )
    assert result2.path is None


# ---- N3: start equals goal ---------------------------------------------------------

def test_start_equals_goal_trivial_route(graph):
    for algo in (bfs.bfs_shortest_path, astar.astar_shortest_path):
        result = algo(graph, "main_hall", "main_hall")
        assert result.path == ["main_hall"]
        assert result.hops == 0
        assert result.cost == 0.0


# ---- N4: invalid node names ---------------------------------------------------------

def test_invalid_node_raises(graph):
    with pytest.raises(ValueError):
        bfs.bfs_shortest_path(graph, "entrance", "nonexistent_room")
    with pytest.raises(ValueError):
        astar.astar_shortest_path(graph, "nonsense", "library")


def test_route_service_converts_invalid_to_error(graph):
    result = route_service.run_route(graph, "entrance", "bad_room", "A*")
    assert result.error is not None and "Unknown location" in result.error
    assert not result.found


# ---- N5: BFS returns minimum hops ----------------------------------------------------

def test_bfs_minimizes_hops(graph):
    result = bfs.bfs_shortest_path(graph, "entrance", "cafe")
    assert result.path is not None
    # known topology: entrance->main_hall->reception->cafe = 3 hops
    assert result.hops == 3
    assert len(result.path) == 4


# ---- N6: A* returns minimum weighted cost ----------------------------------------------

def test_astar_minimizes_cost(graph):
    result = astar.astar_shortest_path(
        graph, "entrance", "library", euclidean_heuristic(graph, "library")
    )
    assert result.path is not None
    assert result.cost > 0
    # brute-force check: no simpler way than comparing to the best found
    # by exhaustive exploration of this small graph
    assert result.cost == _min_cost_exhaustive(graph, "entrance", "library")


def _min_cost_exhaustive(graph, start, goal):
    best = float("inf")
    stack = [(start, 0.0, {start})]
    while stack:
        node, cost, seen = stack.pop()
        if node == goal:
            best = min(best, cost)
            continue
        for n, w in graph.neighbours(node):
            if n not in seen:
                stack.append((n, cost + w, seen | {n}))
    return best


# ---- N7: BFS and A* agree on optimal weighted cost --------------------------------------

def test_bfs_astar_agree_on_cost(graph):
    # BFS path cost >= A* cost; A* (admissible) must equal the optimum,
    # and BFS cost is the cost of its (possibly longer) route.
    b = bfs.bfs_shortest_path(graph, "entrance", "science_lab")
    a = astar.astar_shortest_path(
        graph, "science_lab", "entrance", euclidean_heuristic(graph, "entrance")
    )
    assert b.path is not None and a.path is not None
    optimum = _min_cost_exhaustive(graph, "entrance", "science_lab")
    assert a.cost == pytest.approx(optimum)
    assert b.cost >= a.cost


def test_bfs_astar_same_when_symmetric(graph):
    # On routes where hop-count and weight optima coincide, both agree
    b = bfs.bfs_shortest_path(graph, "entrance", "reception")
    a = astar.astar_shortest_path(
        graph, "entrance", "reception", euclidean_heuristic(graph, "reception")
    )
    assert b.path is not None and a.path is not None
    assert b.cost == pytest.approx(a.cost)


# ---- N8: explored counts and timing reported --------------------------------------------

def test_explored_counts_and_timing_reported(graph):
    b = bfs.bfs_shortest_path(graph, "entrance", "library")
    a = astar.astar_shortest_path(
        graph, "entrance", "library", euclidean_heuristic(graph, "library")
    )
    assert len(b.explored_order) >= len(a.explored_order)  # A* explores <= BFS
    assert b.explored_order and a.explored_order

    rb, ra = route_service.compare_routes(graph, "entrance", "library")
    assert rb.time_ms >= 0 and ra.time_ms >= 0
    assert rb.explored_count > 0 and ra.explored_count > 0
    assert rb.found and ra.found


# ---- graph data validation ----------------------------------------------------------------

def test_graph_loads_sample_map(graph):
    assert "entrance" in graph.nodes
    assert graph.display_name("entrance") == "Main Entrance"
    assert len(graph.nodes) == 15
    # undirected edges are stored both ways
    assert any(n == "main_hall" for n, _ in graph.neighbours("entrance"))


def test_cached_graph_is_shared():
    g1 = route_service.get_graph(DEFAULT_MAP_PATH)
    g2 = route_service.get_graph(DEFAULT_MAP_PATH)
    assert g1 is g2

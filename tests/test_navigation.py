"""Navigation demo tests: map schema, algorithms, profiles, and instructions."""

from types import SimpleNamespace

import pytest

from modules.navigation import (
    astar,
    bfs,
    instructions,
    profiles,
    route_service,
    scenarios,
)
from modules.navigation.graph_data import (
    DEFAULT_MAP_PATH,
    euclidean_heuristic,
    floor_label,
    load_graph,
)
from ui import map_view

MAP = "assets/sample_maps/sample_building.txt"


@pytest.fixture(scope="module")
def graph():
    return load_graph(MAP)


@pytest.fixture(scope="module")
def scenario_map(graph):
    return scenarios.all_scenarios(graph)


# ---- map schema -----------------------------------------------------------------


def test_graph_loads_sample_map(graph):
    assert "entrance" in graph.nodes
    assert graph.display_name("entrance") == "Main Entrance"
    assert len(graph.nodes) == 29
    assert len(graph.edges) == 39
    assert graph.floors() == [0, 1]
    assert graph.kind("entrance") == "entrance"
    assert graph.kind("lift_g") == "lift"
    assert graph.floor("meeting_room") == 1
    assert "accessible_toilet" in graph.tags("accessible_toilet_g")
    assert floor_label(0) == "Ground floor"
    assert floor_label(1) == "Upper floor"

    # undirected edges are stored both ways
    assert any(n == "main_hall" for n, _ in graph.neighbours("entrance"))
    assert graph.edge_attrs("main_hall", "entrance") == graph.edge_attrs(
        "entrance", "main_hall"
    )


def test_vertical_edges_are_lift_or_stairs(graph):
    for a, b, attrs in graph.edge_items():
        if graph.floor(a) != graph.floor(b):
            assert attrs["type"] in {"lift", "stairs"}


def test_same_floor_weights_are_admissible(graph):
    for a, b, attrs in graph.edge_items():
        if graph.floor(a) != graph.floor(b):
            continue
        ax, ay = graph.coords(a)
        bx, by = graph.coords(b)
        straight = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        assert attrs["weight"] >= straight - 1e-9


def test_cached_graph_is_shared():
    g1 = route_service.get_graph(DEFAULT_MAP_PATH)
    g2 = route_service.get_graph(DEFAULT_MAP_PATH)
    assert g1 is g2


def test_map_validation_rejects_bad_edges(tmp_path):
    header = "node a | A | 0 | 0 | 0\nnode b | B | 1 | 0 | 0\n"

    bad_unknown = tmp_path / "unknown.txt"
    bad_unknown.write_text(header + "edge a c | 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown node"):
        load_graph(str(bad_unknown))

    bad_weight = tmp_path / "weight.txt"
    bad_weight.write_text(header + "edge a b | 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        load_graph(str(bad_weight))

    bad_floor = tmp_path / "floor.txt"
    bad_floor.write_text(
        "node a | A | 0 | 0 | 0\nnode b | B | 1 | 0 | 1\nedge a b | 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lift or stairs"):
        load_graph(str(bad_floor))

    bad_crowd = tmp_path / "crowd.txt"
    bad_crowd.write_text(header + "edge a b | 1 | corridor | 2 | 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="crowd"):
        load_graph(str(bad_crowd))


# ---- algorithms ------------------------------------------------------------------


def test_bfs_finds_valid_route(graph):
    result = bfs.bfs_shortest_path(graph, "entrance", "library")
    assert result.path is not None
    assert result.path[0] == "entrance" and result.path[-1] == "library"
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


def test_unreachable_goal_explicit(graph):
    result = bfs.bfs_shortest_path(graph, "entrance", "annex_office")
    assert result.path is None
    assert result.explored_order

    result2 = astar.astar_shortest_path(
        graph, "entrance", "annex_lobby", euclidean_heuristic(graph, "annex_lobby")
    )
    assert result2.path is None


def test_start_equals_goal_trivial_route(graph):
    for algo in (bfs.bfs_shortest_path, astar.astar_shortest_path):
        result = algo(graph, "main_hall", "main_hall")
        assert result.path == ["main_hall"]
        assert result.hops == 0
        assert result.cost == 0.0


def test_invalid_node_raises(graph):
    with pytest.raises(ValueError):
        bfs.bfs_shortest_path(graph, "entrance", "nonexistent_room")
    with pytest.raises(ValueError):
        astar.astar_shortest_path(graph, "nonsense", "library")


def test_route_service_converts_invalid_to_error(graph):
    result = route_service.run_route(graph, "entrance", "bad_room", "A*")
    assert result.error is not None and "Unknown location" in result.error
    assert not result.found


def test_bfs_minimizes_hops(graph):
    result = bfs.bfs_shortest_path(graph, "entrance", "cafe")
    assert result.path is not None
    # known topology: entrance -> main_hall -> reception -> cafe
    assert result.hops == 3
    assert len(result.path) == 4


def test_astar_minimizes_cost(graph):
    result = astar.astar_shortest_path(
        graph, "entrance", "library", euclidean_heuristic(graph, "library")
    )
    assert result.path is not None
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


def test_bfs_and_astar_agree_on_optimum(graph):
    b = bfs.bfs_shortest_path(graph, "entrance", "science_lab")
    a = astar.astar_shortest_path(
        graph, "science_lab", "entrance", euclidean_heuristic(graph, "entrance")
    )
    assert b.path is not None and a.path is not None
    assert b.cost >= a.cost
    assert a.cost == pytest.approx(_min_cost_exhaustive(graph, "entrance", "science_lab"))


def test_allowed_filter_reroutes_around_closure(graph):
    normal = bfs.bfs_shortest_path(graph, "entrance", "cafe")
    allowed, _ = profiles.make_edge_rules(
        graph, "fastest", blocked_nodes={"main_hall"}
    )
    closed = bfs.bfs_shortest_path(graph, "entrance", "cafe", allowed=allowed)
    assert normal.path is not None and closed.path is not None
    assert "main_hall" not in closed.path
    assert closed.hops > normal.hops or closed.cost > normal.cost


def test_explored_counts_and_timing_reported(graph):
    b = bfs.bfs_shortest_path(graph, "entrance", "library")
    a = astar.astar_shortest_path(
        graph, "entrance", "library", euclidean_heuristic(graph, "library")
    )
    assert b.explored_order and a.explored_order
    assert len(a.explored_order) <= len(b.explored_order)

    rb, ra = route_service.compare_routes(graph, "entrance", "library")
    assert rb.time_ms >= 0 and ra.time_ms >= 0
    assert rb.explored_count > 0 and ra.explored_count > 0
    assert rb.found and ra.found


# ---- profiles and scenarios ---------------------------------------------------------


def test_profile_metadata_complete():
    assert set(profiles.PROFILE_ORDER) == set(profiles.PROFILES)
    for key in profiles.PROFILE_ORDER:
        assert profiles.PROFILES[key].label
        assert profiles.PROFILES[key].description


def test_step_free_route_excludes_stairs(graph):
    result = route_service.plan_route(
        graph, "entrance", "meeting_room", profile_key="step_free"
    )
    assert result.found
    assert result.stairs_count == 0
    assert result.lift_count == 1
    assert all(
        (graph.edge_attrs(a, b) or {}).get("type") != "stairs"
        for a, b in zip(result.path, result.path[1:])
    )
    assert result.detour_m > 0
    assert "lift" in result.explanation.lower()


def test_fastest_route_can_use_stairs(graph):
    result = route_service.plan_route(
        graph, "entrance", "meeting_room", profile_key="fastest"
    )
    assert result.found
    assert result.stairs_count == 1
    assert result.lift_count == 0


def test_quiet_profile_penalises_crowds(graph):
    fastest = route_service.plan_route(graph, "entrance", "cafe")
    quiet = route_service.plan_route(graph, "entrance", "cafe", profile_key="quiet")
    assert fastest.found and quiet.found
    assert quiet.cost > fastest.cost
    assert "crowd" in quiet.explanation.lower()


def test_lift_out_makes_step_free_unreachable(graph, scenario_map):
    lift_out = scenario_map["lift_out"]
    result = route_service.plan_route(
        graph,
        "entrance",
        "meeting_room",
        profile_key="step_free",
        blocked_edges=lift_out.blocked_edges,
    )
    assert not result.found
    assert result.unreachable
    assert "lift is out of service" in result.unreachable_reason


def test_lift_out_keeps_fastest_route(graph, scenario_map):
    result = route_service.plan_route(
        graph,
        "entrance",
        "meeting_room",
        profile_key="fastest",
        blocked_edges=scenario_map["lift_out"].blocked_edges,
    )
    assert result.found
    assert result.stairs_count == 1


def test_stairs_closed_keeps_step_free_route(graph, scenario_map):
    result = route_service.plan_route(
        graph,
        "entrance",
        "meeting_room",
        profile_key="step_free",
        blocked_edges=scenario_map["stairs_closed"].blocked_edges,
    )
    assert result.found
    assert result.stairs_count == 0
    assert result.lift_count == 1


def test_node_closure_reroutes_route(graph, scenario_map):
    result = route_service.plan_route(
        graph,
        "entrance",
        "cafe",
        blocked_nodes=scenario_map["main_hall_closed"].blocked_nodes,
    )
    assert result.found
    assert "main_hall" not in result.path
    assert "closure" in result.explanation.lower()


def test_closed_start_or_goal_is_an_error(graph):
    result = route_service.plan_route(
        graph, "main_hall", "cafe", blocked_nodes={"main_hall"}
    )
    assert result.error and "Start location" in result.error
    result = route_service.plan_route(
        graph, "entrance", "main_hall", blocked_nodes={"main_hall"}
    )
    assert result.error and "Destination" in result.error


def test_bfs_rejects_weighted_profile(graph):
    result = route_service.plan_route(
        graph, "entrance", "meeting_room", profile_key="step_free", algorithm="BFS"
    )
    assert not result.found
    assert result.error and "BFS ignores edge weights" in result.error


def test_normalize_blocked_edges(graph):
    normalized = profiles.normalize_blocked_edges(
        [("main_hall", "entrance"), ("entrance", "main_hall"), ("bad",), None]
    )
    assert normalized == frozenset({("entrance", "main_hall")})


def test_blocked_summary_names_locations(graph):
    labels = profiles.blocked_summary(
        graph, [("entrance", "main_hall")], ["corridor_c"]
    )
    assert any("Main Entrance" in label for label in labels)
    assert any("Corridor C" in label for label in labels)


# ---- route result and instructions ---------------------------------------------------


def test_route_result_reports_profile_details(graph):
    result = route_service.plan_route(
        graph, "entrance", "meeting_room", profile_key="step_free"
    )
    assert result.path_display[0] == "Main Entrance"
    assert result.path_display[-1] == "Meeting Room"
    assert result.hops == len(result.path) - 1
    assert result.eta_s > 0
    assert result.min_width_m is not None
    assert result.constraints == ["Stairs excluded"]
    assert result.explored_ids
    assert len(result.explored_ids) == result.explored_count


def test_unreachable_annex_has_explanation(graph):
    result = route_service.plan_route(graph, "entrance", "annex_office")
    assert not result.found
    assert result.unreachable
    assert result.unreachable_reason
    assert result.explanation == result.unreachable_reason


def test_build_steps_covers_route(graph):
    result = route_service.plan_route(graph, "entrance", "meeting_room")
    steps = instructions.build_steps(graph, result.path, 1.1)
    assert len(steps) == len(result.path)
    assert steps[0].action == "start"
    assert steps[0].instruction.startswith("Start at Main Entrance")
    assert steps[-1].action == "arrive"
    assert steps[-1].remaining_m == 0
    assert steps[-1].cumulative_m == pytest.approx(result.cost)
    assert steps[-1].eta_s == result.eta_s


def test_floor_change_step_mentions_lift(graph):
    result = route_service.plan_route(
        graph, "entrance", "meeting_room", profile_key="step_free"
    )
    steps = instructions.build_steps(graph, result.path, 1.1)
    lift_step = next(step for step in steps if step.action == "lift")
    assert "lift" in lift_step.instruction.lower()
    assert "upper floor" in lift_step.instruction.lower()


def test_single_node_route_is_arrival():
    graph = load_graph(MAP)
    steps = instructions.build_steps(graph, ["main_hall"], 1.1)
    assert len(steps) == 1
    assert steps[0].action == "arrive"
    assert steps[0].remaining_m == 0


def test_format_helpers():
    assert instructions.format_distance(3.7) == "3.7 m"
    assert instructions.format_distance(12.2) == "12 m"
    assert instructions.format_duration(45) == "45 s"
    assert instructions.format_duration(60) == "1 min"
    assert instructions.format_duration(90) == "1 min 30 s"


def test_eta_includes_stair_penalty(graph):
    result = route_service.plan_route(graph, "entrance", "meeting_room")
    walking_only = result.cost / 1.1
    assert result.eta_s > walking_only


def test_invalid_speed_raises(graph):
    with pytest.raises(ValueError):
        instructions.build_steps(graph, ["entrance", "main_hall"], 0)


def test_route_speech_summary_is_honest(graph):
    result = route_service.plan_route(
        graph, "entrance", "meeting_room", profile_key="step_free"
    )
    summary = instructions.route_speech_summary(
        graph, result.path, stairs=0, lifts=1
    )
    assert "Simulated route" in summary
    assert "No stairs on this route" in summary
    assert "lift" in summary.lower()


# ---- map figure ------------------------------------------------------------------------


def test_map_figure_has_two_floors_and_route(graph):
    result = route_service.plan_route(graph, "entrance", "meeting_room")
    figure = map_view.build_route_figure(
        graph,
        path=result.path,
        start="entrance",
        goal="meeting_room",
        current="stairs_u",
    )
    names = [trace.name for trace in figure.data]
    assert "Route" in names
    assert "Current step" in names
    assert len(figure.layout.annotations) >= 2


def test_map_figure_shows_closure_handles_only_when_requested(graph):
    without = map_view.build_route_figure(graph, start="entrance", goal="cafe")
    with_handles = map_view.build_route_figure(
        graph,
        start="entrance",
        goal="cafe",
        blocked_edges=[("main_hall", "reception")],
        show_closures=True,
    )
    assert "Click to close / reopen" not in [t.name for t in without.data]
    assert "Click to close / reopen" in [t.name for t in with_handles.data]


def test_decode_selection_accepts_streamlit_dict_and_object():
    event = {"selection": {"points": [{"customdata": ["node", "entrance"]}]}}
    assert map_view.decode_selection(event) == ("node", "entrance")

    event_object = SimpleNamespace(
        selection=SimpleNamespace(
            points=[SimpleNamespace(customdata=["edge", "entrance", "main_hall"])]
        )
    )
    assert map_view.decode_selection(event_object) == (
        "edge",
        "entrance",
        "main_hall",
    )
    assert map_view.decode_selection({"selection": {"points": []}}) is None


def test_label_scope_controls_map_annotations(graph):
    result = route_service.plan_route(graph, "entrance", "staff_room")
    names = {graph.display_name(node) for node in graph.ids()}

    def label_count(scope):
        figure = map_view.build_route_figure(
            graph,
            path=result.path,
            start="entrance",
            goal="staff_room",
            label_scope=scope,
        )
        return sum(1 for a in figure.layout.annotations if a.text in names)

    assert label_count("all") == len(graph.nodes)
    assert label_count("route") == len(set(result.path))
    assert label_count("none") == 0
    with pytest.raises(ValueError):
        map_view.build_route_figure(graph, label_scope="bad")


def test_label_placement_has_no_overlaps(graph):
    xs = [graph.coords(node)[0] for node in graph.ids()]
    ys = [graph.coords(node)[1] for node in graph.ids()]
    x_range = (min(xs) - 4.0, max(xs) + 4.0)
    y_range = (min(ys) - 3.0, max(ys) + 3.0)
    width_px, height_px = 620.0, 285.0
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]

    for floor in graph.floors():
        nodes = [node for node in graph.ids() if graph.floor(node) == floor]
        positions = map_view._place_labels(
            graph, nodes, set(nodes), x_range, y_range,
            width_px=width_px, height_px=height_px,
        )
        boxes = []
        for node, (lx, ly, _dx, _dy) in positions.items():
            text = graph.display_name(node)
            width = min(
                x_span * 0.42,
                max(1.4, (len(text) * 5.1 + 8.0) / (width_px / x_span)),
            )
            height = max(0.75, 14.0 / (height_px / y_span))
            boxes.append(
                (lx - width / 2, ly - height / 2, lx + width / 2, ly + height / 2)
            )
        for index, first in enumerate(boxes):
            for second in boxes[index + 1:]:
                assert map_view._box_overlap(first, second) == 0


def test_bfs_and_astar_paths_can_differ_on_maps(graph):
    bfs_result = route_service.plan_route(
        graph,
        "entrance",
        "staff_room",
        algorithm="BFS",
        include_baseline=False,
    )
    astar_result = route_service.plan_route(
        graph,
        "entrance",
        "staff_room",
        algorithm="A*",
        include_baseline=False,
    )
    assert bfs_result.found and astar_result.found
    assert bfs_result.path != astar_result.path
    assert bfs_result.cost >= astar_result.cost

    bfs_figure = map_view.build_route_figure(
        graph,
        path=bfs_result.path,
        start="entrance",
        goal="staff_room",
        label_scope="route",
        route_name="BFS path",
        route_color="#B45309",
    )
    astar_figure = map_view.build_route_figure(
        graph,
        path=astar_result.path,
        start="entrance",
        goal="staff_room",
        label_scope="route",
        route_name="A* path",
        route_color=map_view.ROUTE,
    )
    assert "BFS path" in [trace.name for trace in bfs_figure.data]
    assert "A* path" in [trace.name for trace in astar_figure.data]

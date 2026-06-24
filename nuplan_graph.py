"""NuPlan lane-graph utilities: subgraph loading and NetworkX visualization.

Public API
----------
load_nuplan_subgraph(nuplan_map, ego_point, radius, depth_limit)
    Build a BFS-bounded directed lane/connector subgraph around ego.
    Returns (small_G, object_map, object_types, ego_id).

nuplan_pos(graph, object_map)
    Compute centroid-based {node_id: (x, y)} positions for NetworkX drawing.

draw_nuplan_subgraph(draw_G, pos, object_types, ego_id, ...)
    Draw lane/connector nodes with default color coding. Returns IPython.display.SVG.

draw_nuplan_path(draw_G, pos, object_types, ego_id, path, ...)
    Draw the subgraph with a highlighted witness path. Returns IPython.display.SVG.
"""

import io
import math

import IPython.display as display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

# ── node/edge palette ─────────────────────────────────────────────────────────
_FC = {
    'lane':      ('#90caf9', '#1565c0'),
    'connector': ('#ffccbc', '#bf360c'),
    'ego':       ('#ffeb3b', '#f57f17'),
}


# ── subgraph loading ──────────────────────────────────────────────────────────

def load_nuplan_subgraph(nuplan_map, ego_point, radius: float, depth_limit: int):
    """Build a BFS-bounded directed lane subgraph around *ego_point*.

    Parameters
    ----------
    nuplan_map:   NuPlanMap instance (from NuPlanMapFactory.build_map_from_name)
    ego_point:    Point2D of the ego vehicle's rear axle
    radius:       map-query radius in metres
    depth_limit:  BFS depth from the ego lane/connector

    Returns
    -------
    small_G      : nx.DiGraph – the BFS-bounded subgraph
    object_map   : dict[str, map_obj] – node_id → NuPlan map object
    object_types : dict[str, str]     – node_id → 'lane' | 'connector'
    ego_id       : str | None         – the node containing the ego vehicle
    """
    direction_layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    map_objects = nuplan_map.get_proximal_map_objects(ego_point, radius, direction_layers)

    object_map: dict = {}
    object_types: dict = {}
    G_full = nx.DiGraph()

    for layer in direction_layers:
        for obj in map_objects.get(layer, []):
            try:
                oid = str(obj.id)
                object_map[oid] = obj
                object_types[oid] = 'lane' if layer == SemanticMapLayer.LANE else 'connector'
                G_full.add_node(oid, type=object_types[oid])
            except Exception:
                continue

    for oid, obj in object_map.items():
        for out in getattr(obj, 'outgoing_edges', []):
            out_id = str(out.id)
            if out_id in G_full:
                G_full.add_edge(oid, out_id)

    ego_obj = (
        nuplan_map.get_one_map_object(ego_point, SemanticMapLayer.LANE)
        or nuplan_map.get_one_map_object(ego_point, SemanticMapLayer.LANE_CONNECTOR)
    )
    ego_id = str(ego_obj.id) if ego_obj else None

    sub_nodes: set = set()
    if ego_id and ego_id in G_full:
        sub_nodes.update(nx.bfs_tree(G_full, ego_id, depth_limit=depth_limit).nodes())

    small_G = G_full.subgraph(sub_nodes).copy()
    return small_G, object_map, object_types, ego_id


# ── position helper ───────────────────────────────────────────────────────────

def nuplan_pos(graph, object_map) -> dict:
    """Return centroid-based ``{node_id: (x, y)}`` positions for NetworkX drawing.

    Nodes whose geometry centroid is non-finite are silently skipped; callers
    should restrict drawing to ``graph.subgraph(pos.keys()).copy()``.
    """
    pos = {}
    for node in graph.nodes():
        if node not in object_map:
            continue
        obj = object_map[node]
        geom = obj.polygon if hasattr(obj, 'polygon') else obj.geometry
        c = geom.centroid
        if math.isfinite(c.x) and math.isfinite(c.y):
            pos[node] = (c.x, c.y)
    return pos


# ── internal helpers ──────────────────────────────────────────────────────────

def _set_equal_limits(ax, pos, margin: float = 1.15) -> None:
    if not pos:
        return
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmid = (min(xs) + max(xs)) / 2
    ymid = (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * margin
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)


def _fig_to_svg(fig) -> display.SVG:
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return display.SVG(buf.getvalue())


# ── drawing functions ─────────────────────────────────────────────────────────

def draw_nuplan_subgraph(
    draw_G,
    pos: dict,
    object_types: dict,
    ego_id,
    *,
    title: str = 'NuPlan subgraph — ego neighbourhood',
    figsize: tuple = (10, 8),
    node_size: int = 600,
    arrowsize: int = 20,
    font_size: int = 7,
) -> display.SVG:
    """Draw a NuPlan lane subgraph with matplotlib/NetworkX.

    Lane nodes are blue, connector nodes are orange, the ego node is yellow.
    Returns an ``IPython.display.SVG`` object — use as the last expression in a
    notebook cell (or pass to ``display.display()``) to render it inline.
    """
    fig, ax = plt.subplots(figsize=figsize)

    lane_nodes = [n for n in draw_G if object_types.get(n) == 'lane']
    connector_nodes = [n for n in draw_G if object_types.get(n) == 'connector']

    nx.draw_networkx_nodes(draw_G, pos, nodelist=lane_nodes,
                           node_color=_FC['lane'][0], edgecolors=_FC['lane'][1],
                           node_size=node_size, linewidths=1.5, ax=ax)
    nx.draw_networkx_nodes(draw_G, pos, nodelist=connector_nodes,
                           node_color=_FC['connector'][0], edgecolors=_FC['connector'][1],
                           node_size=node_size, linewidths=1.5, ax=ax)
    if ego_id and ego_id in draw_G:
        nx.draw_networkx_nodes(draw_G, pos, nodelist=[ego_id],
                               node_color=_FC['ego'][0], edgecolors=_FC['ego'][1],
                               node_size=int(node_size * 1.17), linewidths=2.0, ax=ax)

    nx.draw_networkx_edges(draw_G, pos, edge_color='#546e7a', arrows=True,
                           arrowsize=arrowsize, arrowstyle='->', width=1.5,
                           connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_labels(draw_G, pos, font_size=font_size, font_family='monospace', ax=ax)

    legend = [
        mpatches.Patch(facecolor=_FC['lane'][0],      edgecolor=_FC['lane'][1],      label='Lane'),
        mpatches.Patch(facecolor=_FC['connector'][0], edgecolor=_FC['connector'][1], label='Lane connector'),
        mpatches.Patch(facecolor=_FC['ego'][0],       edgecolor=_FC['ego'][1],       label='Ego'),
    ]
    ax.legend(handles=legend, loc='upper left', fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    _set_equal_limits(ax, pos)
    ax.grid(True, alpha=0.3)

    return _fig_to_svg(fig)


def draw_nuplan_path(
    draw_G,
    pos: dict,
    object_types: dict,
    ego_id,
    path,
    *,
    destination_node=None,
    title: str | None = None,
    goal_label: str | None = None,
    figsize: tuple = (10, 8),
    node_size: int = 650,
    arrowsize: int = 20,
    font_size: int = 7,
) -> display.SVG | None:
    """Draw a NuPlan subgraph with a highlighted witness path.

    Parameters
    ----------
    path             : list of node IDs in traversal order (the lasso stem)
    destination_node : node to colour green; defaults to ``path[-1]``
    goal_label       : legend label for the destination node

    Returns
    -------
    ``IPython.display.SVG`` — use as the last expression in a notebook cell.
    Returns ``None`` and prints a message when *path* is ``None``.
    """
    if path is None:
        print(f'No satisfying path — {title or "UNSAT"}')
        return None

    if destination_node is None:
        destination_node = path[-1] if path else None

    path_set = set(path)
    path_edges_set = set(zip(path, path[1:]))
    _goal_label = goal_label or (str(destination_node) if destination_node else 'goal')

    node_colors = []
    for n in draw_G.nodes():
        if n == destination_node:
            node_colors.append('#43a047')   # green  — goal
        elif n in path_set:
            node_colors.append('#ff7043')   # orange — on path
        elif n == ego_id:
            node_colors.append('#ffeb3b')   # yellow — ego
        elif object_types.get(n) == 'connector':
            node_colors.append('#ffccbc')
        else:
            node_colors.append('#90caf9')

    ec = ['#e53935' if (u, v) in path_edges_set else '#546e7a' for u, v in draw_G.edges()]
    ew = [3.5        if (u, v) in path_edges_set else 1.5       for u, v in draw_G.edges()]

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(draw_G, pos, node_color=node_colors,
                           edgecolors='#37474f', node_size=node_size, linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(draw_G, pos, edge_color=ec, width=ew,
                           arrows=True, arrowsize=arrowsize, arrowstyle='->',
                           connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_labels(draw_G, pos, font_size=font_size, font_family='monospace', ax=ax)

    legend = [
        mpatches.Patch(facecolor='#90caf9', edgecolor='#37474f', label='Lane'),
        mpatches.Patch(facecolor='#ffccbc', edgecolor='#37474f', label='Lane connector'),
        mpatches.Patch(facecolor='#ffeb3b', edgecolor='#37474f', label='Ego (initial)'),
        mpatches.Patch(facecolor='#ff7043', edgecolor='#37474f', label='Path'),
        mpatches.Patch(facecolor='#43a047', edgecolor='#37474f', label=f'Goal  [{_goal_label}]'),
    ]
    ax.legend(handles=legend, loc='upper left', fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    _set_equal_limits(ax, pos)
    ax.grid(True, alpha=0.3)

    return _fig_to_svg(fig)

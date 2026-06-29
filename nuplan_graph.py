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

animate_nuplan_path(draw_G, pos, object_types, ego_id, path, window=..., ...)
    Animate a node-by-node traversal of the path with a sliding highlight window.
    Returns an inline IPython display (HTML5/JS) or writes a GIF when output_path is given.
"""

import io
import math

import IPython.display as display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation
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
    highlight_nodes: dict | None = None,
    title: str | None = None,
    goal_label: str | None = None,
    figsize: tuple = (10, 8),
    node_size: int = 650,
    offpath_node_size: int | None = None,
    arrowsize: int = 20,
    font_size: int = 7,
) -> display.SVG | None:
    """Draw a NuPlan subgraph with a highlighted witness path.

    Parameters
    ----------
    path              : list of node IDs in traversal order (the lasso stem)
    destination_node  : node to colour green; defaults to ``path[-1]``
    highlight_nodes   : optional ``{node_id: (facecolor, legend_label)}`` for extra
                        nodes to colour distinctly (e.g. a waypoint in purple). These
                        win over the path/ego/type colours but not over the green goal.
    goal_label        : legend label for the destination node
    node_size         : size of nodes on the path (ego and destination included)
    offpath_node_size : size of nodes *not* on the path; defaults to ``node_size``
                        (set smaller, e.g. 80–150, to make the path pop on large graphs)

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

    highlight_nodes = highlight_nodes or {}
    path_set = set(path)
    path_edges_set = set(zip(path, path[1:]))
    _goal_label = goal_label or (str(destination_node) if destination_node else 'goal')
    _offpath_sz = offpath_node_size if offpath_node_size is not None else node_size

    node_colors = []
    node_sizes  = []
    for n in draw_G.nodes():
        on_path = (n == destination_node or n in path_set or n == ego_id or n in highlight_nodes)
        node_sizes.append(node_size if on_path else _offpath_sz)
        if n == destination_node:
            node_colors.append('#43a047')         # green  — goal
        elif n in highlight_nodes:
            node_colors.append(highlight_nodes[n][0])  # custom — e.g. waypoint
        elif n == ego_id:
            node_colors.append('#ffeb3b')         # yellow — ego
        elif n in path_set:
            node_colors.append('#ff7043')         # orange — on path
        elif object_types.get(n) == 'connector':
            node_colors.append('#ffccbc')
        else:
            node_colors.append('#90caf9')

    ec = ['#e53935' if (u, v) in path_edges_set else '#546e7a' for u, v in draw_G.edges()]
    ew = [3.5        if (u, v) in path_edges_set else 1.5       for u, v in draw_G.edges()]

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(draw_G, pos, node_color=node_colors,
                           edgecolors='#37474f', node_size=node_sizes, linewidths=1.5, ax=ax)
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
    # Append extra highlighted-node legend entries (deduplicated by label).
    _seen_labels = set()
    for _color, _label in highlight_nodes.values():
        if _label not in _seen_labels:
            legend.append(mpatches.Patch(facecolor=_color, edgecolor='#37474f', label=_label))
            _seen_labels.add(_label)
    ax.legend(handles=legend, loc='upper left', fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    _set_equal_limits(ax, pos)
    ax.grid(True, alpha=0.3)

    return _fig_to_svg(fig)


def draw_nuplan_nodes(
    draw_G,
    pos: dict,
    object_types: dict,
    ego_id,
    highlight_nodes: dict,
    *,
    title: str | None = None,
    figsize: tuple = (14, 11),
    base_node_size: int = 30,
    highlight_node_size: int = 650,
    arrowsize: int = 8,
    font_size: int = 8,
) -> display.SVG:
    """Draw the whole NuPlan directed graph with selected nodes highlighted.

    Unlike :func:`draw_nuplan_path`, no path is assumed — every node is drawn as a
    small grey dot with faint edges, and only the nodes in *highlight_nodes* (plus
    the ego) are enlarged, coloured, and labelled. This is the natural fallback for
    an **infeasible** specification: it shows *where* the (unconnected) nodes of
    interest sit in the overall graph.

    Parameters
    ----------
    highlight_nodes : ``{node_id: (facecolor, legend_label)}``

    Returns an ``IPython.display.SVG`` (use as the last cell expression).
    """
    highlight_nodes = highlight_nodes or {}

    node_colors, node_sizes = [], []
    for n in draw_G.nodes():
        if n in highlight_nodes:
            node_colors.append(highlight_nodes[n][0])
            node_sizes.append(highlight_node_size)
        elif n == ego_id:
            node_colors.append('#ffeb3b')        # yellow — ego
            node_sizes.append(highlight_node_size)
        else:
            node_colors.append('#cfd8dc')        # light grey — context
            node_sizes.append(base_node_size)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(draw_G, pos, node_color=node_colors,
                           edgecolors='#90a4ae', node_size=node_sizes, linewidths=0.7, ax=ax)
    nx.draw_networkx_edges(draw_G, pos, edge_color='#b0bec5', width=0.7,
                           arrows=True, arrowsize=arrowsize, arrowstyle='->',
                           connectionstyle='arc3,rad=0.1', alpha=0.5, ax=ax)

    # Label only the highlighted nodes and the ego, to keep the dense graph readable.
    label_nodes = {n: n for n in draw_G.nodes() if n in highlight_nodes or n == ego_id}
    nx.draw_networkx_labels(draw_G, pos, labels=label_nodes,
                            font_size=font_size, font_family='monospace', ax=ax)

    legend = [mpatches.Patch(facecolor='#cfd8dc', edgecolor='#90a4ae', label='Other nodes')]
    if ego_id in draw_G and ego_id not in highlight_nodes:
        legend.append(mpatches.Patch(facecolor='#ffeb3b', edgecolor='#37474f', label='Ego (initial)'))
    _seen = set()
    for _color, _label in highlight_nodes.values():
        if _label not in _seen:
            legend.append(mpatches.Patch(facecolor=_color, edgecolor='#37474f', label=_label))
            _seen.add(_label)
    ax.legend(handles=legend, loc='upper left', fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    _set_equal_limits(ax, pos)
    ax.grid(True, alpha=0.3)

    return _fig_to_svg(fig)


# ── animation ─────────────────────────────────────────────────────────────────

# Colours for the traversal animation (head + fading trail).
_ANIM_HEAD_COLOR = '#d50000'    # crimson — the node the ego is currently at
_ANIM_TRAIL_COLOR = '#ff7043'   # orange  — recently-visited nodes still in the window


def _xy_limits(pos: dict, margin: float = 1.15):
    """Fixed (xlim, ylim) for the whole animation so the view never jumps."""
    if not pos:
        return (0, 1), (0, 1)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmid = (min(xs) + max(xs)) / 2
    ymid = (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6) / 2 * margin
    return (xmid - half, xmid + half), (ymid - half, ymid + half)


def animate_nuplan_path(
    draw_G,
    pos: dict,
    object_types: dict,
    ego_id,
    path,
    *,
    window: int = 1,
    destination_node=None,
    highlight_nodes: dict | None = None,
    title: str | None = None,
    goal_label: str | None = None,
    figsize: tuple = (10, 8),
    node_size: int = 650,
    offpath_node_size: int | None = None,
    arrowsize: int = 20,
    font_size: int = 7,
    fps: float = 2.0,
    output_path: str | None = None,
    show_labels: bool = True,
    label_window_only: bool = False,
    repeat: bool = True,
):
    """Animate a node-by-node traversal of *path* with a sliding highlight window.

    Frame ``i`` advances the *head* (the node the ego currently occupies) to
    ``path[i]`` and lights the trailing *window* — the last ``window`` visited
    nodes — as a fading comet:

    - the head ``path[i]`` is drawn crimson and enlarged;
    - the ``window - 1`` nodes behind it form an orange trail, fading with age;
    - nodes that have dropped out of the window revert to their base style, so
      the highlighted band visibly *slides* along the route;
    - the destination stays green and the ego node yellow throughout, as in
      :func:`draw_nuplan_path`.

    Parameters
    ----------
    path              : list of node IDs in traversal order (the lasso stem; a
                        repeated tail node, e.g. a destination self-loop, simply
                        keeps the head parked there for those frames).
    window            : number of trailing nodes kept highlighted (``>= 1``).
                        ``1`` shows only the current node; a large value keeps the
                        whole travelled prefix lit.
    destination_node  : node to colour green; defaults to ``path[-1]``.
    highlight_nodes   : optional ``{node_id: (facecolor, legend_label)}`` for extra
                        fixed markers (e.g. a waypoint in purple).
    fps               : frames per second for both the inline player and any GIF.
    output_path       : when given, also write the animation to this file. A
                        ``.gif`` extension uses Pillow; ``.mp4`` is used if ffmpeg
                        is available, otherwise it falls back to a sibling ``.gif``.
    show_labels       : draw node-ID labels.
    label_window_only : when True (and *show_labels*), label only the head/trail
                        nodes each frame — keeps dense graphs readable.
    repeat            : loop the animation in the inline player.

    Returns
    -------
    An ``IPython.display`` object that renders inline in a notebook cell:
    ``HTML`` (HTML5/JS player) by default, or ``Image``/``Video`` when a file was
    written. Returns ``None`` and prints a message when *path* is empty/None.
    """
    if not path:
        print(f'No path to animate — {title or ""}')
        return None
    if window < 1:
        window = 1

    if destination_node is None:
        destination_node = path[-1]
    highlight_nodes = highlight_nodes or {}
    _goal_label = goal_label or (str(destination_node) if destination_node else 'goal')
    _offpath_sz = offpath_node_size if offpath_node_size is not None else int(node_size * 0.35)

    nodes = list(draw_G.nodes())
    edges = list(draw_G.edges())
    xlim, ylim = _xy_limits(pos)

    # Static legend — fixed across frames.
    legend = [
        mpatches.Patch(facecolor='#90caf9', edgecolor='#37474f', label='Lane'),
        mpatches.Patch(facecolor='#ffccbc', edgecolor='#37474f', label='Lane connector'),
        mpatches.Patch(facecolor='#ffeb3b', edgecolor='#37474f', label='Ego (initial)'),
        mpatches.Patch(facecolor=_ANIM_TRAIL_COLOR, edgecolor='#37474f',
                       label=f'Trail (window={window})'),
        mpatches.Patch(facecolor=_ANIM_HEAD_COLOR, edgecolor='#37474f', label='Current node'),
        mpatches.Patch(facecolor='#43a047', edgecolor='#37474f', label=f'Goal  [{_goal_label}]'),
    ]
    _seen_labels = set()
    for _color, _label in highlight_nodes.values():
        if _label not in _seen_labels:
            legend.append(mpatches.Patch(facecolor=_color, edgecolor='#37474f', label=_label))
            _seen_labels.add(_label)

    fig, ax = plt.subplots(figsize=figsize)

    def _node_style(n, head, window_ids):
        """(facecolor, size) for node *n* given the current head and window set."""
        if n == head:
            return _ANIM_HEAD_COLOR, int(node_size * 1.3)
        if n == destination_node:
            return '#43a047', node_size               # goal — always green
        if n in highlight_nodes:
            return highlight_nodes[n][0], node_size    # fixed marker (e.g. waypoint)
        if n in window_ids:
            return _ANIM_TRAIL_COLOR, node_size        # fading trail
        if n == ego_id:
            return '#ffeb3b', node_size                # initial node
        if object_types.get(n) == 'connector':
            return '#ffccbc', _offpath_sz
        return '#90caf9', _offpath_sz                  # plain lane / unknown

    def update(i):
        ax.clear()
        head = path[i]
        lo = max(0, i - window + 1)
        window_seq = path[lo:i + 1]          # head + trailing window (by index)
        window_ids = set(window_seq)
        window_edges = set(zip(window_seq, window_seq[1:]))

        node_colors, node_sizes = [], []
        for n in nodes:
            c, s = _node_style(n, head, window_ids)
            node_colors.append(c)
            node_sizes.append(s)

        ec = [_ANIM_HEAD_COLOR if (u, v) in window_edges else '#cfd8dc' for u, v in edges]
        ew = [3.5 if (u, v) in window_edges else 1.0 for u, v in edges]

        nx.draw_networkx_nodes(draw_G, pos, nodelist=nodes, node_color=node_colors,
                               edgecolors='#37474f', node_size=node_sizes,
                               linewidths=1.5, ax=ax)
        nx.draw_networkx_edges(draw_G, pos, edgelist=edges, edge_color=ec, width=ew,
                               arrows=True, arrowsize=arrowsize, arrowstyle='->',
                               connectionstyle='arc3,rad=0.1', ax=ax)
        if show_labels:
            if label_window_only:
                labels = {n: n for n in window_ids if n in pos}
                if destination_node in pos:
                    labels[destination_node] = destination_node
            else:
                labels = {n: n for n in nodes}
            nx.draw_networkx_labels(draw_G, pos, labels=labels,
                                    font_size=font_size, font_family='monospace', ax=ax)

        ax.legend(handles=legend, loc='upper left', fontsize=10)
        step_title = f'{title + "  —  " if title else ""}step {i}/{len(path) - 1}: {head}'
        ax.set_title(step_title, fontsize=13, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        return []

    interval = 1000.0 / fps if fps > 0 else 500.0
    anim = animation.FuncAnimation(
        fig, update, frames=len(path), interval=interval, blit=False, repeat=repeat
    )

    result = None
    if output_path is not None:
        out = output_path
        if out.lower().endswith('.mp4') and 'ffmpeg' not in animation.writers.list():
            out = out[:-4] + '.gif'
            print(f'ffmpeg not available — writing GIF instead: {out}')
        if out.lower().endswith('.gif'):
            anim.save(out, writer=animation.PillowWriter(fps=fps))
            result = display.Image(filename=out)
        else:
            anim.save(out, writer='ffmpeg', fps=fps)
            result = display.Video(out, embed=True)
        print(f'rendered animation to {out}')
    else:
        result = display.HTML(anim.to_jshtml(fps=fps))

    plt.close(fig)
    return result

"""High-level LTL model-checking pipeline over a NuPlan lane subgraph.

This module collapses the repetitive "build Kripke model → translate LTL to a
Büchi automaton → form the product → run the emptiness check → project the
lasso stem → plot it" boilerplate into two calls:

    solution = solve_for_path(graph, object_types, ego_id, pos, formula, prop_nodes)
    if solution is not None:
        visualize_solution(solution, figsize=(14, 10))

``solve_for_path`` returns ``None`` when the specification is infeasible
(the product automaton is empty) and a :class:`PathSolution` otherwise.

Public API
----------
solve_for_path(graph, object_types, ego_id, pos, formula, prop_nodes=None, ...)
    Run the full model-checking pipeline; return a PathSolution or None.

visualize_solution(solution, **flags)
    Render the satisfying path on the NuPlan subgraph (delegates to
    ``draw_nuplan_path``). ``flags`` are forwarded as drawing options.

animate_solution(solution, window=1, output_path=None, **flags)
    Render a node-by-node traversal video of the path with a sliding highlight
    window (delegates to ``animate_nuplan_path``). Returns an inline HTML5/JS
    player, or writes a GIF/MP4 when ``output_path`` is given.

build_nuplan_kripke(graph, object_types, ego_id, prop_nodes, ...)
    Build just the labelled Kripke model (used internally by solve_for_path).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .buchi_automaton import kripke_buchi_product
from .gpvw_ltl_to_buchi import ltl_to_buchi_gpvw
from .kripke import KripkeModel
from .nuplan_graph import animate_nuplan_path, draw_nuplan_nodes, draw_nuplan_path

# Structural propositions derived from the graph itself (not user-assigned).
_STRUCTURAL_PROPS = frozenset({'lane', 'connector', 'unknown', 'ego', 'fork'})

# Default colours for named (user-assigned) propositions when highlighted.
_PROP_COLORS = {
    'waypoint':    '#7e57c2',   # purple
    'destination': '#43a047',   # green (also the default goal colour)
}
_FALLBACK_PROP_COLOR = '#26a69a'   # teal — for any other named proposition


def _normalize_prop_nodes(prop_nodes: Optional[dict]) -> Dict[str, set]:
    """Coerce ``{prop: node_id | iterable_of_ids}`` into ``{prop: set_of_ids}``."""
    norm: Dict[str, set] = {}
    for prop, value in (prop_nodes or {}).items():
        norm[prop] = {value} if isinstance(value, str) else set(value)
    return norm


@dataclass
class PathSolution:
    """A feasible witness for an LTL specification on a NuPlan subgraph.

    Attributes
    ----------
    formula        : the LTL formula that was checked
    path           : lasso stem projected to Kripke (graph) states, in order
    cycle          : lasso cycle projected to Kripke states
    kripke         : the labelled KripkeModel (with stutter self-loops)
    buchi          : the Büchi automaton B(formula)
    product        : the product automaton M × B
    prop_nodes     : ``{named_prop: {node_id, ...}}`` (structural props excluded)
    goal_node      : node coloured green in the visualization
    goal_label     : legend label for the goal node
    draw_G/pos/object_types/ego_id : visualization context (see nuplan_graph)
    """

    formula: object
    path: List[str]
    cycle: List[str]
    kripke: KripkeModel
    buchi: object
    product: object
    prop_nodes: Dict[str, set]
    draw_G: object
    pos: dict
    object_types: dict
    ego_id: object
    goal_node: Optional[str] = None
    goal_label: Optional[str] = None

    def describe(self) -> None:
        """Print the lasso stem step-by-step with proposition tags."""
        print(f'\nRESULT: SAT — lasso stem ({len(self.path) - 1} step(s)):')
        for i, s in enumerate(self.path):
            tags = []
            if s == self.ego_id:
                tags.append('initial')
            for prop, ids in self.prop_nodes.items():
                if s in ids:
                    tags.append(prop.upper())
            labels = sorted(self.kripke.labeling.get(s, frozenset()))
            tag = f'  [{", ".join(tags)}]' if tags else ''
            print(f'  step {i:2d}: {s}  {labels}{tag}')


def build_nuplan_kripke(
    graph,
    object_types: dict,
    ego_id,
    prop_nodes: Optional[dict] = None,
    *,
    add_fork: bool = True,
    self_loops: bool = True,
) -> KripkeModel:
    """Build a labelled Kripke model from a NuPlan lane subgraph.

    Each node is labelled with its structural type (``lane``/``connector``),
    ``ego`` on the ego node, ``fork`` where ``out_degree > 1`` (if *add_fork*),
    and any named propositions assigned via *prop_nodes*.

    Self-loops (stuttering) are added to every node when *self_loops* is True so
    the transition relation is total — required for Büchi acceptance of ``F``/``G``.
    """
    norm = _normalize_prop_nodes(prop_nodes)

    def node_to_props(node_id) -> set:
        props = {object_types.get(node_id, 'unknown')}
        if node_id == ego_id:
            props.add('ego')
        if add_fork and graph.out_degree(node_id) > 1:
            props.add('fork')
        for prop, ids in norm.items():
            if node_id in ids:
                props.add(prop)
        return props

    kripke = KripkeModel()
    for node in graph.nodes():
        props = node_to_props(node)
        if node == ego_id:
            kripke.add_initial_state(node, props)
        else:
            kripke.add_state(node, props)
    for src, dst in graph.edges():
        kripke.add_transition(src, dst)
    if self_loops:
        for node in graph.nodes():
            kripke.add_transition(node, node)   # stuttering — total relation
    return kripke


def solve_for_path(
    graph,
    object_types: dict,
    ego_id,
    pos: dict,
    formula,
    prop_nodes: Optional[dict] = None,
    *,
    add_fork: bool = True,
    self_loops: bool = True,
    goal_node: Optional[str] = None,
    goal_label: Optional[str] = None,
    verbose: bool = True,
) -> Optional[PathSolution]:
    """Model-check *formula* against the NuPlan subgraph and return a witness.

    Pipeline: build the labelled Kripke model → translate *formula* to a Büchi
    automaton via GPVW → form the product automaton → run the SCC emptiness
    check → project the lasso stem back to graph (Kripke) states.

    Parameters
    ----------
    graph         : nx.DiGraph subgraph (e.g. from ``load_nuplan_subgraph``)
    object_types  : ``{node_id: 'lane' | 'connector'}``
    ego_id        : the initial state (ego's lane/connector)
    pos           : ``{node_id: (x, y)}`` positions (from ``nuplan_pos``); used to
                    restrict the drawable subgraph stored on the solution
    formula       : an LTL formula (see ``tutorials.buchi.ltl``)
    prop_nodes    : ``{named_prop: node_id | iterable_of_ids}`` — assigns named
                    atomic propositions used by *formula* to specific nodes
    goal_node     : node to colour green in the plot; inferred when omitted
                    (prefers the ``destination`` proposition, else ``path[-1]``)

    Returns
    -------
    PathSolution on SAT, or ``None`` when the specification is infeasible.
    """
    norm = _normalize_prop_nodes(prop_nodes)

    kripke = build_nuplan_kripke(
        graph, object_types, ego_id, norm, add_fork=add_fork, self_loops=self_loops
    )
    buchi = ltl_to_buchi_gpvw(formula)
    product = kripke_buchi_product(kripke, buchi)

    if verbose:
        n_trans = sum(len(product.transitions.get(s, [])) for s in product.states)
        print(f'Formula:  {formula}')
        print(f'B(φ): {len(buchi.states)} states   '
              f'M×B: {len(product.states)} states, {n_trans} transitions')

    is_empty, prefix_states, cycle_states = product.check_emptiness_with_states()

    if is_empty:
        if verbose:
            print('\nRESULT: UNSAT — φ is not satisfiable on this model '
                  '(no execution trace satisfies the specification).')
        return None

    path = [s for s, _q in prefix_states]
    cycle = [s for s, _q in cycle_states]

    # Choose the node to colour green: prefer an explicit goal, then the
    # 'destination' proposition, then the last node on the stem.
    if goal_node is None:
        if 'destination' in norm and norm['destination']:
            goal_node = next(iter(norm['destination']))
        elif path:
            goal_node = path[-1]
    if goal_label is None:
        goal_label = 'destination' if 'destination' in norm else str(goal_node)

    draw_G = graph.subgraph(pos.keys()).copy()

    solution = PathSolution(
        formula=formula, path=path, cycle=cycle, kripke=kripke,
        buchi=buchi, product=product, prop_nodes=norm,
        draw_G=draw_G, pos=pos, object_types=object_types, ego_id=ego_id,
        goal_node=goal_node, goal_label=goal_label,
    )
    if verbose:
        solution.describe()
    return solution


def visualize_solution(solution: PathSolution, *, path_only: bool = False, **flags):
    """Render a satisfying path on the NuPlan subgraph.

    Named propositions other than the goal (e.g. ``waypoint``) are highlighted
    with distinct colours and added to the legend. All *flags* are forwarded to
    :func:`tutorials.buchi.nuplan_graph.draw_nuplan_path` (``figsize``,
    ``node_size``, ``offpath_node_size``, ``font_size``, ``arrowsize``,
    ``title``, …).

    Parameters
    ----------
    path_only : when True, draw only the nodes on the witness path (plus any
                highlighted nodes) instead of the whole subgraph. Recommended for
                large graphs (hundreds of nodes) where the full plot is cluttered.

    Returns an ``IPython.display.SVG`` (use as the last cell expression).
    """
    if solution is None:
        print('No solution to visualize.')
        return None

    # Build extra highlights for named props that are not the green goal node.
    highlight_nodes = {}
    for prop, ids in solution.prop_nodes.items():
        if prop == 'destination':
            continue
        color = _PROP_COLORS.get(prop, _FALLBACK_PROP_COLOR)
        for node_id in ids:
            if node_id in solution.draw_G and node_id != solution.goal_node:
                highlight_nodes[node_id] = (color, f'{prop} ({node_id})')

    draw_G, pos = solution.draw_G, solution.pos
    if path_only:
        keep = [n for n in solution.path if n in pos]
        keep += [n for n in highlight_nodes if n in pos and n not in keep]
        if solution.goal_node in pos and solution.goal_node not in keep:
            keep.append(solution.goal_node)
        draw_G = solution.draw_G.subgraph(keep).copy()
        pos = {n: solution.pos[n] for n in draw_G.nodes()}

    flags.setdefault('title', f'φ = {solution.formula}')
    return draw_nuplan_path(
        draw_G, pos, solution.object_types, solution.ego_id,
        solution.path,
        destination_node=solution.goal_node,
        goal_label=solution.goal_label,
        highlight_nodes=highlight_nodes,
        **flags,
    )


def animate_solution(
    solution: PathSolution,
    *,
    window: int = 1,
    path_only: bool = False,
    output_path: Optional[str] = None,
    **flags,
):
    """Render a node-by-node traversal video of a satisfying path.

    Animates the witness as a sliding highlight that walks the route one node per
    frame: the current node is drawn crimson, the previous ``window - 1`` nodes
    trail behind it in orange, and earlier nodes fade back to their base style so
    the lit band visibly slides forward. The goal stays green and the ego yellow,
    matching :func:`visualize_solution`.

    Parameters
    ----------
    window        : number of trailing nodes kept highlighted (``>= 1``). ``1``
                    spotlights only the current node; larger values leave a longer
                    comet tail of recently-visited nodes.
    path_only     : draw only the nodes on the witness path (plus highlighted/goal
                    nodes) — recommended for large graphs, as in
                    :func:`visualize_solution`.
    output_path   : when given, also write the animation to this file (``.gif`` via
                    Pillow, or ``.mp4`` when ffmpeg is present). When omitted, an
                    inline HTML5/JS player is returned.

    All other *flags* are forwarded to
    :func:`tutorials.buchi.nuplan_graph.animate_nuplan_path` (``figsize``,
    ``node_size``, ``offpath_node_size``, ``font_size``, ``arrowsize``, ``fps``,
    ``show_labels``, ``label_window_only``, ``title``, …).

    Returns an ``IPython.display`` object (use as the last cell expression), or
    ``None`` when *solution* is ``None``.
    """
    if solution is None:
        print('No solution to animate.')
        return None

    # Build extra highlights for named props that are not the green goal node.
    highlight_nodes = {}
    for prop, ids in solution.prop_nodes.items():
        if prop == 'destination':
            continue
        color = _PROP_COLORS.get(prop, _FALLBACK_PROP_COLOR)
        for node_id in ids:
            if node_id in solution.draw_G and node_id != solution.goal_node:
                highlight_nodes[node_id] = (color, f'{prop} ({node_id})')

    frames_path = list(solution.path)

    draw_G, pos = solution.draw_G, solution.pos
    if path_only:
        keep = [n for n in frames_path if n in pos]
        keep += [n for n in highlight_nodes if n in pos and n not in keep]
        if solution.goal_node in pos and solution.goal_node not in keep:
            keep.append(solution.goal_node)
        draw_G = solution.draw_G.subgraph(keep).copy()
        pos = {n: solution.pos[n] for n in draw_G.nodes()}

    flags.setdefault('title', f'φ = {solution.formula}')
    return animate_nuplan_path(
        draw_G, pos, solution.object_types, solution.ego_id,
        frames_path,
        window=window,
        destination_node=solution.goal_node,
        goal_label=solution.goal_label,
        highlight_nodes=highlight_nodes,
        output_path=output_path,
        **flags,
    )


def visualize_nodes(graph, object_types, ego_id, pos, prop_nodes, **flags):
    """Show the named-proposition nodes on the overall NuPlan directed graph.

    The natural fallback when :func:`solve_for_path` returns ``None``: with no
    satisfying path to draw, this highlights where the (unconnected) nodes of
    interest sit in the full graph — the ego, plus each node assigned in
    *prop_nodes* (``destination`` green, ``waypoint`` purple, others teal).

    *flags* are forwarded to
    :func:`tutorials.buchi.nuplan_graph.draw_nuplan_nodes` (``figsize``,
    ``base_node_size``, ``highlight_node_size``, ``font_size``, ``title``, …).

    Returns an ``IPython.display.SVG`` (use as the last cell expression).
    """
    norm = _normalize_prop_nodes(prop_nodes)
    highlight_nodes = {}
    for prop, ids in norm.items():
        color = _PROP_COLORS.get(prop, _FALLBACK_PROP_COLOR)
        for node_id in ids:
            if node_id in pos:
                highlight_nodes[node_id] = (color, f'{prop} ({node_id})')

    draw_G = graph.subgraph(pos.keys()).copy()
    return draw_nuplan_nodes(draw_G, pos, object_types, ego_id, highlight_nodes, **flags)

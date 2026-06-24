"""Visualize Büchi automata, Kripke models, and GPVW intermediate nodes using Graphviz.

Graphviz produces publication-quality diagrams with excellent handling of self-loops,
hierarchical layout, and edge label positioning. This module provides a simpler, cleaner
alternative to the networkx/matplotlib renderer.
"""

from __future__ import annotations

from typing import Hashable, Iterable, List, Mapping, Optional, Union

import graphviz

NodePropositions = Union[Mapping[Hashable, Iterable], type(lambda: None)]


def letter_to_str(letter: Optional[Iterable]) -> str:
    """Render a set of propositions as {p, q} or ∅."""
    if letter is None:
        return ""
    elements = sorted(str(element) for element in letter)
    return "{" + ", ".join(elements) + "}" if elements else "∅"


def draw(
    automaton,
    title: str = "",
    output_path: Optional[str] = None,
    node_propositions: Optional[NodePropositions] = None,
    show_edge_labels: bool = True,
    use_xlabels: bool = True,
    initial_color: str = "#ffeb3b",
    accepting_color: str = "#66bb6a",
    format: str = "svg",
    size: str = "8,6",
    nodesep: float = 0.5,
    ranksep: float = 0.75,
    edge_colors: Optional[Mapping[str, str]] = None,
    legend_title: str = "Legend",
    highlight_path: Optional[list] = None,
    highlight_node_color: str = "#ff7043",
    highlight_edge_color: str = "#e53935",
):
    """Draw ``automaton`` (Büchi or Kripke) using Graphviz and optionally save it.

    Each node is labelled with the predicates that hold there when ``node_propositions`` is
    given, and otherwise with its state index ``q0, q1, …``.

    :param title: graph label/title.
    :param output_path: if given, save to this path (without extension; format is appended).
    :param node_propositions: a mapping ``state → predicates`` giving the propositions
        that hold at each state; drawn inside the node.
    :param show_edge_labels: draw the transition letter on each edge.
    :param use_xlabels: place edge labels as external labels (xlabel) rather than inline;
        recommended for orthogonal splines to avoid Graphviz placement warnings.
    :param initial_color: fill colour for initial states (hex).
    :param accepting_color: fill colour for accepting states (hex).
    :param format: output format ('svg', 'pdf', 'png', etc.); only used if output_path given.
    :param size: canvas size as "width,height" (inches); e.g. "12,8" for larger diagrams.
    :param nodesep: minimum separation between nodes in inches.
    :param ranksep: minimum separation between ranks in inches.
    :param edge_colors: mapping from edge label string (e.g. ``"{a}"``) to a colour (hex or name).
        When provided, a colour legend is automatically added to the diagram.
    :param legend_title: title shown at the top of the colour legend box.
    :param highlight_path: optional list of states forming a path to highlight.
    :param highlight_node_color: fill colour for states on the highlighted path.
    :param highlight_edge_color: colour for edges on the highlighted path.
    :return: the Graphviz ``Digraph`` object (call ``.view()`` to open in default viewer).
    """
    g = graphviz.Digraph(
        graph_attr={
            "rankdir": "TB",
            "overlap": "false",
            "splines": "ortho",
            "bgcolor": "transparent",
            "size": size,
            "nodesep": str(nodesep),
            "ranksep": str(ranksep),
            "forcelabels": "true",
        },
        node_attr={
            "shape": "circle",
            "style": "filled",
            "fillcolor": "#90caf9",
            "color": "#37474f",
            "fontname": "Courier",
            "fontsize": "9",
        },
        edge_attr={
            "color": "#546e7a",
            "fontname": "Courier",
            "fontsize": "8",
        },
    )

    if title:
        g.attr(label=title, labelloc="top")

    # Collect node info.
    states = list(automaton.states)
    initial = set(automaton.initial_states)
    accepting = set(automaton.accepting_states)
    order = sorted(states, key=lambda s: str(s))
    state_ids = {s: f"q{i}" for i, s in enumerate(order)}

    path_nodes: set = set(highlight_path) if highlight_path else set()
    path_edges: set = set()
    if highlight_path and len(highlight_path) > 1:
        path_edges = set(zip(highlight_path, highlight_path[1:]))

    # Add nodes.
    for state in order:
        node_id = state_ids[state]
        fillcolor = "#90caf9"
        if state in initial:
            fillcolor = initial_color
        if state in accepting:
            fillcolor = accepting_color
        if state in path_nodes:
            fillcolor = highlight_node_color

        # Node label: propositions (if provided) or state index.
        if node_propositions is not None:
            if callable(node_propositions):
                held = node_propositions(state)
            else:
                held = node_propositions.get(state)
            if held is not None:
                held_str = held if isinstance(held, str) else letter_to_str(held)
                label = held_str
            else:
                label = node_id
        else:
            label = node_id

        # Double-border for accepting states.
        if state in accepting:
            g.node(node_id, label=label, fillcolor=fillcolor, penwidth="2.5")
        else:
            g.node(node_id, label=label, fillcolor=fillcolor, penwidth="1.5")

    # Add edges.
    for src in order:
        src_id = state_ids[src]
        for letter, dst in automaton.successors(src):
            dst_id = state_ids[dst]
            letter_str = letter_to_str(letter)
            on_path = (src, dst) in path_edges
            color = highlight_edge_color if on_path else (edge_colors or {}).get(letter_str, "#546e7a")
            attrs = {"color": color, "fontcolor": color}
            if on_path:
                attrs["penwidth"] = "3.0"
            if show_edge_labels:
                if use_xlabels:
                    g.edge(src_id, dst_id, xlabel=letter_str, **attrs)
                else:
                    g.edge(src_id, dst_id, label=letter_str, **attrs)
            else:
                g.edge(src_id, dst_id, **attrs)

    # Add colour legend when edge_colors is provided.
    if edge_colors:
        header = f'<TR><TD COLSPAN="2"><B>{legend_title}</B></TD></TR>'
        rows = "".join(
            f'<TR>'
            f'<TD BGCOLOR="{color}" WIDTH="18"> </TD>'
            f'<TD ALIGN="LEFT"><FONT COLOR="{color}"><B>{lbl}</B></FONT></TD>'
            f'</TR>'
            for lbl, color in edge_colors.items()
        )
        html = f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="3" CELLPADDING="4">{header}{rows}</TABLE>>'
        g.node(
            "__legend__",
            label=html,
            shape="none",
            margin="0",
            fontname="Courier",
            fontsize="8",
        )

    if output_path:
        g.render(output_path, format=format, cleanup=True)
        print(f"rendered to {output_path}.{format}")

    return g


def draw_gpvw_nodes(
    nodes: List,
    untils: List,
    title: str = "",
    size: str = "10,7",
    nodesep: float = 0.6,
    ranksep: float = 0.9,
    initial_color: str = "#ffeb3b",
    accepting_color: str = "#66bb6a",
    default_color: str = "#90caf9",
    output_path: Optional[str] = None,
    format: str = "svg",
) -> graphviz.Digraph:
    """Render the intermediate GPVW node graph.

    Each node is drawn as an HTML-table label showing its ``old`` and ``next``
    bookkeeping sets (``now`` is always ∅ for finalized nodes).  Each transition
    ``q → q'`` is labeled with the alphabet letter determined by the atoms in the
    ``old`` set of the *post-state* ``q'`` (the node being entered).

    :param nodes:    list of :class:`~gpvw_ltl_to_buchi.GPVWNode` objects
                     returned by :func:`~gpvw_ltl_to_buchi.ltl_to_gba_gpvw`.
    :param untils:   Until sub-formulas of the NNF formula (from
                     :func:`~gpvw_ltl_to_buchi._all_untils`); used to compute
                     which nodes belong to each accepting set.
    :param title:    diagram title.
    :param size:     Graphviz canvas ``"width,height"`` in inches.
    :param nodesep:  minimum node separation.
    :param ranksep:  minimum rank separation.
    :param initial_color:   fill colour for initial nodes.
    :param accepting_color: fill colour for nodes in every accepting set
                            (those that discharge all Until obligations).
    :param default_color:   fill colour for non-initial, non-accepting nodes.
    :param output_path: if given, save to this path (extension appended).
    :param format:   output format (``"svg"``, ``"pdf"``, etc.).
    :return: the Graphviz :class:`~graphviz.Digraph` object.
    """
    from .gpvw_ltl_to_buchi import INIT, Atom, Not

    # ── Precompute accepting sets ─────────────────────────────────────────────
    # A node is "fully accepting" if it belongs to every per-Until accepting set.
    per_until_acc: List[set] = []
    for u in untils:
        per_until_acc.append(
            {n.name for n in nodes if (u not in n.old) or (u.right in n.old)}
        )
    if per_until_acc:
        all_accepting = set.intersection(*per_until_acc)
    else:
        all_accepting = {n.name for n in nodes}   # no Until → all accepting

    initial_nodes = {n.name for n in nodes if INIT in n.incoming}

    # ── Build node-id map ─────────────────────────────────────────────────────
    order = sorted(nodes, key=lambda n: n.name)
    node_label = {n.name: f"q{i}" for i, n in enumerate(order)}

    # ── Graphviz graph ────────────────────────────────────────────────────────
    g = graphviz.Digraph(
        graph_attr={
            "rankdir": "LR",
            "bgcolor": "transparent",
            "size": size,
            "nodesep": str(nodesep),
            "ranksep": str(ranksep),
            "forcelabels": "true",
            "splines": "true",
        },
        node_attr={
            "shape": "none",
            "fontname": "Courier",
            "fontsize": "9",
            "margin": "0",
        },
        edge_attr={
            "color": "#546e7a",
            "fontname": "Courier",
            "fontsize": "8",
        },
    )

    if title:
        g.attr(label=title, labelloc="top", fontname="Courier", fontsize="11")

    def _fmt_fset(fset) -> str:
        """Format a frozenset of LTL formulas as a comma-separated string."""
        if not fset:
            return "∅"
        return ", ".join(sorted(str(f) for f in fset))

    def _letter_str(node) -> str:
        """Alphabet letter emitted on outgoing transitions from *node*."""
        atoms = sorted(f.name for f in node.old if isinstance(f, Atom))
        return "{" + ", ".join(atoms) + "}" if atoms else "∅"

    # ── Draw invisible initial arrow source ───────────────────────────────────
    # When several nodes are initial, a single dummy source state fans out to
    # each of them; those entry edges follow the same post-node labelling rule
    # (the letter is the ``old`` atoms of the initial node being entered).
    multi_init = len(initial_nodes) > 1
    g.node("__init__", label="", shape="none", width="0", height="0")

    # ── Draw each finalized GPVW node ─────────────────────────────────────────
    for node in order:
        nid = str(node.name)
        qlabel = node_label[node.name]
        is_init = node.name in initial_nodes
        is_acc = node.name in all_accepting

        if is_init and is_acc:
            header_bg = "#ffc107"    # amber — both
        elif is_init:
            header_bg = initial_color
        elif is_acc:
            header_bg = accepting_color
        else:
            header_bg = default_color

        border = 'BORDER="3"' if is_acc else 'BORDER="1"'

        old_str = _fmt_fset(node.old)
        nxt_str = _fmt_fset(node.next_set)

        ann = []
        if is_init:
            ann.append("initial")
        if is_acc:
            ann.append("accepting")
        ann_str = f" ({', '.join(ann)})" if ann else ""

        html = (
            f'<<TABLE {border} CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">'
            f'<TR><TD BGCOLOR="{header_bg}"><B>{qlabel}{ann_str}</B></TD></TR>'
            f'<TR><TD ALIGN="LEFT" BGCOLOR="white">'
            f'<B>old:</B> {old_str}</TD></TR>'
            f'<TR><TD ALIGN="LEFT" BGCOLOR="#f5f5f5">'
            f'<B>now:</B> ∅</TD></TR>'
            f'<TR><TD ALIGN="LEFT" BGCOLOR="white">'
            f'<B>nxt:</B>  {nxt_str}</TD></TR>'
            f'</TABLE>>'
        )
        g.node(nid, label=html)

        if is_init:
            init_label = _letter_str(node) if multi_init else ""
            g.edge("__init__", nid, arrowhead="normal", style="", label=init_label)

    # ── Draw transitions ──────────────────────────────────────────────────────
    # The letter on q → q' is the conjunction of atomic propositions in the
    # ``old`` set of the *post-state* q' (the node being entered), matching the
    # GPVW convention where a node's incoming transitions are labelled by the
    # local theory that holds upon arrival.
    for src_node in order:
        src_id = str(src_node.name)
        for tgt_node in order:
            if src_node.name in tgt_node.incoming:
                tgt_id = str(tgt_node.name)
                g.edge(src_id, tgt_id, label=_letter_str(tgt_node))

    if output_path:
        g.render(output_path, format=format, cleanup=True)
        print(f"rendered to {output_path}.{format}")

    return g

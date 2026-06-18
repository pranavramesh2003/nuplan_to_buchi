"""Visualize Büchi automata and Kripke models using Graphviz.

Graphviz produces publication-quality diagrams with excellent handling of self-loops,
hierarchical layout, and edge label positioning. This module provides a simpler, cleaner
alternative to the networkx/matplotlib renderer.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping, Optional, Union

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

    # Add nodes.
    for state in order:
        node_id = state_ids[state]
        fillcolor = "#90caf9"
        if state in initial:
            fillcolor = initial_color
        if state in accepting:
            fillcolor = accepting_color

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
            color = (edge_colors or {}).get(letter_str, "#546e7a")
            attrs = {"color": color, "fontcolor": color}
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

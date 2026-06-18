"""Tarjan's strongly-connected-components algorithm (iterative).

A *strongly connected component* (SCC) of a directed graph is a maximal set of nodes
that are all mutually reachable. Tarjan's algorithm finds every SCC in a single
depth-first traversal in ``O(V + E)`` time using "low-link" values.

This implementation is **iterative** (it manages an explicit DFS stack) so it does not
hit Python's recursion limit on large graphs. It is a self-contained graph utility and
knows nothing about Buchi automata; the emptiness check reuses it.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Mapping

Node = Hashable


def tarjan_scc(graph: Mapping[Node, Iterable[Node]]) -> List[List[Node]]:
    """Return the strongly connected components of a directed graph.

    :param graph: adjacency mapping ``node -> iterable of successor nodes``. Nodes that
        only ever appear as successors are treated as having no outgoing edges.
    :return: a list of SCCs, each a list of nodes. Components are emitted in reverse
        topological order (a component appears before the components it can reach),
        which is the natural output order of Tarjan's algorithm.
    """
    # Collect every node, including those that appear only as a successor.
    nodes: List[Node] = list(graph.keys())
    seen = set(nodes)
    for successors in graph.values():
        for successor in successors:
            if successor not in seen:
                seen.add(successor)
                nodes.append(successor)

    indices: Dict[Node, int] = {}
    low_links: Dict[Node, int] = {}
    on_stack: Dict[Node, bool] = {}
    scc_stack: List[Node] = []
    components: List[List[Node]] = []
    counter = 0

    for root in nodes:
        if root in indices:
            continue

        # Each work-stack frame is (node, iterator over its successors).
        work_stack: List[tuple] = [(root, iter(graph.get(root, ())))]
        indices[root] = low_links[root] = counter
        counter += 1
        scc_stack.append(root)
        on_stack[root] = True

        while work_stack:
            node, successors = work_stack[-1]
            advanced = False
            for successor in successors:
                if successor not in indices:
                    # Tree edge: descend into the successor.
                    indices[successor] = low_links[successor] = counter
                    counter += 1
                    scc_stack.append(successor)
                    on_stack[successor] = True
                    work_stack.append((successor, iter(graph.get(successor, ()))))
                    advanced = True
                    break
                if on_stack.get(successor, False):
                    # Back/cross edge to a node still on the stack.
                    low_links[node] = min(low_links[node], indices[successor])
            if advanced:
                continue

            # All successors of ``node`` processed: it is a root of an SCC iff its
            # low-link equals its own index.
            if low_links[node] == indices[node]:
                component: List[Node] = []
                while True:
                    member = scc_stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                components.append(component)

            work_stack.pop()
            if work_stack:
                parent = work_stack[-1][0]
                low_links[parent] = min(low_links[parent], low_links[node])

    return components

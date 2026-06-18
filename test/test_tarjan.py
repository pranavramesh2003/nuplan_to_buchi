import unittest

from tutorials.buchi.tarjan import tarjan_scc


def _scc_set(graph):
    """Run Tarjan and return the components as a comparable set of frozensets."""
    return {frozenset(component) for component in tarjan_scc(graph)}


class TestTarjanSCC(unittest.TestCase):
    """Tests for the iterative Tarjan strongly-connected-components algorithm."""

    def test_classic_example(self) -> None:
        """A 3-cycle {a,b,c} feeding into a trivial {d}."""
        graph = {
            "a": ["b"],
            "b": ["c"],
            "c": ["a", "d"],
            "d": [],
        }
        self.assertEqual(_scc_set(graph), {frozenset({"a", "b", "c"}), frozenset({"d"})})

    def test_single_self_loop(self) -> None:
        """A self-loop node is its own (size-1) SCC."""
        self.assertEqual(_scc_set({"x": ["x"]}), {frozenset({"x"})})

    def test_isolated_node_is_trivial_scc(self) -> None:
        """A node with no edges still forms a trivial SCC."""
        self.assertEqual(_scc_set({"x": []}), {frozenset({"x"})})

    def test_successor_only_node_included(self) -> None:
        """Nodes appearing only as successors are still partitioned."""
        self.assertEqual(_scc_set({"a": ["b"]}), {frozenset({"a"}), frozenset({"b"})})

    def test_reverse_topological_order(self) -> None:
        """Tarjan emits a component before the components it can reach."""
        graph = {"a": ["b"], "b": ["c"], "c": []}
        order = [frozenset(component) for component in tarjan_scc(graph)]
        self.assertLess(order.index(frozenset({"c"})), order.index(frozenset({"a"})))
        self.assertLess(order.index(frozenset({"b"})), order.index(frozenset({"a"})))

    def test_nested_cycles(self) -> None:
        """Two cycles sharing a connection collapse correctly; exercises the DFS stack."""
        graph = {
            1: [2],
            2: [3],
            3: [1, 4],
            4: [5],
            5: [6],
            6: [4],
        }
        self.assertEqual(_scc_set(graph), {frozenset({1, 2, 3}), frozenset({4, 5, 6})})

    def test_all_strongly_connected(self) -> None:
        graph = {1: [2], 2: [3], 3: [1]}
        self.assertEqual(_scc_set(graph), {frozenset({1, 2, 3})})

    def test_deep_chain_no_recursion_error(self) -> None:
        """A long chain must not overflow the (iterative) implementation."""
        n = 5000
        graph = {i: [i + 1] for i in range(n)}
        graph[n] = []
        components = tarjan_scc(graph)
        self.assertEqual(len(components), n + 1)


if __name__ == "__main__":
    unittest.main()

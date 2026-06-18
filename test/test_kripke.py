import unittest

from tutorials.buchi.kripke import KripkeModel


class TestKripkeModel(unittest.TestCase):
    """Tests for the Kripke model (S, →, L)."""

    def test_states_and_labeling(self) -> None:
        m = KripkeModel()
        m.add_state("s0", {"p"})
        m.label("s1", {"p", "q"})
        self.assertEqual(m.states, {"s0", "s1"})
        self.assertEqual(m.labeling["s0"], frozenset({"p"}))
        self.assertEqual(m.labeling["s1"], frozenset({"p", "q"}))

    def test_transitions_are_unlabeled(self) -> None:
        m = KripkeModel()
        m.add_transition("s0", "s1")
        self.assertEqual(m.successors("s0"), [(frozenset(), "s1")])
        self.assertEqual(m.successors("s1"), [])

    def test_add_transition_registers_endpoints_with_empty_label(self) -> None:
        m = KripkeModel()
        m.add_transition("s0", "s1")
        self.assertEqual(m.states, {"s0", "s1"})
        self.assertEqual(m.labeling["s0"], frozenset())

    def test_reachable_all_when_no_initial(self) -> None:
        m = KripkeModel()
        m.add_transition("a", "b")
        m.add_state("c")
        self.assertEqual(m.reachable_states(), {"a", "b", "c"})

    def test_reachable_from_initial(self) -> None:
        m = KripkeModel()
        m.add_initial_state("a")
        m.add_transition("a", "b")
        m.add_state("c")  # not reachable from a
        self.assertEqual(m.reachable_states(), {"a", "b"})

    def test_accepting_states_always_empty(self) -> None:
        m = KripkeModel()
        m.add_state("s0", {"p"})
        self.assertEqual(m.accepting_states, set())


if __name__ == "__main__":
    unittest.main()

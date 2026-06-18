import unittest

from tutorials.buchi.generalized_buchi import GeneralizedBuchiAutomaton
from tutorials.buchi.omega_word import OmegaWord


class TestDegeneralization(unittest.TestCase):
    """Tests for GBA -> BA degeneralization."""

    def test_no_accepting_sets_makes_every_state_accepting(self) -> None:
        """With zero constraints every infinite run is accepting."""
        gba = GeneralizedBuchiAutomaton()
        gba.add_initial_state("q0")
        gba.add_transition("q0", {"a"}, "q0")
        buchi = gba.to_buchi()

        self.assertEqual(buchi.accepting_states, buchi.states)
        self.assertFalse(buchi.is_empty())
        self.assertTrue(buchi.accepts(OmegaWord(prefix=[], loop=[{"a"}])))

    def test_single_set_matches_plain_buchi(self) -> None:
        """One accepting set degeneralizes to the same (state, 0) acceptance."""
        gba = GeneralizedBuchiAutomaton()
        gba.add_initial_state("q0")
        gba.add_transition("q0", {"a"}, "q1")
        gba.add_transition("q1", {"b"}, "q1")
        gba.add_accepting_set({"q1"})
        buchi = gba.to_buchi()

        self.assertEqual(len(buchi.states), 2)  # k = 1 copy
        self.assertFalse(buchi.is_empty())
        self.assertTrue(buchi.accepts(OmegaWord(prefix=[{"a"}], loop=[{"b"}])))

    def test_two_sets_require_both_infinitely_often(self) -> None:
        """A run must cycle the counter through both F0 and F1 forever."""
        gba = GeneralizedBuchiAutomaton()
        gba.add_initial_state("a")
        gba.add_transition("a", {"x"}, "b")
        gba.add_transition("b", {"y"}, "a")
        gba.add_accepting_set({"a"})  # F0
        gba.add_accepting_set({"b"})  # F1
        buchi = gba.to_buchi()

        # k = 2 -> two copies of {a, b}.
        self.assertEqual(len(buchi.states), 4)
        # The only run alternates a, b, a, b, ... visiting both sets infinitely often.
        self.assertFalse(buchi.is_empty())
        self.assertTrue(buchi.accepts(OmegaWord(prefix=[], loop=[{"x"}, {"y"}])))

    def test_unsatisfiable_set_makes_language_empty(self) -> None:
        """If one accepting set is unreachable on every cycle, the language is empty."""
        gba = GeneralizedBuchiAutomaton()
        gba.add_initial_state("a")
        gba.add_transition("a", {"x"}, "b")
        gba.add_transition("b", {"y"}, "a")
        gba.add_accepting_set({"a"})
        gba.add_accepting_set(set())  # never satisfiable -> counter can never complete
        buchi = gba.to_buchi()

        self.assertTrue(buchi.is_empty())

    def test_counter_advances_only_when_leaving_current_set(self) -> None:
        """From copy i, the counter advances iff the source is in F_i."""
        gba = GeneralizedBuchiAutomaton()
        gba.add_initial_state("a")
        gba.add_transition("a", {"x"}, "b")
        gba.add_transition("b", {"y"}, "a")
        gba.add_accepting_set({"a"})  # F0 = {a}
        gba.add_accepting_set({"b"})  # F1 = {b}
        buchi = gba.to_buchi()

        # a is in F0: (a, 0) --x--> (b, 1) advances; b not in F0: (b, 0) --y--> (a, 0).
        self.assertIn((frozenset({"x"}), ("b", 1)), buchi.successors(("a", 0)))
        self.assertIn((frozenset({"y"}), ("a", 0)), buchi.successors(("b", 0)))
        # In copy 1: b is in F1 so (b, 1) advances back to 0; a not in F1 so (a, 1) stays.
        self.assertIn((frozenset({"y"}), ("a", 0)), buchi.successors(("b", 1)))
        self.assertIn((frozenset({"x"}), ("b", 1)), buchi.successors(("a", 1)))


if __name__ == "__main__":
    unittest.main()

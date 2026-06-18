import unittest

from tutorials.buchi.buchi_automaton import BuchiAutomaton
from tutorials.buchi.omega_word import OmegaWord


class TestBuchiEmptiness(unittest.TestCase):
    """Tests for the SCC-based emptiness check and the witness construction."""

    def test_non_empty_with_witness(self) -> None:
        """Accepting state on a reachable cycle: non-empty, with an accepted witness."""
        a = BuchiAutomaton()
        a.add_initial_state("q0")
        a.add_accepting_state("q1")
        a.add_transition("q0", {"a"}, "q1")
        a.add_transition("q1", {"b"}, "q1")  # accepting self-loop

        result = a.check_emptiness()
        self.assertFalse(result.is_empty)
        self.assertIsNotNone(result.witness)
        self.assertIn("q1", result.accepting_scc)
        self.assertTrue(a.accepts(result.witness))

    def test_self_loop_witness_loop_length_one(self) -> None:
        """A single accepting state with a self-loop yields a length-1 loop witness."""
        a = BuchiAutomaton()
        a.add_initial_state("q0")
        a.add_accepting_state("q0")
        a.add_transition("q0", {"a"}, "q0")

        result = a.check_emptiness()
        self.assertFalse(result.is_empty)
        self.assertEqual(result.witness, OmegaWord(prefix=[], loop=[{"a"}]))

    def test_empty_no_cycle(self) -> None:
        """Accepting state reachable only on an acyclic path: empty."""
        a = BuchiAutomaton()
        a.add_initial_state("q0")
        a.add_accepting_state("q2")
        a.add_transition("q0", {"a"}, "q1")
        a.add_transition("q1", {"b"}, "q2")  # q2 is a dead-end

        result = a.check_emptiness()
        self.assertTrue(result.is_empty)
        self.assertIsNone(result.witness)

    def test_empty_cycle_without_accepting_state(self) -> None:
        """A reachable cycle exists but visits no accepting state: empty."""
        a = BuchiAutomaton()
        a.add_initial_state("q0")
        a.add_accepting_state("q2")  # accepting, but unreachable / off the cycle
        a.add_transition("q0", {"a"}, "q1")
        a.add_transition("q1", {"b"}, "q0")  # cycle q0 <-> q1, neither accepting

        self.assertTrue(a.is_empty())

    def test_empty_unreachable_accepting_cycle(self) -> None:
        """An accepting cycle that is not reachable from the initial states: empty."""
        a = BuchiAutomaton()
        a.add_initial_state("q0")
        a.add_transition("q0", {"a"}, "q0")  # reachable, non-accepting self-loop
        a.add_accepting_state("q9")
        a.add_transition("q9", {"b"}, "q9")  # accepting cycle, but disconnected

        self.assertTrue(a.is_empty())

    def test_witness_prefix_traverses_to_scc(self) -> None:
        """Witness stem reaches the accepting SCC across several states."""
        a = BuchiAutomaton()
        a.add_initial_state("s0")
        a.add_transition("s0", {"p"}, "s1")
        a.add_transition("s1", {"q"}, "s2")
        a.add_accepting_state("s2")
        a.add_transition("s2", {"r"}, "s3")
        a.add_transition("s3", {"t"}, "s2")  # accepting cycle s2 -> s3 -> s2

        result = a.check_emptiness()
        self.assertFalse(result.is_empty)
        # Stem: s0 --p--> s1 --q--> s2 ; loop: s2 --r--> s3 --t--> s2
        self.assertEqual(result.witness.prefix, (frozenset({"p"}), frozenset({"q"})))
        self.assertEqual(result.witness.loop, (frozenset({"r"}), frozenset({"t"})))
        self.assertTrue(a.accepts(result.witness))

    def test_no_initial_state_is_empty(self) -> None:
        a = BuchiAutomaton()
        a.add_accepting_state("q0")
        a.add_transition("q0", {"a"}, "q0")
        self.assertTrue(a.is_empty())


class TestBuchiAccepts(unittest.TestCase):
    """Tests for the lasso-word verifier accepts()."""

    def _automaton(self) -> BuchiAutomaton:
        # q0 --a--> q1 (accepting) --b--> q1 ; q0 --a--> q0 (non-accepting self-loop)
        a = BuchiAutomaton()
        a.add_initial_state("q0")
        a.add_accepting_state("q1")
        a.add_transition("q0", {"a"}, "q0")
        a.add_transition("q0", {"a"}, "q1")
        a.add_transition("q1", {"b"}, "q1")
        return a

    def test_accepts_lasso_through_accepting_state(self) -> None:
        a = self._automaton()
        self.assertTrue(a.accepts(OmegaWord(prefix=[{"a"}], loop=[{"b"}])))

    def test_rejects_loop_that_avoids_accepting_state(self) -> None:
        """Looping forever on the non-accepting q0 self-loop is not accepted."""
        a = self._automaton()
        self.assertFalse(a.accepts(OmegaWord(prefix=[], loop=[{"a"}])))

    def test_rejects_word_with_no_matching_transition(self) -> None:
        a = self._automaton()
        self.assertFalse(a.accepts(OmegaWord(prefix=[{"z"}], loop=[{"b"}])))

    def test_accepts_with_custom_object_letters(self) -> None:
        """Letters may be sets of arbitrary objects, not just strings."""

        class Prop:
            def __init__(self, name: str) -> None:
                self.name = name

            def __hash__(self) -> int:
                return hash(self.name)

            def __eq__(self, other: object) -> bool:
                return isinstance(other, Prop) and other.name == self.name

        p, q = Prop("p"), Prop("q")
        a = BuchiAutomaton()
        a.add_initial_state("s0")
        a.add_accepting_state("s1")
        a.add_transition("s0", {p}, "s1")
        a.add_transition("s1", {q}, "s1")

        self.assertTrue(a.accepts(OmegaWord(prefix=[{p}], loop=[{q}])))
        self.assertFalse(a.check_emptiness().is_empty)


if __name__ == "__main__":
    unittest.main()

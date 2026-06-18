import unittest

from tutorials.buchi.buchi_automaton import BuchiAutomaton, intersect
from tutorials.buchi.omega_word import OmegaWord


def _infinitely_often(symbol: str, other: str) -> BuchiAutomaton:
    """A deterministic Buchi automaton over {{symbol}, {other}} for 'symbol i.o.'.

    State s1 is entered exactly after reading ``{symbol}``, so visiting s1 infinitely
    often is equivalent to reading ``{symbol}`` infinitely often.
    """
    a = BuchiAutomaton()
    a.add_initial_state("s0")
    a.add_accepting_state("s1")
    a.add_transition("s0", {symbol}, "s1")
    a.add_transition("s0", {other}, "s0")
    a.add_transition("s1", {symbol}, "s1")
    a.add_transition("s1", {other}, "s0")
    return a


class TestIntersectionStructure(unittest.TestCase):
    """Structural checks on the Q1 × Q2 × {1,2} product."""

    def setUp(self) -> None:
        self.a1 = _infinitely_often("a", "b")
        self.a2 = _infinitely_often("b", "a")
        self.product = intersect(self.a1, self.a2)

    def test_state_space_is_full_product_with_bit(self) -> None:
        expected = {(q1, q2, bit) for q1 in self.a1.states for q2 in self.a2.states for bit in (1, 2)}
        self.assertEqual(self.product.states, expected)
        self.assertEqual(len(self.product.states), len(self.a1.states) * len(self.a2.states) * 2)

    def test_initial_states_use_bit_one(self) -> None:
        expected = {(q1, q2, 1) for q1 in self.a1.initial_states for q2 in self.a2.initial_states}
        self.assertEqual(self.product.initial_states, expected)

    def test_accepting_states_are_F1_times_Q2_times_one(self) -> None:
        expected = {(q1, q2, 1) for q1 in self.a1.accepting_states for q2 in self.a2.states}
        self.assertEqual(self.product.accepting_states, expected)

    def test_toggle_bit_one_to_two_only_on_a1_goal(self) -> None:
        """From bit 1, the bit flips to 2 iff the A1-component is an A1 goal state."""
        # s1 is an A1 goal -> outgoing bit-1 edges must land in bit 2.
        for letter, dst in self.product.successors(("s1", "s0", 1)):
            self.assertEqual(dst[2], 2)
        # s0 is not an A1 goal -> outgoing bit-1 edges stay in bit 1.
        for letter, dst in self.product.successors(("s0", "s0", 1)):
            self.assertEqual(dst[2], 1)

    def test_toggle_bit_two_to_one_only_on_a2_goal(self) -> None:
        """From bit 2, the bit flips to 1 iff the A2-component is an A2 goal state."""
        for letter, dst in self.product.successors(("s0", "s1", 2)):
            self.assertEqual(dst[2], 1)
        for letter, dst in self.product.successors(("s0", "s0", 2)):
            self.assertEqual(dst[2], 2)

    def test_transition_relation_synchronizes_on_shared_letter(self) -> None:
        """No product edge exists when the two components disagree on the letter read."""
        # From (s0, s0, *): A1 reads {a}->s1 and {b}->s0; A2 reads {b}->s1 and {a}->s0.
        # Common letters {a} and {b} both exist, so edges exist for each.
        letters = {letter for letter, _ in self.product.successors(("s0", "s0", 1))}
        self.assertEqual(letters, {frozenset({"a"}), frozenset({"b"})})


class TestIntersectionLanguage(unittest.TestCase):
    """L(intersect(A1, A2)) = L(A1) ∩ L(A2)."""

    def test_accepts_iff_both_accept(self) -> None:
        a1 = _infinitely_often("a", "b")  # a infinitely often
        a2 = _infinitely_often("b", "a")  # b infinitely often
        product = intersect(a1, a2)

        both = OmegaWord(prefix=[], loop=[{"a"}, {"b"}])  # a and b both i.o.
        only_a = OmegaWord(prefix=[], loop=[{"a"}])       # a i.o., b finitely
        only_b = OmegaWord(prefix=[{"a"}], loop=[{"b"}])  # b i.o., a finitely

        # Component automata behave as intended.
        self.assertTrue(a1.accepts(both) and a2.accepts(both))
        self.assertTrue(a1.accepts(only_a) and not a2.accepts(only_a))
        self.assertTrue(not a1.accepts(only_b) and a2.accepts(only_b))

        # Product accepts exactly the words both accept.
        self.assertTrue(product.accepts(both))
        self.assertFalse(product.accepts(only_a))
        self.assertFalse(product.accepts(only_b))

    def test_non_empty_intersection_has_witness(self) -> None:
        a1 = _infinitely_often("a", "b")
        a2 = _infinitely_often("b", "a")
        product = intersect(a1, a2)

        result = product.check_emptiness()
        self.assertFalse(result.is_empty)
        self.assertIsNotNone(result.witness)
        self.assertTrue(product.accepts(result.witness))
        # The witness lies in both component languages.
        self.assertTrue(a1.accepts(result.witness))
        self.assertTrue(a2.accepts(result.witness))

    def test_empty_intersection_when_letters_never_agree(self) -> None:
        """A1 forces {a} forever, A2 forces {b} forever: no common run, empty product."""
        a1 = BuchiAutomaton()
        a1.add_initial_state("p")
        a1.add_accepting_state("p")
        a1.add_transition("p", {"a"}, "p")

        a2 = BuchiAutomaton()
        a2.add_initial_state("r")
        a2.add_accepting_state("r")
        a2.add_transition("r", {"b"}, "r")

        product = intersect(a1, a2)
        self.assertTrue(product.is_empty())

    def test_intersection_with_self_equals_self_language(self) -> None:
        """Intersecting an automaton with itself preserves acceptance."""
        a = _infinitely_often("a", "b")
        product = intersect(a, a)
        word = OmegaWord(prefix=[], loop=[{"a"}])
        self.assertEqual(a.accepts(word), product.accepts(word))
        self.assertFalse(product.is_empty())

    def test_method_matches_function(self) -> None:
        a1 = _infinitely_often("a", "b")
        a2 = _infinitely_often("b", "a")
        word = OmegaWord(prefix=[], loop=[{"a"}, {"b"}])
        self.assertEqual(a1.intersect(a2).accepts(word), intersect(a1, a2).accepts(word))


if __name__ == "__main__":
    unittest.main()

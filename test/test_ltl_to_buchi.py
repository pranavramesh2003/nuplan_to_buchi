import unittest
from itertools import combinations, product

from tutorials.buchi.ltl import Atom, F, G, Implies, Next, Not, Or, R, U, W, X, atoms, satisfies
from tutorials.buchi.ltl_to_buchi import ltl_to_buchi, ltl_to_gba
from tutorials.buchi.omega_word import OmegaWord

P, Q = Atom("p"), Atom("q")


def _enumerate_lassos(ap, max_stem=2, max_loop=2):
    """All lassos over the alphabet 2^ap with bounded stem/loop length."""
    ordered = sorted(ap)
    alphabet = [frozenset(s) for r in range(len(ordered) + 1) for s in combinations(ordered, r)]
    words = []
    for stem_len in range(max_stem + 1):
        for loop_len in range(1, max_loop + 1):
            for stem in product(alphabet, repeat=stem_len):
                for loop in product(alphabet, repeat=loop_len):
                    words.append(OmegaWord(prefix=list(stem), loop=list(loop)))
    return words


class TestLTLToBuchiCorrectness(unittest.TestCase):
    """The generated automaton accepts a word iff the word satisfies the formula.

    Cross-checked against the independent reference semantics ``ltl.satisfies`` over an
    exhaustive set of small lasso words built from the formula's own alphabet.
    """

    FORMULAS = [
        P,
        Not(P),
        X(P),
        X(X(P)),
        F(P),
        G(P),
        U(P, Q),
        Not(U(P, Q)),
        R(P, Q),
        W(P, Q),
        G(F(P)),                 # infinitely often p
        F(G(P)),                 # eventually always p
        G(Implies(P, F(Q))),     # response: every p is eventually followed by q
        G(Implies(P, X(Q))),     # every p is immediately followed by q
        Or(G(F(P)), G(F(Q))),    # p infinitely often, or q infinitely often
        U(P, G(Q)),
        F(P) & G(Q),
    ]

    def test_accepts_matches_semantics(self) -> None:
        for formula in self.FORMULAS:
            ap = atoms(formula) or {"p"}
            buchi = ltl_to_buchi(formula)
            for word in _enumerate_lassos(ap):
                with self.subTest(formula=str(formula), word=str(word)):
                    self.assertEqual(buchi.accepts(word), satisfies(formula, word))


class TestLTLToBuchiStructure(unittest.TestCase):
    """Structural / emptiness sanity checks on the translation."""

    def test_satisfiable_formula_is_non_empty(self) -> None:
        self.assertFalse(ltl_to_buchi(G(F(P))).is_empty())
        self.assertFalse(ltl_to_buchi(U(P, Q)).is_empty())

    def test_contradiction_is_empty(self) -> None:
        """p ∧ ¬p has no model, so its automaton is empty."""
        self.assertTrue(ltl_to_buchi(P & Not(P)).is_empty())

    def test_g_false_is_empty(self) -> None:
        """G(p ∧ ¬p) is unsatisfiable."""
        self.assertTrue(ltl_to_buchi(G(P & Not(P))).is_empty())

    def test_witness_satisfies_formula(self) -> None:
        """The emptiness witness for a satisfiable formula actually models it."""
        formula = G(Implies(P, F(Q)))
        buchi = ltl_to_buchi(formula)
        result = buchi.check_emptiness()
        self.assertFalse(result.is_empty)
        self.assertTrue(satisfies(formula, result.witness))

    def test_gba_one_accepting_set_per_until(self) -> None:
        """The generalized automaton has exactly one accepting set per Until subformula."""
        # F p = true U p, and G(F p) = ¬(true U ¬(true U p)) -> two Until subformulas.
        gba = ltl_to_gba(G(F(P)))
        self.assertEqual(len(gba.accepting_sets), 2)
        # A purely propositional formula has no Until -> no acceptance constraints.
        self.assertEqual(len(ltl_to_gba(P & Q).accepting_sets), 0)

    def test_negation_complements_language(self) -> None:
        """A word satisfies φ iff it is rejected by the automaton for ¬φ."""
        formula = U(P, Q)
        buchi_pos = ltl_to_buchi(formula)
        buchi_neg = ltl_to_buchi(Not(formula))
        for word in _enumerate_lassos({"p", "q"}):
            self.assertNotEqual(
                buchi_pos.accepts(word),
                buchi_neg.accepts(word),
                f"φ and ¬φ both decided the same way on {word}",
            )


if __name__ == "__main__":
    unittest.main()

import unittest

from tutorials.buchi.ltl import (
    And,
    Always,
    Atom,
    BoolConst,
    Eventually,
    F,
    G,
    Implies,
    Next,
    Not,
    Or,
    Release,
    U,
    Until,
    WeakUntil,
    X,
    atoms,
    satisfies,
    subformulas,
)
from tutorials.buchi.omega_word import OmegaWord


class TestLTLFormula(unittest.TestCase):
    """Formula construction, equality/hashing, sugar, subformulas, core rewriting."""

    def test_equality_and_hashing(self) -> None:
        self.assertEqual(Atom("p"), Atom("p"))
        self.assertNotEqual(Atom("p"), Atom("q"))
        self.assertNotEqual(Atom("p"), Next(Atom("p")))  # different node types
        self.assertEqual(len({Atom("p"), Atom("p"), Atom("q")}), 2)

    def test_operator_sugar(self) -> None:
        p, q = Atom("p"), Atom("q")
        self.assertEqual(p & q, And(p, q))
        self.assertEqual(p | q, Or(p, q))
        self.assertEqual(~p, Not(p))
        self.assertEqual(p >> q, Implies(p, q))

    def test_subformulas_and_atoms(self) -> None:
        p, q = Atom("p"), Atom("q")
        formula = U(p, X(q))
        self.assertEqual(subformulas(formula), {formula, p, X(q), q})
        self.assertEqual(atoms(formula), {"p", "q"})

    def test_to_core_uses_only_core_operators(self) -> None:
        p, q = Atom("p"), Atom("q")
        core = G(p >> F(q)).to_core()
        allowed = (Atom, BoolConst, Not, And, Next, Until)
        for sub in subformulas(core):
            self.assertIsInstance(sub, allowed, f"{sub} is not a core operator")

    def test_to_core_preserves_semantics(self) -> None:
        """A formula and its core rewriting agree on every test word."""
        p, q = Atom("p"), Atom("q")
        formulas = [G(p), F(q), Release(p, q), WeakUntil(p, q), p >> X(q), G(F(p))]
        words = _enumerate_lassos({"p", "q"}, max_stem=1, max_loop=2)
        for formula in formulas:
            core = formula.to_core()
            for word in words:
                self.assertEqual(
                    satisfies(formula, word),
                    satisfies(core, word),
                    f"{formula} vs core on {word}",
                )


class TestLTLSemantics(unittest.TestCase):
    """Spot-checks of the reference semantics over lasso words."""

    def setUp(self) -> None:
        self.p, self.q = Atom("p"), Atom("q")

    def test_atom(self) -> None:
        self.assertTrue(satisfies(self.p, OmegaWord(prefix=[{"p"}], loop=[set()])))
        self.assertFalse(satisfies(self.p, OmegaWord(prefix=[set()], loop=[set()])))

    def test_next(self) -> None:
        word = OmegaWord(prefix=[set(), {"p"}], loop=[set()])
        self.assertTrue(satisfies(X(self.p), word))
        self.assertFalse(satisfies(self.p, word))

    def test_eventually_and_always(self) -> None:
        eventually = OmegaWord(prefix=[set(), set()], loop=[{"p"}])
        self.assertTrue(satisfies(F(self.p), eventually))
        always = OmegaWord(prefix=[], loop=[{"p"}])
        self.assertTrue(satisfies(G(self.p), always))
        self.assertFalse(satisfies(G(self.p), OmegaWord(prefix=[{"p"}], loop=[set()])))

    def test_until(self) -> None:
        # p holds until q becomes true.
        good = OmegaWord(prefix=[{"p"}, {"p"}, {"q"}], loop=[set()])
        self.assertTrue(satisfies(U(self.p, self.q), good))
        # q never becomes true -> until fails.
        bad = OmegaWord(prefix=[{"p"}], loop=[{"p"}])
        self.assertFalse(satisfies(U(self.p, self.q), bad))

    def test_gf_is_infinitely_often(self) -> None:
        infinitely = OmegaWord(prefix=[], loop=[{"p"}, set()])
        self.assertTrue(satisfies(G(F(self.p)), infinitely))
        finitely = OmegaWord(prefix=[{"p"}], loop=[set()])
        self.assertFalse(satisfies(G(F(self.p)), finitely))


def _enumerate_lassos(ap, max_stem=1, max_loop=2):
    """All lassos with stem length <= max_stem and loop length in [1, max_loop]."""
    from itertools import combinations, product

    ordered = sorted(ap)
    alphabet = [frozenset(s) for r in range(len(ordered) + 1) for s in combinations(ordered, r)]
    words = []
    for stem_len in range(max_stem + 1):
        for loop_len in range(1, max_loop + 1):
            for stem in product(alphabet, repeat=stem_len):
                for loop in product(alphabet, repeat=loop_len):
                    words.append(OmegaWord(prefix=list(stem), loop=list(loop)))
    return words


if __name__ == "__main__":
    unittest.main()

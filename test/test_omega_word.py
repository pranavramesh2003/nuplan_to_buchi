import unittest

from tutorials.buchi.omega_word import OmegaWord


class TestOmegaWord(unittest.TestCase):
    """Tests for the OmegaWord lasso representation."""

    def test_letters_normalized_to_frozenset(self) -> None:
        """Each letter (an arbitrary iterable) is stored as a frozenset, in tuples."""
        word = OmegaWord(prefix=[{"a"}, ["b", "b"]], loop=[("c",)])
        self.assertEqual(word.prefix, (frozenset({"a"}), frozenset({"b"})))
        self.assertEqual(word.loop, (frozenset({"c"}),))
        self.assertIsInstance(word.prefix, tuple)
        self.assertIsInstance(word.loop, tuple)

    def test_empty_loop_raises(self) -> None:
        """An omega-word needs a non-empty recurrent part."""
        with self.assertRaises(ValueError):
            OmegaWord(prefix=[{"a"}], loop=[])

    def test_empty_prefix_allowed(self) -> None:
        """A purely periodic word (empty stem) is valid."""
        word = OmegaWord(prefix=[], loop=[{"a"}])
        self.assertEqual(word.prefix, ())
        self.assertEqual(word.period, 1)

    def test_unroll_crosses_prefix_loop_boundary(self) -> None:
        """unroll(n) takes the prefix first, then cycles the loop."""
        word = OmegaWord(prefix=[{"a"}], loop=[{"b"}, {"c"}])
        self.assertEqual(word.unroll(0), [])
        self.assertEqual(
            word.unroll(5),
            [frozenset({"a"}), frozenset({"b"}), frozenset({"c"}), frozenset({"b"}), frozenset({"c"})],
        )

    def test_unroll_negative_raises(self) -> None:
        word = OmegaWord(prefix=[], loop=[{"a"}])
        with self.assertRaises(ValueError):
            word.unroll(-1)

    def test_equality_and_hash(self) -> None:
        """Structural equality on (prefix, loop); equal words hash equally."""
        a = OmegaWord(prefix=[{"x"}], loop=[{"y"}])
        b = OmegaWord(prefix=[["x"]], loop=[("y",)])
        c = OmegaWord(prefix=[], loop=[{"y"}])
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b, c}), 2)

    def test_str_renders_lasso(self) -> None:
        word = OmegaWord(prefix=[{"a"}], loop=[{"b"}])
        self.assertEqual(str(word), "{a}·({b})ω")
        periodic = OmegaWord(prefix=[], loop=[{"c"}])
        self.assertEqual(str(periodic), "({c})ω")


if __name__ == "__main__":
    unittest.main()

"""Representation of infinite (omega-) words as lassos.

A Buchi automaton reads *infinite* words. While the set of infinite words over an
alphabet is uncountable, the words that matter for emptiness/model-checking are the
*ultimately periodic* ones: a finite non-recurrent prefix ``u`` followed by a finite
block ``v`` repeated forever. We write such a word as ``u . v^omega`` and call it a
*lasso* (the prefix is the stem, the loop is the cycle).

Every letter of the word is a **set** of arbitrary objects (the classic
"a letter is a subset of the atomic propositions" model). Letters are normalized to
``frozenset`` so that words are hashable and comparable.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

# A single letter of the alphabet: a set of arbitrary (hashable) objects.
Letter = frozenset


class OmegaWord:
    """An ultimately periodic infinite word ``prefix . loop^omega`` (a lasso).

    :param prefix: the non-recurrent part ``u`` (may be empty); a sequence of letters,
        where each letter is any iterable of objects.
    :param loop: the recurrent part ``v`` that repeats forever; must be non-empty.
    :raises ValueError: if ``loop`` is empty (an infinite word needs something to repeat).
    """

    def __init__(self, prefix: Sequence[Iterable], loop: Sequence[Iterable]) -> None:
        loop_letters = tuple(self._as_letter(letter) for letter in loop)
        if not loop_letters:
            raise ValueError("The loop (recurrent part) of an omega-word must be non-empty.")
        self.prefix: Tuple[Letter, ...] = tuple(self._as_letter(letter) for letter in prefix)
        self.loop: Tuple[Letter, ...] = loop_letters

    @staticmethod
    def _as_letter(letter: Iterable) -> Letter:
        """Normalize an arbitrary iterable into a ``frozenset`` letter."""
        return frozenset(letter)

    @property
    def period(self) -> int:
        """Length of the recurrent block ``v``."""
        return len(self.loop)

    @property
    def is_ultimately_periodic(self) -> bool:
        """Lasso words are by construction ultimately periodic; always ``True``."""
        return True

    def unroll(self, n: int) -> List[Letter]:
        """Return the first ``n`` letters of the infinite word.

        Letters are taken from the prefix first, then from the loop cycled indefinitely.

        :param n: number of letters to materialize (``n >= 0``).
        :return: a list of the first ``n`` letters.
        """
        if n < 0:
            raise ValueError("Cannot unroll a negative number of letters.")
        letters: List[Letter] = list(self.prefix[:n])
        while len(letters) < n:
            remaining = n - len(letters)
            letters.extend(self.loop[:remaining])
        return letters

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OmegaWord):
            return NotImplemented
        return self.prefix == other.prefix and self.loop == other.loop

    def __hash__(self) -> int:
        return hash((self.prefix, self.loop))

    @staticmethod
    def _format_letter(letter: Letter) -> str:
        """Render a single letter as ``{a, b}`` with a stable element order."""
        if not letter:
            return "{}"
        elements = sorted(str(element) for element in letter)
        return "{" + ", ".join(elements) + "}"

    def __str__(self) -> str:
        loop = "(" + "".join(self._format_letter(letter) for letter in self.loop) + ")ω"
        if not self.prefix:
            return loop
        prefix = "".join(self._format_letter(letter) for letter in self.prefix)
        return f"{prefix}·{loop}"

    def __repr__(self) -> str:
        return f"OmegaWord(prefix={list(self.prefix)!r}, loop={list(self.loop)!r})"

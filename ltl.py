"""Linear Temporal Logic (LTL): syntax, rewriting to a core fragment, and semantics.

LTL formulas are built over atomic propositions with the Boolean connectives and the
temporal operators ``X`` (next), ``U`` (until), ``F`` (eventually), ``G`` (always),
``R`` (release) and ``W`` (weak until). Formulas are immutable, hashable values so they
can serve as dictionary keys and set members (the LTL→Büchi construction relies on this).

Two things live here besides the syntax:

* :meth:`LTLFormula.to_core` rewrites any formula into the **core fragment**
  ``{atom, true, ¬, ∧, X, U}``. Every other operator is syntactic sugar over it
  (``F φ = true U φ``, ``G φ = ¬(true U ¬φ)``, ``φ R ψ = ¬(¬φ U ¬ψ)``, …). The
  tableau construction in :mod:`ltl_to_buchi` only has to handle the core.
* :func:`satisfies` is a direct, operator-complete semantics over *lasso* words
  (:class:`omega_word.OmegaWord`). It is independent of the automaton construction and is
  used to cross-check that the generated Büchi automaton accepts exactly the models of φ.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Set, Tuple

from .omega_word import OmegaWord


class LTLFormula:
    """Base class for LTL formulas; provides operator sugar and shared helpers."""

    # --- operator overloading so formulas can be written as a & b, a | b, ~a, a >> b ---
    def __and__(self, other: "LTLFormula") -> "And":
        return And(self, other)

    def __or__(self, other: "LTLFormula") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def __rshift__(self, other: "LTLFormula") -> "Implies":
        return Implies(self, other)

    def to_core(self) -> "LTLFormula":
        """Rewrite into the core fragment ``{atom, true, ¬, ∧, X, U}``."""
        raise NotImplementedError

    def __str__(self) -> str:
        return _format(self)


# --------------------------------------------------------------------------- atoms
@dataclass(frozen=True)
class Atom(LTLFormula):
    """An atomic proposition, identified by name."""

    name: str

    def to_core(self) -> LTLFormula:
        return self


@dataclass(frozen=True)
class BoolConst(LTLFormula):
    """The Boolean constant ``true`` (``value=True``) or ``false`` (``value=False``)."""

    value: bool

    def to_core(self) -> LTLFormula:
        return self if self.value else Not(BoolConst(True))


TRUE = BoolConst(True)
FALSE = BoolConst(False)


# ----------------------------------------------------------------- boolean operators
@dataclass(frozen=True)
class Not(LTLFormula):
    operand: LTLFormula

    def to_core(self) -> LTLFormula:
        return Not(self.operand.to_core())


@dataclass(frozen=True)
class And(LTLFormula):
    left: LTLFormula
    right: LTLFormula

    def to_core(self) -> LTLFormula:
        return And(self.left.to_core(), self.right.to_core())


@dataclass(frozen=True)
class Or(LTLFormula):
    left: LTLFormula
    right: LTLFormula

    def to_core(self) -> LTLFormula:
        # a ∨ b = ¬(¬a ∧ ¬b)
        return Not(And(Not(self.left.to_core()), Not(self.right.to_core())))


@dataclass(frozen=True)
class Implies(LTLFormula):
    left: LTLFormula
    right: LTLFormula

    def to_core(self) -> LTLFormula:
        # a → b = ¬(a ∧ ¬b)
        return Not(And(self.left.to_core(), Not(self.right.to_core())))


# ---------------------------------------------------------------- temporal operators
@dataclass(frozen=True)
class Next(LTLFormula):
    operand: LTLFormula

    def to_core(self) -> LTLFormula:
        return Next(self.operand.to_core())


@dataclass(frozen=True)
class Until(LTLFormula):
    left: LTLFormula
    right: LTLFormula

    def to_core(self) -> LTLFormula:
        return Until(self.left.to_core(), self.right.to_core())


@dataclass(frozen=True)
class Eventually(LTLFormula):
    operand: LTLFormula

    def to_core(self) -> LTLFormula:
        # F a = true U a
        return Until(BoolConst(True), self.operand.to_core())


@dataclass(frozen=True)
class Always(LTLFormula):
    operand: LTLFormula

    def to_core(self) -> LTLFormula:
        # G a = ¬F¬a = ¬(true U ¬a)
        return Not(Until(BoolConst(True), Not(self.operand.to_core())))


@dataclass(frozen=True)
class Release(LTLFormula):
    left: LTLFormula
    right: LTLFormula

    def to_core(self) -> LTLFormula:
        # a R b = ¬(¬a U ¬b)
        return Not(Until(Not(self.left.to_core()), Not(self.right.to_core())))


@dataclass(frozen=True)
class WeakUntil(LTLFormula):
    left: LTLFormula
    right: LTLFormula

    def to_core(self) -> LTLFormula:
        # a W b = ¬(¬b U (¬a ∧ ¬b))
        a, b = self.left.to_core(), self.right.to_core()
        return Not(Until(Not(b), And(Not(a), Not(b))))


# ----------------------------------------------------------------- convenience builders
def Var(name: str) -> Atom:
    """Shorthand for :class:`Atom`."""
    return Atom(name)


def X(formula: LTLFormula) -> Next:
    return Next(formula)


def F(formula: LTLFormula) -> Eventually:
    return Eventually(formula)


def G(formula: LTLFormula) -> Always:
    return Always(formula)


def U(left: LTLFormula, right: LTLFormula) -> Until:
    return Until(left, right)


def R(left: LTLFormula, right: LTLFormula) -> Release:
    return Release(left, right)


def W(left: LTLFormula, right: LTLFormula) -> WeakUntil:
    return WeakUntil(left, right)


# --------------------------------------------------------------------------- helpers
def _children(formula: LTLFormula) -> List[LTLFormula]:
    """Immediate subformulas of ``formula``."""
    if isinstance(formula, (Atom, BoolConst)):
        return []
    if isinstance(formula, (Not, Next, Eventually, Always)):
        return [formula.operand]
    if isinstance(formula, (And, Or, Implies, Until, Release, WeakUntil)):
        return [formula.left, formula.right]
    raise TypeError(f"Unknown formula type: {type(formula).__name__}")


def subformulas(formula: LTLFormula) -> Set[LTLFormula]:
    """Return ``formula`` together with all of its (transitive) subformulas."""
    result: Set[LTLFormula] = {formula}
    for child in _children(formula):
        result |= subformulas(child)
    return result


def atoms(formula: LTLFormula) -> Set[str]:
    """Return the names of all atomic propositions occurring in ``formula``."""
    return {sub.name for sub in subformulas(formula) if isinstance(sub, Atom)}


def _format(formula: LTLFormula) -> str:
    """Render a formula as a readable infix/prefix string (mostly for diagnostics)."""
    if isinstance(formula, Atom):
        return formula.name
    if isinstance(formula, BoolConst):
        return "true" if formula.value else "false"
    if isinstance(formula, Not):
        return f"¬{_format(formula.operand)}"
    if isinstance(formula, Next):
        return f"X {_format(formula.operand)}"
    if isinstance(formula, Eventually):
        return f"F {_format(formula.operand)}"
    if isinstance(formula, Always):
        return f"G {_format(formula.operand)}"
    binary = {And: "∧", Or: "∨", Implies: "→", Until: "U", Release: "R", WeakUntil: "W"}
    symbol = binary[type(formula)]
    return f"({_format(formula.left)} {symbol} {_format(formula.right)})"


# --------------------------------------------------------------------------- semantics
def satisfies(formula: LTLFormula, word: OmegaWord) -> bool:
    """Evaluate the LTL semantics of ``formula`` at position 0 of a lasso ``word``.

    This is an operator-complete reference semantics over ultimately periodic words. A
    lasso ``u·v^omega`` has finitely many distinct positions: the ``len(u)`` stem
    positions followed by a cycle over the ``len(v)`` loop positions. Each subformula's
    truth value at each position is computed bottom-up; the temporal operators are
    resolved by a fixed-point iteration over those positions (least fixed point for
    ``U``/``F``, greatest for ``G``/``R``/``W``).

    :return: ``True`` iff ``word`` satisfies ``formula``.
    """
    stem = len(word.prefix)
    period = word.period
    num_positions = stem + period

    def letter_at(index: int) -> frozenset:
        return word.prefix[index] if index < stem else word.loop[(index - stem) % period]

    def successor(index: int) -> int:
        # The last position loops back to the first loop position (start of the cycle).
        return index + 1 if index < num_positions - 1 else stem

    # Process subformulas smallest-first so each operator sees finalized operand values.
    ordered = sorted(subformulas(formula), key=lambda f: len(subformulas(f)))
    value: Dict[Tuple[int, LTLFormula], bool] = {}

    for sub in ordered:
        positions = range(num_positions)
        if isinstance(sub, Atom):
            for i in positions:
                value[(i, sub)] = sub.name in letter_at(i)
        elif isinstance(sub, BoolConst):
            for i in positions:
                value[(i, sub)] = sub.value
        elif isinstance(sub, Not):
            for i in positions:
                value[(i, sub)] = not value[(i, sub.operand)]
        elif isinstance(sub, And):
            for i in positions:
                value[(i, sub)] = value[(i, sub.left)] and value[(i, sub.right)]
        elif isinstance(sub, Or):
            for i in positions:
                value[(i, sub)] = value[(i, sub.left)] or value[(i, sub.right)]
        elif isinstance(sub, Implies):
            for i in positions:
                value[(i, sub)] = (not value[(i, sub.left)]) or value[(i, sub.right)]
        elif isinstance(sub, Next):
            for i in positions:
                value[(i, sub)] = value[(successor(i), sub.operand)]
        else:
            _fixpoint_temporal(sub, positions, successor, value)

    return value[(0, formula)]


def _fixpoint_temporal(sub, positions, successor, value) -> None:
    """Resolve a temporal operator over the lasso positions by fixed-point iteration."""
    greatest = isinstance(sub, (Always, Release, WeakUntil))
    for i in positions:
        value[(i, sub)] = greatest  # least fixed point starts at False, greatest at True.

    def step(i: int) -> bool:
        nxt = value[(successor(i), sub)]
        if isinstance(sub, Eventually):
            return value[(i, sub.operand)] or nxt
        if isinstance(sub, Always):
            return value[(i, sub.operand)] and nxt
        if isinstance(sub, Until):
            return value[(i, sub.right)] or (value[(i, sub.left)] and nxt)
        if isinstance(sub, Release):
            return value[(i, sub.right)] and (value[(i, sub.left)] or nxt)
        if isinstance(sub, WeakUntil):
            return value[(i, sub.right)] or (value[(i, sub.left)] and nxt)
        raise TypeError(f"Not a temporal operator: {type(sub).__name__}")

    changed = True
    while changed:
        changed = False
        for i in positions:
            new_value = step(i)
            if new_value != value[(i, sub)]:
                value[(i, sub)] = new_value
                changed = True

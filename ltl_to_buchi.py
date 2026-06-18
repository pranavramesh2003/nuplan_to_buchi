"""Translation of an LTL formula into an equivalent Büchi automaton.

The classic tableau / *closure* construction (Vardi–Wolper, as presented in Baier &
Katoen, *Principles of Model Checking*):

1. Rewrite ``φ`` into the core fragment ``{atom, true, ¬, ∧, X, U}`` (see
   :meth:`ltl.LTLFormula.to_core`).
2. Form the **closure** ``cl(φ)`` — every core subformula together with its negation.
3. Enumerate the **elementary sets**: maximal, locally consistent subsets ``B ⊆ cl(φ)``.
   Each such set is a state and represents a guess about which subformulas hold "now".
4. Build a **generalized Büchi automaton** over those states:
   * alphabet letters are sets of true atomic propositions; a state ``B`` emits the single
     letter ``B ∩ AP``;
   * transitions encode the one-step semantics of ``X`` and the expansion law of ``U``
     (``φ1 U φ2 ≡ φ2 ∨ (φ1 ∧ X(φ1 U φ2))``);
   * there is one accepting set per ``Until`` subformula, forcing every "until" promise to
     eventually be fulfilled.
5. **Degeneralize** the GBA to an ordinary Büchi automaton (see
   :mod:`generalized_buchi`).

The resulting automaton accepts exactly the lasso words (and, in general, the ω-words)
that satisfy ``φ`` — verified against the reference semantics :func:`ltl.satisfies`.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, FrozenSet, List, Set

from .buchi_automaton import BuchiAutomaton
from .generalized_buchi import GeneralizedBuchiAutomaton
from .ltl import And, Atom, BoolConst, LTLFormula, Next, Not, Until, subformulas

# An elementary set: the subformulas (of the closure) held to be true at a state.
ElementarySet = FrozenSet[LTLFormula]


def negate(formula: LTLFormula) -> LTLFormula:
    """Negation with double-negation collapse (``¬¬ψ`` is identified with ``ψ``)."""
    return formula.operand if isinstance(formula, Not) else Not(formula)


def closure(core_formula: LTLFormula) -> Set[LTLFormula]:
    """The closure ``cl(φ)``: core subformulas and their negations, plus ``true``."""
    result: Set[LTLFormula] = set(subformulas(core_formula))
    result.add(BoolConst(True))
    for sub in list(result):
        result.add(negate(sub))
    return result


def _evaluate(formula: LTLFormula, assignment: Dict[LTLFormula, bool]) -> bool:
    """Truth of a core formula given truth values for the freely-chosen subformulas.

    ``assignment`` fixes the value of every atom, ``X``- and ``U``-subformula; the value
    of ``true``, ``¬`` and ``∧`` is then determined.
    """
    if isinstance(formula, BoolConst):
        return formula.value
    if isinstance(formula, (Atom, Next, Until)):
        return assignment[formula]
    if isinstance(formula, Not):
        return not _evaluate(formula.operand, assignment)
    if isinstance(formula, And):
        return _evaluate(formula.left, assignment) and _evaluate(formula.right, assignment)
    raise TypeError(f"Non-core formula reached evaluation: {type(formula).__name__}")


def elementary_sets(core_formula: LTLFormula) -> List[ElementarySet]:
    """Enumerate the elementary sets of ``cl(core_formula)``.

    The independent choices are the truth values of the atoms and the ``X``/``U``
    subformulas; ``∧`` and ``¬`` are then forced. Assignments are kept only when they are
    *locally consistent* w.r.t. the until-expansion law.
    """
    cl = closure(core_formula)
    subs = subformulas(core_formula) | {BoolConst(True)}
    choices = [s for s in subs if isinstance(s, (Atom, Next, Until))]
    untils = [s for s in subs if isinstance(s, Until)]

    sets: List[ElementarySet] = []
    for bits in product((False, True), repeat=len(choices)):
        assignment = dict(zip(choices, bits))

        # Local until-consistency: φ2 ⇒ (φ1 U φ2); (φ1 U φ2) ∧ ¬φ2 ⇒ φ1.
        consistent = True
        for u in untils:
            holds = assignment[u]
            right = _evaluate(u.right, assignment)
            left = _evaluate(u.left, assignment)
            if right and not holds:
                consistent = False
                break
            if holds and not right and not left:
                consistent = False
                break
        if not consistent:
            continue

        members = frozenset(f for f in cl if _evaluate(f, assignment))
        sets.append(members)

    return sets


def ltl_to_gba(formula: LTLFormula) -> GeneralizedBuchiAutomaton:
    """Build a generalized Büchi automaton recognising the models of ``formula``."""
    core = formula.to_core()
    subs = subformulas(core) | {BoolConst(True)}
    ap = sorted({s.name for s in subs if isinstance(s, Atom)})
    nexts = [s for s in subs if isinstance(s, Next)]
    untils = [s for s in subs if isinstance(s, Until)]
    states = elementary_sets(core)

    gba = GeneralizedBuchiAutomaton()
    for state in states:
        gba.add_state(state)
    for state in states:
        if core in state:
            gba.add_initial_state(state)

    for source in states:
        label = frozenset(name for name in ap if Atom(name) in source)
        for target in states:
            # Next: X ψ ∈ source  ⟺  ψ ∈ target.
            if any((n in source) != (n.operand in target) for n in nexts):
                continue
            # Until expansion: (φ1 U φ2) ∈ source ⟺ φ2 ∈ source ∨ (φ1 ∈ source ∧ U ∈ target).
            valid = True
            for u in untils:
                expanded = (u.right in source) or (u.left in source and u in target)
                if (u in source) != expanded:
                    valid = False
                    break
            if valid:
                gba.add_transition(source, label, target)

    # One accepting set per until: states that do not owe an unfulfilled φ1 U φ2.
    for u in untils:
        gba.add_accepting_set({s for s in states if (u not in s) or (u.right in s)})

    return gba


def ltl_to_buchi(formula: LTLFormula) -> BuchiAutomaton:
    """Translate ``formula`` into an ordinary Büchi automaton accepting its models.

    Alphabet letters are ``frozenset`` of the atomic-proposition names that hold at a
    step (a letter is a subset of the formula's atoms). The returned automaton accepts a
    lasso word iff that word satisfies ``formula`` (see :func:`ltl.satisfies`).
    """
    return ltl_to_gba(formula).to_buchi()

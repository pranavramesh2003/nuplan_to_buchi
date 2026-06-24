"""On-the-fly LTL → Büchi automaton translation (GPVW 1996).

R. Gerth, D. Peled, M. Y. Vardi, and P. Wolper.
"Simple On-the-fly Automatic Verification of Linear Temporal Logic."
PSTV/FORTE 1996.

The algorithm works on formulas in Negation Normal Form (NNF) and builds states
on-the-fly by expanding sets of obligations.  Each node in the intermediate graph
carries three bookkeeping sets:

  old   – formulas already processed at this step ("decided true here")
  now   – formulas yet to be processed at this step
  next  – formulas deferred to the next step

When ``now`` becomes empty, the node is finalized and added to the Generalized
Büchi Automaton (GBA).  Two nodes with the same ``old`` and ``next`` are merged.
Accepting sets: one per Until sub-formula ``φ U ψ``, containing states where
either ``φ U ψ ∉ old`` (no obligation) or ``ψ ∈ old`` (obligation discharged).
"""

from __future__ import annotations

from itertools import chain, combinations
from dataclasses import dataclass
from typing import FrozenSet, List, Set, Tuple

from .buchi_automaton import BuchiAutomaton
from .generalized_buchi import GeneralizedBuchiAutomaton
from .ltl import (
    And, Atom, BoolConst, Eventually, Always, Implies, LTLFormula,
    Next, Not, Or, Release, Until, WeakUntil, subformulas,
)

# ── Sentinel for the initial incoming source ──────────────────────────────────
INIT = "init"


# ── NNF conversion ────────────────────────────────────────────────────────────

def to_nnf(phi: LTLFormula) -> LTLFormula:
    """Rewrite *phi* into Negation Normal Form — push ¬ inward to literals."""
    if isinstance(phi, (Atom, BoolConst)):
        return phi
    if isinstance(phi, Not):
        return _neg(phi.operand)
    if isinstance(phi, And):
        return And(to_nnf(phi.left), to_nnf(phi.right))
    if isinstance(phi, Or):
        return Or(to_nnf(phi.left), to_nnf(phi.right))
    if isinstance(phi, Next):
        return Next(to_nnf(phi.operand))
    if isinstance(phi, Until):
        return Until(to_nnf(phi.left), to_nnf(phi.right))
    if isinstance(phi, Release):
        return Release(to_nnf(phi.left), to_nnf(phi.right))
    if isinstance(phi, Eventually):        # F φ = true U φ
        return Until(BoolConst(True), to_nnf(phi.operand))
    if isinstance(phi, Always):            # G φ = false R φ
        return Release(BoolConst(False), to_nnf(phi.operand))
    if isinstance(phi, Implies):
        return to_nnf(Or(Not(phi.left), phi.right))
    if isinstance(phi, WeakUntil):         # φ W ψ = (φ U ψ) ∨ (false R φ)
        a, b = to_nnf(phi.left), to_nnf(phi.right)
        return Or(Until(a, b), Release(BoolConst(False), a))
    raise TypeError(f"Unknown formula type: {type(phi).__name__}")


def _neg(phi: LTLFormula) -> LTLFormula:
    """NNF of ¬*phi* — push the negation inward one step."""
    if isinstance(phi, BoolConst):
        return BoolConst(not phi.value)
    if isinstance(phi, Atom):
        return Not(phi)
    if isinstance(phi, Not):
        return to_nnf(phi.operand)
    if isinstance(phi, And):
        return Or(_neg(phi.left), _neg(phi.right))
    if isinstance(phi, Or):
        return And(_neg(phi.left), _neg(phi.right))
    if isinstance(phi, Next):
        return Next(_neg(phi.operand))
    if isinstance(phi, Until):
        return Release(_neg(phi.left), _neg(phi.right))
    if isinstance(phi, Release):
        return Until(_neg(phi.left), _neg(phi.right))
    if isinstance(phi, Eventually):
        return Release(BoolConst(False), _neg(phi.operand))
    if isinstance(phi, Always):
        return Until(BoolConst(True), _neg(phi.operand))
    if isinstance(phi, Implies):
        return And(to_nnf(phi.left), _neg(phi.right))
    if isinstance(phi, WeakUntil):
        # ¬(φ W ψ) = ¬(φ U ψ) ∧ ¬(G φ) = (¬φ R ¬ψ) ∧ (true U ¬φ)
        a, b = _neg(phi.left), _neg(phi.right)
        return And(Release(a, b), Until(BoolConst(True), a))
    raise TypeError(f"Unknown formula type: {type(phi).__name__}")


# ── Intermediate node ─────────────────────────────────────────────────────────

@dataclass
class GPVWNode:
    """An intermediate node produced by the GPVW expansion.

    After the expansion terminates, ``now`` is always ∅ and each node becomes a
    state of the Generalized Büchi Automaton.

    Attributes
    ----------
    name:     unique integer identifier
    incoming: set of node names (ints) or the ``INIT`` sentinel whose transitions
              enter this node
    now:      formulas yet to be processed at this step (empty in final nodes)
    old:      formulas already processed ("what holds at this step")
    next_set: formulas forwarded to the next step
    """
    name: int
    incoming: Set
    now: FrozenSet[LTLFormula]
    old: FrozenSet[LTLFormula]
    next_set: FrozenSet[LTLFormula]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_untils(phi: LTLFormula) -> List[Until]:
    """Collect every Until sub-formula of *phi* (in a fixed deterministic order)."""
    return sorted(
        [s for s in subformulas(phi) if isinstance(s, Until)],
        key=str,
    )


def _all_atom_names(phi: LTLFormula) -> FrozenSet[str]:
    """Return the names of every Atom appearing in *phi*."""
    return frozenset(s.name for s in subformulas(phi) if isinstance(s, Atom))


def _compatible_letters(
    old: FrozenSet[LTLFormula],
    all_atoms: FrozenSet[str],
) -> List[FrozenSet[str]]:
    """All alphabet letters consistent with the literals in *old*.

    An atom ``p`` is *forced true* if ``Atom(p) ∈ old``, *forced false* if
    ``Not(Atom(p)) ∈ old``, and *free* otherwise.  Returns one letter per
    choice of truth-value for the free atoms.
    """
    pos = frozenset(f.name for f in old if isinstance(f, Atom))
    neg = frozenset(
        f.operand.name for f in old
        if isinstance(f, Not) and isinstance(f.operand, Atom)
    )
    free = all_atoms - pos - neg
    return [
        pos | frozenset(extra)
        for extra in chain.from_iterable(
            combinations(sorted(free), r) for r in range(len(free) + 1)
        )
    ]


# ── Core GPVW expansion ───────────────────────────────────────────────────────

def _expand(
    q: GPVWNode,
    nodes: List[GPVWNode],
    counter: List[int],
) -> List[GPVWNode]:
    """Recursively expand node *q* and return the updated Nodes list."""

    def fresh() -> int:
        counter[0] += 1
        return counter[0]

    # ── Base case: now = ∅ ────────────────────────────────────────────────────
    if not q.now:
        # Merge with an existing node that has the same old and next.
        for existing in nodes:
            if existing.old == q.old and existing.next_set == q.next_set:
                existing.incoming |= q.incoming
                return nodes
        # No match: add this node to Nodes and start expanding the next step.
        nodes = nodes + [q]
        nxt = GPVWNode(
            name=fresh(),
            incoming={q.name},
            now=q.next_set,
            old=frozenset(),
            next_set=frozenset(),
        )
        return _expand(nxt, nodes, counter)

    # ── Pick the lexicographically smallest formula from now ──────────────────
    eta = min(q.now, key=str)
    now_rest = q.now - {eta}
    old_new = q.old | {eta}

    # ── Case analysis on η ────────────────────────────────────────────────────
    if isinstance(eta, BoolConst):
        if eta.value:   # true: trivially satisfied, continue
            return _expand(
                GPVWNode(q.name, q.incoming, now_rest, old_new, q.next_set),
                nodes, counter,
            )
        else:           # false: dead branch
            return nodes

    if isinstance(eta, (Atom, Not)):   # literal (atom or negated atom)
        neg = Not(eta) if isinstance(eta, Atom) else eta.operand
        if neg in q.old:               # contradiction with old
            return nodes
        return _expand(
            GPVWNode(q.name, q.incoming, now_rest, old_new, q.next_set),
            nodes, counter,
        )

    if isinstance(eta, And):
        new_now = now_rest | (frozenset({eta.left, eta.right}) - old_new)
        return _expand(
            GPVWNode(q.name, q.incoming, new_now, old_new, q.next_set),
            nodes, counter,
        )

    if isinstance(eta, Next):
        return _expand(
            GPVWNode(q.name, q.incoming, now_rest, old_new, q.next_set | {eta.operand}),
            nodes, counter,
        )

    if isinstance(eta, Until):
        phi, psi = eta.left, eta.right
        # Branch A: ψ holds now (Until discharged immediately).
        q1 = GPVWNode(
            fresh(), set(q.incoming),
            now_rest | (frozenset({psi}) - old_new),
            old_new, q.next_set,
        )
        # Branch B: φ holds now and Until is deferred to next step.
        q2 = GPVWNode(
            fresh(), set(q.incoming),
            now_rest | (frozenset({phi}) - old_new),
            old_new, q.next_set | {eta},
        )
        nodes = _expand(q1, nodes, counter)
        return _expand(q2, nodes, counter)

    if isinstance(eta, Release):
        phi, psi = eta.left, eta.right
        # Branch A: both φ ∧ ψ hold now (Release terminates).
        q1 = GPVWNode(
            fresh(), set(q.incoming),
            now_rest | (frozenset({phi, psi}) - old_new),
            old_new, q.next_set,
        )
        # Branch B: ψ holds now and Release is deferred to next step.
        q2 = GPVWNode(
            fresh(), set(q.incoming),
            now_rest | (frozenset({psi}) - old_new),
            old_new, q.next_set | {eta},
        )
        nodes = _expand(q1, nodes, counter)
        return _expand(q2, nodes, counter)

    if isinstance(eta, Or):
        q1 = GPVWNode(
            fresh(), set(q.incoming),
            now_rest | (frozenset({eta.left}) - old_new),
            old_new, q.next_set,
        )
        q2 = GPVWNode(
            fresh(), set(q.incoming),
            now_rest | (frozenset({eta.right}) - old_new),
            old_new, q.next_set,
        )
        nodes = _expand(q1, nodes, counter)
        return _expand(q2, nodes, counter)

    raise TypeError(
        f"Unexpected formula type in NNF expansion: {type(eta).__name__}: {eta}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def ltl_to_gba_gpvw(
    formula: LTLFormula,
) -> Tuple[GeneralizedBuchiAutomaton, List[GPVWNode]]:
    """Translate *formula* to a GBA via the GPVW on-the-fly algorithm.

    Parameters
    ----------
    formula:
        Any LTL formula (converted internally to NNF).

    Returns
    -------
    gba:
        The Generalized Büchi Automaton whose language equals the models of
        *formula*.  States are the integer ``name`` fields of the GPVW nodes.
    nodes:
        The final GPVW nodes (``now = ∅``), each carrying its ``old`` and
        ``next_set`` bookkeeping sets for inspection and visualization.
    """
    nnf = to_nnf(formula)
    untils = _all_untils(nnf)
    all_atoms = _all_atom_names(nnf)
    counter = [0]

    def fresh() -> int:
        counter[0] += 1
        return counter[0]

    init_node = GPVWNode(
        name=fresh(),
        incoming={INIT},
        now=frozenset({nnf}),
        old=frozenset(),
        next_set=frozenset(),
    )
    final_nodes = _expand(init_node, [], counter)

    # ── Build the GBA ─────────────────────────────────────────────────────────
    gba = GeneralizedBuchiAutomaton()

    for node in final_nodes:
        gba.add_state(node.name)

    for node in final_nodes:
        if INIT in node.incoming:
            gba.add_initial_state(node.name)

    # Transitions: for each node q, add one transition per compatible letter
    # to every node q' that has q.name in its incoming set.
    for node in final_nodes:
        for label in _compatible_letters(node.old, all_atoms):
            for target in final_nodes:
                if node.name in target.incoming:
                    gba.add_transition(node.name, label, target.name)

    # Accepting sets: one per Until sub-formula.
    # A node satisfies the condition for "φ U ψ" if it carries no unfulfilled
    # Until obligation (U ∉ old) or the right-hand side already holds (ψ ∈ old).
    for u in untils:
        acc = {
            n.name for n in final_nodes
            if (u not in n.old) or (u.right in n.old)
        }
        gba.add_accepting_set(acc)

    return gba, final_nodes


def formula_untils(formula: LTLFormula) -> List[Until]:
    """Return the Until sub-formulas of *formula*'s NNF (used for accepting-set annotation)."""
    return _all_untils(to_nnf(formula))


def ltl_to_buchi_gpvw(formula: LTLFormula) -> BuchiAutomaton:
    """Translate *formula* to an ordinary Büchi automaton via GPVW.

    This is the main entry point.  The returned automaton accepts a lasso word
    iff that word satisfies *formula* (cross-checked against
    :func:`ltl.satisfies`).
    """
    gba, _ = ltl_to_gba_gpvw(formula)
    return gba.to_buchi()

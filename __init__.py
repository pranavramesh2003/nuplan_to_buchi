"""Buchi automata: construction, omega-word (lasso) representation, and emptiness.

See :mod:`buchi_automaton` for the automaton and SCC-based emptiness check,
:mod:`omega_word` for the infinite-word representation, :mod:`tarjan` for the underlying
strongly-connected-components algorithm, :mod:`generalized_buchi` for generalized
acceptance and degeneralization, and :mod:`ltl` / :mod:`gpvw_ltl_to_buchi` for linear
temporal logic and its (GPVW on-the-fly) translation to Buchi automata.
"""

from .buchi_automaton import BuchiAutomaton, EmptinessResult, intersect, kripke_buchi_product
from .generalized_buchi import GeneralizedBuchiAutomaton
from .kripke import KripkeModel

# Visualization modules — not auto-exported, imported explicitly by users.
# See visualization.py (networkx/matplotlib) or visualization_graphviz.py (Graphviz/DOT)
from .ltl import (
    F,
    G,
    R,
    U,
    W,
    X,
    And,
    Always,
    Atom,
    BoolConst,
    Eventually,
    Implies,
    LTLFormula,
    Next,
    Not,
    Or,
    Release,
    Until,
    Var,
    WeakUntil,
    satisfies,
)
from .gpvw_ltl_to_buchi import ltl_to_buchi_gpvw, ltl_to_gba_gpvw, to_nnf, GPVWNode, INIT, formula_untils
from .omega_word import OmegaWord
from .tarjan import tarjan_scc

__all__ = [
    "BuchiAutomaton",
    "EmptinessResult",
    "GeneralizedBuchiAutomaton",
    "kripke_buchi_product",
    "KripkeModel",
    "OmegaWord",
    "intersect",
    "tarjan_scc",
    # LTL
    "LTLFormula",
    "Atom",
    "Var",
    "BoolConst",
    "Not",
    "And",
    "Or",
    "Implies",
    "Next",
    "Eventually",
    "Always",
    "Until",
    "Release",
    "WeakUntil",
    "X",
    "F",
    "G",
    "U",
    "R",
    "W",
    "satisfies",
    "ltl_to_buchi_gpvw",
    "ltl_to_gba_gpvw",
    "to_nnf",
    "GPVWNode",
    "INIT",
    "formula_untils",
]

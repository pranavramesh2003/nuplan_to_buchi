"""Generalized Büchi automata and their degeneralization to ordinary Büchi automata.

A **generalized Büchi automaton** (GBA) has the same shape as a Büchi automaton except
that acceptance is given by *several* accepting sets ``F_0, …, F_{k-1}``: a run is
accepting iff it visits **each** ``F_i`` infinitely often. GBAs arise naturally from the
LTL tableau construction (one accepting set per ``Until`` subformula).

A single Büchi condition cannot directly express "visit every ``F_i`` infinitely often",
so we **degeneralize**: take ``k`` copies of the state space indexed by a counter
``i ∈ {0, …, k-1}`` that records which accepting set we are currently waiting for, advance
the counter when the current ``F_i`` is seen, and accept whenever the counter completes a
full cycle. This is exactly the ``{1, 2}`` toggle of
:func:`buchi_automaton.intersect` generalized from two sets to ``k``: indeed, intersecting
two Büchi automata is the same as degeneralizing the product equipped with the two
accepting sets ``F1 × Q2`` and ``Q1 × F2``.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Set, Tuple

from .buchi_automaton import BuchiAutomaton
from .omega_word import Letter

State = Hashable


class GeneralizedBuchiAutomaton:
    """A Büchi automaton with multiple accepting sets (generalized acceptance)."""

    def __init__(self) -> None:
        self.states: Set[State] = set()
        self.initial_states: Set[State] = set()
        # Each accepting set must be visited infinitely often by an accepting run.
        self.accepting_sets: List[Set[State]] = []
        self.transitions: Dict[State, List[Tuple[Letter, State]]] = {}

    # ------------------------------------------------------------------ builders
    def add_state(self, state: State) -> None:
        self.states.add(state)
        self.transitions.setdefault(state, [])

    def add_initial_state(self, state: State) -> None:
        self.add_state(state)
        self.initial_states.add(state)

    def add_transition(self, src: State, letter: Iterable, dst: State) -> None:
        self.add_state(src)
        self.add_state(dst)
        self.transitions[src].append((frozenset(letter), dst))

    def add_accepting_set(self, states: Iterable[State]) -> None:
        """Register an accepting set ``F_i`` (its members must already be states)."""
        self.accepting_sets.append(set(states))

    def successors(self, state: State) -> List[Tuple[Letter, State]]:
        return self.transitions.get(state, [])

    # ------------------------------------------------------------------ degeneralize
    def to_buchi(self) -> BuchiAutomaton:
        """Return an ordinary Büchi automaton accepting the same language.

        With ``k`` accepting sets the result has states ``(q, i)`` for ``i ∈ {0..k-1}``;
        the counter advances ``i → (i+1) mod k`` on transitions leaving a state in
        ``F_i`` and the accepting states are ``{(q, 0) : q ∈ F_0}``. A run that visits
        ``{(·, 0)} ∩ F_0`` infinitely often must cycle the counter forever, i.e. visit
        every ``F_i`` infinitely often. With ``k = 0`` (no constraints) every run is
        accepting, so all states are made accepting.
        """
        buchi = BuchiAutomaton()
        sets = self.accepting_sets

        if not sets:
            for state in self.states:
                buchi.add_state(state)
            for state in self.initial_states:
                buchi.add_initial_state(state)
            for state in self.states:
                buchi.add_accepting_state(state)
            for src in self.states:
                for letter, dst in self.successors(src):
                    buchi.add_transition(src, letter, dst)
            return buchi

        k = len(sets)
        for state in self.states:
            for i in range(k):
                buchi.add_state((state, i))
        for state in self.initial_states:
            buchi.add_initial_state((state, 0))
        for state in sets[0]:
            buchi.add_accepting_state((state, 0))
        for src in self.states:
            for i in range(k):
                advance = src in sets[i]
                next_i = (i + 1) % k if advance else i
                for letter, dst in self.successors(src):
                    buchi.add_transition((src, i), letter, (dst, next_i))
        return buchi

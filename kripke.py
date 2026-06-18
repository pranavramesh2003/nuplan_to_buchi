"""Kripke models — the state-labeled transition systems of temporal logic.

A **Kripke model** is a triple ``M = (S, →, L)``:

* ``S`` — a set of states/worlds,
* ``→ ⊆ S × S`` — a (directed) transition/accessibility relation, **unlabeled**,
* ``L : S → 2^AP`` — a labelling assigning to each state the set of atomic propositions
  that *hold* there.

Unlike a Büchi automaton (whose propositions sit on the *transitions* as letters), a Kripke
model carries its propositions on the *states*. This class is intentionally light; its main
use here is to be drawn by :func:`visualization.draw` with ``node_propositions=model.labeling``
and ``show_edge_labels=False`` (propositions inside the nodes, plain directed edges).

It exposes ``states`` / ``initial_states`` / ``accepting_states`` / ``successors`` so it can
reuse the same renderer as the automata; ``accepting_states`` is always empty (a Kripke
model has no acceptance condition) and ``initial_states`` is optional.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, FrozenSet, Hashable, Iterable, List, Set, Tuple

State = Hashable


class KripkeModel:
    """A Kripke model ``(S, →, L)``: states with proposition labels and unlabeled edges."""

    def __init__(self) -> None:
        self.states: Set[State] = set()
        self.initial_states: Set[State] = set()
        # Always empty — kept so the visualizer can treat a model like an automaton.
        self.accepting_states: Set[State] = set()
        self.labeling: Dict[State, FrozenSet] = {}  # L: state -> propositions holding there
        self._transitions: Dict[State, Set[State]] = {}

    def add_state(self, state: State, propositions: Iterable = ()) -> None:
        """Register a state with the propositions that hold there (``L(state)``)."""
        self.states.add(state)
        self._transitions.setdefault(state, set())
        self.labeling.setdefault(state, frozenset(propositions))

    def label(self, state: State, propositions: Iterable) -> None:
        """Set ``L(state)`` to ``propositions`` (registering the state if needed)."""
        self.add_state(state)
        self.labeling[state] = frozenset(propositions)

    def add_initial_state(self, state: State, propositions: Iterable = ()) -> None:
        """Register ``state`` (with its label) and mark it initial."""
        self.add_state(state, propositions)
        self.initial_states.add(state)

    def add_transition(self, src: State, dst: State) -> None:
        """Add a directed (unlabeled) edge ``src → dst``."""
        self.add_state(src)
        self.add_state(dst)
        self._transitions[src].add(dst)

    def successors(self, state: State) -> List[Tuple[FrozenSet, State]]:
        """Return ``(empty-letter, target)`` pairs — edges carry no letters in a Kripke model."""
        return [(frozenset(), dst) for dst in sorted(self._transitions.get(state, ()), key=str)]

    def reachable_states(self) -> Set[State]:
        """States reachable from the initial states (or all states if none are marked)."""
        if not self.initial_states:
            return set(self.states)
        visited: Set[State] = set(self.initial_states)
        queue: Deque[State] = deque(self.initial_states)
        while queue:
            state = queue.popleft()
            for _, dst in self.successors(state):
                if dst not in visited:
                    visited.add(dst)
                    queue.append(dst)
        return visited

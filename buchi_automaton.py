"""Non-deterministic Buchi automata and an SCC-based emptiness check.

A **Buchi automaton** ``A = (Q, Sigma, delta, Q0, F)`` reads infinite words. A run is an
infinite sequence of states consistent with the transition relation ``delta``; it is
*accepting* iff it visits the accepting set ``F`` **infinitely often**. The language
``L(A)`` is the set of infinite words admitting some accepting run.

Here a letter of ``Sigma`` is a **set** of arbitrary objects (the "subset of atomic
propositions" model), normalized to ``frozenset``.

**Emptiness.** ``L(A) != ∅`` iff there is an accepting state that is reachable from an
initial state *and* lies on a cycle. Equivalently: a reachable, non-trivial strongly
connected component that contains an accepting state. We find SCCs with Tarjan's
algorithm (see :mod:`tarjan`) and, when the language is non-empty, reconstruct a concrete
witness lasso ``u . v^omega`` (see :class:`omega_word.OmegaWord`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, FrozenSet, Hashable, Iterable, List, Optional, Set, Tuple

from .omega_word import Letter, OmegaWord
from .tarjan import tarjan_scc

State = Hashable


@dataclass
class EmptinessResult:
    """Outcome of an emptiness check.

    :param is_empty: ``True`` iff the automaton accepts no word.
    :param witness: an accepted lasso word when non-empty, otherwise ``None``.
    :param accepting_scc: the strongly connected component that witnesses non-emptiness
        (a reachable non-trivial SCC containing an accepting state), otherwise ``None``.
    """

    is_empty: bool
    witness: Optional[OmegaWord] = None
    accepting_scc: Optional[List[State]] = None


class BuchiAutomaton:
    """A non-deterministic Buchi automaton with set-labeled transitions."""

    def __init__(self) -> None:
        self.states: Set[State] = set()
        self.initial_states: Set[State] = set()
        self.accepting_states: Set[State] = set()
        # src -> list of (letter, dst); letters are frozensets.
        self.transitions: Dict[State, List[Tuple[Letter, State]]] = {}

    # ------------------------------------------------------------------ builders
    def add_state(self, state: State) -> None:
        """Register a state (idempotent)."""
        self.states.add(state)
        self.transitions.setdefault(state, [])

    def add_initial_state(self, state: State) -> None:
        """Register ``state`` and mark it initial."""
        self.add_state(state)
        self.initial_states.add(state)

    def add_accepting_state(self, state: State) -> None:
        """Register ``state`` and add it to the accepting set ``F``."""
        self.add_state(state)
        self.accepting_states.add(state)

    def add_transition(self, src: State, letter: Iterable, dst: State) -> None:
        """Add a transition ``src --letter--> dst``.

        :param letter: any iterable of objects; normalized to a ``frozenset``.
        """
        self.add_state(src)
        self.add_state(dst)
        self.transitions[src].append((frozenset(letter), dst))

    # ------------------------------------------------------------------ queries
    def successors(self, state: State) -> List[Tuple[Letter, State]]:
        """Return the outgoing ``(letter, dst)`` transitions of ``state``."""
        return self.transitions.get(state, [])

    def reachable_states(self) -> Set[State]:
        """Return all states reachable from the initial states (BFS)."""
        visited: Set[State] = set(self.initial_states)
        queue: Deque[State] = deque(self.initial_states)
        while queue:
            state = queue.popleft()
            for _, dst in self.successors(state):
                if dst not in visited:
                    visited.add(dst)
                    queue.append(dst)
        return visited

    def state_graph(self, restrict: Optional[Set[State]] = None) -> Dict[State, Set[State]]:
        """Return the letter-erased successor graph, optionally restricted to a node set.

        :param restrict: if given, only edges between nodes in this set are kept.
        """
        graph: Dict[State, Set[State]] = {}
        nodes = restrict if restrict is not None else self.states
        for state in nodes:
            targets: Set[State] = set()
            for _, dst in self.successors(state):
                if restrict is None or dst in restrict:
                    targets.add(dst)
            graph[state] = targets
        return graph

    # ------------------------------------------------------------------ intersection
    def intersect(self, other: "BuchiAutomaton") -> "BuchiAutomaton":
        """Return an automaton recognising ``L(self) ∩ L(other)`` (see :func:`intersect`)."""
        return intersect(self, other)

    # ------------------------------------------------------------------ emptiness
    def check_emptiness(self) -> EmptinessResult:
        """Decide whether ``L(A)`` is empty and, if not, return a witness lasso.

        :return: an :class:`EmptinessResult`.
        """
        reachable = self.reachable_states()
        graph = self.state_graph(restrict=reachable)

        for component in tarjan_scc(graph):
            scc_set = set(component)
            if not self._is_non_trivial(component, scc_set, graph):
                continue
            accepting = [state for state in component if state in self.accepting_states]
            if not accepting:
                continue
            # Non-empty: build a witness lasso through an accepting state of this SCC.
            witness = self._build_witness(accepting[0], scc_set, reachable)
            return EmptinessResult(is_empty=False, witness=witness, accepting_scc=component)

        return EmptinessResult(is_empty=True, witness=None, accepting_scc=None)

    def is_empty(self) -> bool:
        """Convenience boolean form of :meth:`check_emptiness`."""
        return self.check_emptiness().is_empty

    @staticmethod
    def _is_non_trivial(
        component: List[State], scc_set: Set[State], graph: Dict[State, Set[State]]
    ) -> bool:
        """A SCC carries an infinite run iff it has a cycle: size > 1 or a self-loop."""
        if len(component) > 1:
            return True
        only = component[0]
        return only in graph.get(only, set())

    def _build_witness(
        self, accepting_state: State, scc_set: Set[State], reachable: Set[State]
    ) -> OmegaWord:
        """Construct a lasso ``u . v^omega`` through ``accepting_state`` inside its SCC."""
        prefix = self._shortest_path_letters(self.initial_states, accepting_state, reachable)
        loop = self._shortest_cycle_letters(accepting_state, scc_set)
        witness = OmegaWord(prefix=prefix, loop=loop)
        # Internal sanity check: the reconstructed word must actually be accepted.
        assert self.accepts(witness), "Constructed witness is not accepted (internal error)."
        return witness

    def _shortest_path_letters(
        self, sources: Iterable[State], target: State, allowed: Set[State]
    ) -> List[Letter]:
        """BFS shortest letter-path from any source to ``target`` within ``allowed``.

        :return: the sequence of letters along the path (empty if a source equals target).
        """
        sources = set(sources)
        if target in sources:
            return []
        # parent[state] = (previous_state, letter_into_state)
        parent: Dict[State, Tuple[State, Letter]] = {}
        visited: Set[State] = set(sources)
        queue: Deque[State] = deque(sources)
        while queue:
            state = queue.popleft()
            for letter, dst in self.successors(state):
                if dst not in allowed or dst in visited:
                    continue
                visited.add(dst)
                parent[dst] = (state, letter)
                if dst == target:
                    return self._reconstruct(parent, sources, target)
                queue.append(dst)
        raise ValueError(f"No path to {target!r}; automaton is inconsistent.")

    @staticmethod
    def _reconstruct(
        parent: Dict[State, Tuple[State, Letter]], sources: Set[State], target: State
    ) -> List[Letter]:
        """Walk ``parent`` pointers from ``target`` back to a source, collecting letters."""
        letters: List[Letter] = []
        node = target
        while node not in sources:
            prev, letter = parent[node]
            letters.append(letter)
            node = prev
        letters.reverse()
        return letters

    def _shortest_cycle_letters(self, state: State, scc_set: Set[State]) -> List[Letter]:
        """Shortest letter-path ``state -> ... -> state`` (length >= 1) inside ``scc_set``."""
        best: Optional[List[Letter]] = None
        for first_letter, succ in self.successors(state):
            if succ not in scc_set:
                continue
            if succ == state:
                # Self-loop: a length-1 cycle, can't be beaten.
                return [first_letter]
            rest = self._shortest_path_letters({succ}, state, scc_set)
            candidate = [first_letter] + rest
            if best is None or len(candidate) < len(best):
                best = candidate
        if best is None:
            raise ValueError(f"State {state!r} has no cycle within its SCC (internal error).")
        return best

    def _shortest_path_states(
        self, sources: Iterable[State], target: State, allowed: Set[State]
    ) -> List[State]:
        """BFS shortest state sequence from any source to ``target`` within ``allowed``.

        :return: list of states ``[source, ..., target]`` (inclusive), length >= 1.
        """
        sources_set = set(sources)
        if target in sources_set:
            return [target]
        parent: Dict[State, State] = {}
        visited: Set[State] = set(sources_set)
        queue: Deque[State] = deque(sources_set)
        while queue:
            state = queue.popleft()
            for _, dst in self.successors(state):
                if dst not in allowed or dst in visited:
                    continue
                visited.add(dst)
                parent[dst] = state
                if dst == target:
                    path: List[State] = []
                    node = dst
                    while node not in sources_set:
                        path.append(node)
                        node = parent[node]
                    path.append(node)
                    path.reverse()
                    return path
                queue.append(dst)
        raise ValueError(f"No path to {target!r}; automaton is inconsistent.")

    def _shortest_cycle_states(self, state: State, scc_set: Set[State]) -> List[State]:
        """Shortest state sequence forming a cycle ``state → … → state`` inside ``scc_set``.

        :return: list ``[state, ..., state]`` of length >= 2.
        """
        best: Optional[List[State]] = None
        for _, succ in self.successors(state):
            if succ not in scc_set:
                continue
            if succ == state:
                return [state, state]
            try:
                rest = self._shortest_path_states({succ}, state, scc_set)
            except ValueError:
                continue
            candidate = [state] + rest
            if best is None or len(candidate) < len(best):
                best = candidate
        if best is None:
            raise ValueError(f"State {state!r} has no cycle within its SCC (internal error).")
        return best

    def check_emptiness_with_states(
        self,
    ) -> Tuple[bool, Optional[List[State]], Optional[List[State]]]:
        """Emptiness check that also returns the concrete state trace of the witness lasso.

        Uses Tarjan's SCC algorithm (via :func:`tarjan.tarjan_scc`) to find the first
        non-trivial reachable SCC containing an accepting state, then reconstructs the
        shortest stem and shortest cycle through that state.

        :return: a triple ``(is_empty, prefix_states, cycle_states)``:

          - ``is_empty``: ``True`` when the language is empty.
          - ``prefix_states``: when non-empty, the state sequence
            ``[initial, …, accepting]`` (inclusive) — the *stem* of the lasso.
          - ``cycle_states``: when non-empty, ``[accepting, …, accepting]``
            (inclusive, length ≥ 2) — the *loop* of the lasso.

        To obtain the finite satisfying path (truncated before the repetitive part),
        use only ``prefix_states``.
        """
        reachable = self.reachable_states()
        graph = self.state_graph(restrict=reachable)
        for component in tarjan_scc(graph):
            scc_set = set(component)
            if not self._is_non_trivial(component, scc_set, graph):
                continue
            accepting = [s for s in component if s in self.accepting_states]
            if not accepting:
                continue
            acc_state = accepting[0]
            prefix_states = self._shortest_path_states(self.initial_states, acc_state, reachable)
            cycle_states = self._shortest_cycle_states(acc_state, scc_set)
            return False, prefix_states, cycle_states
        return True, None, None

    # ------------------------------------------------------------------ verifier
    def accepts(self, word: OmegaWord) -> bool:
        """Check whether the automaton has an accepting run on the lasso ``word``.

        The word is ``u . v^omega``. We first compute the set of states reachable after
        reading the prefix ``u``, then search the product of the automaton with the loop
        ``v`` (configurations ``(state, phase)`` where ``phase`` indexes into ``v``). The
        word is accepted iff some reachable configuration lies on a product cycle that
        passes through an accepting state -- i.e. ``F`` is visited infinitely often.
        """
        current: Set[State] = set(self.initial_states)
        for letter in word.prefix:
            current = {dst for q in current for lab, dst in self.successors(q) if lab == letter}
            if not current:
                return False

        period = word.period
        # Product graph over configurations (state, phase).
        start: Set[Tuple[State, int]] = {(q, 0) for q in current}
        product: Dict[Tuple[State, int], Set[Tuple[State, int]]] = {}
        visited: Set[Tuple[State, int]] = set(start)
        queue: Deque[Tuple[State, int]] = deque(start)
        while queue:
            node = queue.popleft()
            state, phase = node
            letter = word.loop[phase]
            next_phase = (phase + 1) % period
            targets: Set[Tuple[State, int]] = set()
            for lab, dst in self.successors(state):
                if lab == letter:
                    child = (dst, next_phase)
                    targets.add(child)
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)
            product[node] = targets

        # Reachable accepting product-cycle <=> non-trivial SCC with an accepting state.
        for component in tarjan_scc(product):
            scc_set = set(component)
            non_trivial = len(component) > 1 or component[0] in product.get(component[0], set())
            if non_trivial and any(state in self.accepting_states for state, _ in component):
                return True
        return False


# A product state: (state of A1, state of A2, tracking bit in {1, 2}).
ProductState = Tuple[State, State, int]


def _group_by_letter(
    transitions: List[Tuple[Letter, State]]
) -> Dict[Letter, List[State]]:
    """Index a list of ``(letter, dst)`` transitions by letter for synchronized joins."""
    by_letter: Dict[Letter, List[State]] = {}
    for letter, dst in transitions:
        by_letter.setdefault(letter, []).append(dst)
    return by_letter


def intersect(a1: BuchiAutomaton, a2: BuchiAutomaton) -> BuchiAutomaton:
    """Build a Buchi automaton recognising ``L(a1) ∩ L(a2)``.

    Standard intersection (Choueka) construction. A single Buchi acceptance condition
    cannot directly demand that *both* ``F1`` and ``F2`` be visited infinitely often, so
    the product carries a **tracking bit** ``{1, 2}`` recording which automaton's accepting
    set we are currently waiting to see:

    - **States:** ``Q1 × Q2 × {1, 2}`` (the bit ``1`` watches ``A1``'s goal, ``2`` watches
      ``A2``'s goal). The transition relation ``Δ ⊆ (Q1×Q2×{1,2}) × Σ × (Q1×Q2×{1,2})`` is
      a genuine relation: it is partial (no edge when the two letters disagree) and
      non-deterministic (it inherits every choice of ``Δ1`` and ``Δ2``).
    - **Initial:** ``Q0,1 × Q0,2 × {1}`` — start while watching ``A1``.
    - **Transitions:** synchronized on a *common* letter ``a`` (so ``(q1,a,q1') ∈ Δ1`` and
      ``(q2,a,q2') ∈ Δ2``), with the bit toggled on the *source* component:

      ===========================  ====================================  ===========
      from                          condition                             to bit
      ===========================  ====================================  ===========
      ``(q1, q2, 1)``               ``q1 ∈ F1``                            ``2``
      ``(q1, q2, 1)``               ``q1 ∉ F1``                            ``1``
      ``(q1, q2, 2)``               ``q2 ∈ F2``                            ``1``
      ``(q1, q2, 2)``               ``q2 ∉ F2``                            ``2``
      ===========================  ====================================  ===========

    - **Accepting:** ``F1 × Q2 × {1}``. Each visit there fires a ``1 → 2`` toggle, and the
      only way back to bit ``1`` is through an ``F2`` state in bit ``2``; so visiting the
      accepting set infinitely often forces *both* ``F1`` and ``F2`` infinitely often.

    The full ``Q1 × Q2 × {1, 2}`` state space is materialized as specified; unreachable
    product states are harmless and are ignored by :meth:`BuchiAutomaton.check_emptiness`.

    :param a1: the first automaton.
    :param a2: the second automaton.
    :return: a new :class:`BuchiAutomaton` whose language is ``L(a1) ∩ L(a2)``.
    """
    product = BuchiAutomaton()

    # States: the full product space Q1 × Q2 × {1, 2}.
    for q1 in a1.states:
        for q2 in a2.states:
            product.add_state((q1, q2, 1))
            product.add_state((q1, q2, 2))

    # Initial states: Q0,1 × Q0,2 × {1}.
    for q1 in a1.initial_states:
        for q2 in a2.initial_states:
            product.add_initial_state((q1, q2, 1))

    # Accepting states: F1 × Q2 × {1}.
    for q1 in a1.accepting_states:
        for q2 in a2.states:
            product.add_accepting_state((q1, q2, 1))

    # Transition relation, synchronized on a shared letter, with the tracking-bit toggle.
    a2_successors = {q2: _group_by_letter(a2.successors(q2)) for q2 in a2.states}
    for q1 in a1.states:
        q1_is_goal = q1 in a1.accepting_states
        a1_succ = a1.successors(q1)
        for q2 in a2.states:
            q2_is_goal = q2 in a2.accepting_states
            by_letter = a2_successors[q2]
            for letter, q1_next in a1_succ:
                for q2_next in by_letter.get(letter, ()):
                    # bit 1: toggle to 2 once A1's goal is seen, else stay in 1.
                    product.add_transition(
                        (q1, q2, 1), letter, (q1_next, q2_next, 2 if q1_is_goal else 1)
                    )
                    # bit 2: toggle back to 1 once A2's goal is seen, else stay in 2.
                    product.add_transition(
                        (q1, q2, 2), letter, (q1_next, q2_next, 1 if q2_is_goal else 2)
                    )

    return product


# Product state: (Kripke state, Büchi state).
KripkeBuchiState = Tuple[State, State]


def kripke_buchi_product(kripke: object, buchi: BuchiAutomaton) -> BuchiAutomaton:
    """Build the synchronous product automaton ``M × B`` for automata-theoretic model checking.

    Given a Kripke model ``M = (S, S₀, →, L)`` and a Büchi automaton ``B = (Q, Σ, δ, Q₀, F)``
    over the atomic propositions of an LTL formula ``¬φ``, the product ``M × B`` is a Büchi
    automaton whose language is non-empty iff ``M`` has a path that *satisfies* ``φ``.

    **Construction** (source-labelling convention, matching :func:`gpvw_ltl_to_buchi.ltl_to_buchi_gpvw`):

    - **States**: ``S × Q``.
    - **Initial**: ``S₀ × Q₀``.
    - **Accepting**: ``S × F``.
    - **Transitions**: ``(s, q) → (s', q')`` whenever ``s → s'`` in ``M`` and
      ``q --L_B(s)--> q'`` in ``B``, where ``L_B(s) = L(s) ∩ AP(B)`` is the Kripke label
      restricted to ``B``'s atomic propositions.

    The self-loops on the Kripke model (one per state) ensure every state has at least one
    successor, so the product is *total*, and every non-trivial SCC in the product corresponds
    to an infinite path in ``M``.

    :param kripke: a :class:`~kripke.KripkeModel` (duck-typed: needs ``states``,
        ``initial_states``, ``labeling``, and ``successors``).
    :param buchi: Büchi automaton built from ``¬φ`` (or from ``φ`` directly to find a
        witness *satisfying* ``φ``).
    :return: a new :class:`BuchiAutomaton` representing ``M × B``.
    """
    # Collect B's APs from transition labels so we can project L(s) correctly.
    buchi_aps: FrozenSet = frozenset(
        p for q in buchi.states for letter, _ in buchi.successors(q) for p in letter
    )

    # Index Büchi transitions: (q, letter) → [q'] for fast lookup.
    b_by_letter: Dict[Tuple, List] = {}
    for q in buchi.states:
        for letter, q_next in buchi.successors(q):
            b_by_letter.setdefault((q, letter), []).append(q_next)

    product = BuchiAutomaton()

    # All product states.
    for s in kripke.states:
        for q in buchi.states:
            product.add_state((s, q))

    # Initial product states.
    for s0 in kripke.initial_states:
        for q0 in buchi.initial_states:
            product.add_initial_state((s0, q0))

    # Accepting product states: those whose Büchi component is accepting.
    for s in kripke.states:
        for q in buchi.accepting_states:
            product.add_accepting_state((s, q))

    # Transitions: synchronized on L_B(s) = L(s) ∩ AP(B).
    for s in kripke.states:
        s_b_label: FrozenSet = frozenset(kripke.labeling.get(s, frozenset())) & buchi_aps
        for _, s_next in kripke.successors(s):
            for q in buchi.states:
                for q_next in b_by_letter.get((q, s_b_label), ()):
                    product.add_transition((s, q), s_b_label, (s_next, q_next))

    return product

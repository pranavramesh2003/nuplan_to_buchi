# Büchi automata

A self-contained Python (≥ 3.9) toolkit for non-deterministic Büchi automata: ω-word
representation, emptiness checking, intersection, generalized acceptance, an LTL→Büchi
translation, and a NetworkX visualizer. The core (everything except `visualization.py`)
uses **only the standard library**; the visualizer additionally needs `networkx` and
`matplotlib`.

## Features

| Feature | Entry point | Notes |
|---|---|---|
| Büchi automaton with a relational transition function `Δ` | `BuchiAutomaton` | non-deterministic, partial; states are any hashable, letters are sets |
| Infinite words as lassos `u·vᵒ̬` | `OmegaWord` | ultimately-periodic ω-words; `unroll(n)` |
| Emptiness check + witness | `BuchiAutomaton.check_emptiness` / `is_empty` | Tarjan SCC; returns an accepted witness lasso and the accepting SCC |
| Membership / run verification | `BuchiAutomaton.accepts(word)` | product-with-loop search for an accepting run |
| Tarjan strongly-connected components | `tarjan_scc(graph)` | iterative (no recursion-depth limit) |
| Intersection `L(A₁) ∩ L(A₂)` | `intersect(a1, a2)` / `BuchiAutomaton.intersect` | `Q₁×Q₂×{1,2}` product with tracking bit |
| Generalized Büchi acceptance + degeneralization | `GeneralizedBuchiAutomaton`, `.to_buchi()` | `k` accepting sets → counter construction |
| LTL syntax, core rewriting, semantics | `ltl.py` (`Atom`, `Not`, `X`, `F`, `G`, `U`, `R`, `W`, …) | operator sugar (`&`, `\|`, `~`, `>>`); `satisfies(φ, word)` reference semantics |
| LTL → Büchi translation | `ltl_to_buchi(φ)`, `ltl_to_gba(φ)` | closure / elementary-sets tableau → GBA → BA |
| Visualization (Kripke-diagram view) | `visualization.draw(...)` | states→nodes, transitions→edges, propositions shown |

## Package layout

```
buchi/
├── omega_word.py          OmegaWord — infinite words as lassos (prefix + loop)
├── tarjan.py              tarjan_scc — iterative strongly-connected components
├── buchi_automaton.py     BuchiAutomaton, EmptinessResult, intersect
├── generalized_buchi.py   GeneralizedBuchiAutomaton + degeneralization to_buchi()
├── ltl.py                 LTL formulas, to_core(), subformulas, satisfies() semantics
├── ltl_to_buchi.py        closure, elementary_sets, ltl_to_gba, ltl_to_buchi
├── visualization.py       to_networkx / draw  (needs networkx + matplotlib)
├── examples/              visualization_demo.ipynb — visual tour
└── test/                  unittest suites (one per module)
```

## Internal representation

The automaton is a plain mutable object built incrementally; nothing is frozen except the
letters. All four pieces of `A = (Q, Σ, δ, Q₀, F)` are stored explicitly on
`BuchiAutomaton`:

```python
self.states:           Set[State]                               # Q  — any hashable values
self.initial_states:   Set[State]                               # Q₀ ⊆ Q
self.accepting_states: Set[State]                               # F  ⊆ Q
self.transitions:      Dict[State, List[Tuple[Letter, State]]]  # δ as an adjacency list
```

- **States** (`State`) are arbitrary *hashables* — strings, ints, or tuples such as the
  intersection's `(q1, q2, bit)` and the degeneralized `(state, counter)`. The package
  never assumes anything about a state beyond hashability/equality.
- **Letters** (`Letter = frozenset`) are *sets of arbitrary objects* — the "subset of
  atomic propositions" model. `add_transition` normalizes any iterable to a `frozenset`,
  so `{"a"}`, `["a"]`, or a set of custom objects all work and compare by value.
- **Transitions are a relation, not a function.** `δ` is kept as an adjacency list
  `state → [(letter, target), …]`. This makes it both **partial** (a state may have no
  outgoing edge for some letter) and **non-deterministic** (the same `(state, letter)` may
  appear with several targets, and a state may carry several letters). `successors(s)`
  returns the raw `(letter, target)` list. Keeping the letter on every edge is what lets
  the emptiness check reconstruct a concrete witness word.

Builder API: `add_state`, `add_initial_state`, `add_accepting_state`,
`add_transition(src, letter, dst)` (auto-registers `src`/`dst`). Derived views:
`reachable_states()` (BFS from `Q₀`) and `state_graph(restrict=…)` (the letter-erased
successor graph handed to Tarjan).

**`OmegaWord`** stores two tuples of `frozenset` letters, `prefix` (stem `u`) and `loop`
(cycle `v`, required non-empty); it is hashable and value-comparable.

**`EmptinessResult`** is a dataclass `(is_empty: bool, witness: Optional[OmegaWord],
accepting_scc: Optional[List[State]])`.

**`GeneralizedBuchiAutomaton`** mirrors `BuchiAutomaton` but replaces the single `F` with
`accepting_sets: List[Set[State]]`; `to_buchi()` degeneralizes it to the representation
above.

## Concepts

- **Büchi automaton** `A = (Q, Σ, δ, Q₀, F)` reads *infinite* words. A run is accepting
  iff it visits the accepting set `F` **infinitely often**. `L(A)` is the set of words
  with an accepting run.
- **Letter.** Every letter of `Σ` is a *set* of arbitrary objects (the "subset of atomic
  propositions" model), stored internally as a `frozenset`.
- **ω-word as a lasso.** The words that matter are *ultimately periodic*: a non-recurrent
  prefix `u` followed by a recurrent block `v` repeated forever — written `u·vᵒ̬` and
  represented by `OmegaWord(prefix=u, loop=v)`.

## Emptiness check

> `L(A) ≠ ∅` **iff** some accepting state is reachable from an initial state *and* lies on
> a cycle — equivalently, a reachable **non-trivial** SCC (size > 1, or a single node with
> a self-loop) contains an accepting state.

`check_emptiness()` finds SCCs with **Tarjan's algorithm** (`tarjan.py`) and, when the
language is non-empty, reconstructs a concrete witness lasso: a BFS shortest path from an
initial state to an accepting state (the stem `u`) plus a shortest cycle through that state
inside its SCC (the loop `v`).

## Usage

```python
from tutorials.buchi import BuchiAutomaton

a = BuchiAutomaton()
a.add_initial_state("q0")
a.add_accepting_state("q1")
a.add_transition("q0", {"a"}, "q1")
a.add_transition("q1", {"b"}, "q1")   # accepting self-loop

result = a.check_emptiness()
print(result.is_empty)   # False
print(result.witness)    # {a}·({b})ω
assert a.accepts(result.witness)
```

## Intersection

`intersect(a1, a2)` builds an automaton for `L(A1) ∩ L(A2)` using the standard product
with a tracking bit `{1, 2}`: states `Q1 × Q2 × {1, 2}`, a transition relation
`Δ ⊆ (Q1×Q2×{1,2}) × Σ × (Q1×Q2×{1,2})` synchronized on a shared letter, and accepting set
`F1 × Q2 × {1}`. The bit records which automaton's goal we're waiting on and toggles
`1→2` on an `F1` state, `2→1` on an `F2` state — so the accepting set is hit infinitely
often iff both `F1` and `F2` are.

## Generalized Büchi automata

`GeneralizedBuchiAutomaton` allows several accepting sets `F₀,…,F_{k-1}` (a run must visit
*each* infinitely often). `to_buchi()` **degeneralizes** to an ordinary Büchi automaton via
a counter `i ∈ {0..k-1}` that advances when the current `Fᵢ` is seen — the `k`-set
generalization of the intersection toggle.

## LTL → Büchi

`ltl_to_buchi(formula)` translates a linear-temporal-logic formula into a Büchi automaton
accepting exactly its models, via the closure / elementary-sets tableau (`ltl_to_buchi.py`):
formula → core fragment `{atom, true, ¬, ∧, X, U}` → elementary sets → generalized Büchi
automaton (one accepting set per `Until`) → degeneralized Büchi automaton.

```python
from tutorials.buchi import Atom, G, F, ltl_to_buchi, satisfies

p, q = Atom("p"), Atom("q")
phi = G(p >> F(q))                      # every p is eventually followed by q
buchi = ltl_to_buchi(phi)
print(buchi.is_empty())                 # False (satisfiable)
witness = buchi.check_emptiness().witness
assert satisfies(phi, witness)          # the witness lasso models phi
```

`ltl.satisfies(formula, word)` is an independent operator-complete LTL semantics over lasso
words, used to cross-check the translation (the test suite verifies `accepts == satisfies`
over thousands of words).

## Visualization

Two visualization engines are available:

### NetworkX / Matplotlib (`visualization.py`)

Renders any `BuchiAutomaton` or `GeneralizedBuchiAutomaton` as a NetworkX graph using
matplotlib. Keeps two distinct notions of propositions:

- **propositions read on a transition** (the letter of `Σ`) → **edge labels**
- **predicates that hold at a state** (a Kripke labelling) → **inside the node**

Initial states are **yellow**, accepting states are **green (double ring)**, and highlighted
sets (accepting SCC / witness) are **red**.

```python
from tutorials.buchi.visualization import draw, letter_labeling

aut = ltl_to_buchi(G(p >> F(q)))
draw(aut, title="φ = G(p → F q)", reachable_only=True,
     node_propositions=letter_labeling(aut, reachable_only=True), show_edge_labels=False)
```

### Graphviz / DOT (`visualization_graphviz.py`)

Uses **Graphviz** (the DOT language) for publication-quality diagrams with:
- Excellent self-loop rendering (natural arcs on the node)
- Hierarchical automatic layout
- Professional edge label positioning
- Vector output (SVG/PDF) that scales to any size

```python
from tutorials.buchi.visualization_graphviz import draw
import IPython.display as display

g = draw(buchi, title="My automaton", initial_color="#ffeb3b")
display.SVG(g.pipe(format='svg'))  # Inline in Jupyter
g.render('/tmp/automaton', format='pdf')  # Save to file
```

Requires: `pip install graphviz` and the Graphviz binary (`apt-get install graphviz` on Ubuntu).

See [`examples/visualization_demo.ipynb`](examples/visualization_demo.ipynb) (NetworkX) and
[`examples/visualization_graphviz.ipynb`](examples/visualization_graphviz.ipynb) (Graphviz)
for visual tours.

## Tests

```bash
python -m pytest tutorials/buchi/test/ -v
# or
python -m unittest discover -s tutorials/buchi/test -v
```

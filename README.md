# LTL Model Checking on NuPlan Lane Graphs

This library **solves Linear Temporal Logic (LTL) formulae over [NuPlan](https://www.nuscenes.org/nuplan)
autonomous-driving lane graphs**. Point it at a NuPlan map and an ego pose, write a
temporal-logic specification of where the vehicle should go — *eventually reach the
destination*, *visit a waypoint then stay in the goal lane*, *avoid forks until past the
intersection* — and it returns a concrete route through the road network that satisfies
the spec, or proves that none exists. Satisfying routes can be plotted on the map or
rendered as a **node-by-node traversal video**.

The automata-theoretic machinery — Büchi automata, ω-words, Tarjan SCC emptiness,
LTL→Büchi translation — is the **verification engine underneath**; it is fully reusable
on its own, but the headline use case is driving-scenario model checking.

```
   NuPlan map + ego pose
            │  load_nuplan_subgraph        (BFS-bounded lane/connector DiGraph)
            ▼
   lane/connector graph  ──build_nuplan_kripke──▶  Kripke model  M = (S, →, L)
                                                          │
   LTL formula  φ  ──────ltl_to_buchi_gpvw──────▶  Büchi automaton  B(φ)
                                                          │  kripke_buchi_product
                                                          ▼
                                                   product  M × B
                                                          │  Tarjan SCC emptiness
                                                          ▼
                              SAT → lasso witness route    |    UNSAT → ∅
                                                          │
                          visualize_solution / animate_solution
                                                          ▼
                                            static plot   |   traversal video
```

The whole pipeline is wrapped in **two calls**:

```python
solution = solve_for_path(graph, object_types, ego_id, pos, formula, prop_nodes)
if solution is not None:
    visualize_solution(solution)          # or: animate_solution(solution, window=3)
```

`solve_for_path` returns a `PathSolution` (the witness route + context) on SAT, or
`None` when the specification is infeasible.

## Quick start

```python
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db

from tutorials.buchi import F, Var
from tutorials.buchi.nuplan_scenarios import get_scenarios
from tutorials.buchi.nuplan_graph import load_nuplan_subgraph, nuplan_pos
from tutorials.buchi.nuplan_modelcheck import solve_for_path, visualize_solution, animate_solution

# 1. Load a scenario + its map, and locate the ego vehicle.
scenario = get_scenarios(split='mini', scenario_types=['accelerating_at_traffic_light'])[1]
nuplan_map = NuPlanMapFactory(
    get_maps_db(map_root=scenario.map_root, map_version=scenario.map_version)
).build_map_from_name(scenario.map_api.map_name)
rear = scenario.get_ego_state_at_iteration(0).rear_axle
ego_point = Point2D(rear.x, rear.y)

# 2. Build a BFS-bounded lane subgraph around the ego, with drawing positions.
G, object_map, object_types, ego_id = load_nuplan_subgraph(
    nuplan_map, ego_point, radius=200, depth_limit=10)
pos = nuplan_pos(G, object_map)

# 3. Specify an LTL goal and model-check it.  φ = F(destination): eventually reach 67471.
solution = solve_for_path(G, object_types, ego_id, pos,
                          F(Var('destination')), prop_nodes={'destination': '67471'})

# 4. Plot the satisfying route — and render the traversal as a video.
if solution is not None:
    visualize_solution(solution, title='F(destination)')
    animate_solution(solution, window=3, fps=2, output_path='traversal.mp4')
```

See [`examples/visualize_nuplan.ipynb`](examples/visualize_nuplan.ipynb) for the full tour
(reachability, nested `F(waypoint ∧ F(G(destination)))`, feasible vs infeasible queries,
and the traversal animation).

## Atomic propositions

The LTL formula talks about **atomic propositions** that hold at lane-graph nodes. There
are two kinds:

| Kind | Propositions | Source |
|---|---|---|
| **Structural** (auto-labelled from the graph) | `lane`, `connector`, `unknown`, `ego`, `fork` | node's map-object type; `ego` on the ego node; `fork` where `out_degree > 1` |
| **Named** (user-assigned) | `destination`, `waypoint`, … any name | `prop_nodes={'destination': '67471', 'waypoint': ['64385', …]}` |

`prop_nodes` maps a proposition name to a node id (or an iterable of ids). Named props are
what your formula refers to; structural props are always available. Example formulae:

```python
from tutorials.buchi import F, G, X, And, Or, Not, Var
F(Var('destination'))                                   # eventually reach the destination
F(And(Var('lane'), Not(Var('fork'))))                   # eventually reach a plain (non-fork) lane
F(And(Var('waypoint'), F(G(Var('destination')))))       # reach a waypoint, then stay in the goal
```

---

# `nuplan_graph.py` — lane-graph loading & visualization

The NuPlan front end: turn a NuPlan map into a drawable lane/connector graph, and render
it (static or animated) with NetworkX + matplotlib. All drawing functions return an
`IPython.display.SVG` (use as the last expression in a notebook cell to render inline),
except `animate_nuplan_path`, which returns a video/player object.

## Loading

### `load_nuplan_subgraph(nuplan_map, ego_point, radius, depth_limit)`

Build a **BFS-bounded directed lane subgraph** around the ego vehicle.

```python
small_G, object_map, object_types, ego_id = load_nuplan_subgraph(
    nuplan_map, ego_point, radius=200, depth_limit=10)
```

Queries `nuplan_map.get_proximal_map_objects` for `LANE` and `LANE_CONNECTOR` layers within
`radius` metres of `ego_point`, wires them into a `networkx.DiGraph` via each object's
`outgoing_edges`, finds the node containing the ego, and keeps everything reachable within
`depth_limit` BFS hops of it.

- **`nuplan_map`** — a `NuPlanMap` (from `NuPlanMapFactory.build_map_from_name`).
- **`ego_point`** — `Point2D` of the ego's rear axle.
- **`radius`** *(float)* — map-query radius in metres.
- **`depth_limit`** *(int)* — BFS depth from the ego lane/connector.

**Returns** `(small_G, object_map, object_types, ego_id)`:

| | Type | Meaning |
|---|---|---|
| `small_G` | `nx.DiGraph` | the BFS-bounded subgraph (nodes are string ids) |
| `object_map` | `dict[str, map_obj]` | node id → underlying NuPlan map object |
| `object_types` | `dict[str, str]` | node id → `'lane'` or `'connector'` |
| `ego_id` | `str \| None` | the node containing the ego (`None` if not on the map) |

### `nuplan_pos(graph, object_map)`

Compute **centroid-based `{node_id: (x, y)}` positions** for drawing.

```python
pos = nuplan_pos(small_G, object_map)
draw_G = small_G.subgraph(pos.keys()).copy()   # restrict to drawable nodes
```

Uses each object's `polygon` (or `geometry`) centroid. Nodes whose centroid is non-finite
are **silently skipped**, so callers should restrict drawing to `graph.subgraph(pos.keys())`
— every drawing function below expects `pos` to cover the graph it is handed.

## Static drawing

### `draw_nuplan_subgraph(draw_G, pos, object_types, ego_id, *, title, figsize=(10,8), node_size=600, arrowsize=20, font_size=7)`

Draw the lane subgraph with default colour coding: **lane = blue**, **connector = orange**,
**ego = yellow** (slightly enlarged). A legend, equal-aspect axes, and a faint grid are
added. The plain "here is the neighbourhood" view, with no path.

```python
draw_nuplan_subgraph(draw_G, pos, object_types, ego_id,
                     title='NuPlan subgraph — ego neighbourhood')
```

### `draw_nuplan_path(draw_G, pos, object_types, ego_id, path, *, destination_node=None, highlight_nodes=None, title=None, goal_label=None, figsize=(10,8), node_size=650, offpath_node_size=None, arrowsize=20, font_size=7)`

Draw the subgraph with a **highlighted witness path** — the static counterpart of
`animate_nuplan_path`, and what `visualize_solution` delegates to.

```python
draw_nuplan_path(draw_G, pos, object_types, ego_id, path,
                 destination_node='67471', offpath_node_size=120)
```

- **`path`** — list of node ids in traversal order (the lasso stem). Pass `None` to get a
  printed "no satisfying path" message and a `None` return.
- **`destination_node`** — node coloured **green**; defaults to `path[-1]`.
- **`highlight_nodes`** — `{node_id: (facecolor, legend_label)}` for extra markers (e.g. a
  waypoint in purple). These win over the path/ego/type colours but not over the green goal.
- **`goal_label`** — legend label for the destination node.
- **`offpath_node_size`** — size of nodes **not** on the path; defaults to `node_size`. Set
  it smaller (≈ 80–150) to make the path pop on large graphs.

**Colour precedence**: green goal → custom `highlight_nodes` → yellow ego → orange path →
dim base (blue lane / orange connector). Path edges are drawn **red and thick**.

### `draw_nuplan_nodes(draw_G, pos, object_types, ego_id, highlight_nodes, *, title=None, figsize=(14,11), base_node_size=30, highlight_node_size=650, arrowsize=8, font_size=8)`

Draw the **whole graph as small grey dots** with only selected nodes enlarged, coloured,
and labelled — the natural fallback for an **infeasible** spec: there is no path to draw, so
this shows *where* the (unconnected) nodes of interest sit. `visualize_nodes` delegates here.

```python
draw_nuplan_nodes(draw_G, pos, object_types, ego_id,
                  highlight_nodes={'68280': ('#43a047', 'destination (68280)')})
```

- **`highlight_nodes`** — `{node_id: (facecolor, legend_label)}`; these plus the ego are the
  only enlarged/labelled nodes. Everything else is a faint context dot.

## Animation

### `animate_nuplan_path(draw_G, pos, object_types, ego_id, path, *, window=1, destination_node=None, highlight_nodes=None, title=None, goal_label=None, figsize=(10,8), node_size=650, offpath_node_size=None, arrowsize=20, font_size=7, fps=2.0, output_path=None, show_labels=True, label_window_only=False, repeat=True)`

Animate a **node-by-node traversal** of `path` with a **sliding highlight window**. Frame
`i` advances the *head* (the node the ego currently occupies) to `path[i]` and lights the
trailing window as a fading comet:

- the **head** `path[i]` is drawn **crimson** and enlarged;
- the **`window − 1` nodes behind it** form an **orange** trail;
- nodes that have dropped out of the window **revert to their base style**, so the
  highlighted band visibly *slides* along the route;
- the destination stays **green** and the ego **yellow**, matching `draw_nuplan_path`.

```python
animate_nuplan_path(draw_G, pos, object_types, ego_id, path,
                    window=3, fps=6, output_path='traversal.mp4')
```

| Parameter | Effect |
|---|---|
| `window` *(int ≥ 1)* | trailing-highlight length; `1` spotlights only the current node, a large value keeps the whole travelled prefix lit |
| `fps` *(float)* | frames per second for both the inline player and any written file |
| `output_path` | also write a file: `.mp4` (H.264 via ffmpeg) or `.gif` (Pillow). Omit for an inline player |
| `show_labels` | draw node-id labels (default `True`) |
| `label_window_only` | when labelling, label only the head/trail nodes each frame — keeps long paths legible |
| `repeat` | loop the animation in the inline player |
| `destination_node` / `highlight_nodes` / `goal_label` / `figsize` / `node_size` / `offpath_node_size` / `arrowsize` / `font_size` | as in `draw_nuplan_path` |

**Returns** an `IPython.display` object that renders inline in a notebook cell:

- **no `output_path`** → an `HTML` **HTML5/JS player** (self-contained, needs **no ffmpeg**);
- **`output_path='….mp4'`** → writes H.264 via ffmpeg and returns an inline `Video`;
- **`output_path='….gif'`** → writes a GIF via Pillow and returns an inline `Video`.

`.mp4` requires ffmpeg on the `PATH` (`apt-get install ffmpeg`); if it is missing the writer
**falls back to a sibling `.gif`** with a printed notice. Returns `None` (and prints) when
`path` is empty.

**Example output** — the 115-step witness for `φ = F(64385 ∧ F(G(63680)))` on a 700-node
subgraph (example 6a in the notebook), animated with `window=5`, `path_only=True`:

<video src="examples/traversal_6a.mp4" controls muted loop width="640"></video>

## Internal helpers

`_set_equal_limits` / `_xy_limits` (equal-aspect, non-jumping axis bounds), `_fig_to_svg`
(figure → inline SVG), and the animation palette constants `_ANIM_HEAD_COLOR` /
`_ANIM_TRAIL_COLOR` are private and not part of the public API.

---

# `nuplan_modelcheck.py` — the model-checking pipeline

Collapses *build Kripke model → translate LTL to Büchi → form product → emptiness check →
project the lasso stem → plot/animate* into a few calls.

| Function | Purpose |
|---|---|
| `solve_for_path(graph, object_types, ego_id, pos, formula, prop_nodes=None, *, add_fork=True, self_loops=True, goal_node=None, goal_label=None, verbose=True)` | run the whole pipeline; return a `PathSolution` on SAT or `None` on UNSAT |
| `PathSolution` | dataclass holding the witness `path`/`cycle`, the `kripke`/`buchi`/`product`, `prop_nodes`, and drawing context; `.describe()` prints the lasso step-by-step |
| `build_nuplan_kripke(graph, object_types, ego_id, prop_nodes=None, *, add_fork=True, self_loops=True)` | build just the labelled `KripkeModel` (adds stutter self-loops so `F`/`G` acceptance works) |
| `visualize_solution(solution, *, path_only=False, **flags)` | plot the witness route (delegates to `draw_nuplan_path`); `path_only` draws only the path's nodes |
| `animate_solution(solution, *, window=1, path_only=False, output_path=None, **flags)` | render the traversal video (delegates to `animate_nuplan_path`) |
| `visualize_nodes(graph, object_types, ego_id, pos, prop_nodes, **flags)` | UNSAT fallback — show the prop nodes on the full graph (delegates to `draw_nuplan_nodes`) |

Named props are coloured automatically: `destination` green, `waypoint` purple, any other
named prop teal. `**flags` on `visualize_solution`/`animate_solution` are forwarded to the
underlying `nuplan_graph` drawing routine (`figsize`, `node_size`, `offpath_node_size`,
`font_size`, `arrowsize`, `title`, and for animation `fps`, `label_window_only`, …).

---

# The Büchi verification engine

The standard-library core that powers the pipeline. Reusable for general LTL / ω-automata
work independent of NuPlan.

## Features

| Feature | Entry point | Notes |
|---|---|---|
| Büchi automaton with a relational transition function `Δ` | `BuchiAutomaton` | non-deterministic, partial; states are any hashable, letters are sets |
| Infinite words as lassos `u·vᵒ̬` | `OmegaWord` | ultimately-periodic ω-words; `unroll(n)` |
| Emptiness check + witness | `BuchiAutomaton.check_emptiness` / `is_empty` | Tarjan SCC; returns an accepted witness lasso and the accepting SCC |
| Emptiness with concrete states | `BuchiAutomaton.check_emptiness_with_states` | returns `(is_empty, prefix_states, cycle_states)` — used to project the route |
| Membership / run verification | `BuchiAutomaton.accepts(word)` | product-with-loop search for an accepting run |
| Tarjan strongly-connected components | `tarjan_scc(graph)` | iterative (no recursion-depth limit) |
| Intersection `L(A₁) ∩ L(A₂)` | `intersect(a1, a2)` / `BuchiAutomaton.intersect` | `Q₁×Q₂×{1,2}` product with tracking bit |
| Kripke × Büchi product | `kripke_buchi_product(kripke, buchi)` | the model-checking product `M × B` |
| Generalized Büchi acceptance + degeneralization | `GeneralizedBuchiAutomaton`, `.to_buchi()` | `k` accepting sets → counter construction |
| LTL syntax, NNF, semantics | `ltl.py` (`Atom`/`Var`, `Not`, `X`, `F`, `G`, `U`, `R`, `W`, …) | operator sugar (`&`, `\|`, `~`, `>>`); `satisfies(φ, word)` reference semantics |
| LTL → Büchi translation | `ltl_to_buchi_gpvw(φ)`, `ltl_to_gba_gpvw(φ)` | GPVW on-the-fly tableau → GBA → BA |
| Kripke model `M = (S, →, L)` | `KripkeModel` | `add_state` / `add_initial_state` / `add_transition`; `labeling` maps state → propositions |

## Concepts

- **Büchi automaton** `A = (Q, Σ, δ, Q₀, F)` reads *infinite* words. A run is accepting iff
  it visits the accepting set `F` **infinitely often**. `L(A)` is the set of words with an
  accepting run.
- **Letter.** Every letter of `Σ` is a *set* of atomic propositions, stored as a `frozenset`.
- **ω-word as a lasso.** The words that matter are *ultimately periodic*: a non-recurrent
  prefix `u` followed by a recurrent block `v` repeated forever — written `u·vᵒ̬` and
  represented by `OmegaWord(prefix=u, loop=v)`.
- **Model checking** a Kripke model `M` against `φ`: build `B(φ)`, form the product `M × B`,
  and test `L(M × B)` for emptiness. Non-empty ⇒ `M` has an execution satisfying `φ`, and the
  witness lasso projects back to a concrete route through `M`.

## Emptiness check

> `L(A) ≠ ∅` **iff** some accepting state is reachable from an initial state *and* lies on a
> cycle — equivalently, a reachable **non-trivial** SCC (size > 1, or a single self-looping
> node) contains an accepting state.

`check_emptiness()` finds SCCs with **Tarjan's algorithm** (`tarjan.py`) and, when non-empty,
reconstructs a concrete witness lasso: a BFS shortest path from an initial state to an
accepting state (stem `u`) plus a shortest cycle through it inside its SCC (loop `v`).

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

## LTL → Büchi

`ltl_to_buchi_gpvw(formula)` translates a linear-temporal-logic formula into a Büchi
automaton accepting exactly its models, via the **GPVW** (Gerth–Peled–Vardi–Wolper)
on-the-fly tableau (`gpvw_ltl_to_buchi.py`): formula → negation normal form → generalized
Büchi automaton (one accepting set per `Until`) → degeneralized Büchi automaton.

```python
from tutorials.buchi import Atom, G, F, ltl_to_buchi_gpvw, satisfies

p, q = Atom("p"), Atom("q")
phi = G(p >> F(q))                      # every p is eventually followed by q
buchi = ltl_to_buchi_gpvw(phi)
print(buchi.is_empty())                 # False (satisfiable)
witness = buchi.check_emptiness().witness
assert satisfies(phi, witness)          # the witness lasso models phi
```

`ltl.satisfies(formula, word)` is an independent operator-complete LTL semantics over lasso
words, used to cross-check the translation (the test suite verifies `accepts == satisfies`
over many words).

## Intersection & generalized acceptance

`intersect(a1, a2)` builds an automaton for `L(A1) ∩ L(A2)` via the standard `Q1 × Q2 × {1,2}`
product with a tracking bit that toggles on `F1`/`F2` states, so the accepting set is hit
infinitely often iff both `F1` and `F2` are. `GeneralizedBuchiAutomaton` allows several
accepting sets `F₀,…,F_{k-1}` (a run must visit *each* infinitely often); `to_buchi()`
**degeneralizes** to an ordinary Büchi automaton via a counter `i ∈ {0..k-1}` — the `k`-set
generalization of the intersection toggle.

## Visualizing automata (Graphviz)

`visualization_graphviz.py` renders any `BuchiAutomaton` / `GeneralizedBuchiAutomaton` /
`KripkeModel` as a **Graphviz/DOT** diagram (publication-quality self-loops, hierarchical
layout, SVG/PDF). Initial states are **yellow**, accepting states **green (double ring)**, and
a `highlight_path` is drawn in red.

```python
from tutorials.buchi.visualization_graphviz import draw
import IPython.display as display

g = draw(buchi, title="φ = G(p → F q)", initial_color="#ffeb3b")
display.SVG(g.pipe(format='svg'))        # inline in Jupyter
g.render('/tmp/automaton', format='pdf') # save to file
```

Requires `pip install graphviz` and the Graphviz binary (`apt-get install graphviz`). (NuPlan
lane graphs are drawn instead with the NetworkX/matplotlib functions in `nuplan_graph.py`.)

# Package layout

```
buchi/
├── ltl.py                    LTL AST (Atom/Var, Not, And, Or, X, F, G, U, R, W) + satisfies()
├── gpvw_ltl_to_buchi.py      GPVW on-the-fly LTL→Büchi (to_nnf, ltl_to_gba_gpvw, ltl_to_buchi_gpvw)
├── buchi_automaton.py        BuchiAutomaton, EmptinessResult, intersect, kripke_buchi_product
├── generalized_buchi.py      GeneralizedBuchiAutomaton + degeneralization to_buchi()
├── kripke.py                 KripkeModel — (S, →, L)
├── omega_word.py             OmegaWord — infinite words as lassos (prefix + loop)
├── tarjan.py                 tarjan_scc — iterative strongly-connected components
├── nuplan_scenarios.py       get_scenarios — wrapper over NuPlan's scenario builder
├── nuplan_graph.py           lane-subgraph loading + NetworkX drawing & traversal animation
├── nuplan_modelcheck.py      solve_for_path / visualize_solution / animate_solution pipeline
├── visualization_graphviz.py Graphviz/DOT renderer for Büchi automata & Kripke models
├── examples/                 visualize_nuplan.ipynb, visualize_LTL_buchi.ipynb, visualization_graphviz.ipynb
└── test/                     unittest suites (one per core module)
```

## Dependencies

- **Engine** (`ltl`, `gpvw_ltl_to_buchi`, `buchi_automaton`, `generalized_buchi`, `kripke`,
  `omega_word`, `tarjan`): **standard library only**.
- **NuPlan layer** (`nuplan_scenarios`, `nuplan_graph`, `nuplan_modelcheck`): `networkx`,
  `matplotlib`, the `nuplan-devkit`, and the NuPlan dataset + maps (paths configurable via
  `NUPLAN_DATA_ROOT` / `NUPLAN_MAPS_ROOT` / `NUPLAN_MAP_VERSION` — see `nuplan_scenarios.py`).
  `.mp4` animation additionally needs **ffmpeg**; `.gif` needs **Pillow**.
- **Automaton diagrams** (`visualization_graphviz`): `graphviz` + the Graphviz binary.

## Tests

```bash
python -m pytest tutorials/buchi/test/ -v
# or
python -m unittest discover -s tutorials/buchi/test -v
```

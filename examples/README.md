# Examples — visualization

[`visualization_demo.ipynb`](visualization_demo.ipynb) visualizes two kinds of diagram as
**NetworkX** graphs:

1. **Büchi automaton** — a small automaton accepting the ω-word `a·b·(a·b·c)ᵒ̬`. Propositions
   live on the *transitions*, so they are shown as **edge labels**; nodes are indexed
   `q0, q1, …`. The **start state is yellow** and the **accepting state is green**
   (drawn with a double ring).
2. **Kripke model** `(S, →, L)` — propositions live on the *states*, so they are drawn
   **inside the nodes**; the edges are plain directed arrows **with no labels**.

The renderer is [`../visualization.py`](../visualization.py):

```python
from tutorials.buchi.visualization import draw

# Büchi automaton: edge labels on, custom initial/accepting colours.
draw(buchi, initial_color="#ffeb3b", accepting_color="#66bb6a")

# Kripke model: propositions inside nodes, no edge labels.
draw(kripke, node_propositions=kripke.labeling, show_edge_labels=False)
```

A node shows the **predicates holding there** when `node_propositions` is supplied,
otherwise its **state index**. Initial states are coloured (`initial_color`), accepting
states are coloured + double-ringed (`accepting_color`), and a highlighted set (e.g. an
accepting SCC) is red.

## Running

The notebook adds the repo root to `sys.path` itself, so it runs from any directory:

```bash
jupyter notebook tutorials/buchi/examples/visualization_demo.ipynb
# or headless:
jupyter nbconvert --to notebook --execute --inplace \
    tutorials/buchi/examples/visualization_demo.ipynb
```

# Graphviz Visualization

[`visualization_graphviz.ipynb`](visualization_graphviz.ipynb) uses **Graphviz** (the DOT language) to render Büchi automata and Kripke models. Graphviz produces publication-quality diagrams with:

- **Excellent self-loop rendering** — arcs positioned naturally on the node
- **Hierarchical layout** — clean, readable structure automatically
- **Professional edge label positioning** — labels sit cleanly on edges
- **Vector output** (SVG/PDF) — scales to any size without quality loss

The visualizer lives in [`../visualization_graphviz.py`](../visualization_graphviz.py):

```python
from tutorials.buchi.visualization_graphviz import draw

# Render a Büchi automaton to an SVG graphviz object.
g = draw(buchi, title="My automaton",
         initial_color="#ffeb3b", accepting_color="#66bb6a")

# Display inline in Jupyter:
import IPython.display as display
display.SVG(g.pipe(format='svg'))

# Or save to a file:
g.render('/tmp/my_automaton', format='pdf')
```

Both the Python `graphviz` package and the Graphviz binary (dot) must be installed:

```bash
pip install graphviz
# On macOS: brew install graphviz
# On Ubuntu: apt-get install graphviz
# On Windows: https://www.graphviz.org/download/
```

The notebook auto-detects the repo root and runs from any directory. Open it with:

```bash
jupyter notebook tutorials/buchi/examples/visualization_graphviz.ipynb
```

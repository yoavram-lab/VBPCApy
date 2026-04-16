# Plotting

VBPCApy provides three convenience plotting functions. They require matplotlib,
installed via the `plot` extra.

## Install

```bash
pip install vbpca_py[plot]
```

## Scree plot

Shows the explained variance ratio for each component with an optional
cumulative line:

```python
from vbpca_py import VBPCA
from vbpca_py.plotting import scree_plot

model = VBPCA(n_components=10, maxiters=200)
model.fit(X)

fig = scree_plot(model, cumulative=True)
```

Pass an existing axes to embed in a subplot:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
scree_plot(model, ax=ax, cumulative=True)
ax.set_title("My scree plot")
plt.show()
```

## Loadings bar plot

Visualise which features contribute most to a given component:

```python
from vbpca_py.plotting import loadings_barplot

fig = loadings_barplot(model, component=0, top_n=15)
```

| Parameter | Description |
|-----------|-------------|
| `component` | Zero-indexed component to plot (default: 0) |
| `top_n` | Show only the N highest-magnitude features |
| `feature_names` | List of feature name strings for the x-axis |

## Variance explained plot

Absolute variance (not ratio) per component, useful for comparing across
models:

```python
from vbpca_py.plotting import variance_explained_plot

fig = variance_explained_plot(model)
```

## Customisation

All three functions return a `matplotlib.figure.Figure` and accept an optional
`ax` parameter. Use standard matplotlib API to customise colours, labels,
saving, etc.:

```python
fig = scree_plot(model)
fig.savefig("scree.png", dpi=150, bbox_inches="tight")
```

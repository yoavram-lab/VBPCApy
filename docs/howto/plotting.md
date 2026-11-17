# Plotting

VBPCApy provides three convenience plotting functions. They require matplotlib,
installed via the `plot` extra.

## Install

```bash
pip install "vbpca-py[plot]"
```

The examples plot a model fitted to 200 samples and 30 features with a
rank-four signal:

```python
import numpy as np

from vbpca_py import VBPCA

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 4)) @ rng.standard_normal((4, 30))
X += 0.3 * rng.standard_normal(X.shape)

model = VBPCA(n_components=10, maxiters=200, random_state=0).fit(X)
```

## Scree plot

Shows the explained variance ratio for each component with an optional
cumulative line:

```python
from vbpca_py.plotting import scree_plot

fig = scree_plot(model, cumulative=True)
```

Pass an existing axes to embed in a subplot:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
scree_plot(model, ax=ax, cumulative=True)
ax.set_title("My scree plot")
```

## Loadings bar plot

Visualise which features contribute most to a given component:

```python
from vbpca_py.plotting import loadings_barplot

names = [f"feature {j}" for j in range(X.shape[1])]
fig = loadings_barplot(model, component=0, top_n=15, feature_names=names)
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

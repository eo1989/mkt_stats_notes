---
title: "IPyWidgets test"
format: html
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: WinPy3.12
    language: python
    name: win_pyenv3.12
---

```python
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML
```

```python
data = {
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "value": [10, 20, 30, 40, 50],
}
df = pd.DataFrame(data)
```

```python
name_dropdown = widgets.Dropdown(
    options=["All"] + df["name"].unique().tolist(),
    value="All",
    description="Name:",
    style={"description_width": "initial"},
)
```

```python
def filter_df(name):
    if name == "All":
        filtered_df = df
    else:
        filtered_df = df[df["name"] == name]
    display(HTML(filtered_df.to_html(index=False)))
```

```python
out = widgets.interactive_output(filter_df, {"name": name_dropdown})
```

```python
display(name_dropdown, out)
```

```python

```

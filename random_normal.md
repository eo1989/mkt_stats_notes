---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

```

```python
days = 18
sims = 1_000
stock_symb = 'AAPL'
stock_price = 100
drift = 0.0004
sigma = 0.018
call_strike = 110.
call_price = 0.84
call_be = call_strike + call_price
```

```python
setup = np.arange(sims*days).reshape((sims, days), order='C')
draw = np.zeros_like(setup, dtype="float32")
price = np.zeros_like(setup, dtype="float32")

```

```python
np.random.seed(742)
draw = np.random.standard_normal([sims, days])
```

```python
ito_adj_drift = drift - ((sigma**2)/2)
f"{ito_adj_drift:.7f}"
```

more on the drift adjustment.
$(\mu - \frac{\sigma^2}{2})$ .. is necessary because the following two conditions are true:
- Even though $EV(\Epsilon) = 0$, and $e^{0} = 1$ because we have a skewed log-normal transformation, $EV(e ^{\epsilon}) > 1$.
- $\mu$ can be set to zero in this context, and it often is, but will see why you may need to change it.

```python

```

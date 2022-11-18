# %%
import datetime as dt
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pd.options.plotting.backend = "plotly"
pio.templates.default = 'plotly_dark'
pio.renderers.default = 'notebook_connected'

sns.set_style('darkgrid')
# sns.style.use('seaborn-deep')
plt.rcParams['figure.dpi'] = 150


# %%
def kelly(p, w=1, l=1):
    return p / l - (1 - p) / w


kelly(0.7)

# %% [markdown]
# ## Probability matching is suboptimal
#
# Below we examine the case of an individual playing a game with the following rules:
# - *n* iterations
# - each iteration is a flip of a biased coin with $p = 0.8$
# - in each iteration, an individual bets \\$1 on either H or T. They lose their bet if they are wrong, win their bet if they are right.
# Obviously the dominant strategy is to bet \\$1 on H every time if $p > 0.5$, else bet \\$1 on $T$ every time.
# Probability matching" is the observed bias in which people tend to bet H with a frequency approx equal to the probability of H, i.e betting heads 80% of the time.
# Consider a situation where the coin has known bias $p >= 0.5$ and we bet heads with frequency $f$. Elementary arithmetic shows that the EV per iteration is $1 - 2(f+p-2fp)$.
# In the "\optimal"\ case $f=1$, we recover $EV = 2p-1$. Probability matching yields $(2p-1)^2$.

# %%
# Analysis
ps = np.linspace(0.5, 1, 100)
df = pd.DataFrame({
    "optimal": 2 * ps - 1,
    "matching": (2 * ps - 1)**2
},
                  index=ps)
df.plot()

# %% [markdown]
# For a given true probability, you can see that the EV is linear in f. e.g if $p = 0.8$
# you end up with $EV = 1.2f - 0.6$
# This is verified via simulation (below)

# %%


def flips(p=0.5, n=10_000):
    # codes H/T as +/- 1
    a = np.random.binomial(1, p, (n, ))
    return np.where(a == 0, -1, a)


# %%

true_prob = 0.8
s_prob = 1
y = flips(true_prob)
s = flips(s_prob)


def eval_strategy(s_prob, true_prob=0.8, n=10_000):
    y = flips(true_prob, n)
    s = flips(s_prob, n)
    return (y * s).cumsum()


# %%

n = 100_000
ps = np.arange(0.5, 1.0, 0.05)
res = []
for s_prob in ps:
    res.append(eval_strategy(s_prob, 0.8, n)[-1] / n)

pd.DataFrame({"analytical": 1.2 * ps - 0.6, "sim": res}, index=ps).plot()
# %%

# plot out wealth growth
ps = np.arange(0.5, 1.0, 0.05)
res = {}
for s_prob in ps:
    res[f"{s_prob:.2f}"] = eval_strategy(s_prob, 0.8, 1000)

pd.DataFrame(res).plot()
# %%


def kelly_fraction(p, W):
    return p - (1 - p) / W


Ws = np.linspace(0.5, 10, 100)
fs = kelly_fraction(0.75, Ws)

fig, ax = plt.subplots()
ax.plot(Ws, fs, label='kelly')
ax.set_ylim(0.4, 1)
ax.axhline(0.75,
           xmin=0,
           xmax=9,
           color='k',
           linestyle='--',
           label='prob matching')
ax.set_xlabel('W')
ax.set_ylabel('f*')
ax.legend()
plt.show()

# %%

kelly_fraction(0.75, 5)
# 0.7
# %%


def kelly_simul(true_prob=0.8, n=100_000):
    y = flips(true_prob, n)


# %%

y = flips(true_prob, n)
print(f"{y}")
# y is an array([1, 1, 1, ..., -1, 1, 1])

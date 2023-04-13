# %%

from decimal import DivisionByZero
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

sns.set_style('ticks')
sns.set_context('notebook', rc = {'lines.linewidth': 0.6})
plt.figure(figsize = (8, 6))

# %%
def N(x: float|np.ndarray) -> float|np.ndarray:
	""" Normal distribution function """
	return norm.cdf(x)

# def S(self, S: float) -> float|DivisionByZero:
# 	if S >= 0:
# 		return S
# 	elif S <= 0:
# 		return DivisionByZero("S must be greater than 0")
# def S(self, S: float) -> Any:
# 	self.S = S
# 	return S if S >= 0 else DivisionByZero("S must be greater than 0")

def K(self, K: float) -> Any:
	self.K = K
	return K if K >= 0 else exit

def opt(S: float, K: float, r: float, sigma: float, ttm: float, c_or_p: str) -> float:
	"""
	Euro call/put price calc, again, using BSM & python types
	S: spot
	K: strike
	r: risk free rate
	sigma: volatility
	ttm: time to maturity
	c_or_p: call or put
	"""
	d1 = (np.log(S/K) + (r + 0.5*(sigma**2)) * ttm) / (sigma * np.sqrt(sigma))
	d2 = d1 - sigma * np.sqrt(ttm)

	if (c_or_p == "call"):
		return (S*N(d1) - K*np.exp(-r * ttm)*N(d2))
	else:
		return (K*np.exp(-r*ttm) * N(-d2) - S*N(-d1))
# %%
"""
Option price w/ respect to underlying:
\delta = \frac{\partial V}{\partial S}
\gamma = \frac{\partial^2 V}{\partial S^2}

\frac{\delta p}{\delta S} = \frac{delta c}{\delta S}
"""
s = [i for i in range(0, 101)]
c = [opt(S = i, K = 50, r = 0.04, sigma = 0.3, ttm = 1, c_or_p="call") for i in range(0, 101)]
p = [opt(S = i, K = 50, r = 0.04, sigma = 0.3, ttm = 1, c_or_p="put") for i in range(0, 101)]

df = pd.DataFrame(data = {'Underlying': s, 'Call':c, 'Put': p})
sns.lineplot(data = df)
plt.axvline(x = 50, ls = '-.', c = 'black')

# %%
sns.lineplot(data = df, x = 'Underlying', y = 'Call', color = 'green')
plt.axvline(x = 50, ls = '-', c = 'black')
# %%
sns.lineplot(data = df, x = 'Underlying', y = 'Put', color = 'red')
plt.axvline(x = 50, ls = '-', c = 'black')
#%%
"""
Option Price w.r.t. strike:
"""
s = [i for i in range(1, 101)]
k = [i for i in range(1, 101)]
c = [opt(S = 50, K = i, r = 0.05, sigma = 0.3, ttm = 1, c_or_p = "call") for i in range(1, 101)]
p = [opt(S = 50, K = i, r = 0.05, sigma = 0.3, ttm = 1, c_or_p = "put") for i in range(1, 101)]

df = pd.DataFrame(data = {'Underlying': s, 'Strike': k, 'Call': c, 'Put': p})
# %%
sns.lineplot(data = df, x = 'Strike', y = 'Call', color = 'green')
plt.axvline(x = 50, ls = '-', c = 'black')
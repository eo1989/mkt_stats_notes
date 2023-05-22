# %%
"""
1. GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
	- Model that captures the volatility clustering in (financial) time series data, by modeling the conditional variance as a function of past values of the return series and past values of the conditional variance.
	- Commonly used to model volatility in financial markets.
		- The `Arch` library is used to create a list of returns for a financial time series, then fit a GARCH(1,1) model to the returns, and prints out a summary of the results.
		- This summary includes the estimated parameters, the standard errors, the log-likelihood, and the Akaike and Bayesian information criteria. The `p` and `q` arguments specify the autoregressive and moving average order of the model, respectively.
	example:
"""
from ast import With
import arch as ar

returns = [0.03, -0.02, 0.05, 0.04, -0.05, -0.01, 0.03]
model = ar.arch_model(returns, mean="constant", vol="GARCH", p=1, q=1)
result = model.fit()
print(f"result summary:\n{result.summary()}")

# %%
"""
2. EGARCH (Exponential GARCH)
	- A variant of the GARCH model that models the log of the conditional variance instead of the conditional variance itself.
 	- This allows the conditional variance to be negative, which can be useful in financial applications, it's a more flexible model.
"""

model = ar.arch_model(returns, mean="constant", vol="EGARCH", p=1, q=1)
result = model.fit()
print(f"result summary:\n{result.summary()}")


# %%
"""
3. ARCH (Autoregressive Conditional Heteroskedasticity)
	- A simpler model that assumes that the conditional variance is a function of past squared residuals (squared `returns`) of the `return` series.
"""
model = ar.arch_model(returns, mean="constant", vol="ARCH", p=1)
result = model.fit()
print(f"{result.summary()}")

# %%
"""
4. SV (Stochastic Volatility) Models
	- A class of models that capture the time-varying volatility in financial time series data by modeling the volatility as an unobserved stochastic process.
"""
# import pystan
# example used pystan but that would require linux/macos
import os
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm

rng = np.random.RandomState(1234)
az.style.use("arviz-darkgrid")
MSFT = f"C:\\Users\\eorlo\\Desktop\\ASX_Portfolio\\eo_practice\\mkt_stats_notes\\Interactive-Data-Visualization-with-Python\\datasets\\chap5_data\\microsoft_stock.csv"
# print(MSFT) --> checks out it works
try:
    # returns = pd.read_csv("C:\Users\eorlo\Desktop\ASX_Portfolio\eo_practice\mkt_stats_notes\Interactive-Data-Visualization-with-Python\datasets\chap5_data\microsoft_stock.csv", index_col = 'date')
    returns = pd.read_csv(MSFT, index_col="date")
except FileNotFoundError:
    # returns = pd.read_csv(pm.get_data("microsoft_stock.csv"), index_col="date")
    returns = pd.read_csv(pm.get_data(MSFT), index_col="date")

# returns["change"] = np.log(returns["close"]).diff()
returns["change"] = np.log(returns["adj_close"]).diff()
returns = returns.drop(columns=["open", "high", "low", "volume", "close"]).dropna()
# returns = returns.dropna()
returns.head()
# %%
fig, ax = plt.subplots(figsize=(14, 4))
returns.plot(y="change", label="MSFT", ax=ax, lw=0.5)
ax.set(xlabel="time", ylabel="returns")
ax.legend()


# %%
# Specify the model in PyMC mirrors its statistical specification
def bake_stochastic_vol_model(data):
    with pm.Model(coords={"time": data.index.values}) as model:
        step_size = pm.Exponential("step_size", 10)
        volatility = pm.GaussianRandomWalk(
            "volatility", sigma=step_size, dims="time", init_dist=pm.Normal.dist(0, 100)
        )
        nu = pm.Exponential("nu", 0.1)
        returns = pm.StudentT(
            "returns",
            nu=nu,
            lam=np.exp(-2 * volatility),
            observed=data["change"],
            dims="time",
        )
    return model


os.add_dll_directory("C:/Users/eorlo/scoop/apps/gcc/current/bin")
stochastic_vol_model = bake_stochastic_vol_model(returns)

# %%
pm.model_to_graphviz(stochastic_vol_model)
# Everything checks out. awwwright.
# %%
with stochastic_vol_model:
    idata = pm.sample_prior_predictive(500, random_seed=rng)

prior_predictive = az.extract(idata, group="prior_predictive")
# %%

fig, ax = plt.subplots(figsize=(14, 4))
returns["change"].plot(ax = ax, lw = 1, color = 'black')
ax.plot(prior_predictive['returns'][:, 0::10], "g", alpha = 0.5, lw = 1, zorder = -10)

max_observed, max_simulated = np.max(np.abs(returns["change"])), np.max(np.abs(prior_predictive["returns"].values))

ax.set_title(f"Maximum observed: {max_observed:.2g}\nMax simulated: {max_simulated:.2g}(!)")
# %%

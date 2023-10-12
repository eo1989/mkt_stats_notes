# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import warnings; warnings.filterwarnings("ignore", module = "seaborn\..*")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set_theme()
colors = sns.color_palette()
colors += colors

no_port_mgrs = 12

no_returns = np.random.normal(loc = 20, scale = 8, size = no_port_mgrs)
means = np.random.normal(loc = 0.1, scale = 0.05, size = no_port_mgrs)
stds = np.random.normal(loc = 0.3, scale = 0.05, size = no_port_mgrs)
skews = np.random.randn(no_port_mgrs) * 0.5

ports = [stats.pearson3(loc = mean, scale = std, skew = sk) for mean, std, sk in zip(means, stds, skews)]

port_mgr = np.arange(no_port_mgrs)
shape_dim = [3, 4]

port_mgr = port_mgr.reshape(shape_dim)
fig, axs = plt.subplots(nrows=3, ncols=4, figsize = (12, 11))

port_returns = {}

for (x, y), port in np.ndenumerate(port_mgr):
    years = int(no_returns[port])
    port_returns[port] = ports[port].rvs(size = int(years))
    sns_dist = sns.histplot(port_returns[port], bins = 10, kde = True, ax = axs[x, y], color = colors[port_mgr[x][y]])
    axs[x, y].set_title(f"Mgr {port_mgr[x][y]}, {years} yrs")
    axs[x, y].set_xlim(-1, 1)

fig.suptitle('Comparison of portfolio managers returns')

#
#
#
#
#
#
#

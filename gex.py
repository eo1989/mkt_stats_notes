# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime as dt
import scipy as sc
from datetime import timedelta, date
import itertools as itr

# %%
# df = pd.read_csv("data/spy_3.csv", header=0, index_col = 0)
# df.head()


# %%
# This is bad. Do not do it this way. us pandas instead of str manipulation.
df2 = pd.read_csv("data/spy_3.csv", sep='\s+', header=None, skiprows=0)
df2.head()

# %%
df2.columns = ['FirstCol']
spot_price = df2['FirstCol'][0].split(',')[2]
spot_price

# %%
spot_price = float(spot_price.split('Last: ')[1])

# %%

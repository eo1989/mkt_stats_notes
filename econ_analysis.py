# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3.9.6 64-bit
#     language: python
#     name: python3
# ---



# +
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from fredapi import Fred as fr
pd.set_option('display.max_columns', 500)

color_pallette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.style.use(["fivethirtyeight", "../presentation.mplstyle"])
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.titlesize"] = 24
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


fred_key = os.environ['FRED_API']
# -

fred = fr(api_key=fred_key)

# search for economic data
sp_search = fred.search('S&P', order_by = 'popularity')
sp_search.head()

sp500 = fred.get_series(series_id='SP500')
sp500.plot(title = 'S&P 500', lw = 2)

fred.search('unemployment')


unrate = fred.get_series('UNRATE')
unrate.plot()

unemp_df = fred.search('Unemployment Rate State', filter=('frequency', 'Monthly'))
unemp_df

unemp_df = unemp_df.query('seasonal_adjustment == "Seasonally Adjusted" and units == "Percent"')
unemp_df

# unemp_df['title'].str.contains('Unemployment Rate')
unemp_df = unemp_df.loc[unemp_df['title'].str.contains('Unemployment Rate')]
unemp_df.shape

# +
all_results = []

for myid in unemp_df.index:
	results = fred.get_series(myid)
	results = results.to_frame(name = myid)
	all_results.append(results)
# -

unemp_df.tail(10)

# +
all_results = []

for myid in unemp_df.index:
	results = fred.get_series(myid)
	results = results.to_frame(name = myid)
	all_results.append(results)
unemp_results = pd.concat(all_results, axis=1).drop(['UNRATE','LNS14000089', 'LRHUTTTTUSM156S', 'LRUNTTTTUSM156S', 'LRUN24TTUSM156S', 'LNS14000026', 'CWSTUR', 'LNS14000025', 'LNS14023705', 'USAURAMS', 'LNS14000315', 'LNS14000006', 'LNS14000031', 'M0892AUSM156SNBR', 'M0892BUSM156SNBR', 'LNS14000024', 'LNS14000009', 'LNS14000002', 'LNS14000003', 'U2RATE', 'LNS14000001', 'LNS14027662', 'LRUN64TTUSM156S', 'LNS14027660', 'LNS14032183', 'LNS14000012', 'LNS14024887', 'LNS14000018', 'LASMT261982000000003', 'LNS14000060', 'LASMT391746000000003', 'LNS14000028', 'LNS14000036', 'LNS14000032', 'LNS14027659', 'LNS14027689', 'LNS14024230', 'LNS14000048', 'LNS14000029'], axis =1)
unemp_states = unemp_results.copy()# redo this programmatically, function or loop through w/ number of chars >~5 to be excluded

# +
unemp_results.dropna(axis = 1)
unemp_states = unemp_states.dropna()

# id_to_state = unemp_df['title'].str.replace('Unemployment Rate in ','').to_dict()
unemp_states.columns
# -

# Plot states nemployment rate
px.line(unemp_states)

## Pull April 2020 unemployment rate per state
ax = unemp_states.loc[unemp_states.index == '2020-02-01'].T. \
	sort_values('2020-02-01'). \
	plot(kind='barh', figsize = (8, 12), width = 0.7, edgecolor = 'black', \
		 title = 'Unemployment Rate by State, April 2020')
ax.legend().remove()
ax.set_xlabel('% Unemployed')
plt.show()


part_df = fred.search('participation rate state', filter = ('frequency', 'Monthly'))
part_df = part_df.query('seasonal_adjustment == "Seasonally Adjusted" and units == "Percent"')
part_df

# +
# part_id_to_state = part_df['title'].str.replace('Labor Force Participation Rate for ','').to_dict()

all_results = []

for myid in part_df.index:
	results = fred.get_series(myid)
	results = results.to_frame(name = myid)
	all_results.append(results)
part_states = pd.concat(all_results, axis=1)
# .drop(['UNRATE','LNS14000089', 'LRHUTTTTUSM156S', 'LRUNTTTTUSM156S', 'LRUN24TTUSM156S', 'LNS14000026', 'CWSTUR', 'LNS14000025', 'LNS14023705', 'USAURAMS', 'LNS14000315', 'LNS14000006', 'LNS14000031', 'M0892AUSM156SNBR', 'M0892BUSM156SNBR', 'LNS14000024', 'LNS14000009', 'LNS14000002', 'LNS14000003', 'U2RATE', 'LNS14000001', 'LNS14027662', 'LRUN64TTUSM156S', 'LNS14027660', 'LNS14032183', 'LNS14000012', 'LNS14024887', 'LNS14000018', 'LASMT261982000000003', 'LNS14000060', 'LASMT391746000000003', 'LNS14000028', 'LNS14000036', 'LNS14000032', 'LNS14027659', 'LNS14027689', 'LNS14024230', 'LNS14000048', 'LNS14000029'], axis =1)
# -

part_id_to_state = part_df['title'].str.replace('Labor Force Participation Rate ','').to_dict()
part_states.columns = [part_id_to_state[c] for c in part_states.columns]

part_states



# %%
import yfinance as yf
import pandas as pd
from datetime import datetime, date

_today = date.today().strftime("%Y-%m-%d")
print(f"today is '{_today}'")

# %%
symbols = [
    # "MSFT",
    "AAPL",
    "AMD",
    # "GOOG",
    # "AMZN",
    "UPST",
    "TTD",
    "OXY",
    # "^VIX",
    "QQQ",
    "SPY",
]
symbol_data = yf.download(symbols, start="2022-01-01", end=_today)
print(symbol_data.head())
# %%
# format the columns
colnames = symbol_data.columns
indexes = symbol_data.index

print(colnames)
# %%
print(indexes)

# %%
# Split the data by type of prices, adj close, open, etc
adj_close_df = symbol_data["Adj Close"]
close_df = symbol_data["Close"]
high_df = symbol_data["High"]
low_df = symbol_data["Low"]
open_df = symbol_data["Open"]
volume_df = symbol_data["Volume"]

price_type = [adj_close_df, close_df, high_df, low_df, open_df, volume_df]
# %%

for df in price_type:
    print(f"Adjust close: {adj_close_df.round(2)}")
    print(f"Close: {close_df.round(2)}")
    print(f"High: {high_df.round(2)}")
    print(f"Low: {low_df.round(2)}")
    print(f"Open: {open_df.round(2)}")
    print(f"Close: {close_df.round(2)}")
# df
# %%
# split based on symbol:
symbol_data.columns = symbol_data.columns.swaplevel(0, 1)
symbol_data.sort_index(axis=1, level=0, inplace=True)

print(symbol_data)
# %%

symbol_data_dfs = {}

for symbol in symbol_data:
    symbol_data_dfs[symbol] = symbol_data[symbol].copy()
    symbol_data_dfs[symbol]["Volume"] /= 1_000_000
    symbol_data_dfs[symbol] = symbol_data_dfs[symbol].round(0)


for symbol, df in symbol_data.items():
    print(f"Data for symbol: {symbol}")
    print(df)

# %%

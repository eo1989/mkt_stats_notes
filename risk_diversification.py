# %%
start = "2020-01-01"
end = "2022-12-30"

assets = [
    "AAPL", "MSFT", "AMZN", "META", "GOOG", "LULU", "JPM", "BAC", "NVDA",
    "AMD", "MRNA", "VZ", "TSLA", "DE", "DIS", "TGT", "WMT", "ZION", "C", "ZM",
    "LUV", "WFC", "BA", "APA", "CVX", "XOM", "OXY", "MO", "PLTR", "NFLX"
]
assets.sort()

# %%
import pandas as pd
import yfinance as yf
import riskfolio as rp

pd.options.display.float_format = '{:.3f}'.format
yf.pdr_override()

assets.sort()

data = yf.download(assets, start=start, end=end)
_data = data.copy(deep=True)
# _data
# %%
# _data = data.loc[:, ('Adj Close', slice(None))]
# _data

# %%
returns1 = data['Adj Close'].pct_change().dropna()
ret2 = _data['Adj Close'].pct_change().dropna()


port1 = rp.Portfolio(returns=returns1)
port2 = rp.Portfolio(returns=ret2)

port1.assets_stats(method_mu='hist', method_cov='hist', d=0.94)
port2.assets_stats(method_mu='hist', method_cov='hist', d=0.94)

w_rp = port1.rp_optimization(
    model='Classic',  # historical
    rm='MV',  # mean-variance optimization
    hist=True,  # use historical scenarios
    rf=0,  # risk-free rate
    b=None  # dont use constraints
)
# %%
display(w_rp.T)
# %%
ax = rp.plot_pie(w=w_rp)
# %%
ax = rp.plot_risk_con(w_rp, cov=port1.cov, returns=port1.returns, rm='MV', rf=0)
# %%
port2a = rp.Portfolio(returns=ret2, lowerret=0.008)
# port.lowerret = 0.008
w_rp_c = port2a.rp_optimization(
    model = 'Classic',
    rm = "MV",
    hist = True,
    rf = 0,
    b = None
)
# %%
ax = rp.plot_risk_con(w=w_rp_c, cov=port2.cov, returns=port2.returns, rm='MV', rf = 0)

# %%

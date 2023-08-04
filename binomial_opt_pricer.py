import numpy as np


def binomial_tree(S, K, T, r, sigma, N, option_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            tree[j, i] = S * (u ** (i - j)) * (d**j)
    option = np.zeros((N + 1, N + 1))
    if option_type == "call":
        option[:, N] = np.maximum(np.zeros(N + 1), tree[:, N] - K)
    elif option_type == "put":
        option[:, N] = np.maximum(np.zeros(N + 1), K - tree[:, N])
    else:
        raise ValueError("Invalid option type")
    for i in np.arange(N - 1, -1, -1):
        for j in np.arange(0, i + 1):
            if option_type == "call":
                option[j, i] = np.maximum(
                    np.zeros(N + 1),
                    np.exp(-r * dt)
                    * (p * option[j, i + 1] + (1 - p) * option[j + 1, i + 1])
                    - K,
                )
            elif option_type == "put":
                option[j, i] = np.maximum(
                    np.zeros(N + 1),
                    np.exp(-r * dt)
                    * (p * option[j, i + 1] + (1 - p) * option[j + 1, i + 1])
                    - K,
                )
    return option[0, 0]


# Example usage:
S = 100  # current stock price
K = 105  # strike price
T = 1  # time to expiration (in years)
r = 0.05  # risk-free interest rate
sigma = 0.2  # stock price volatility
N = 100  # number of time periods
call_price = binomial_tree(S, K, T, r, sigma, N, "call")
put_price = binomial_tree(S, K, T, r, sigma, N, "put")
print(
    "Binomial option prices: call = {:.2f}, put = {:.2f}".format(call_price, put_price)
)

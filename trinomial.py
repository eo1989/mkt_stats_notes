import numpy as np


def trinomial_tree(S, K, T, r, sigma, N, option_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(2 * dt))
    d = 1 / u
    m = 1
    pu = (np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2)))**2 / (
        (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))**2
        * 2 + (np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2)))**2)
    pd = ((np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-r * dt / 2)) *
          (np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2))) /
          ((np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))
           **2 * 2 +
           (np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2)))**2))
    pm = 1 - pu - pd
    tree = np.zeros((2 * N + 1, 2 * N + 1))
    for i in range(2 * N + 1):
        for j in range(max(0, i - 2 * N), min(i + 1, 2 * N + 1)):
            tree[j, i] = S * (u**(i - j)) * (d**j)
    option = np.zeros((2 * N + 1, 2 * N + 1))
    if option_type == "call":
        option[:, 2 * N] = np.maximum(np.zeros(2 * N + 1), tree[:, 2 * N] - K)
    elif option_type == "put":
        option[:, 2 * N] = np.maximum(np.zeros(2 * N + 1), K - tree[:, 2 * N])
    else:
        raise ValueError("Invalid option type")
    for i in np.arange(2 * N - 1, -1, -1):
        for j in np.arange(max(0, i - 2 * N), min(i + 1, 2 * N + 1)):
            if option_type == "call":
                option[j, i] = np.maximum(
                    np.zeros(2 * N + 1),
                    np.exp(-r * dt) *
                    (pu * option[j - 1, i + 1] + pm * option[j, i + 1] +
                     pd * option[j + 1, i + 1]) - K,
                )
            elif option_type == "put":
                option[j, i] = np.maximum(
                    np.zeros(2 * N + 1),
                    np.exp(-r * dt) *
                    (pu * option[j - 1, i + 1] + pm * option[j, i + 1] +
                     pd * option[j + 1, i + 1]) - K,
                )
    return option[N, N]


# Example usage:
S = 418.35  # current stock price
K = 430  # strike price
T = 7 / 365  # time to expiration (in years)
r = 0.05  # risk-free interest rate
sigma = 0.176  # stock price volatility
N = 100  # number of time periods
call_price = trinomial_tree(S, K, T, r, sigma, N, "call")
put_price = trinomial_tree(S, K, T, r, sigma, N, "put")
print("Trinomial option prices: call = {:.2f}, put = {:.2f}".format(
    call_price, put_price))

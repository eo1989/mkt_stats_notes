# From book: Modern Multiparadigm Software Architectures
# and Design Patterns with Examples and Applications
# in C++, C#, and Python. Volume I
# by Daniel J. Duffy (planned pub 12/2023)
# TODO: Get this book!

import abc
import time as tm
import math as mat
import cmath as cmat
import random
import numpy as np
from numpy.random import Generator, Philox, PCG64, MT19937, PCG64DXSM
from scipy.stats import norm
from abc import ABC

from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# import threading??


class Rng(ABC):
    @property
    @abc.abstractmethod
    def value(self) -> None:
        pass


class PhiloxRng(Rng):
    def __init__(self, seed=1234):
        self.rg = Generator(Philox(seed))

    def value(self):
        return self.rg.standard_normal()

    def __call__(self):
        return self.value()


class PCG64Rng(Rng):
    def __init__(self, seed=1234):
        self.rg = Generator(PCG64(seed))

    def value(self):
        return self.rg.standard_normal()

    def __call__(self):
        return self.value()


class PCG64DXSMRng(Rng):
    def __init__(self, seed=1234):
        self.rg = Generator(PCG64DXSM(seed))

    def value(self):
        return self.rg.standard_normal()

    def __call__(self):
        return self.value()


class MT199337Rng(Rng):
    def __init__(self, seed=1234):
        self.rg = Generator(MT19937(seed))

    def value(self):
        return self.rg.standard_normal()

    def __call__(self):
        return self.value()


class GaussRng(Rng):
    def value(self):
        return random.gauss(0, 1)

    def __call__(self):
        return self.value()


# Payoff
def Payoff(x, K):
    return max(K - x, 0.0)


t0 = tm.time()

# init option data
r = 0.055
d = 0.0
sig = 0.31
# T = 0.25
# T = 0.0436  # ~11 days to exp in 252 trading day calendar
T = 0.0301  # ~11 days to exp in 365 day calendar
K = 150.00
S_0 = 142.00


# exact
N = norm.cdf


def CallPrice(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r * T) * N(d2)


# print('Exact Call: {CallPrice(S_0, K, T, r, sig)}')


def PutPrice(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r * sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * N(d2) - S * N(d1)


# print('Exact Put: {PutPrice(S_0, K, T, r, sig)}')

V_old = S_0

# NT = time steps, NSIM = number of simulations
NT = 300
# NSIM = 1_000_000
NSIM = 10_000  # make it much smaller

# discrete params, look into 3doptions.py
dt = T / NT
# why not just use np?
sqrk = mat.sqrt(dt)


def computePrice(NSIM, NT, rg):
    sumPriceT = 0.0
    for i in range(1, NSIM):
        V_old = S_0
        for j in range(0, NT):
            V_new = V_old + (dt * (r - d) * V_old) + (sqrk * sig * V_old * rg())
            V_old = V_new
        sumPriceT += Payoff(V_new, K)
    return sumPriceT


# family of random number generators
rg1 = PhiloxRng()
rg2 = PCG64Rng()
rg3 = GaussRng()
rg4 = MT199337Rng()


# reference: numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html#numpy.random.PCG64DXSM
rg5 = PCG64DXSMRng()

# TODO: put all of this into a function to maintain DRY principles
print(f"Sequential MC")
price = mat.exp(-r * T) * computePrice(NSIM, NT, rg5) / NSIM
print(price)

price = mat.exp(-r * T) * computePrice(NSIM, NT, rg4) / NSIM
print(price)

price = mat.exp(-r * T) * computePrice(NSIM, NT, rg3) / NSIM
print(price)

price = mat.exp(-r * T) * computePrice(NSIM, NT, rg2) / NSIM
print(price)

price = mat.exp(-r * T) * computePrice(NSIM, NT, rg1) / NSIM
print(price)

t1 = tm.time()
print(f"time to compute: {t1 - t0}")


# Parallel version
def computePrice2(NSIM, NT, rg):
    return mat.exp(-r * T) * computePrice(NSIM, NT, rg) / NSIM


if __name__ == "__main__":
    print(f"Exact Put: {PutPrice(S_0, K, T, r, sig)}")
    print(f"Exact Call: {CallPrice(S_0, K, T, r, sig)}")
    print(f"Estimated processing time: [300, 500] seconds, pending on NS & NT")
    t0 = tm.time()
    mp.set_start_method("spawn")

    with ProcessPoolExecutor(max_workers=5) as pool:
        fut1 = pool.submit(computePrice2, NSIM, NT, rg1)
        fut2 = pool.submit(computePrice2, NSIM, NT, rg2)
        fut3 = pool.submit(computePrice2, NSIM, NT, rg3)
        fut4 = pool.submit(computePrice2, NSIM, NT, rg4)
        fut5 = pool.submit(computePrice2, NSIM, NT, rg5)

        print(f"{fut1.result()}")
        print(f"{fut2.result()}")
        print(f"{fut3.result()}")
        print(f"{fut4.result()}")
        print(f"{fut5.result()}")

        t1 = tm.time()
        print(f"time to compute in parallel: {t1 - t0}")

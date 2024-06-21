import numba as nb
import numpy as np

"""
AOT Compilation version
If all else fails, use numbas Ahead-Of-Time compilation. This is much more error prone though,
and much harder to trace errors.
"""
from numba.pycc import CC

cc = CC("app_funcs")


# @nb.jit(nopython=True)
# @nb.njit((nb.float64[:], nb.float64[:], nb.float64[:]))
# @nb.njit((nb.float64[:], nb.float64[:], nb.float64[:]), cache=True)
@cc.export("transform_data", "f8[:], f8[:], f8[:]")
def transform_data(x, y, z):
    output = np.empty(x.shape[0])

    for i in range(x.shape[0]):
        if x[i] < y[i]:
            val = (y[i] ** 2 + z[i] ** 2) ** 0.5
        else:
            val = (x[i] ** 2 + z[i] ** 2) ** 0.5

        output[i] = val

    return output


if __name__ == "__main__":
    cc.compile()
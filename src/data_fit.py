import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.plot import *

def reaclib_exp(t9, a0, a1, a2, a3, a4, a5, a6):
    """Rate format of REACLIB library."""
    params = [a0, a1, a2, a3, a4, a5, a6]
    s = params[0]
    for i in range(1, 6):
        s += params[i]*t9**((2*i-5)/3)
    s += params[6]*np.log(t9)
    return s

#def q_parameter


z = 11
n = 13

idx = 15

res, cov = op.curve_fit(reaclib_exp, templist, np.log(z_array[idx, :]))

print(res)

plot = [np.exp(reaclib_exp(t, res[0], res[1], res[2], res[3], res[4], res[5], res[6])) for t in templist]

plt.plot(templist, plot)
plt.plot(templist, z_array[idx, :])
plt.yscale("log")
plt.savefig("test3.png")

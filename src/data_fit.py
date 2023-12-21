import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def q_parameters(tempQpoint, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
                 a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20,
                 a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
                 a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42,
                 a43, a44, a45, a46, a47, a48):
    q_val, t9 = tempQpoint
    params = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
                 a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20,
                 a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
                 a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42,
                 a43, a44, a45, a46, a47, a48]
    val_list = []
    for i in range(0, 7):
        val_list.append(reaclib_exp(q_val, params[7*i], params[7*i+1], params[7*i+2], params[7*i+3], params[7*i+4], params[7*i+5], params[7*i+6]))
    return reaclib_exp(t9, val_list[0], val_list[1], val_list[2], val_list[3], val_list[4], val_list[5], val_list[6])



z = 11
n = 13

# the code for fitting the grid
QG, TG = np.meshgrid(column_q_sort, np.array(templist))
z = z_array.ravel()
points = np.vstack((QG.ravel(), TG.ravel()))
res, cov = op.curve_fit(q_parameters, points, np.log(z))

print(res)
#plot = [np.exp(q_parameters(t, res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11],
#                            res[12], res[13], res[14], res[15], res[16], res[17], res[18], res[19], res[20], res[21], res[22], res[23],
 #                           res[24], res[25], res[26], res[27], res[28], res[29], res[30], res[31], res[32], res[33], res[34],
  #                          res[35], res[36], res[37], res[38], res[39], res[40], res[41], res[42], res[43], res[44], res[45], res[46],
   #                         res[47], res[48])) for t in column_q_sort]


X = np.arange(np.min(TG), np.max(TG), 0.05)
Y = np.arange(np.min(QG), np.max(QG), 0.01)
X, Y = np.meshgrid(X, Y)
ZFit = np.exp(q_parameters((X, Y), res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11],
                            res[12], res[13], res[14], res[15], res[16], res[17], res[18], res[19], res[20], res[21], res[22], res[23],
                            res[24], res[25], res[26], res[27], res[28], res[29], res[30], res[31], res[32], res[33], res[34],
                            res[35], res[36], res[37], res[38], res[39], res[40], res[41], res[42], res[43], res[44], res[45], res[46],
                            res[47], res[48]))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(TG, QG, z_array.transpose(), cmap='plasma')
ax.set_zlim(0,np.max(z_array))
ax.set_xlabel("Temperature [GK]")
ax.set_ylabel("Q-value [MeV]")

ax.plot_surface(X, Y, ZFit)

#plt.plot(column_q_sort, plot)
#plt.plot(column_q_sort, z_array[:, idx])
#plt.yscale("log")
plt.savefig("test3.png")

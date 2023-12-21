import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from src.plot import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def reaclib_exp(t9, a0, a1, a2, a3, a4, a5, a6):
    """Rate format of REACLIB library."""
    params = [a0, a1, a2, a3, a4, a5, a6]
    s = params[0]
    for i in range(1, 6):
        s += params[i]*t9**((2*i-5)/3)
    s += params[6]*np.log(t9)
    return s



z = 11
n = 13

# the code for fitting the grid
QG, TG = np.meshgrid(column_q_sort, np.array(templist))
z = z_array.ravel()


dataset = [[], []]
for q_idx, Q_val in enumerate(column_q_sort):
    for t_idx, T_val in enumerate(templist):
        dataset[0].append((Q_val, T_val))
        #z_array[q_idx, t_idx] = (Q_val**3 - 10*(T_val - 5)**3)
        dataset[1].append((z_array[q_idx, t_idx]))

opt = Adam(learning_rate=0.01)

model = Sequential()
model.add(Dense(128, activation='relu',input_dim=2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))   
model.add(Dense(32,activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer=opt)
# dataset[0], dataset[1]
model.fit(dataset[0], dataset[1], epochs=600, batch_size=30, verbose=2)


plt.plot(model.history.history['loss'], color='blue')
plt.savefig("NNLoss")


X = np.arange(np.min(TG), np.max(TG), 0.1)
Y = np.arange(np.min(QG), np.max(QG), 0.1)

testdata = []
for q_idx, y in enumerate(Y):
    testdata.append([])
    for t_idx, x in enumerate(X):
        testdata[-1].append((y, x))

X, Y = np.meshgrid(X, Y)

print(z_array.shape, len(testdata))
ZFit = np.empty((len(testdata), len(testdata[0])))
predictlist = []
for idx, ya in enumerate(testdata):
    #for idx2, xy in enumerate(ya):
    temp = model.predict(ya, verbose=1)
    predictlist.append(temp)
      #  ZFit[idx, idx2] = temp[0]
        # should probably find a better way to do this print
    
ZFit.flat[:] = predictlist

print(model.predict([(0.05, 10.75)]))
print(model.predict([(0.15, 10.75)]))
print(model.predict([(0.05, 5.75)]))

print(model.summary())


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(TG, QG, z_array.transpose(), cmap='plasma', alpha=0.5)
#ax.set_zlim(0,np.max(z_array))
ax.set_xlabel("Temperature [GK]")
ax.set_ylabel("Q-value [MeV]")

ax.plot_surface(X, Y, ZFit, color="blue")

#plt.plot(column_q_sort, plot)
#plt.plot(column_q_sort, z_array[:, idx])
#plt.yscale("log")
plt.savefig("test3.png")

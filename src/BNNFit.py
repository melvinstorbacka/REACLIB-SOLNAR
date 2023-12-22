import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from src.plot import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp

def reaclib_exp(t9, a0, a1, a2, a3, a4, a5, a6):
    """Rate format of REACLIB library."""
    params = [a0, a1, a2, a3, a4, a5, a6]
    s = params[0]
    for i in range(1, 6):
        s += params[i]*t9**((2*i-5)/3)
    s += params[6]*np.log(t9)
    return s



z = 50
n = 123

# the code for fitting the grid
QG, TG = np.meshgrid(column_q_sort, np.array(templist))
z = z_array.ravel()


dataset = [[], []]
for q_idx, Q_val in enumerate(column_q_sort):
    for t_idx, T_val in enumerate(templist):
        for ld_idx in range(0, 6):
            dataset[0].append((Q_val, T_val))
            if z_array[ld_idx, q_idx, t_idx] != 0:
                z_array[ld_idx, q_idx, t_idx] = np.log2(z_array[ld_idx, q_idx, t_idx])
            else:
                z_array[ld_idx, q_idx, t_idx] = np.log2(1e-30) # this is supposed to approximate 0...
            dataset[1].append((z_array[ld_idx, q_idx, t_idx]))

def create_inputs():
    inputs = {}
    inputs["QTTuple"] = layers.Input(
        name = "QTTuple", shape=(2,)
    )
    return inputs

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential([tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(n), scale_diag=tf.ones(n)
        )
    )])
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential([tfp.layers.VariableLayer(
        tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfp.layers.MultivariateNormalTriL(n)])
    return posterior_model


def create_model(train_size):
    

    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(
            units=8,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        ), tfp.layers.DenseVariational(
            units=8,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",),
            layers.Dense(units=1)])

    return model



def run_fit(model, loss, QT, Z):

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    model_history = model.fit(QT, Z, epochs=150, verbose=2)
    print("Model training finished.")
    _, rmse = model.evaluate(QT, Z, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print(model.summary())

    plt.plot((model_history.history['loss']), color='blue')
    plt.plot((model_history.history['root_mean_squared_error']), color='orange')
    plt.savefig("NNLoss")

#opt = keras.optimizers.RMSprop(learning_rate=0.01)

#model = Sequential()
#model.add(Dense(32, activation='sigmoid',input_dim=2))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(32, activation='relu'))   
#model.add(Dense(32,activation='sigmoid'))
#model.add(Dense(1))


#model.compile(loss='mae', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])
# dataset[0], dataset[1]
#model.fit(dataset[0], dataset[1], epochs=400, batch_size=20, verbose=2)

mse_loss = keras.losses.MeanAbsoluteError()
bnn_model = create_model(108*21*6)
run_fit(bnn_model, mse_loss, dataset[0], dataset[1])


X = np.arange(np.min(TG), np.max(TG), 0.1)
Y = np.arange(np.min(QG), np.max(QG), 0.1)

testdata = []
for q_idx, y in enumerate(Y):
    testdata.append([])
    for t_idx, x in enumerate(X):
        testdata[-1].append((y, x))

X, Y = np.meshgrid(X, Y)

"""
print(z_array.shape, len(testdata))
ZFit = np.empty((len(testdata), len(testdata[0])))
predictlist = []
for idx, ya in enumerate(testdata):
    #for idx2, xy in enumerate(ya):
    temp = model.predict(ya, verbose=1)
    predictlist.append(temp)
      #  ZFit[idx, idx2] = temp[0]
        # should probably find a better way to do this print
"""
    

def plot_2Dprediction(model, iterations=100, Q_idx=10):
    tempsLin = np.arange(0.0001, 10, 0.03)
    plotarray = [(column_q_sort[Q_idx], t) for t in tempsLin]
    plt.clf()
    for _ in range(iterations):
        plt.plot(tempsLin, (model.predict(plotarray)), color="green")
    plt.plot(templist, (z_array[0, Q_idx, :]), color="red", linewidth=4)
    plt.plot(templist, (z_array[1, Q_idx, :]), color="red", linewidth=4)
    plt.plot(templist, (z_array[2, Q_idx, :]), color="red", linewidth=4)
    plt.plot(templist, (z_array[3, Q_idx, :]), color="red", linewidth=4)
    plt.plot(templist, (z_array[4, Q_idx, :]), color="red", linewidth=4)
    plt.plot(templist, (z_array[5, Q_idx, :]), color="red", linewidth=4)


    plt.savefig("constQ.png")


"""
ZFit.flat[:] = predictlist



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(TG, QG, 2**(z_array.transpose()), cmap='plasma', alpha=0.5)
ax.set_zlim(0,2**(np.max(z_array)))
ax.set_xlabel("Temperature [GK]")
ax.set_ylabel("Q-value [MeV]")

ax.plot_surface(X, Y, 2**(ZFit), color="blue")

#plt.plot(column_q_sort, plot)
#plt.plot(column_q_sort, z_array[:, idx])
#plt.yscale("log")
plt.savefig("test3.png")



print(column_q_sort[-2])
"""



plot_2Dprediction(bnn_model, 100, 10)



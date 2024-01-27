import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp


mae_loss = keras.losses.MeanAbsoluteError()


def reaclib_exp(t9, a0, a1, a2, a3, a4, a5, a6):
    """Rate format of REACLIB library.
    t9          : Temperature in Gigakelvin
    a0,...,a6   : Parameters of REACLIB function"""
    params = [a0, a1, a2, a3, a4, a5, a6]
    s = params[0]
    for i in range(1, 6):
        s += params[i]*t9**((2*i-5)/3)
    s += params[6]*np.log(t9)
    return s


def read_data(n, z):
    """Reads the data from ./data/{n}-{z}/, and outputs a (Q, T)-array and Rate list, as fit dataset.
    n       : neutron number
    z       : proton number"""

    dir_path = f"./data/{z}-{n}/"
    files = os.listdir(dir_path)
    files.sort()

    QT_points = [[] for i in range(1, 7)]
    rate_points = [[] for i in range(1, 7)]
    templist = [[] for i in range(1, 7)]
    qlist = [[] for i in range(1, 7)]

    non_exp_files = []

    for file_path in files:
        if "exp" not in file_path:
            non_exp_files.append(file_path)

    files = non_exp_files

    for file_path in files:
        Q, idx, ld_idx = float(file_path.split("|")[1]), int(file_path.split("|")[2]), int(file_path.split("|")[3])
        with open(dir_path + file_path, "r") as f:
            f.readline()
            f.readline()
            line = f.readline()
            line = line.split(" ")
            if len(line) == 5:
                    temperature, rate = float(line[2]), float(line[3])
            else:
                temperature, rate = float(line[1]), float(line[2])
            QT_points[ld_idx-1].append((Q, temperature))
            if temperature not in templist[ld_idx - 1]:
                templist[ld_idx-1].append(temperature)
            ind = templist[ld_idx - 1].index(temperature)
            qlist[ld_idx - 1].append(Q)
            if rate != 0:
                rate = np.log2(rate)
            else:
                rate = np.log2(1e-30)
            rate_points[ld_idx-1].append(rate)
            while True:
                line = f.readline()
                if not line or "Q" in line:
                    break
                line = line.split(" ")
                if len(line) == 5:
                    temperature, rate = float(line[2]), float(line[3])
                else:
                    temperature, rate = float(line[1]), float(line[2])
                if temperature not in templist[ld_idx - 1]:
                    templist[ld_idx-1].append(temperature)
                if rate != 0:
                    rate = np.log2(rate)
                else:
                    rate = np.log2(1e-30)
                rate_points[ld_idx-1].append(rate)
                ind = templist[ld_idx - 1].index(temperature)
                QT_points[ld_idx-1].append((Q, temperature))

    print(np.array(rate_points).shape)
    print(np.array(QT_points).shape)
    #print(len(rate_points[0]))
    #print(templist)
    #print(qlist)
        

    # these arrays need to be flattened or select one of the dimensions (different LD models)
    return np.array(QT_points), np.array(rate_points), qlist[1], templist[1]



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


def create_probabilistic_bnn_model(train_size):

    model = keras.Sequential([keras.layers.Input(shape=(2,)),
        tfp.layers.DenseVariational(
                units=4,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / train_size,
                activation="sigmoid",),
        tfp.layers.DenseVariational(
                units=4,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / train_size,
                activation="sigmoid",),
        layers.Dense(units=2),
        tfp.layers.IndependentNormal(1)])

    return model


def create_bnn_model(train_size):

    model = keras.Sequential([keras.layers.Input(shape=(2,)),
        tfp.layers.DenseVariational(
                units=4,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / train_size,
                activation="sigmoid",),
        tfp.layers.DenseVariational(
                units=4,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / train_size,
                activation="sigmoid",),
        layers.Dense(units=1)])

    return model


def create_standard_nn_model():

    model = keras.Sequential([keras.layers.Input(shape=(2,)),
        layers.Dense(
                units=32,
                activation="sigmoid"),
        layers.Dense(
                units=32,
                activation="sigmoid"),
        layers.Dense(units=1)])

    return model


def negative_loglikelihood(targets, estimated_distribution):
    """Loss function of negative log-likelihood."""
    return -estimated_distribution.log_prob(targets)


def fit_data(model, loss, QT_array, rate_array, train_size, batch_size, ld_idx=None):
    """Fits the NN model to the rate surface provided.
    model       : NN model to be used for the fit
    loss        : loss function to be used
    QT_array    : array of (Q, T) points of rates
    rate_array  : array of rates for (Q, T) points
    train_size  : number of data points
    batch_size  : batch size"""
    
    if ld_idx is not None:
        train_size = 108*21
    else:
        train_size = 108*21*6
    batch_size = 2*108
    epochs = 7500

    initial_learning_rate = 0.025
    final_learning_rate = 0.00010
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
    steps_per_epoch = int(train_size/batch_size)

    # exponential decay learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=steps_per_epoch,
                decay_rate=learning_rate_decay_factor,
                staircase=True
                )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    print(rate_array.shape)
    if ld_idx:
        Z_train = rate_array[ld_idx, :]
        QT_train = QT_array[ld_idx, :, :]
    else:
        Z_train = rate_array
        QT_train = QT_array

    model_history = model.fit(QT_train, Z_train, epochs=epochs, batch_size=batch_size, verbose=2)
    print("Model training finished.")
    _, rmse = model.evaluate(QT_train, Z_train, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print(model.summary())

    return model_history


def plot_probabilistic_bnn(model, n, z, Q=None, q_idx=None, rate_data=None, templist=None):

    if q_idx and rate_data is not None and templist is not None:
        plt.plot(templist, np.log10(2**(rate_data[0, q_idx, :])), color="red", linewidth=1)
        plt.plot(templist, np.log10(2**(rate_data[0, q_idx, :])), color="red", linewidth=1)
        plt.plot(templist, np.log10(2**(rate_data[0, q_idx, :])), color="red", linewidth=1)
        plt.plot(templist, np.log10(2**(rate_data[0, q_idx, :])), color="red", linewidth=1)
        plt.plot(templist, np.log10(2**(rate_data[0, q_idx, :])), color="red", linewidth=1)
        plt.plot(templist, np.log10(2**(rate_data[0, q_idx, :])), color="red", linewidth=1, label="TALYS Data")

    if Q:
        tempsLin = np.arange(0.0001, 10, 0.03)
        plotarray = [(Q, t) for t in tempsLin]
        #for _ in range(iterations):
        #   plt.plot(tempsLin, (model.predict(plotarray)), color="green")
        prediction_distribution = model(np.array(plotarray))
        prediction_mean = prediction_distribution.mean().numpy()
        prediction_stdv = prediction_distribution.stddev().numpy()

        plt.plot(tempsLin, np.log10(2**prediction_mean), color="royalblue", label="Mean Fit, μ")
        plt.fill_between(tempsLin, (np.log10(2**(prediction_mean + prediction_stdv))).flatten(), (np.log10(2**(prediction_mean - prediction_stdv))).flatten(), color="lightsteelblue", label="μ±σ")

    plt.title(f"Reaction Rate vs. Temperature for {n=},{z=} and Q={Q} MeV")
    plt.xlabel("Temperature [GK]")
    plt.ylabel("log10 Reaction rate")
    plt.legend()
    plt.savefig("constQ.png")
    plt.clf()


def plot_bnn(model, n, z, iterations=100, ld_idx=None, Q=None, q_idx=None, rate_data=None, templist=None):

    if ld_idx and q_idx and rate_data is not None and templist is not None:
        plt.plot(templist, np.log10(2**(rate_data[ld_idx, q_idx, :])), color="red", linewidth=1, label="TALYS Data")

    if Q:
        tempsLin = np.arange(0.0001, 10, 0.03)
        plotarray = [(Q, t) for t in tempsLin]
        for _ in range(iterations):
           plt.plot(tempsLin, (model.predict(plotarray)), color="lightsteelblue", label="BNN Predictions")

    plt.title(f"Reaction Rate vs. Temperature for {n=},{z=} and Q={Q} MeV")
    plt.xlabel("Temperature [GK]")
    plt.ylabel("log10 Reaction rate")
    plt.legend()
    plt.savefig("constQ.png")
    plt.clf()


def plot_standard_nn(model, n, z, ld_idx=None, q_idx=None, rate_data=None, templist=None, qlist=None, name="plots/test.png"):

    plt.scatter(templist, np.log10(2**(rate_data[ld_idx, q_idx*len(templist):(q_idx+1)*len(templist)])), color="red", linewidth=1, label="TALYS Data")

    if qlist:
        Q = qlist[q_idx]
        tempsLin = np.arange(0.0001, 10, 0.03)
        plotarray = [(Q, t) for t in tempsLin]
        plt.plot(tempsLin, np.log10(2**(model.predict(plotarray))), color="royalblue", label="NN Predictions")

    plt.title(f"Reaction Rate vs. Temperature for {n=},{z=} and Q={Q} MeV")
    plt.xlabel("Temperature [GK]")
    plt.ylabel("log10 Reaction rate")
    plt.legend()
    plt.savefig(name)
    plt.clf()


def plot3d_standard_nn(model, n, z, ld_idx=None, Q=None, num_q=None, q_step=None, q_list=None, rate_data=None, templist=None):
    

    q_fine_grid = np.arange(Q - np.floor(num_q/2 - 1)*q_step, Q + np.ceil(num_q/2)*q_step, 0.1)
    temp_fine_grid = np.arange(0.001, 10, 0.1)

    print(q_fine_grid)

    #for t in temp_fine_grid:
        #for q in q_fine_grid:
    
    plotarray = [(q, t) for t in temp_fine_grid for q in q_fine_grid]


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if ld_idx and rate_data is not None and q_list is not None and templist is not None:
        QG, TG = np.meshgrid(np.array(q_list), np.array(templist))
        Z_plot = np.reshape(rate_data[ld_idx, :], (len(q_list), len(templist)))#rate_data[ld_idx, :].ravel()

        ax.plot_surface(TG, QG, np.log10(2**(Z_plot.transpose())), cmap='plasma', alpha=0.8, label="TALYS Data")
        ax.set_zlim(0,np.log10(2**(np.max(rate_data))))


    plt.title(f"Reaction Rate vs. Temperature vs. Q-value for {n=},{z=}")
    ax.set_xlabel("Temperature [GK]")
    ax.set_ylabel("Q-value [MeV]")
    ax.set_zlabel("Log10(Reaction Rate)")

    calc_list = [(q, t) for q in q_fine_grid for t in temp_fine_grid]

    QGF, TGF = np.meshgrid(q_fine_grid, temp_fine_grid)

    plot_list = np.log10(2**(model.predict(calc_list)))

    plot_list = np.reshape(plot_list, TGF.shape)

    ax.plot_surface(TGF, QGF, plot_list.transpose(), color="blue", alpha=0.5, label="NN Fit")

    #plt.plot(column_q_sort, plot)
    #plt.plot(column_q_sort, z_array[:, idx])
    #plt.yscale("log")
    plt.savefig("test4.png")
    plt.clf()

def save_probabilistic_bnn(model):
    pass


def save_bnn(model):
    pass


def save_standard_nn(model):
    pass


def load_probabilistic_bnn(model):
    pass


def load_bnn(model):
    pass


def load_standard_nn(model):
    pass


def reaclib_fit(model):
    pass

def plot_loss(model_history):
    plt.plot(np.log10(model_history.history['loss']), color='blue', label="Loss")
    plt.plot(np.log10(model_history.history['root_mean_squared_error']), color='orange', label="RMSE")
    plt.legend()
    plt.title("Loss and RMSE as function of training epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Log10")
    plt.savefig("NNLoss")
    plt.clf()


def main():
    """Just have to figure out what to put here :P"""


mae_loss = keras.losses.MeanAbsoluteError()
train_size = 108*21*6
bnn_model = create_standard_nn_model()

data = read_data(123, 82)

print(len(data[2]), len(data[3]))

print(data[2])

history = fit_data(bnn_model, keras.losses.MeanAbsoluteError(), data[0], data[1], train_size, 32, ld_idx=1)

plot_loss(history)

for i in range(0, 20):
    qlist = data[2].copy()
    qlist.sort()
    j = data[2].index(qlist[i])
    plot_standard_nn(bnn_model, 123, 82, 1, j, data[1], data[3], data[2], f"plots/plot{i}.png")

plot3d_standard_nn(bnn_model, 123, 82, 1, 8.08666, 21, 0.5, data[2], data[1], data[3])
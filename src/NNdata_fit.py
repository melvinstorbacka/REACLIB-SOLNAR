import numpy as np
from src.plot import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp

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


def create_standard_nn_model(train_size):

    model = keras.Sequential([keras.layers.Input(shape=(2,)),
        layers.Dense(
                units=4,
                activation="sigmoid"),
        layers.Dense(
                units=4,
                activation="sigmoid"),
        layers.Dense(units=1)])

    return model


def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


mae_loss = keras.losses.MeanAbsoluteError()


def fit_data(model, loss, QT_array, rate_array, train_size, batch_size):
    """Fits the NN model to the rate surface provided.
    model       : NN model to be used for the fit
    loss        : loss function to be used
    QT_array    : array of (Q, T) points of rates
    rate_array  : array of rates for (Q, T) points
    train_size  : number of data points
    batch_size  : batch size"""
    
    train_size = 108*21*6
    batch_size = 32
    epochs = 1200

    initial_learning_rate = 0.1
    final_learning_rate = 0.00075
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
    steps_per_epoch = int(train_size/batch_size)

    # exponential decay learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=steps_per_epoch,
                decay_rate=learning_rate_decay_factor,
                staircase=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    return None


def plot_probabilistic_bnn(model, Q):
    pass


def plot_bnn(model, Q):
    pass


def plot_standard_nn(model, Q):
    pass


def save_probabilistic_bnn(model):
    pass


def save_bnn(model):
    pass


def save_standard_nn(model):
    pass


def reaclib_fit(model):
    pass
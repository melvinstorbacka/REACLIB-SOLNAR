import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as op

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import multiprocessing
from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
from data_read import read


mae_loss = keras.losses.MeanAbsoluteError()


def read_data(n, z):
    """Reads the data from ./data/{n}-{z}/, and outputs a (Q, T)-array and Rate list, as fit dataset.
    n       : neutron number
    z       : proton number
    
    Output:
    QT_points   : List of points in format (Q, Temperature)
    rate_points : List of rate points corresponding to the (Q, T) values in QT_points.
                  Note that the format is (Q_1, T_1), (Q_1, T_2), ..., (Q_1, T_108), (Q_2, T_1)...
    qlist       : List of the Q-values
    templist    : List of the temperatures"""

    dir_path = f"./data/{z}-{n}/"
    files = os.listdir(dir_path)
    files.sort()

    non_exp_files = []
    
    # TODO: add checking if central is double - remove one of them

    for file_path in files:
        if "exp" not in file_path:
            non_exp_files.append(file_path)

    files = non_exp_files

    QT_points, rate_points, qlist, templist, errorlist = read(files, dir_path)

    return QT_points, rate_points, qlist, templist, errorlist
    



def prior(kernel_size, bias_size, dtype=None):
    """Provides a prior distribution for the Bayesian Neural Network"""
    n = kernel_size + bias_size
    prior_model = keras.Sequential([tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(n), scale_diag=tf.ones(n)
        )
    )])
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    """Provides a posterior distribution for the Bayesian Neural Network"""
    n = kernel_size + bias_size
    posterior_model = keras.Sequential([tfp.layers.VariableLayer(
        tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfp.layers.MultivariateNormalTriL(n)])
    return posterior_model


def create_probabilistic_bnn_model(train_size):
    """"Creates a probabilistic Bayesian Neural Network"""
    model = keras.Sequential([keras.layers.Input(shape=(2,)),
        tfp.layers.DenseVariational(
                units=32,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / train_size,
                activation="sigmoid",),
        tfp.layers.DenseVariational(
                units=32,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / train_size,
                activation="sigmoid",),
        layers.Dense(units=2),
        tfp.layers.IndependentNormal(1)])

    return model


def create_bnn_model(train_size):
    """Creates a standard Bayesian Neural Network (i.e. random outputs within uncertainty)"""
    model = keras.Sequential([keras.layers.Input(shape=(2,)),
        tfp.layers.DenseVariational(
                units=8,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / train_size,
                activation="sigmoid",),
        tfp.layers.DenseVariational(
                units=8,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / train_size,
                activation="sigmoid",),
        layers.Dense(units=2),
        tfp.layers.IndependentNormal(1)])

    return model


def create_standard_nn_model():
    """Creates a standard Neural Network for fitting the rates with high accuracy."""
    model = keras.Sequential([layers.Input(shape=(2,)),
        layers.Dense(
                units=32, # 32
                activation="sigmoid"),
        layers.Dense(
                units=32, # 32
                activation="sigmoid"),
        layers.Dense(units=1)])

    return model


def geometric_mean(y_tensor): # test
    logx = tf.math.log(y_tensor)
    return tf.exp(tf.reduce_mean(logx))



@keras.saving.register_keras_serializable()
def mae_loss_no_zero_rates(y_true, y_pred):
    """Custom mean absolute error loss function used for fitting the rates. Standard MAE except for
    rates = 0, in which case we replace these with the log(minimal rate value for the nucleus) - 5, and 
    give them a much smaller weight in the loss calculations. This is done to focus on the behaviour at non-zero rates
    (since zero rates -> -infinity in log format), while still maintaining the negative trend of the rates."""
    
    sub_const = tf.constant(5.0)

    weighted_loss = tf.where(tf.math.is_inf(y_true), 1e-100*((tf.abs(y_pred - tf.reduce_min(y_true - sub_const)))), tf.abs(y_pred - y_true))

    #loss = (tf.abs(y_pred - y_true))

    #weighted_loss = tf.where(tf.math.equal(y_true, match_value), 1e-45*loss, loss)

    final_loss = tf.reduce_mean(weighted_loss)

    return final_loss


@keras.saving.register_keras_serializable()
def test_loss(y_true, y_pred):
    """Custom mean absolute error loss function used for fitting the rates. Standard MAE except for
    rates = 0, in which case we replace these with the log(minimal rate value for the nucleus) - 5, and 
    give them a much smaller weight in the loss calculations. This is done to focus on the behaviour at non-zero rates
    (since zero rates -> -infinity in log format), while still maintaining the negative trend of the rates."""
    
    sub_const = tf.constant(5.0)

    minimal_ytrue = tf.reduce_min(y_true)

    weighted_loss = tf.where(tf.math.is_inf(y_true), 1e-100*((tf.abs(y_pred - (minimal_ytrue - sub_const)))) - minimal_ytrue, tf.abs(y_pred - y_true) - minimal_ytrue)

    #loss = (tf.abs(y_pred - y_true))

    #weighted_loss = tf.where(tf.math.equal(y_true, match_value), 1e-45*loss, loss)

    final_loss = tf.reduce_mean(weighted_loss)

    return final_loss




@keras.saving.register_keras_serializable()
def mape_loss_no_zero(y_true, y_pred):
    """Mean average percentage error metric for evualuating the fit, disregarding rates = 0."""

    #mae = tf.abs(tf.math.subtract(2**y_pred, 2**y_true))

    sub_const = tf.constant(5.0)

    minimal_ytrue = tf.math.reduce_min(y_true)

    weighted_log_loss = tf.where(tf.math.is_inf(y_true), tf.math.log(1e-32*tf.abs(tf.divide(tf.subtract(tf.pow(2.0, y_pred), tf.pow(2.0, minimal_ytrue - sub_const)), tf.pow(2.0, tf.reduce_min(y_true - sub_const))))), tf.math.log(tf.abs(tf.divide(tf.subtract(tf.pow(2.0, y_pred), tf.pow(2.0, y_true)), 1)))) #tf.pow(2.0, y_true)))))

    #weighted_loss = tf.where(tf.math.greater(weighted_log_loss, 10), tf.constant(1e10), tf.math.exp(weighted_log_loss))

    final_loss = tf.reduce_mean(weighted_log_loss)

    return final_loss

@keras.saving.register_keras_serializable()
def mape_no_zero_rates(y_true, y_pred):
    """Mean average percentage error metric for evualuating the fit, disregarding rates = 0."""

    #mae = tf.abs(tf.math.subtract(2**y_pred, 2**y_true))

    match_value = tf.constant(np.inf)

    weighted_loss = tf.where(tf.math.is_inf(y_true), 0.0, tf.abs(tf.divide(tf.subtract(tf.pow(2.0, y_pred), tf.pow(2.0, y_true)), tf.pow(2.0, y_true))))

    final_mape = tf.divide(tf.reduce_sum(weighted_loss), tf.cast(tf.math.count_nonzero(weighted_loss), tf.float32))*100

    return final_mape


@keras.saving.register_keras_serializable()
def rmse_no_zero_rates(y_true, y_pred): 
    """Root mean square error metric for evualuating the fit, disregarding rates = 0."""
    
    rms = tf.math.squared_difference(y_pred, y_true)

    match_value = tf.constant(np.inf)

    weighted_loss = tf.where(tf.math.is_inf(y_true), 0.0, rms)

    final_rms = tf.sqrt(tf.divide(tf.reduce_sum(weighted_loss), tf.cast(tf.math.count_nonzero(weighted_loss), tf.float32)))

    return final_rms 


def negative_loglikelihood(targets, estimated_distribution):
    """Loss function of negative log-likelihood."""
    return -estimated_distribution.log_prob(targets)


def fit_data(model, loss, QT_array, rate_array, train_size, batch_size, ld_idxp1=None):
    """Fits the NN model to the rate surface provided.
    model       : NN model to be used for the fit
    loss        : loss function to be used
    QT_array    : array of (Q, T) points of rates
    rate_array  : array of rates for (Q, T) points
    train_size  : number of data points with 6 LD models
    batch_size  : batch size
    ld_idxp1    : level density model number +1 (corresponding to correct numbers in TALYS manual)"""
    

    if ld_idxp1 is not None:
        ld_idx = ld_idxp1 - 1
        train_size = train_size/6
    batch_size = 2*108  # (times two for standard NN at least)
    epochs = 7500       # 7500 for standard NN fit

    initial_learning_rate = 0.025   # 0.025 good learning rate for standard NN
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
        metrics=[rmse_no_zero_rates, mape_no_zero_rates],
    )

    print("Start training the model...")
    if ld_idxp1:
        if len(rate_array[ld_idx]) != 108*21:
            print("Incorrect dimensions, returning.", flush=True)
            return
        Z_train = rate_array[ld_idx, :]
        QT_train = QT_array[ld_idx, :, :]
    else:
        Z_train = rate_array.flatten()
        QT_train = QT_array.reshape(-1, QT_array.shape[-1])


    model_history = model.fit(QT_train, Z_train, epochs=epochs, batch_size=batch_size, verbose=1)
    print("Model training finished.")
    _, rmse, mae = model.evaluate(QT_train, Z_train, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    return model_history, rmse, mae


def plot_probabilistic_bnn(model, n, z, q_idxplusone=None, rate_data=None, templist=None, qlist=None, name="plots/test.png"):
    
    q_idx = q_idxplusone - 1
    Q = qlist[q_idx]

    if q_idxplusone and rate_data is not None and templist is not None:
        plt.plot(templist, np.log10(2**(rate_data[0, q_idx*len(templist):(q_idx+1)*len(templist)])), color="red", linewidth=1)
        plt.plot(templist, np.log10(2**(rate_data[1, q_idx*len(templist):(q_idx+1)*len(templist)])), color="red", linewidth=1)
        plt.plot(templist, np.log10(2**(rate_data[2, q_idx*len(templist):(q_idx+1)*len(templist)])), color="red", linewidth=1)
        plt.plot(templist, np.log10(2**(rate_data[3, q_idx*len(templist):(q_idx+1)*len(templist)])), color="red", linewidth=1)
        plt.plot(templist, np.log10(2**(rate_data[4, q_idx*len(templist):(q_idx+1)*len(templist)])), color="red", linewidth=1)
        plt.plot(templist, np.log10(2**(rate_data[5, q_idx*len(templist):(q_idx+1)*len(templist)])), color="red", linewidth=1, label="TALYS Data")

    if Q:
        tempsLin = np.arange(0.0001, 10, 0.03)
        plotarray = [(Q, t) for t in tempsLin]

        prediction_distribution = model(np.array(plotarray))
        prediction_mean = prediction_distribution.mean().numpy()
        prediction_stdv = prediction_distribution.stddev().numpy()

        plt.plot(tempsLin, np.log10(2**prediction_mean), color="royalblue", label="Mean Fit, μ")
        plt.fill_between(tempsLin, (np.log10(2**(prediction_mean + prediction_stdv))).flatten(), (np.log10(2**(prediction_mean - prediction_stdv))).flatten(), color="lightsteelblue", label="μ±σ")

    plt.title(f"Reaction Rate vs. Temperature for {n=},{z=} and Q={Q} MeV")
    plt.xlabel("Temperature [GK]")
    plt.ylabel("log10 Reaction rate")
    plt.legend()
    plt.savefig(name)
    plt.clf()


def plot_bnn(model, n, z, iterations=100, q_idxplusone=None, rate_data=None, templist=None, qlist=None, name="plots/test.png"):

    q_idx = q_idxplusone - 1

    if  q_idxplusone and rate_data is not None and templist is not None:
        plt.scatter(templist, np.log10(2**(rate_data[0, q_idx*len(templist):(q_idx+1)*len(templist)])), color="red", linewidth=1, label="TALYS Data")
        for ld_idx in range(1, 6): 
            plt.scatter(templist, np.log10(2**(rate_data[ld_idx, q_idx*len(templist):(q_idx+1)*len(templist)])), color="red", linewidth=1)

    if qlist:
        Q = qlist[q_idx]
        tempsLin = np.arange(0.0001, 10, 0.03)
        plotarray = [(Q, t) for t in tempsLin]
        plt.plot(tempsLin, np.log10(2**(model.predict(plotarray))), color="lightsteelblue", label="BNN Predictions")
        for _ in range(iterations - 1):
           plt.plot(tempsLin, np.log10(2**(model.predict(plotarray))), color="lightsteelblue")

    plt.title(f"Reaction Rate vs. Temperature for {n=},{z=} and Q={Q} MeV")
    plt.xlabel("Temperature [GK]")
    plt.ylabel("log10 Reaction rate")
    plt.legend()
    plt.savefig(name)
    plt.clf()


def plot_standard_nn(model, n, z, ld_idx=None, q_idxplusone=None, rate_data=None, templist=None, qlist=None, name="plots/test.png"):

    q_idx = q_idxplusone - 1

    fig, axs = plt.subplots(2)

    axs[0].scatter(templist, (2**(rate_data[ld_idx, q_idx*len(templist):(q_idx+1)*len(templist)])), color="red", linewidth=1, label="TALYS Data")

    if qlist:
        Q = qlist[q_idx]
        tempsLin = np.arange(0.00001, 10, 0.03)
        plotarray = [(Q, t) for t in tempsLin]
        predList = model.predict(plotarray)
        axs[0].plot(tempsLin, (2**(predList)), color="royalblue", label="NN Predictions")

    axs[0].set_title(f"Reaction Rate vs. Temperature for {n=},{z=} and Q={Q} MeV")
    axs[0].set_ylabel("log10 Reaction rate")
    axs[0].legend()
    axs[0].set_yscale("log")


    pred_discrete_temps = 2**model.predict([(Q, T) for T in templist])
    axs[1].sharex(axs[0])
    axs[1].plot(templist, [NN/talys - 1 for talys, NN in zip(2**rate_data[ld_idx, q_idx*len(templist):(q_idx+1)*len(templist)], pred_discrete_temps)],
                 color="lightsteelblue", label=f"Neural Network Predictions vs. Talys %-deviations", linestyle='--', marker='o')
    axs[1].plot([0, 10], [0, 0], color="gray", linestyle='dashed')
    axs[1].set_xlabel("Temperature [GK]")
    axs[1].set_ylabel("Percentual Deviation vs. Talys")
    axs[1].set_ylim(-0.5, 0.5)
    plt.savefig(name)
    plt.clf()


def plot3d_standard_nn(model, n, z, ld_idx=None, Q=None, num_q=None, q_step=None, q_list=None, rate_data=None, templist=None, name="3DPlot.png"):
    

    q_fine_grid = np.arange(Q - np.floor(num_q/2 - 1)*q_step, Q + np.ceil(num_q/2)*q_step, 0.1)
    temp_fine_grid = np.arange(0.001, 10, 0.1)
    
    plotarray = [(q, t) for t in temp_fine_grid for q in q_fine_grid]


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    #if ld_idx and rate_data is not None and q_list is not None and templist is not None:
    QG, TG = np.meshgrid(np.array(q_list), np.array(templist))
    Z_plot = np.reshape(rate_data[ld_idx, :], (len(q_list), len(templist)))

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

    ax.plot_surface(TGF, QGF, plot_list, color="blue", alpha=0.5, label="NN Fit")

    plt.savefig(name)
    plt.clf()


def save_probabilistic_bnn(model):
    """Used to save parameters for a probabilistic network. Will be implemented later, 
    when appropiate use case has been found. // 2024-02-07"""
    pass


def save_bnn(model):
    """Used to save parameters for a BNN. Will be implemented later, 
    when appropiate use case has been found. // 2024-02-07"""
    pass


def save_standard_nn(model, n, z, ld_idx, experimental=False):
    """Used to save parameters of standard NN fit. Saves the model in as
    /NNParameters/{z}-{n}/ld_idx.h5
    model   :: fit standard NN model
    n       :: neutron number
    z       :: proton number
    ld_idx  :: level density model number in TALYS"""

    # TODO: add separate structure for experimental nuclei

    if not os.path.exists("NNParameters/"):
        os.mkdir("NNParameters/")

    if not os.path.exists(f"NNParameters/{z}-{n}"):
        os.mkdir(f"NNParameters/{z}-{n}")

    model.save(f"NNParameters/{z}-{n}/{ld_idx}.keras")

    return None


def load_probabilistic_bnn(model):
    """Used to load parameters for a probabilistic network. Will be implemented later, 
    when appropiate use case has been found. // 2024-02-07"""
    pass


def load_bnn(model):
    """Used to save parameters for a BNN. Will be implemented later, 
    when appropiate use case has been found. // 2024-02-07"""
    pass


def load_standard_nn(n, z, ld_idx):
    """Used to load parameters of standard NN fit. Loads the model in
    /NNParameters/{z}-{n}/ld_idx/{Q}.json. Returns the NN model loaded in Keras.
    n       :: neutron number
    z       :: proton number
    ld_idx  :: level density model number in TALYS
    Q       :: Q-value in MeV"""
    
    # add different treatment for experimental fit

    model = keras.models.load_model(f'NNParameters/{z}-{n}/{ld_idx}.keras', custom_objects={ 'loss': mae_loss_no_zero_rates, 'accuracy' : rmse_no_zero_rates })

    return model


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


def reaclib_fit(model, Q):
    """Fit the mean NN model values to the Reaclib format, for a specific slice of Q-value."""
    fitY = []
    fitQT = []
    fitX = []
    for T in np.arange(0.0001, 10, 0.001):
        fitQT.append((Q, T))
        fitX.append(T)
    fitY = [np.log(2**rate[0]) for rate in model.predict(fitQT)]

    res, cov = op.curve_fit(reaclib_exp, fitX, fitY)

    return res


def reaclib_output_rate(model, file_path, Q):
    """Append reaclib fit constants to file in file_path."""
    pass


def reaclib_total_output(model, file_path, list_of_nuclei, list_of_masses):
    """Outputs the total reaclib fit file for all nuclei in the list
    TODO: warning if nuclei are missing"""

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

    nz_list = []

    for dirname in os.listdir("data/"):
        temp = dirname.split("-")
        nz_list.append([int(temp[1]), int(temp[0])])

    arguments = nz_list
            
    # parallel computation
    num_cores = multiprocessing.cpu_count()
    print(f"Running with {num_cores} cores.")
    pool = multiprocessing.Pool(num_cores)

    #for argument in arguments:
     #   print(argument)
      #  fit_and_save(argument)

    pool.map_async(fit_and_save, arguments)

    pool.close()
    pool.join()
              


def fit_and_save(args):
    nz = args
    train_size = 108*21*6
    n, z = nz
    data = read_data(n, z)
    print(data[-1])
    for ld_idx in range(6):
        ld_idxp1 = 1 + ld_idx
        if ld_idxp1 not in data[-1]:
            print(ld_idxp1)
            if os.path.exists(f"NNParameters/{z}-{n}/{ld_idx}.keras"):
                print("Skipping, already exists.", flush=True)
                continue
            nn_model = create_standard_nn_model()
            history, rmse, mae = fit_data(nn_model, test_loss, data[0], data[1], train_size, 2*108, ld_idxp1) # testing custom loss function
            save_standard_nn(nn_model, n, z, ld_idx)
            with open("NNParameters/model_data.utf8", "a+") as f:
                if not os.path.getsize("NNParameters/model_data.utf8"):
                    f.write("ld, n   , z   , rmse            , mae           \n")
                f.write(f"{ld_idxp1:>2}, {n:<4}, {z:<4}, {rmse:<15}, {mae:<15}%  \n")
    return


#main()

#main()




    #plot_loss(history)



#fit_and_save([84, 43])

n, z = 84, 43
data = read_data(n, z)

ld_idx = 0 # check indices...

bnn_model = load_standard_nn(n, z, ld_idx)

for i in range(0, 20):
    qlist = data[2].copy()
    qlist.sort()
    j = data[2].index(qlist[i])
    plot_standard_nn(bnn_model, n, z, ld_idx=ld_idx, q_idxplusone=j+1, rate_data=data[1], templist=data[3], qlist=data[2], name = f"plots/plot{i}.png")

Q = 1.0

plotX = []
plotT = []
for T in np.arange(0.001, 10, 0.0001):
   plotX.append((Q, T))
   plotT.append(T)

a0, a1, a2, a3, a4, a5, a6 = (reaclib_fit(bnn_model, Q))

plotReac = [reaclib_exp(t, a0, a1, a2, a3, a4, a5, a6) for t in plotT]

predList = bnn_model.predict(plotX)

print("RMS:" + str(np.sqrt(np.mean(np.array(plotReac) - np.array([np.log(2**rate[0]) for rate in predList]))**2)))


plotRel = [np.exp(plotreac)/2**(pred[0]) for plotreac, pred in zip(plotReac, predList)]

plt.plot(plotT, [2**rate[0] for rate in predList], color="red", label="NNFit")
plt.plot(plotT, np.exp(plotReac), color="blue", label="REACLIB Polynomial Fit")
plt.plot(plotT, plotRel, label="REACLIB/NNFit")
plt.legend()
plt.yscale("log")
plt.savefig("test.png")


    #
    #    model, n, z, iterations=100, ld_idx=None, q_idx=None, rate_data=None, templist=None, qlist=None, name="plots/test.png"):
    #

    #plot3d_standard_nn(bnn_model, 79, 56, 1, data[2][10], 10, 0.5, data[2], data[1], data[3])







# TODO:
#
# * Test different loss function, with some sort of weighting to different rates somehow.
#       - Now trying MAPE, will probably have to revert back though.
#
#
#
#

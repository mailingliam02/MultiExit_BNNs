import numpy as np

# https://github.com/Basher4/IVIMNET/blob/master/IVIMNET/simulations.py
# which itself is originally from oliverchampion i think
def sim_signal(SNR, bvalues, sims=100000, Dmin=0.5 / 1000, Dmax=2.0 / 1000, fmin=0.1, fmax=0.5, Dsmin=0.05, Dsmax=0.2,
               rician=False, state=123):
    """
    This simulates IVIM curves. Data is simulated by randomly selecting a value of D, f and D* from within the
    predefined range.
    input:
    :param SNR: SNR of the simulated data. If SNR is set to 0, no noise is added
    :param bvalues: 1D Array of b-values used
    :param sims: number of simulations to be performed (need a large amount for training)
    optional:
    :param Dmin: minimal simulated D. Default = 0.0005
    :param Dmax: maximal simulated D. Default = 0.002
    :param fmin: minimal simulated f. Default = 0.1
    :param Dmax: minimal simulated f. Default = 0.5
    :param Dpmin: minimal simulated D*. Default = 0.05
    :param Dpmax: minimal simulated D*. Default = 0.2
    :param rician: boolean giving whether Rician noise is used; default = False
    :return data_sim: 2D array with noisy IVIM signal (x-axis is sims long, y-axis is len(b-values) long)
    :return D: 1D array with the used D for simulations, sims long
    :return f: 1D array with the used f for simulations, sims long
    :return Dp: 1D array with the used D* for simulations, sims long
    """

    # randomly select parameters from predefined range
    rg = np.random.RandomState(state)
    test = rg.uniform(0, 1, (sims, 1))
    D = Dmin + (test * (Dmax - Dmin))
    test = rg.uniform(0, 1, (sims, 1))
    f = fmin + (test * (fmax - fmin))
    test = rg.uniform(0, 1, (sims, 1))
    Dp = Dsmin + (test * (Dsmax - Dsmin))

    # initialise data array
    data_sim = np.zeros([len(D), len(bvalues)])
    bvalues = np.array(bvalues)

    if type(SNR) == tuple:
        test = rg.uniform(0, 1, (sims, 1))
        SNR = np.exp(np.log(SNR[1]) + (test * (np.log(SNR[0]) - np.log(SNR[1]))))
        addnoise = True
    elif SNR == 0:
        addnoise = False
    else:
        SNR = SNR * np.ones_like(Dp)
        addnoise = True

    # loop over array to fill with simulated IVIM data
    for aa in range(len(D)):
        data_sim[aa, :] = ivim(bvalues, D[aa][0], f[aa][0], Dp[aa][0], 1)

    # if SNR is set to zero, don't add noise
    if addnoise:
        # initialise noise arrays
        noise_imag = np.zeros([sims, len(bvalues)])
        noise_real = np.zeros([sims, len(bvalues)])
        # fill arrays
        for i in range(0, sims - 1):
            noise_real[i,] = rg.normal(0, 1 / SNR[i],
                                       (1, len(bvalues)))  # wrong! need a SD per input. Might need to loop to maD noise
            noise_imag[i,] = rg.normal(0, 1 / SNR[i], (1, len(bvalues)))
        if rician:
            # add Rician noise as the square root of squared gaussian distributed real signal + noise and imaginary noise
            data_sim = np.sqrt(np.power(data_sim + noise_real, 2) + np.power(noise_imag, 2))
        else:
            # or add Gaussian noise
            data_sim = data_sim + noise_imag
    else:
        data_sim = data_sim

    # normalise signal
    S0_noisy = np.mean(data_sim[:, bvalues == 0], axis=1)
    data_sim = data_sim / S0_noisy[:, None]
    return data_sim, D, f, Dp

def ivim(bvalues, Dt, Fp, Dp, S0):
    # regular IVIM function
    return (S0 * (Fp * np.exp(-bvalues * Dp) + (1 - Fp) * np.exp(-bvalues * Dt)))
import numpy as np
import pandas as pd

import scipy.stats as st
import scipy.optimize
from scipy.special import erf

import warnings
import tqdm
import numba

import bebi103

# Seed random number generator
rg = np.random.default_rng(3252)






# Gamma Stuff


def log_like_iid_gamma(params, n):
    """Log likelihood for i.i.d. gamma measurements."""
    alpha, beta = params

    if alpha <= 0 or beta <= 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(n, alpha, loc=0, scale=1 / beta))

def mle_iid_gamma(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    gamma measurements, parametrized by alpha, b=1/beta"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_gamma(params, n),
            x0=np.array([2.4, 0.005]),
            args=(n,),
            method="Powell",
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError("Convergence failed with message", res.message)
        
        

def cdf_successive(t, beta1, beta2):
    """Compute the CDF for the Successive Poisson Distribution based on data (t), 
    beta1, and beta2 values."""
    if beta1 != beta2:
        term1 = (beta1 * beta2) / (beta2 - beta1)
        term2 = (1 - np.exp(-beta1 * t)) / beta1
        term3 = (1 - np.exp(-beta2 * t)) / beta2
        f = term1 * (term2 - term3)
    elif beta1 == beta2:
        f = np.exp(-beta2 * t) * (-beta2 * t + np.exp(beta2 * t) - 1)
    return f


# Plotting
def get_hv_plot(b1, b2):
    """Plot data and theoretical curve based on sliders for parameters"""
    p=iqplot.ecdf(
        df_12,
        q="time to catastrophe (s)",
        marker_kwargs={
            "fill_color": "darkblue",
            "line_color": "darkblue",
            "fill_alpha": 0.5,
            "line_alpha": 0.4,
            "legend_label": "ECDF",
        },
        show_legend=True,
        title="Comparing Fit of Gamma Distribution CDFs to Data"
        )

    x = np.linspace(0, 2000, 200)
    gamma_mle = st.gamma.cdf(x, alpha_mle, loc=0, scale=1 / beta_mle)
    model2_mle = cdf_successive(x, b1, b2)
    p.line(x, gamma_mle, line_color="orange", line_width=1.66, legend_label="gamma")
    p.line(x, model2_mle, line_color="green", line_width=1.66, legend_label="model 2")
    
    return p




def gen_gamma(alpha, beta, size):
    """Generates random values from a gamma distribution with 
    specified parameters. Returns an array of these values."""
    return rg.gamma(alpha, 1 / beta, size=size)


# Bootstrapping
def draw_parametric_bs_reps_mle(
    mle_fun, gen_fun, data, args=(), size=1, progress_bar=False
):
    """Draw parametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(*params, size)`.
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    params = mle_fun(data, *args)

    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array(
        [mle_fun(gen_fun(*params, size=len(data), *args)) for _ in iterator]
    )


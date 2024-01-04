import numpy as np
import pandas as pd

import scipy.stats as st
import scipy.optimize
from scipy.special import erf

import warnings
import tqdm
import numba

import bebi103

import hw9_3_functions
from hw9_3_figures import *

# Seed random number generator
rg = np.random.default_rng(3252)




list_conc = df["concentration (uM)"].unique()





def make_mle_dict():
    '''Produces a dictionary with keys of (growth event, bacterium)
    and values of a list of [alpha_mle, beta_mle] from the dataset.'''
    mle_dict={}
    for i in list_conc:
        df_n = df.loc[df["concentration (uM)"] == i]
        n_conc = np.array(df_n["time to catastrophe (s)"])
        alpha_mle, beta_mle = hw9_3_functions.mle_iid_gamma(n_conc)
        mle_dict[i]=[alpha_mle, beta_mle]
    return mle_dict



def make_conf_int_d():
    '''Produces a dictionary with keys of (growth event, bacterium)
    and values of a list of confidence intervals in the form
    [alpha_conf_int, beta_conf_int] from the dataset.'''
    conf_int_d = {}
    for i in list_conc:
        print("Progress for " + i + " Bootstrapping")
        df_n = df.loc[df["concentration (uM)"] == i]
        n_conc = np.array(df_n["time to catastrophe (s)"])
        bs_reps_parametric = hw9_3_functions.draw_parametric_bs_reps_mle(
            hw9_3_functions.mle_iid_gamma, hw9_3_functions.gen_gamma, n_conc, args=(), 
            size=1000, progress_bar=True)
        alpha_conf_int = np.percentile(bs_reps_parametric[:, 0], [2.5, 97.5], axis=0)
        beta_conf_int = np.percentile(bs_reps_parametric[:, 1], [2.5, 97.5], axis=0)
        conf_int_d[i] = [alpha_conf_int, beta_conf_int]
    return conf_int_d



def summaries_a_b(mle_dict, conf_int_d):
    '''
    mle_dict:
    ------------
    Dictionary of parameter values. Keys are a tuple of (growth event, bacterium)
    and values are a list of parameters.
    
    conf_int_d:
    -----------
    Dictionary of parameter confidence intervals. Keys are a tuple of 
    (growth event, bacterium) and values are a list of conf ints for each parameter.
    '''
    summaries_a = []
    summaries_b = []

    for i in mle_dict:
        summaries_a.append({
            "estimate": mle_dict[i][0],
            "conf_int": [conf_int_d[i][0][0], conf_int_d[i][0][1]],
            "label": i
        })
        summaries_b.append({
            "estimate": mle_dict[i][1],
            "conf_int": [conf_int_d[i][1][0], conf_int_d[i][1][1]],
            "label": i
        })
    return(summaries_a, summaries_b)
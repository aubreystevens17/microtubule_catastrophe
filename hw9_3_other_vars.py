import numpy as np
import scipy.stats as st
import pandas as pd
import bokeh.io
import hw9_3_functions


# Seed random number generator
rg = np.random.default_rng(3252)


#Initializing Dataframe
df = pd.read_csv("~/10-bebi103a-2020/data/microtubule_tidied.csv")



# Stuff for plotting
colors = bokeh.palettes.Turbo8



df_12 = df.loc[df["concentration (uM)"] == "12 uM"]
n = np.array(df_12["time to catastrophe (s)"])
alpha_mle_12, beta_mle_12 = hw9_3_functions.mle_iid_gamma(n)



# Gamma distribution
params_mle_g = alpha_mle_12, beta_mle_12
log_l_gamma = hw9_3_functions.log_like_iid_gamma(params_mle_g, n)
akaike_gamma = -2 * log_l_gamma + 4

# Definining our computed beta vals
beta1 = 2/np.mean(n)
beta2 = 2/np.mean(n)

# Successive distribution
params_mle_successive = 2, beta1
log_l_successive = hw9_3_functions.log_like_iid_gamma(params_mle_successive, n)
akaike_successive = -2 * log_l_successive + 2






aic_max = max(akaike_gamma, akaike_successive)
term_base = np.exp(-(akaike_gamma - aic_max)/2) + np.exp(-(akaike_successive - aic_max)/2)
# Gamma
term_g = np.exp(-(akaike_gamma - aic_max)/2)
# Successive
term_s = np.exp(-(akaike_successive - aic_max)/2)

# Weight of gamma model
w_gam = term_g / term_base

# Weight of successive model
w_suc = term_s / term_base





gamma_samples = np.array(
    [rg.gamma(alpha_mle_12, 1 / beta_mle_12, size=len(n)) for _ in range(10000)]
)



t1 = np.array([rg.exponential(1/beta1, size=len(n)) for _ in range(10000)])
t2 = np.array([rg.exponential(1/beta2, size=len(n)) for _ in range(10000)])
successive_samples = t1 + t2
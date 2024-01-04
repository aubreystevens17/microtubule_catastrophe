import numpy as np
import scipy.stats as st
import pandas as pd

import panel as pn
import iqplot
import bokeh.io
import holoviews as hv
import bebi103

import hw9_3_functions
from hw9_3_other_vars import *
hv.extension("bokeh")



# ECDF Plot
colors = bokeh.palettes.Turbo8
ecdf_plot = iqplot.ecdf(
    data=df,
    q="time to catastrophe (s)",
    cats=["concentration (uM)"],
    palette=colors,
    width=700,
    title="ECDF for Each Tubulin Concentration",
    marker_kwargs={"fill_alpha": 0.3},
)
ecdf_plot.legend.location = "bottom_right"


# Stripbox plot
strip = hv.Scatter(
    data=df, kdims=["concentration (uM)"], vdims=["time to catastrophe (s)"],
).opts(
    color="concentration (uM)",
    jitter=0.5,
    alpha=0.5,
    title="Concentration vs. Time to Catastrophe (s)",
    cmap=colors,
)
box = hv.BoxWhisker(
    data=df, kdims=["concentration (uM)"], vdims=["time to catastrophe (s)"],
).opts(
    box_fill_color="concentration (uM)", box_alpha=0.3, outlier_alpha=0, cmap=colors,
)
stripbox = box * strip




# ECDF of 12 uM
p = iqplot.ecdf(
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
    title="Comparing Fit of Each Distribution CDF to Data",
)
x = np.linspace(0, 2000, 200)
gamma_mle = st.gamma.cdf(x, alpha_mle_12, loc=0, scale=1 / beta_mle_12)
successive_mle = hw9_3_functions.cdf_successive(x, beta1, beta2)
p.line(x, gamma_mle, line_color="orange", line_width=1.66, legend_label="Gamma")
p.line(
    x,
    successive_mle,
    line_color="green",
    line_width=1.66,
    legend_label="Successive Poisson",
)
p.legend.location = "bottom_right"
p.legend.visible = True




# Gamma distributed predictive ECDFs
# Predictive
p1 = bebi103.viz.predictive_ecdf(
    samples=gamma_samples,
    data=n,
    discrete=False,
    x_axis_label="n",
    title="Predictive ECDF: Gamma Model",
    plot_width=300,
)

# Difference
p2 = bebi103.viz.predictive_ecdf(
    samples=gamma_samples,
    data=n,
    diff="ecdf",
    discrete=False,
    x_axis_label="n",
    title="ECDF Difference: Gamma Model",
    x_range=p1.x_range,
    plot_width=300,
)



# Other distributed ECDFs
# Predictive
p3 = bebi103.viz.predictive_ecdf(
    samples=successive_samples,
    data=n,
    discrete=False,
    x_axis_label="n",
    title="Predictive ECDF: Successive Poisson Model",
    x_range=p1.x_range,
    plot_width=300,
)

# Difference
p4 = bebi103.viz.predictive_ecdf(
    samples=successive_samples,
    data=n,
    diff="ecdf",
    discrete=False,
    x_axis_label="n",
    title="ECDF Difference: Successive Poisson Model",
    x_range=p1.x_range,
    plot_width=300,
)



def summaries_a_b_plots(summaries_a, summaries_b):
    '''Takes in an input summaries_a and summaries_b, which are two lists of 
    dictionaries and returns two plotting figures as a tuple with pa corresponding
    to summaries_a and pb corresponding to summaries_b
    
    summaries_a and summaries_b:
    ----------------------------
    Each is a list of dictionaries with keys of (growth event, bacterium)
    and values of the computed values of parameters.'''
    # Set up CI plots
    pa = bebi103.viz.confints(
        summaries_a,
        title="Confidence Intervals for Alpha",
        x_axis_label="Number of Arrivals",
        y_axis_label="concentration",
        frame_height=150,
        frame_width=600,
        line_kwargs={"color": "indianred", "alpha": 0.7},
        marker_kwargs={"color": "indianred"},
    )

    pb = bebi103.viz.confints(
        summaries_b,
        title="Confidence Intervals for Beta",
        x_axis_label="Rate of Arrivals (1/s)",
        y_axis_label="concentration",
        frame_height=150,
        frame_width=600,
        line_kwargs={"color": "indianred", "alpha": 0.7},
        marker_kwargs={"color": "indianred"},
    )
    return (pa, pb)






# create beta1 slider
b1_slider = pn.widgets.FloatSlider(start=0, end=.1, step=0.0001, value=beta1, name="b1")

# create beta2 slider
b2_slider = pn.widgets.FloatSlider(
    start=0, end=.1, step=0.0001, value=beta2, name="b2",
)



@pn.depends(
    b1_slider.param.value,
    b2_slider.param.value,
)
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
    gamma_mle = st.gamma.cdf(x, alpha_mle_12, loc=0, scale=1 / beta_mle_12)
    model2_mle = hw9_3_functions.cdf_successive(x, b1, b2)
    p.line(x, gamma_mle, line_color="orange", line_width=1.66, legend_label="gamma")
    p.line(x, model2_mle, line_color="green", line_width=1.66, legend_label="model 2")
    
    return p
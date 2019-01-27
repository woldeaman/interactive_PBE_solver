"""Running the server for interactive PBE-solving."""
import numpy as np
import pbe_solver as pbe
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure


# Set up data
x, y = np.zeros(50), np.zeros(50)
source = ColumnDataSource(data=dict(x=x, y=y))


# Set up plot
plot = figure(plot_height=400, plot_width=400, title="my pbe solution",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)


# Set up widgets
title_text = TextInput(title="Title", value='my pbe solution')
bins = Slider(title="bins", value=100, start=50, end=250, step=10)
temp = Slider(title="temperatur [K]", value=300, start=0, end=500, step=10)
dist = Slider(title="distance [nm]", value=1.0, start=0.1, end=10, step=0.1)
valency = Slider(title="valency", value=1, start=1, end=3, step=1)
sigma = Slider(title="surface charge [e/nm^2]", value=0, start=0.5, end=5, step=0.1)
c_0 = Slider(title="ion concentration [mol/l]", value=0, start=1, end=5, step=0.1)


def solver(bins, temp, distance, valency, sigma, c_0_pre):
    """Run the PBE solver."""
    # TODO: for now without profiles
    rho, eps, pmf_cat, pmf_an = np.zeros(bins), np.ones(bins)*(1/80), np.zeros(bins), np.zeros(bins)
    # convert units
    (zz_hat, kappa, c_0, beta, dz_hat, sigma_hat, rho_hat) = pbe.convert_units(
        bins, temp, distance, valency, sigma, rho, c_0_pre)

    # compute gouy chapman solution to start with
    eps_avg = 1/np.average(eps)  # average epsilon
    psi_start = (sigma_hat/eps_avg)*np.exp(-(zz_hat))  # gouy chapman solution
    # TODO: implement correct formula, probably also depends on eps, pmfs, rho ...
    omega = 2/(1 + np.sqrt(np.pi/bins))  # omega parameter for SOR

    # call iteration procedure
    psi = pbe.iteration_loop(psi_start, omega, dz_hat, sigma_hat, rho_hat, eps,
                             pmf_cat, pmf_an)

    return psi


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = title_text.value


# connect callback function to input element
title_text.on_change('value', update_title)


def update_data(attrname, old, new):

    # Get the current slider values
    b = bins.value
    t = temp.value
    d = dist.value
    z = valency.value
    s = sigma.value
    c = c_0.value

    # solve pbe
    psi = solver(b, t, d, z, s, c)
    source.data = dict(x=psi[:, 0], y=psi[:, 1])


# connect same callback function to all sliders
for w in [bins, temp, dist, valency, sigma, c_0]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = widgetbox(title_text, bins, temp, dist, valency, sigma, c_0)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "pbe-visualization"

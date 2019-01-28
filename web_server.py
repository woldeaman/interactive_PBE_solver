"""Running the server for interactive PBE-solving."""
import numpy as np
import pbe_solver.pbe_solver as pbe  # this needs to be installed beforehand
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.plotting import figure


###########################
#       FUNCTIONS        #
##########################################################################
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
    omega = 2/(1 + np.sqrt(np.pi/bins))  # omega parameter for SOR

    # call iteration procedure
    psi = pbe.iteration_loop(psi_start, omega, dz_hat, sigma_hat, rho_hat, eps,
                             pmf_cat, pmf_an)
    zz = np.linspace(0, distance/2, bins)  # computing z-vector
    (symm_zz, symm_psi,  # compute physical data and plot if in verbos mode
     symm_dens_n, symm_dens_p) = pbe.showData(zz, psi, pmf_an, pmf_cat, c_0, beta,
                                              valency, sigma_hat, plot=False)
    data = {'zz': symm_zz, 'psi': symm_psi,
            'dens_n': symm_dens_n, 'dens_p': symm_dens_p}
    return data


def update_data(attrname, old, new):
    """Callback function to update parameters."""
    # Get the current slider values
    b = bins.value
    t = temp.value
    d = dist.value
    z = valency.value
    s = sigma.value
    c = c_0.value

    # solve pbe
    data = solver(b, t, d, z, s, c)

    # rewrite data
    psi_plt.data_source.data = dict(x=data['zz'], y=data['psi'])
    cat_plt.data_source.data = dict(x=data['zz'], y=data['dens_n'])
    ani_plt.data_source.data = dict(x=data['zz'], y=data['dens_p'])
##########################################################################


####################
#      MAIN        #
##########################################################################
# Set up data
x, y = np.zeros(50), np.zeros(50)
psi = ColumnDataSource(data=dict(x=x, y=y))
dens_neg = ColumnDataSource(data=dict(x=x, y=y))
dens_pos = ColumnDataSource(data=dict(x=x, y=y))

# set up plots
potential = figure(plot_height=500, plot_width=500, title="electrostatic potential")
psi_plt = potential.line('x', 'y', source=psi, line_width=3, line_alpha=0.6)
dens = figure(plot_height=500, plot_width=500, title="ion density")
cat_plt = dens.line('x', 'y', source=dens_pos, line_width=3, line_alpha=0.6, legend='cation density')
ani_plt = dens.line('x', 'y', source=dens_neg, line_width=3, line_alpha=0.6, legend='anion density')


# Set up widgets
bins = Slider(title="bins", value=100, start=50, end=250, step=10)
temp = Slider(title="temperatur [K]", value=300, start=0, end=500, step=10)
dist = Slider(title="distance [nm]", value=1.0, start=0.1, end=10, step=0.1)
valency = Slider(title="valency", value=1, start=1, end=3, step=1)
sigma = Slider(title="surface charge [e/nm^2]", value=0.1, start=0.5, end=5, step=0.1)
c_0 = Slider(title="ion concentration [mol/l]", value=0.5, start=1, end=5, step=0.1)

# connect same callback function to all sliders
for w in [bins, temp, dist, valency, sigma, c_0]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = widgetbox(bins, temp, dist, valency, sigma, c_0)
curdoc().add_root(row(inputs, potential, dens, width=1500))
curdoc().title = "pbe-visualization"
##########################################################################

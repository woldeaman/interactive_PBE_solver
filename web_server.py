"""Running the server for interactive PBE-solving."""
import numpy as np
import pbe_solver.pbe_solver as pbe  # this needs to be installed beforehand
import scipy.constants as sc
from bokeh.io import curdoc
from bokeh.layouts import widgetbox, gridplot
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.plotting import figure

##################
#   ENVIRONMENT  #
##########################################################################
nm_3_to_mols = 1E24/sc.Avogadro  # from nm^3 to mol/l
##########################################################################


###########################
#       FUNCTIONS        #
##########################################################################
def solver(bins, temp, distance, valency, sigma, c_0_pre):
    """Run the PBE solver."""
    # TODO: implement reading of PMFs, epsilon and charge density profiles
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
    pbe_data = solver(b, t, d, z, s, c)

    # rewrite data
    total_data.data = dict(x=pbe_data['zz']-pbe_data['zz'].max()/2, y0=pbe_data['psi'],
                           y1=pbe_data['dens_p']*nm_3_to_mols, y2=pbe_data['dens_n']*nm_3_to_mols)
##########################################################################


####################
#      MAIN        #
##########################################################################
# set up and initiallize widgets
bins = Slider(title="bins", value=100, start=50, end=250, step=10)
temp = Slider(title="temperatur [K]", value=300, start=0, end=500, step=10)
dist = Slider(title="distance [nm]", value=1.0, start=0.1, end=10, step=0.1)
valency = Slider(title="valency", value=1, start=1, end=4, step=1)
sigma = Slider(title="surface charge [e/nm\u00b2]", value=0, start=-5, end=5, step=0.1)
c_0 = Slider(title="ion concentration [mol/l]", value=0, start=1, end=5, step=0.1)
# connect same callback function to all sliders
for w in [bins, temp, dist, valency, sigma, c_0]:
    w.on_change('value', update_data)

# set up and initiallize data
zz = np.linspace(-dist.value/2, dist.value/2, bins.value)
psi_dat, cat_dat, an_dat = np.zeros(bins.value), np.zeros(bins.value), np.zeros(bins.value)
total_data = ColumnDataSource(data=dict(x=zz, y0=psi_dat, y1=cat_dat, y2=an_dat))

# set up figures
tools = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
potential = figure(plot_height=500, plot_width=500, title="electrostatic potential", tools=tools,
                   x_axis_label='z-distance [nm]', y_axis_label='\u03C6 [mV]')
dens = figure(plot_height=500, plot_width=500, title="ion concentration", x_range=potential.x_range,  # linking plots together in x-axis
              x_axis_label='z-distance [nm]', y_axis_label='c [mol/l]', tools=tools)
# create plots
psi_plt = potential.circle('x', 'y0', source=total_data, color='black')
cat_plt = dens.circle('x', 'y1', source=total_data, color='red', legend='cation concentration')
ani_plt = dens.circle('x', 'y2', source=total_data, color='blue', legend='anion concentration')


# Set up layouts and add to document
inputs = widgetbox(bins, temp, dist, valency, sigma, c_0)
all_plots = gridplot([[inputs, potential, dens]])  # make plots linked
curdoc().add_root(all_plots)
curdoc().title = "pbe-visualization"
##########################################################################

"""Running the server for interactive PBE-solving."""
import numpy as np
import pbe_solver.pbe_solver as pbe  # this needs to be installed beforehand
import scipy.constants as sc
from bokeh.io import curdoc
from bokeh.models.layouts import WidgetBox, Row, Column
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.models import FreehandDrawTool
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
    for store in [chrg_dat, eps_dat]:  # update zz-vector of free draw data
        current_dat = store.data
        store.data = dict(xs=[pbe_data['zz']-pbe_data['zz'].max()/2],
                          ys=current_dat['ys'])
    current_pmf = pmf_dat.data
    pmf_dat.data = dict(xs=[pbe_data['zz']-pbe_data['zz'].max()/2]*2,
                        ys=current_pmf['ys'])
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
# plot charge, epsilon and pmf
chrg, eps, pmf_cat, pmf_an = np.zeros(bins.value), np.ones(bins.value)*80, np.zeros(bins.value), np.zeros(bins.value)
chrg_dat, eps_dat = ColumnDataSource(data=dict(xs=[zz], ys=[chrg])), ColumnDataSource(data=dict(xs=[zz], ys=[eps]))
pmf_dat = ColumnDataSource(data=dict(xs=[zz, zz], ys=[pmf_cat, pmf_an]))

# set up figures
tools = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
potential = figure(plot_height=700, plot_width=500, title="electrostatic potential", tools=tools,
                   x_axis_label='z-distance [nm]', y_axis_label='\u03C6(z) [mV]')
dens = figure(plot_height=700, plot_width=500, title="ion concentration", x_range=potential.x_range,  # linking plots together in x-axis
              x_axis_label='z-distance [nm]', y_axis_label='c(z) [mol/l]', tools=tools)
# create drawing plots for charge, epsilon and pmfs
charge = figure(plot_width=300, plot_height=150, x_range=potential.x_range,
                y_axis_label='\u03C1(z) [e/m\u00B3]', tools='reset,save', y_range=(-2, 2))
epsilon = figure(plot_width=300, plot_height=150, x_range=potential.x_range, y_axis_label='\u03B5(z)',
                 tools='reset,save', y_range=(0, 100))
pmfs = figure(plot_width=300, plot_height=150, x_range=potential.x_range, x_axis_label='z-distance [nm]',
              y_axis_label='PMF(z) [k\u0299T]', tools='reset,save', y_range=(-10, 10))

# create plots
psi_plt = potential.line('x', 'y0', source=total_data, color='black', line_width=3)
cat_plt = dens.line('x', 'y1', source=total_data, color='red', legend='cation concentration', line_width=3)
ani_plt = dens.line('x', 'y2', source=total_data, color='blue', legend='anion concentration', line_width=3)
# set up figures with drawing tools
for fig, dat in zip([charge, epsilon], [chrg_dat, eps_dat]):
    plt = fig.multi_line(xs='xs', ys='ys', source=dat, color='black', line_width=3)
    draw_tool = FreehandDrawTool(renderers=[plt], num_objects=1)
    fig.add_tools(draw_tool)
pmf_plt = pmfs.multi_line(xs='xs', ys='ys', source=pmf_dat, color='red', line_width=3)
draw_tool_p = FreehandDrawTool(renderers=[pmf_plt], num_objects=2)
pmfs.add_tools(draw_tool_p)
pmfs.toolbar.active_drag = draw_tool_p

# Set up layouts and add to document
inputs = WidgetBox(bins, temp, dist, valency, sigma, c_0)
draw_inputs = Column(inputs, charge, epsilon, pmfs)
pot_dens_plt = Row(potential, dens)  # make plots linked
all_plots = layout([[draw_inputs, pot_dens_plt]])
curdoc().add_root(all_plots)
curdoc().title = "pbe-visualization"
##########################################################################

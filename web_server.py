"""Running the server for interactive PBE-solving."""
import numpy as np
import pbe_solver.pbe_solver as pbe  # this needs to be installed beforehand
import scipy.constants as sc
import scipy.interpolate as ip
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, FreehandDrawTool, Legend, LegendItem, Span, LinearAxis, Range1d
from bokeh.models.layouts import WidgetBox, Row, Column
from bokeh.models.widgets import Slider, Button
from bokeh.plotting import figure

##################
#   ENVIRONMENT  #
##########################################################################
nm_3_to_mols = 1E24/sc.Avogadro  # from 1/nm^3 to mol/l
##########################################################################


###########################
#       FUNCTIONS        #
##########################################################################
def initialize_widgets():
    """Set up all the widgets."""
    global bins  # make them global to be accessable throughout the script
    bins = Slider(title="Bins", value=100, start=50, end=250, step=10)
    global temp
    temp = Slider(title="Temperatur [K]", value=300, start=0, end=500, step=10)
    global dist
    dist = Slider(title="Distance [nm]", value=5, start=0.1, end=10, step=0.1)
    global valency_cat
    valency_cat = Slider(title="Ion Valency [Cations]", value=1, start=1, end=2, step=1)
    global valency_an
    valency_an = Slider(title="Ion Valency [Anions]", value=1, start=1, end=2, step=1)
    global sigma
    sigma = Slider(title="Surface Charge [e/nm\u00b2]", value=0, start=-5, end=5, step=0.1)
    global c_0
    c_0 = Slider(title="Ion Concentration [mol/l]", value=1, start=1, end=5, step=0.1)
    global c_imp
    c_imp = Slider(title="Impurity Concentration [nmol/l]", value=0, start=0, end=500, step=10)
    global start
    start = Button(label="SOLVE", button_type="primary")
    start.on_click(update_data)  # update plots when clicking button


def initialize_data():
    """Set up and initiallize all data stores."""
    zz = np.linspace(-dist.value/2, dist.value/2, bins.value)
    psi_dat, cat_dat, an_dat, cat_imp, an_imp = np.zeros(bins.value), np.zeros(bins.value), np.zeros(bins.value), np.zeros(bins.value), np.zeros(bins.value)
    global total_data
    total_data = ColumnDataSource(data=dict(x=zz, y0=psi_dat, y1=cat_dat, y2=an_dat, y3=cat_imp, y4=an_imp))
    # plot charge, epsilon and pmf
    zz_half = np.linspace(-dist.value/2, 0, bins.value)
    chrg, eps, pmf_cat, pmf_an, pmf_imp_p, pmf_imp_n = np.zeros(bins.value), np.ones(bins.value)*80, np.zeros(bins.value), np.zeros(bins.value), np.zeros(bins.value), np.zeros(bins.value)
    global chrg_dat, eps_dat, pmf_dat, pmf_imp_dat
    chrg_dat, eps_dat = ColumnDataSource(data=dict(xs=[zz_half], ys=[chrg])), ColumnDataSource(data=dict(xs=[zz_half], ys=[eps]))
    pmf_dat = ColumnDataSource(data=dict(xs=[zz_half, zz_half], ys=[pmf_cat, pmf_an],
                                         colors=['red', 'blue']))
    pmf_imp_dat = ColumnDataSource(data=dict(xs=[zz_half, zz_half], ys=[pmf_imp_p, pmf_imp_n],
                                             colors=['darkred', 'darkblue']))


def build_figures():
    """Create figures."""
    tools = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
    global potential
    potential = figure(plot_height=800, plot_width=800, title="Electrostatic Potential", tools=tools,
                       x_axis_label='z-distance [nm]', y_axis_label='\u03C6(z) [mV]')
    global dens
    dens = figure(plot_height=800, plot_width=800, title="Ion Concentration", x_range=potential.x_range,  # linking plots together in x-axis
                  x_axis_label='z-distance [nm]', y_axis_label='c(z) [mol/l]', tools=tools)
    # second y-axis for impurity density distribution
    dens.extra_y_ranges = {"Impurity": Range1d(start=-100, end=200)}
    dens.add_layout(LinearAxis(y_range_name="Impurity", axis_label='c\u1D62(z) [nmol/l]'), 'right')

    for fig in [potential, dens]:  # make titles a little nicer
        fig.title.text_font_size = '15pt'
        fig.title.text_font_style = 'italic'
        fig.title.align = 'center'

    # create drawing plots for charge, epsilon and pmfs
    global charge
    charge = figure(plot_width=300, plot_height=155, y_axis_label='\u03C1(z) [e/m\u00B3]',
                    tools='pan,ywheel_zoom,save', y_range=(-2, 2))
    global epsilon
    epsilon = figure(plot_width=300, plot_height=155, x_range=charge.x_range, y_axis_label='\u03B5(z)',
                     tools='pan,ywheel_zoom,save', y_range=(-5, 100))
    global pmfs
    pmfs = figure(plot_width=300, plot_height=185, x_range=charge.x_range, x_axis_label='z-distance [nm]',
                  y_axis_label='V(z) [k\u0299T]', tools='pan,ywheel_zoom,save', y_range=(-7.5, 7.5))
    global pmfs_imp
    pmfs_imp = figure(plot_width=300, plot_height=185, x_range=charge.x_range, x_axis_label='z-distance [nm]',
                      y_axis_label='V\u1D62(z) [k\u0299T]', tools='pan,ywheel_zoom,save', y_range=(-15, 5))

    # create plots
    potential.line('x', 'y0', source=total_data, color='black', line_width=3)
    dens.line('x', 'y1', source=total_data, color='red', legend='Cations',
              muted_color='red', muted_alpha=0.2, line_width=3)
    dens.line('x', 'y2', source=total_data, color='blue', legend='Anions',
              muted_color='blue', muted_alpha=0.2, line_width=3)
    dens.legend.location = "top_center"
    dens.legend.click_policy = "mute"  # make density transparent on click
    # plot impurity distribution as well
    dens.line('x', 'y4', source=total_data, color='darkblue', legend='Impurities',
              muted_color='darkblue', muted_alpha=0.2, line_width=3, y_range_name="Impurity")
    dens.line('x', 'y3', source=total_data, color='darkred', legend='Impurity Counterions',
              muted_color='darkred', muted_alpha=0.2, line_width=3, y_range_name="Impurity")

    # set up figures with drawing tools
    for fig, dat in zip([charge, epsilon], [chrg_dat, eps_dat]):
        plt = fig.multi_line(xs='xs', ys='ys', source=dat, color='black', line_width=3)
        draw_tool = FreehandDrawTool(renderers=[plt], num_objects=1)
        fig.add_tools(draw_tool)
        fig.toolbar.active_drag = draw_tool
    # indicating zero charge line
    zero_chrg = Span(location=0, dimension='width', line_dash='dotted', line_color='black',
                     line_width=1)
    charge.add_layout(zero_chrg)
    # indicate water and vacuum epsilon values
    water_eps = Span(location=80, dimension='width', line_dash='dotted', line_color='blue',
                     line_width=1)
    vac_eps = Span(location=1, dimension='width', line_dash='dotted', line_color='black',
                   line_width=1)
    epsilon.add_layout(water_eps)
    epsilon.add_layout(vac_eps)
    # TODO: different colors for lines mess up while drawing...
    for fig, dat, lbls in zip([pmfs, pmfs_imp], [pmf_dat, pmf_imp_dat], [["V\u207A", "V\u207B"], ["V\u1D62\u207A", "V\u1D62\u207B"]]):
        pmf_plt = fig.multi_line(xs='xs', ys='ys', line_color='colors', source=dat, line_width=3)
        draw_tool_p = FreehandDrawTool(renderers=[pmf_plt], num_objects=2)
        fig.add_tools(draw_tool_p)
        fig.toolbar.active_drag = draw_tool_p
        # indicate zero energy line
        zero_pmf = Span(location=0, dimension='width', line_dash='dotted', line_color='black', line_width=1)
        fig.add_layout(zero_pmf)
        # make legend
        legend = Legend(items=[LegendItem(label=lbls[0], renderers=[pmf_plt], index=0),
                               LegendItem(label=lbls[0], renderers=[pmf_plt], index=1)])
        fig.add_layout(legend)
        fig.legend.location = "top_center"
        fig.legend.label_text_font_size = "5pt"


def solver(bins, temp, distance, valency_cat, valency_an, sigma, c_0_pre, c_imp_pre, rho, eps,
           pmf_cat, pmf_an, pmf_imp_cat, pmf_imp_an):
    """Run the PBE solver."""
    # determine maximum valency used for normalization
    val_max = np.max([valency_cat, valency_an])
    # convert units
    (zz_hat, kappa, c_0, c_imp, beta, dz_hat, sigma_hat, rho_hat) = pbe.convert_units(
        bins, temp, distance, sigma, rho, val_max, c_0_pre, c_imp_pre)

    # compute gouy chapman solution to start with
    eps_avg = 1/np.average(eps)  # average epsilon
    psi_start = (sigma_hat/eps_avg)*np.exp(-(zz_hat))  # gouy chapman solution
    omega = 2/(1 + np.sqrt(np.pi/bins))  # omega parameter for SOR

    # call iteration procedure
    psi = pbe.iteration_loop(psi_start, omega, dz_hat, valency_cat, valency_an,
                             sigma_hat, rho_hat, eps, pmf_cat, pmf_an, pmf_imp_cat, pmf_imp_an, c_imp)
    zz = np.linspace(0, distance/2, bins)  # computing z-vector
    (symm_zz, symm_psi,  # compute physical data and plot if in verbos mode
     symm_dens_p, symm_dens_n, symm_imp_p, symm_imp_n) = pbe.showData(zz, psi, pmf_cat, pmf_an, pmf_imp_cat, pmf_imp_an, val_max, c_0, c_imp, beta,
                                                                      valency_cat, valency_an, sigma_hat, plot=False)
    data = {'zz': symm_zz, 'psi': symm_psi,
            'dens_n': symm_dens_n, 'dens_p': symm_dens_p,
            'imp_n': symm_imp_n, 'imp_p': symm_imp_p}
    return data


def interpolate_free_draw_data():
    """Do interpolation to update free draw data and profiles."""
    zz = np.linspace(-dist.value/2, 0, bins.value)
    for store in [chrg_dat, eps_dat]:
        xx, yy = store.data['xs'][0], store.data['ys'][0]  # interpolate data
        xx_unq, idx_unq = np.unique(xx, return_index=True)
        spl = ip.UnivariateSpline(xx_unq[xx_unq.argsort()],
                                  np.array(yy)[idx_unq][xx_unq.argsort()],
                                  s=0, k=1, ext='const')
        store.data = dict(xs=[zz], ys=[spl(zz)])

    # interpolate PMF data
    for dat, colors in zip([pmf_dat, pmf_imp_dat], [['red', 'blue'], ['darkred', 'darkblue']]):
        pmf_spls = []
        current_pmf_dat = dat.data
        for i in range(2):
            xx, yy = current_pmf_dat['xs'][i], current_pmf_dat['ys'][i]  # interpolate pmf data
            xx_unq, idx_unq = np.unique(xx, return_index=True)
            pmf_spls.append(ip.UnivariateSpline(xx_unq[xx_unq.argsort()],
                                                np.array(yy)[idx_unq][xx_unq.argsort()],
                                                s=0, k=1, ext='const'))
        dat.data = dict(xs=[zz]*2, ys=[pmf_spls[i](zz) for i in range(2)], colors=colors)


def update_data():
    """Callback function to update parameters."""
    interpolate_free_draw_data()  # first interpolate free draw data
    # Get the current slider values
    b = bins.value
    t = temp.value
    d = dist.value
    z_cat = valency_cat.value
    z_an = valency_an.value
    s = sigma.value
    c = c_0.value
    c_i = c_imp.value
    r = chrg_dat.data['ys'][0]
    e = 1/eps_dat.data['ys'][0]  # inverse epsilon is input
    p_c = pmf_dat.data['ys'][0]
    p_a = pmf_dat.data['ys'][1]
    p_i_c = pmf_imp_dat.data['ys'][0]
    p_i_a = pmf_imp_dat.data['ys'][1]

    # solve pbe
    pbe_data = solver(b, t, d, z_cat, z_an, s, c, c_i, r, e, p_c, p_a, p_i_c, p_i_a)

    # rewrite data
    total_data.data = dict(x=pbe_data['zz']-pbe_data['zz'].max()/2, y0=pbe_data['psi'],
                           y1=pbe_data['dens_p']*nm_3_to_mols, y2=pbe_data['dens_n']*nm_3_to_mols,
                           y3=pbe_data['imp_p']*nm_3_to_mols*1E9, y4=pbe_data['imp_n']*nm_3_to_mols*1E9,)
##########################################################################


####################
#      MAIN        #
##########################################################################
# set up and initiallize widgets
initialize_widgets()
# set up and initiallize data
initialize_data()
# set up figures
build_figures()

# Set up layouts and add to document
inputs = WidgetBox(start, bins, temp, dist, valency_cat, valency_an, sigma, c_0, c_imp)
draw_inputs = Column(inputs, charge, epsilon, pmfs, pmfs_imp)
pot_dens_plt = Row(potential, dens)  # make plots linked
all_plots = layout([[draw_inputs, pot_dens_plt]])
curdoc().add_root(all_plots)
curdoc().title = "pbe-visualization"
##########################################################################

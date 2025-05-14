import numpy as np
import matplotlib.pyplot as plt
import sp_helper_functions as sph
import os
from astropy.table import Table


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        Galaxy Plotting Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compare_indices()       : Create and save a plot showing both the measured
                          indices and those from the moses catalogue

plot_ppxf_weights()     : Make a map of the template weights derived by ppxf
                          full spectral fitting.
"""


# ____________________________________________________________________________#
# __________________ Create a map of the template weights ____________________#
def plot_ppxf_weights(pp, filename, model='MILES', tag='', fig=None, ax=None,
                      fig_output_dr='', output_filename=None,
                      age_grid_path=None, met_grid_path=None,
                      save=False, **kwargs):

    # Create a plot of the stellar population template weights
    # Create some variable name shortcuts

    # determine path to relevant input data files
    pwd = os.path.dirname(__file__)

    if model == 'MILES':
        if age_grid_path is None:
            age_grid_path = '../../data/MILES_FSF_files/MILES_age_grid.npy'
        if met_grid_path is None:
            met_grid_path = '../../data/MILES_FSF_files/MILES_met_grid.npy'
        if fig_output_dr is None:
            fig_output_dr = '../MOSES_spectra/FINAL/plots/v14/MILES_weights/'

        age_grid = np.log10(np.load(age_grid_path))
        shape    = age_grid.shape

        if ax is None:
            fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
        ax       = [ax]
        ylab     = r'$Age \; [\log_{10}Gyr]$'

    if model == 'BPASS':
        if age_grid_path is None:
            age_grid_path = pwd + ('/../../data/BPASS_FSF_files/'
                                   'BPASS_age_grid.npy')

        if met_grid_path is None:
            met_grid_path = pwd + ('/../../BPASS_FSF_files/'
                                   'BPASS_met_grid.npy')

        if fig_output_dr is None:
            fig_output_dr = ('../MOSES_spectra/FINAL/plots/v14/'
                             'BPASS_weights_2/')

        age_grid = np.log10(np.load(age_grid_path))  # logarithmic scale
        if ax is None:
            fig, ax  = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
        ylab     = r'$Age \; [\log_{10}Gyr]$'

    weights = pp.weights[~pp.gas_component]
    weights = weights.reshape(pp.reg_dim) / weights.sum()
    sp      = sph.mean_age_metal(weights=weights, model=model)
    title   = str(filename + '; regularization = '
                  + '{0:.1f}'.format(pp.regul) + '\n'
                  'mdegree= ' + str(pp.mdegree) + ', degree= '
                  + str(pp.degree)
                  + '\n Age= ' + "{0:.3f}".format(sp[0])
                  + ' Gyrs, [Z/H]=' + "{0:.3f}".format(sp[1]))

    # load metallicity grid
    met_grid = np.load(met_grid_path)

    # Set up the x (met) and y (age) borders
    x, y   = met_grid[0, :], age_grid[:, 0]    # met, age array
    xb, yb = (x[1:] + x[:-1]) / 2, (y[1:] + y[:-1]) / 2  # x, y borders
    xb  = np.hstack([1.5 * x[0] - x[1] / 2, xb, 1.5 * x[-1] - x[-2] / 2])
    yb  = np.hstack([1.5 * y[0] - y[1] / 2, yb, 1.5 * y[-1] - y[-2] / 2])

    # add extra empty dimension so it works in the for loop
    if model == 'MILES':
        weights  = weights[:, :, np.newaxis]

    # start plotting
    for i in range(len(ax)):
        ax[i].pcolormesh(xb, yb, weights[:, :, i], edgecolors='face',
                         vmin=0, vmax=0.4)
        ax[i].plot(met_grid, age_grid, 'w,')  # plot white points grid spacing
        ax[i].plot(sp[1], np.log10(sp[0]), c='r', ms=20)
        ax[i].set_xlabel('[Z/H]')
        ax[0].set_ylabel(ylab)
        ax[0].set_title(title)

    cb = fig.colorbar(ax[i].collections[0], ax=ax[i])
    cb.set_label('% Weighting')

    if save:
        if output_filename is None:
            output_filename = filename.split('.fits')[0] + tag + '.pdf'
        if fig_output_dr != '':
            os.makedirs(os.path.dirname(fig_output_dr), exist_ok=True)
        fig.savefig(fig_output_dr + output_filename, format='pdf')
        plt.close(fig)

    else:
        return(cb)


# ____________________________________________________________________________#
# Create a figure comparing the measured Lick indices to the MOSES catalogue  #
def compare_indices(lick_indices, filename,
                    catalogue='input_files/MOSES_catalogue.fits',
                    save=False,
                    fig_output_dr='../MOSES_spectra/test_LICK_plots/'):

    # Make a plot comparing measured indicies to those in the moses catalogue
    data = Table.read(catalogue)

    # Determine catalogue row corresponding to this object
    if   filename.split('-')[0] == 'spec':
        colname = 'new_filename'
    elif filename.split('-')[0] == 'spSpec':
        colname = 'dr4_filename'
    row      = np.where(data[colname] == filename)[0][0]

    # put all the index information into arrays which can be used easily
    cat_vals = np.array([data[ind][row]           for ind in [*lick_indices]])
    cat_errs = np.array([data[ind + '_ERR'][row]  for ind in [*lick_indices]])
    m_vals   = np.array([lick_indices[ind][0] for ind in [*lick_indices]])
    m_errs   = np.array([lick_indices[ind][1] for ind in [*lick_indices]])

    x = np.arange(len(m_vals))

    # start plotting
    fig, ax = plt.subplots(1, figsize=(8, 6))

    ax.plot(x, m_vals, c='b', marker='.', linewidth=0)
    ax.plot(x, cat_vals, c='r', marker='.', linewidth=0)
    ax.fill_between(x, (m_vals   - m_errs), (m_vals   + m_errs), alpha=0.5)
    ax.fill_between(x, (cat_vals - cat_errs), (cat_vals + cat_errs),
                    alpha=0.5, color='r')
    ax.errorbar(x, m_vals, m_errs, c='b', alpha=0.4, fmt='o', linewidth=0.5,
                markersize=0.1)

    ax.set_xticks(x)
    ax.set_xticklabels([*lick_indices], rotation='vertical')
    ax.set_ylabel('Lick Index Value')
    ax.set_title(filename)

    if save:
        fig.savefig(fig_output_dr + filename.split('.')[0] + '.pdf')

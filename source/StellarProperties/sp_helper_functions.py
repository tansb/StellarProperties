import numpy as np
import astropy.io.ascii as ap_ascii
import ppxf.ppxf_util as ppxf_util
import os
from . import get_from_file_functions as gff
from scipy.constants import c

c = c * 10**-3  # units of km/s


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        Galaxy Helper Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
clean_ppxf_object()     : Delete a bunch of useless attribuets from the ppxf
                          object
determine_goodpixels()  : A wrapper for ppxf_utils.determine_goodpixels()
                          which masks emission lines, but also masks skyline
                          around 5557 Angstroms
get_from_file()         : imports the galaxy spectrum and wavelength data from
                          input file
get_mask()              : returns a mask which currently just cuts off the
                          noisy ends of the galaxy spectrum
get_model_templates()   : Load in the model spectra to be used in the full
                          spectral fitting method. if model='MILES' then this
                          is just a wrapper for miles_utils.miles(), elif
                          model='BPASS' then it is a wrapper for
                          prepare_bpass_class.BPASS().
get_templates()         : Load in the templates spectra, broaden to match the
                          galaxy instrumental resolution by convolving with a
                          gaussian, then logarithmically (log_e) rebin the
                          spectra.
mean_age_metal()        : Return the mean age [Gyrs] and metallicity [Z/H]
                          given the template weights determined by ppxf.
measure_ind()           : Measure the Lick indicies.
mc_ind_errors()         : Estimate the uncertainties on the indicies by
                          randomly drawing noise from the variance spectrum
                          and re-measuring the index.
"""


# ____________________________________________________________________________#
# ______________ Get the necesarry information from file _____________________#
def get_from_file(input_fn, spec_type, **kwargs):
    """A wrapper function which determiens which spectum type it is (e.g.
       SDSS, LEGA-C, MAGPI, SAMI), and then directs to the correct
       get_from_file function

       Each of these get_from_file returns the following:
            (1) z           : the (cosmological) redshift,
            (2) loglam_gal  : logarithmically(10) binned wavelength in AA
            (3) logflux_gal : logarithmically(10) binned flux array
            (4) wdisp       : the wavelength dispersion array.
    """
    if spec_type == 'JWST':
        return gff.JWST_get_from_file(input_fn, **kwargs)

    if spec_type == 'KCWI':
        return gff.KCWI_get_from_file(input_fn, **kwargs)

    elif spec_type == 'GTC_OSIRIS':
        return gff.GTC_OSIRIS_get_from_file(input_fn, **kwargs)

    elif spec_type == 'GTC_OSIRIS_spec2D':
        return gff.GTC_OSIRIS_get_from_file_spec2d(input_fn, **kwargs)

    elif spec_type == 'XSHOOTER':
        return gff.XSHOOTER_get_from_file(input_fn, **kwargs)

    elif spec_type == 'SAMI':
        return gff.SAMI_get_from_file(input_fn, **kwargs)

    elif spec_type == 'SDSS':
        return gff.SDSS_get_from_file(input_fn, **kwargs)

    elif spec_type == 'LEGA-C':
        return gff.LEGAC_get_from_file(input_fn, **kwargs)

    elif spec_type == 'MAGPI':
        return gff.MAGPI_get_from_file(input_fn, **kwargs)

    else:
        raise Exception("What type of spectrum is this? Options are: 'SDSS', "
                        "'LEGA-C', 'SAMI', and 'MAGPI', 'KCWI'")


def set_up_ppxf(loglam_gal, var, FWHM_inst, z, models, mask_emission):

    """ There are n steps to setting up PPXF:
            (1) set velocity scale
            (2) interpolate the galaxy FWHM to tempalte wavelength
            (3) broaden templates to galaxy's interpolated FWHM
            (4) set and mask galaxy data based on tempalate wavelength range
            (5) get the goodpixels and noise arrays
            (6) set the difference in starting velocities dv
                (in NATURAL LOG space) between galaxy and template spectra
    """

    # (1) Set up the velocity scale
    velscale = np.log(loglam_gal[1] / loglam_gal[0]) * c

    # (2) interpolate the galaxy FWHM to tempalte wavelength
    FWHM_gal_interp = np.interp(models.linlam, loglam_gal, FWHM_inst)

    # (3) broaden templates to galaxy's interpolated FWHM
    logflux_temp, loglam_temp = models.broaden(
        velscale=velscale, FWHM_gal_interp=FWHM_gal_interp, reuse=False)

    # (4) set and mask galaxy data based on tempalate wavelength range
    mask = get_mask(loglam_gal, good_range=[loglam_temp[0], loglam_temp[-1]])
    loglam_gal = loglam_gal[mask]
    var        = var[mask]

    # (5) get the goodpixels and noise arrays
    goodpix, ns_clean = get_goodpixels_and_noise(loglam_gal, var, z,
                                                 mask_emission=mask_emission)

    # (6) set the difference in starting velocities dv
    dv = np.log(loglam_temp[0] / loglam_gal[0]) * c  # km/s

    # NOES:
    # (1) velscale == c * d_ln_lambda : velocity scale in km/s per pixels.
    #     Used to set number of output pixels for the logarithmic binning of
    #     the templates. Because there is a constant log_lambda fraction per
    #     pixel ie log10lam_gal[i+1] - log10lam_gal[i] = const for all i, the
    #     velocity scale is constant.
    #
    # (2) To broaden the templates, we must first interpolate the FWHM_inst to
    #     the linear scale of the templates.
    # (3) A mask needs to be applied to the galaxy spectrum, because we can't
    #     use galaxy data outside of the template's wavelength range.
    # (6) Although a mask has been applied to the galaxy spectrum, this is
    #     somewhat rough and there may still be a residual difference between
    #     the starting point of the galaxy and template spectra. Hence we
    #     define dv (in NATURAL LOG space) as the difference in starting
    #     velocity, which is added to the template spectra in PPXF via the
    #     'VSYST' keyword.

    return(loglam_gal, var, mask, goodpix, ns_clean, logflux_temp,
           loglam_temp, dv, velscale)



# ____________________________________________________________________________#
# ______________ Wrapper for ppxf vacuum to air conversion ___________________#

def vacuum_to_air(loglam_gal):

    # convert sdss vacuum wavelengths to air. The wavelength dependence of
    # this correction is so weak, to avoid resampling, it is approximated
    # with a constant factor.
    # i.e lam_air ~= lam_vac * median[lam_air / lam_vac]
    # hence:
    # log10(lam_air) ~= log10(lam_vac) + log10(median[lam_air / lam_vac])
    vac_2_air = np.nanmedian(ppxf_util.vac_to_air(loglam_gal) / loglam_gal)
    loglam_gal += np.log10(vac_2_air)

    return(loglam_gal)


# ____________________________________________________________________________#
# __________________ Define a mask to match tempaltes ________________________#
def get_mask(wavelength, good_range):

    # the wavelength range (in ln[Angstroms]) of templates (from
    # ppxf_example_kinematics_sdss.py)
    # temp_range = [3500., 7429.4]
    # apply a mask to the galaxy arrays to retain only wavelength range in
    # common with templates However, everything before ~4000 Angstroms is
    # real shit, so remove up to 4000 A instead. This extra cut means the mask
    # will need to be applied to the templates as well.

    mask = (wavelength >= good_range[0]) & (wavelength <= good_range[1])
    return mask


# ___________________________________________________________________________#
# _____________________ Get goodpixels and noise array ______________________#
def get_goodpixels_and_noise(loglam_gal, var, z, mask_emission):

    """ Return the goodpixels and a cleaned noise array based on where there
        are infs, nans in the variance array. Replace these with the median
        value and return."""

    # Use (wrapper for) ppxf_util.determine_goodpixels, to identify and
    # exclude the following emission lines: [NOTE not all lines are within
    # the wavelength range of the spectrum] 2x[OII], Hdelta, Hgamma,
    # Hbeta, 2x[OIII], [OI], 2x[NII], Halpha, 2x [SII]
    # Additionaly, the wrapper also masks the prominent skyline at 5557 A.
    if mask_emission is True:
        goodpix0 = determine_goodpixels(lam_gal=loglam_gal, z=z)

    elif mask_emission is False:
        goodpix0 = np.arange(len(loglam_gal))

    ns = np.sqrt(var)  # The NOISE SPECTRUM from file.
    # Exclude in the condition calculation the pixels with inf noise
    # (where 1/var was set to 0; else the standard deviation screws up)
    # establish the condition that pixels which have an uncertainty higher
    # than 5sigma then it will be cut
    # condition    = (np.nanmedian(ns[~np.isinf(ns)])
    #                + 5 * np.nanstd(ns[~np.isinf(ns)])) , (ns > condition)

    high_err_pix = np.where((ns == 0.) | np.isnan(ns) | np.isinf(ns))[0]
    # Catches infs, nans and where the noise spectrum is zero
    # Remove these high error pixels if they are in goodpixels

    goodpix = np.array([pix for pix in goodpix0
                        if pix not in high_err_pix])

    # Even though these pixels have been exluded from goodpix already,
    # ppxf still messes up if there are inf values in the noise spectrum,
    # so just replace these with the median.
    ns_clean = np.copy(ns)
    ns_clean[high_err_pix] = np.nanmedian(ns[goodpix])

    return goodpix, ns_clean


# ___________________________________________________________________________#
# _______ Determine goodpixels to exclude emission and sky lines ____________#
def determine_goodpixels(lam_gal, z):

    # Use ppxf_util function determine_goodpixels to mask emission lines,
    # plus cover up the skyline in the region around 5557 angstroms and cut
    # the crappy end of the spectrum.

    # The observed (i.e not redshift corrected: must undo correction),
    # np.log_e(wavelength) for wavelength in Angstroms. Have already masked
    # the galaxy spectrum and the templates, so for lamRangeTemp can just put
    # some arbitarily large range so it wont further trim the edges of the
    # spectrum
    obs_lam = lam_gal * (z + 1)
    goodpix = ppxf_util.determine_goodpixels(ln_lam=np.log(obs_lam),
                                             lam_range_temp=[100, 20000],
                                             redshift=z)

    # Always replace the region around 5557 (nonredshift corrected) due to the
    # sky emission line
    skyline = np.where((obs_lam > 5565) & (obs_lam < 5590))[0]

    # skyline, list(range(50)), and list(range(len(lam_gal) -30, len(lam_gal))
    # contain the indices of "bad pixels" with respect to lam_gal. However, we
    # need the indicies with respect to *GOODPIX*. The method my_list.index(x)
    # returns the index with respect to my_list of the element with value x.
    # Note: this is to remove the first 50 and last 30 pixels from the
    # spectrum. The if part takes avoids a ValueError if s is not in goodpix.
    inds = [list(goodpix).index(s) for s in list(skyline) if s in goodpix]
    goodpix_final = np.delete(goodpix, inds)

    return goodpix_final



# ____________________________________________________________________________#
# _____ Delete all the useless stuffs that is stored in the ppxf objects _____#
def clean_ppxf_object(pp):

    # A bunch of attributes saved in the ppxf object are somewhat useless,
    # they were used for internal calculations/ or they just save data that
    # can be found elsewhere. In order to reduce the file size of the pickled
    # objects, just remove them.

    # Current level of cleaning (12 attributes) reduces ppxf object file size
    # from 48.9 MB to < 1 MB (factor ≈50 reduction in size!!)

    # Notice that the `star` and `templates` attributes are actually the same
    # data, but for two different versions of `ppxf`.

    useless_attributes = ['component', 'fixall', 'matrix', 'method',
                          'moments', 'njev', 'npix', 'npix_temp',
                          'quiet', 'reddening', 'sigma_diff',
                          'templates_rfft', 'templates', 'star']

    for attr in useless_attributes:
        try:
            delattr(pp, attr)       # delete them!
        except AttributeError:
            pass  # if attribute doesn't exist, ignore it


# ____________________________________________________________________________#
# ______ Measure the LICK index given an input spectrum and deets_____________#
def measure_ind(w, flux, index, table):

    r = np.where(index == table['Index'])[0][0]  # table row for that index

    # The array indices corresponding to the index, blue, and red passbands.
    index_inds = np.where((w <= table[r]['I_upper'])
                          & (w >= table[r]['I_lower']))[0]
    blue_inds  = np.where((w <= table[r]['BC_upper'])
                          & (w >= table[r]['BC_lower']))[0]
    red_inds   = np.where((w <= table[r]['RC_upper'])
                          & (w >= table[r]['RC_lower']))[0]

    # Mean height of the blue and red passbands
    height_red  = np.nanmean(flux[red_inds])
    height_blue = np.nanmean(flux[blue_inds])
    # Wavelength value in the centre of the index, blue, and red passbands.
    red_centre   = np.nanmean(w[red_inds])
    blue_centre  = np.nanmean(w[blue_inds])

    # Slope & intercept of the straight line connecting the mean values in the
    # blue & red passbands. d_spectrum/ d_wavelength, i.e. rise / run
    slope       = (height_red - height_blue) / (red_centre - blue_centre)
    y_intercept = height_blue - slope * blue_centre

    dw = w[1] - w[0]  # d_wavelength

    def ps_line(x):
        # The equation of the straight line connecting the blue and red
        # pseudocontinua
        return(slope * x + y_intercept)

    # Index Equivalent Width
    EW = np.nansum([(1 - flux[i] / ps_line(w[i])) * dw for i in index_inds])

    # If the index is due to a molecular transition, the definition changes
    # slightly and the equivalent width is provided in log10 units.
    if table['Molecular'][r] == 1:

        # if the difference between the wavelength pixel and the defined
        # passband is close to dw (~90% of dw), then term0 is too large by a
        # factor of ~1.02 which may seem small, but leads to a significant
        # offset in the final equivalent width index value. To avoid this, in
        # these instances use the mean of w[index_inds[-1]] and
        # table[r]['I_upper'] to get a better value for term0.

        if abs(w[index_inds[-1]] - table[r]['I_upper']) > 0.9 * dw:
            aa = np.nanmean([w[index_inds[-1]], table[r]['I_upper']])
        else:
            aa = table[r]['I_upper']

        if w[index_inds[0]]  - table[r]['I_lower'] > 0.9 * dw:
            bb = np.nanmean([w[index_inds[0]], table[r]['I_lower']])
        else:
            bb = table[r]['I_lower']

        term_0 = 1 / (aa - bb)
        term_1 = np.nansum([(flux[j] / ps_line(w[j])) * dw
                            for j in index_inds])
        EW     = -2.5 * np.log10(term_0 * term_1)

    return(EW)


# ____________________________________________________________________________#
# ______________ Estimate the errors on the LICK index _______________________#
def mc_ind_errors(rest_wavelength, spectrum, variance, index,
                  index_definitions, n=100):
    # Determine uncertainty on a Lick index measurement by randomly drawing
    # noise from the variance spectrum and re-measuring the index.

    ind_vals = np.zeros((n))
    for i in range(n):
        spec_new    = (np.copy(spectrum)
                       + np.random.randn(len(spectrum)) * np.sqrt(variance))
        ind_vals[i] = measure_ind(rest_wavelength, spec_new, index,
                                  index_definitions)

    err_ind  = np.std(ind_vals)

    return err_ind


# ____________________________________________________________________________#
# ___________ Correct indicies for high dispersion galaxies __________________#
def vdisp_correct(veldisp, width, index):

    # Correct Lick index measurements for observed velocity dispersion if
    # larger than Lick resolution.
    # These tables provide the slope and intercept of the correction (i.e 0th
    # and 1st order corrections), for velocity dispersions starting from
    # 25km/s to 400 km/s in steps of 25kms (so the table has 17 rows). The
    # columns correspond to the different indices (so 25 columns).

    pwd = os.path.dirname(__file__)  # present working directory

    sigcorr_slope = ap_ascii.read(pwd + '/../../data/'
                                        'velocity_dispersion_correction/'
                                        'sigcorr_slope.dat')
    sigcorr_icpt  = ap_ascii.read(pwd + '/../../data/'
                                        'velocity_dispersion_correction/'
                                        'sigcorr_icpt.dat')

    # get the slopes and intercepts for the relavent index.
    slope_arr = sigcorr_slope[index]
    icpt_arr  = sigcorr_icpt[index]

    # array of velocity dispersions from 25km/s to 400km/s
    sigma = np.arange(17) * 25

    if (np.ceil(veldisp) % 25) == 0.:
        # i.e if the velocity dispersion is exactly a multiple of 25
        width = (width * slope_arr[int(np.ceil(veldisp) / 25)]
                 + icpt_arr[int(np.ceil(veldisp) / 25)])
    else:
        # if not, take some weighting of the values above and below the actual
        low   = (width * slope_arr[int(np.ceil(veldisp) / 25)]
                 + icpt_arr[int(np.ceil(veldisp) / 25)])
        high  = (width * slope_arr[int(np.ceil(veldisp) / 25 + 1)]
                 + icpt_arr[int(np.ceil(veldisp) / 25 + 1)])

        width = (((25.  - (veldisp - sigma[int(np.ceil(veldisp) / 25)])) * low
                  + (25. - (sigma[int(np.ceil(veldisp) / 25 + 1)] - veldisp))
                  * high) / 25.)

    return width


# ____________________________________________________________________________#
# ___________ A slightly edited version of the method in miles_util __________#
def mean_age_metal(weights, models, info=False):

    # Return mean Age and Metallicity given the relative weight for each
    # template
    # need two 2d arrays of shape (nages, nmets).
    # models.ages and models.mets are both 1d arrays of length nages, nmets
    # so need to populate the 2d arrays with these 1d arrays
    met_grid     = np.array([models.mets for _ in range(len(models.ages))])
    log_age_grid = np.array([np.log10(models.ages)
                             for _ in range(len(models.mets))]).T

    # AGE in units of Gyrs, MET in [Z/H] = log10(Z/Z_0)
    # Equations (1) and (2) from McDermid+15 :
    #
    # (1)   log10[Age/years] = Σ w_i * log10[t_ssp,i] / Σ w_i
    #
    # (2)           [Z/H]    = Σ w_i * [Z/H]_ssp,i    / Σ w_i
    #
    # NOTE: The grid of model ages is sampled logarithmically,
    # hence the logarithmic form of equation (1).
    #
    # http://adsabs.harvard.edu/abs/2015MNRAS.448.3484M
    w = weights
    log_mean_age = np.sum(w * log_age_grid) / np.sum(weights)
    mean_met     = np.sum(w * met_grid)     / np.sum(weights)
    mean_age     = 10**log_mean_age

    if info:
        print('Weighted <Age> [Gyr]: %.3g' % mean_age)
        print('Weighted <[Z/H]>:     %.3g' % mean_met)

    return mean_age, mean_met

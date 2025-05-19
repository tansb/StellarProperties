import os
import glob
import math
import warnings
import numpy as np
import ppxf.ppxf_util as pu
from astropy.io import fits
from astropy.io import ascii as ap_ascii
from astropy import wcs


class Templates():
    def __init__(self):
        self.templates           = None
        self.FWHM                = None
        self.linlam              = None
        self.templates_convolved = None
        self.FWHM_gal_interp     = None
        self.loglam_temp         = None

    def broaden(self, velscale, FWHM_gal_interp, reuse=False):
        """Broaden the templates by convolving with the difference in
           resolution with galaxy """

        # Firstly, check whether the desired broadening has already been used.
        # If so, then return the broadened templates already in memory. This
        # introduces bugs when analysing many galaxies in parallel, so have
        # introduced the boolean argument 'reuse' allowing this code block to
        # be switched on/off.

        if reuse:
            if self.FWHM_gal_interp is not None:
                if not np.all(np.equal(self.FWHM_gal_interp, FWHM_gal_interp)):
                    warn_mess = ("Using multiple galaxy resolutions on the "
                                 "same templates is inefficient and may "
                                 "generate a race condition.")
                    warnings.warn(warn_mess, RuntimeWarning)

                if (self.templates_convolved is not None
                        and self.loglam_temp is not None):
                    return (self.templates_convolved, self.loglam_temp)

        # Depending on whether the tempaltes are stellar or synthetic, the
        # self.templates array may may have 2 or 3 (or 4 if BPASS) dimensions.
        # If 2, add a superfluous third dimension so that code runs.
        if self.templates.ndim == 2:
            temps = np.expand_dims(self.templates, axis=2)
            temps = np.expand_dims(temps, axis=3)

        if self.templates.ndim == 3:
            temps = np.expand_dims(self.templates, axis=3)

        if self.templates.ndim == 4:
            temps = self.templates

        FWHM_dif  = np.sqrt((FWHM_gal_interp**2 - self.FWHM**2).clip(0))
        # Sigma difference in pixels
        sigma     = FWHM_dif / math.sqrt(8 * np.log(2)) / self.dw
        lam_range = [self.linlam[0], self.linlam[-1]]

        # Test logarithmic rebinning on 1 raw model spectra to determine length
        # of resulting array.
        ssp = temps[:, 0, 0, 0]
        # If the resolution of the instrument is higher than the templates,
        # can't/don't want to convolve the templates with a gaussian, hence
        # the if else below.
        if np.array_equal(np.unique(sigma), np.array([0.])):
            ssp_convolved = ssp

        else:
            #if np.unique(sigma) != 0:
            ssp_convolved = pu.gaussian_filter1d(ssp, sigma)

        sspNew, loglam_temp  = pu.log_rebin(lam_range, ssp_convolved,
                                            velscale=velscale)[:2]
        templates_convolved  = np.ones((sspNew.size, temps.shape[1],
                                        temps.shape[2], temps.shape[3]))

        # Here we make sure the spectra are sorted in both [M/H] and Age along
        # the two axes of the rectangular grid of templates.
        for j in range(temps.shape[1]):
            for k in range(temps.shape[2]):
                for m in range(temps.shape[3]):
                    # convolution variable sigma
                    # If the resolution of the instrument is higher than the
                    # templates, can't/don't want to convolve the templates
                    # with a gaussian, hence the if else below.
                    if np.array_equal(np.unique(sigma), np.array([0.])):
                        # if np.unique(sigma) != 0:
                        ssp0 = temps[:, j, k, m]
                    else:
                        ssp0 = pu.gaussian_filter1d(temps[:, j, k, m], sigma)

                    # (log_e) binning
                    sspNew = pu.log_rebin(lam_range, ssp0,
                                          velscale=velscale)[0]

                    if self.weighting == 'light_weighted':
                        sspNew /= np.mean(sspNew)
                    templates_convolved[:, j, k, m] = sspNew

        # Delete superfluous third dimension if it exists
        # if len(self.templates.shape) == 2:
        templates_convolved = templates_convolved.squeeze()

        # Store these broadened values for future reference.
        self.templates_convolved = templates_convolved
        self.FWHM_gal_interp     = FWHM_gal_interp
        self.loglam_temp         = np.exp(loglam_temp)  # Angstrom units

        if np.array_equal(np.unique(sigma), np.array([0.])):
            # np.unique(sigma) == 0:
            print("NOT BROADENING: TEMPLATES LOWER RESOLUTION THAN INSTRUMENT")

            # Assign the sigma difference to an attribute of the tempalte,
            # So it can be subtracted off when calculating velocity dispersions
            FWHM_dif_abs = np.sqrt(np.abs(FWHM_gal_interp**2 - self.FWHM**2))
            # Sigma difference in pixels
            sigma_dif_abs = FWHM_dif_abs / math.sqrt(8 * np.log(2)) / self.dw
            self.sigma_dif_abs = sigma_dif_abs

        return (self.templates_convolved, self.loglam_temp)
    # ------------------------------------------------------------------------#

    def reduce_wavelength_range(self, lam_min=3000, lam_max=8000):
        """ Especially the EMILES templates have a way bigger wavelength range
            than is currently needed. So chop it to make handling faster. """

        # First assert that the templates haven't already been broadened
        assert self.loglam_temp is None, "This must be done before broadening"

        reduced_range = np.where((self.linlam >= lam_min)
                                 & (self.linlam <= lam_max))[0]
        # Update the template attributes
        self.linlam    = self.linlam[reduced_range]

        if self.templates.ndim == 3:
            self.templates = self.templates[reduced_range, :, :]
        elif self.templates.ndim == 4:
            self.templates = self.templates[reduced_range, :, :, :]
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#


class Stellar_Templates(Templates):

    def __init__(self, model='MILES', lib_dir=None):
        """ Import from file all templates into an 2D array. """

        super().__init__()  # inherit from superclass "Templates"

        self.weighting = None
        self.model = model

        # determine path to relevant input data files
        pwd = os.path.dirname(__file__)
        data_dir = pwd + '/../../../data/Stellar_Templates/'

        # determine path to relevant input data files
        if lib_dir is None:
            if self.model == 'MILES':
                # use the MILES stellar spectra
                lib_dir = data_dir + 'MILES_library_v9.1_FITS/'
                self.FWHM = 2.54  # In units of Angstroms (Beifiori+ 2011)
                # https://ui.adsabs.harvard.edu/abs/2011A%26A...531A.109B/)

            elif self.model == 'CaT':
                # use the CaT specific stellar spectra (8350-9020 AA)
                lib_dir = data_dir + 'CaT_library_v9.1_FITS/'
                self.FWHM = 1.5  # In units of Angstroms
                # http://research.iac.es/proyecto/miles/pages/'
                # 'stellar-libraries/cat-library.php

            else:
                raise ValueError("The model name given isn't recognised. "
                                 "do either model='CaT' or 'MILES' or "
                                 "provide  the path to the templates via "
                                 "lib_dir=<path_to_templates>")

        all_file_names = glob.glob(lib_dir + 's*.fits')

        # Import a single tempalte to get the sizing details
        with fits.open(all_file_names[0]) as hdu:
            h   = hdu[0].header
            ssp = hdu[0].data.squeeze()

        # linearly binned wavelength array in Angstroms
        self.linlam = np.array([h['CRVAL1'] + h['CDELT1'] * i
                                for i in range(h['NAXIS1'])])
        self.dw = h['CDELT1']

        # Initialise an empty array to hold all the templates
        templates       = np.empty((h['NAXIS1'], len(all_file_names)))
        templates[:, 0] = ssp

        for i, file in enumerate(all_file_names[1:], 1):
            with fits.open(file) as hdu:
                templates[:, i] = hdu[0].data.squeeze()

        self.templates = templates
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#


class Synthetic_Templates(Templates):

    def __init__(self, model='MILES', light_weighted=False):
        """ Import from file the 3D array containing the templates. """

        super().__init__()  # inherit from superclass "Templates"

        if light_weighted:
            self.weighting = 'light_weighted'
        else:
            self.weighting = 'mass_weighted'

        # determine path to relevant input data files
        pwd = os.path.dirname(__file__)
        data_dir = pwd + '/../../../data/Synthetic_Templates/'

        if model == 'MILES':
            top_lib = data_dir + 'MILES_FSF_files/'
            lib_dir       = top_lib + 'MILES_BASTI_CH_baseFe/'
            array_path    = top_lib + 'MILES_all_data_3D.npy'
            age_grid_path = top_lib + 'MILES_age_grid.npy'
            met_grid_path = top_lib + 'MILES_met_grid.npy'

            with fits.open(glob.glob(lib_dir + '*.fits')[0]) as file:
                h = file[0].header

            self.dw     = h['CDELT1']  # wavelength spacing of each pixel
            # wavelength array of the templates
            self.linlam = np.array([h['CRVAL1'] + h['CDELT1'] * i
                                    for i in range(h['NAXIS1'])])

        elif model == 'EMILES':
            top_lib = data_dir + 'EMILES_FSF_files/'
            lib_dir       = top_lib + 'EMILES_PADOVA00_BASE_CH_FITS/'
            array_path    = top_lib + 'EMILES_all_data_3D.npy'
            age_grid_path = top_lib + 'EMILES_age_grid.npy'
            met_grid_path = top_lib + 'EMILES_met_grid.npy'

            with fits.open(glob.glob(lib_dir + '*.fits')[0]) as file:
                h = file[0].header

            self.dw     = h['CDELT1']  # wavelength spacing of each pixel
            # wavelength array of the templates
            self.linlam = np.array([h['CRVAL1'] + h['CDELT1'] * i
                                    for i in range(h['NAXIS1'])])

        elif model == 'EMILES-IR':
            # models with just the infra-red part
            top_lib = data_dir + 'EMILES-IR_FSF_files/'
            lib_dir       = top_lib + 'EMILES_BaSTI_bi_baseFe_infrared/'
            array_path    = top_lib + 'EMILES_all_data_3D.npy'
            age_grid_path = top_lib + 'EMILES-IR_age_grid.npy'
            met_grid_path = top_lib + 'EMILES-IR_met_grid.npy'

            with fits.open(glob.glob(lib_dir + '*.fits')[0]) as file:
                h = file[0].header

            self.dw     = h['CDELT1']  # wavelength spacing of each pixel
            # wavelength array of the templates
            self.linlam = np.array([h['CRVAL1'] + h['CDELT1'] * i
                                    for i in range(h['NAXIS1'])])
        else:
            raise ValueError("The model name given isn't recognised")

        self.name      = model
        self.templates = np.load(array_path)      # array of the templates
        self.n_ages    = self.templates.shape[1]  # number of ages
        self.n_mets    = self.templates.shape[2]  # number of metals
        self.FWHM      = 2.3  # in units of Angstroms (Vazdekis et al. 2010)

        # arrays containing the ages and mets of the templates
        self.ages = np.load(age_grid_path)[:, 0]
        self.mets = np.load(met_grid_path)[0, :]
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#


class Elodie_LH_Templates(Templates):
    """
    # Please refer to Boardman et al. (2017) for the resolution
    # http://adsabs.harvard.edu/abs/2017MNRAS.471.4005B
    """

    def __init__(self, lib_dir='../../../templates/templlib/lh_elodie'):

        super().__init__()  # inherit from superclass "Templates"

        # template resolution
        self.FWHM = 0.13

        all_file_names = glob.glob(os.path.join(lib_dir, '*.fits'))

        # Get the size of the templates
        _hdr0_ = fits.getheader(all_file_names[0])
        n_wave = _hdr0_['NAXIS1']

        # Pull out the wavelength range. Need to correct a keyword.
        _hdr0_['CTYPE1'] = 'WAVELENGTH'  # These peeps where thinking what?
        _wcs0_ = wcs.WCS(_hdr0_)

        self.linlam = np.squeeze(
            _wcs0_.wcs_pix2world(np.arange(_wcs0_._naxis[0]), 0))
        self.dw = _hdr0_['CDELT1']

        # Set up the output array.
        templates = np.empty((n_wave, len(all_file_names)))

        for n, fname in enumerate(all_file_names):
            with fits.open(fname) as fhandle:
                templates[:, n] = np.squeeze(
                    self._fill_ELODIE_LH(fhandle['PRIMARY'].data))

        _isnan_   = ~np.isfinite(templates)
        _hasnan_  = np.sum(_isnan_, axis=0)
        _isclean_ = _hasnan_ == 0
        templates = templates[:, _isclean_]

        self.templates = templates
    # ------------------------------------------------------------------------#

    @staticmethod
    def _fill_ELODIE_LH(t):
        """Replace small gaps in the ELODIE LH library to make them viable"""

        is_nan = np.where(~np.isfinite(t))[0]

        gaps = is_nan[1:] - is_nan[:-1]

        # Identify where the sequence of nan's has gaps.
        gaps = np.where(gaps != 1)[0]

        gaps = np.hstack([np.array([-1]), gaps])

        # Add the last interval with an escamotage.
        gaps = np.hstack([gaps, np.array([-1])])

        max_gap = 0

        for i in range(len(gaps) - 1):
            index_a, index_nana = is_nan[gaps[i] + 1] - 1, is_nan[gaps[i] + 1]

            # index_a, index_nana = is_nan[gaps[i]]-1, is_nan[gaps[i]]
            _, index_b = is_nan[gaps[i + 1]], is_nan[gaps[i + 1]] + 1

            if index_b - index_nana > max_gap:
                max_gap = index_b - index_nana
                if max_gap > 50:
                    print(f"Gap ({max_gap} pixels) too large: "
                          "template erased")
                    return t * np.nan

            t_a, t_b = t[index_a], t[index_b]

            idx = np.arange(index_nana, index_b, dtype=np.int)

            def lin_replace(i):
                r = ((t_b - t_a) / np.float(index_b - index_a)
                     * (i - index_a) + t_a)
                return r

            # changed lambda to function for pep8
            # lin_replace = lambda i: ((t_b - t_a) /np.float(index_b - index_a)
            #                           * (i - index_a) + t_a)
            t[idx] = lin_replace(idx)
        return t

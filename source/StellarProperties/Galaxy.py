import numpy as np
import os
import pickle
import math
import astropy.io.ascii as ap_ascii
import ppxf.ppxf_util as pu
from . import sp_helper_functions as sph
from importlib import reload
from astropy.io import fits
from astropy.table import Table
from scipy.constants import c
from ppxf.ppxf import ppxf

reload(sph)
c = c * 10**-3  # units of km/s


"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   Galaxy Method Descriptions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

clean_spectrum()        : Fits ppxf() 2 times to create a template which is
                          used to mask emission
                          lines, sky lines, and discrepant pixels
measure_lick_indices()  : Measures Lick indices
measure_sp_parameters() : Measures the stellar population parameters Age,
                          Metallicity, and [Alpha/Fe] using either the
                          Schiavon (2007) models, or the Thomas, Maraston,
                          Johansson (2010) models.
FSF()                   : Measures the stellar population parameters Age and
                          Metallicity using either the Vazdekis-Miles (2010)
                          [model='MILES'] or BPASS (2018) [model='BPASS']
                          models.
save()                  : Saves the galaxy object using pickle. Files sizes
                          are ~230MB

To obtain lick indices run in the following order:
         - clean_spectrum()
         - measure_lick_indices()
         - optional: measure_sp_parameters(model=model) where model is 'tmj'
                     or 's07'. Alternatively, input any model using the
                     'model_file' keyword
         - save()

To do full spectral fitting run in the following order:
        - clean_spectrum()
        - FSF(model=model) where model is 'MILES' or 'BPASS'
        - save()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   Galaxy Attribute Descriptions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

extra_redshift   = The extra shift in wavelength due to peculiar velocity as
                   determined by ppxf
filename         = The name of the input spectrum file; e.g.
                   'spec-0296-51665-0319.fits'
FWHM_inst        = The (wavelength dependant) instrument resolution (FWHM) in
                   Angstroms
galaxy_final     = The normalised, linearly binned, emmission line corrected
                   FINAL galaxy spectrum, now ready for shit like index
                   measurements
loglam_gal       = Logarithmically (log10) binned, redshift corrected [but not
                   peculiar velocity corrected!] galaxy wavelength in units of
                   Angstroms
lam_gal_rest     = Logarithmically (log10) binned, redshift AND VELOCITY
                   corrected galaxy wavelength in units of Angstroms
lick_indices     = Dictionary containing all the [index values, 1 sigma
                   uncertainty on index value]
lick_sp_s07      = Dictionary containing the stellar population parameters
                   determined from Lick indices and the Schiavon 2007 models
lick_sp_tmj      = Dictionary containing the stellar population parameters
                   determined from Lick indices and the Thomas, Maraston, and
                   Johansson (2010) models
linlam_gal_rest  = Linearly binned, redshift and velocity corrected galaxy
                   wavelength in units of Angstroms
log_galaxy_final = Logarithmically (log10) binned, normalised, emission line
                   corrected FINAL galaxy spectrum, now ready for stuffs like
                   full spectral fitting.
logflux_gal      = Logarithmically (log10) binned, normalised original galaxy
                   spectrum. Still contains emission and sky lines.
nmodels          = the number of models used in the Lick Indices + models
                   stellar population fitting
pp0              = Results from the first ppxf cleaning iteration
pp1              = Results from (second) and final cleaning ppxf iteration
pp_BPASS         = Results from the full spectral fitting using the BPASS
                   models
pp_MILES         = Results from the full spectral fitting using the
                   Vazdekis-Miles models
sig_gal          = Sigma due to velocity dispersion (determined by pPXF) in
                   units of Angstroms
sig_inst         = Sigma due to instrumental broadening in units of Angstroms
sig_tot          = Total Sigma: sig_tot^2 = sig_gal^2 + sig_inst^2 in units of
                   Angstroms
sp_BPASS         = Dictionary containing the stellar population parameters
                   determined from full spectral fitting and the BPASS models
sp_MILES         = Dictionary containing the stellar population parameters
                   determined from full spectral fitting and the
                   Vazdekis-Miles models
var              = variance spectrum from the fits file
weights_BPASS    = The BPASS template weights from ppxf full spectral fitting
weights_MILES    = The MILES tempalte weights from ppxf full spectral fitting
z                = Galaxy redshift taken from input spectral file (ppxf then
                   calculates a more accurate extraredshift, saved under
                   extra_redshift)
sp_MILES.weighting = indicates whether the templates used in the fit were
                   'light_weighted' or 'mass_weighted'
"""


class Galaxy():

    def __init__(self, input_fn, spec_type, **kwargs):
        """ Here, the spectral data is loaded from file, and all the necessary
            information for ppxf is calculated and saved as attributes. """

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.filename  = input_fn.split('/')[-1]

        # Import the relevant information from the fits file. Specifically:
        #    (1) z           : the (cosmological) redshift,
        #    (2) loglam_gal  : logarithmically(10) binned wavelength in AA
        #    (3) logflux_gal : logarithmically(10) binned flux array
        #    (4) FWHM_inst   : instrumental resolution in FWHM in units of AA
        #    (5) var         : variance array (noise = sqrt(var))

        file_info = sph.get_from_file(input_fn, spec_type, **kwargs)
        self.z, loglam_gal, logflux_gal, FWHM_inst, var = file_info

        # Redshift correct the logarithmically binned wavelength. Also need
        # to correct the instrumental sigma for cosmological redshift
        # (see s2.4 Cappellari 2017).lam_emit = lam_obs/(z+1) in log becomes:
        self.loglam_gal = loglam_gal / (self.z + 1)
        self.FWHM_inst  = FWHM_inst / (self.z + 1)

        # set the velocity scale
        # Normalise the flux (Divide out the median). Must also divide out
        # the same value from the noise (i.e sqrt variance array)
        self.logflux_gal = logflux_gal / np.nanmedian(logflux_gal)
        self.var         = var / np.nanmedian(logflux_gal) ** 2

        # set up variable name to be assigned in clean_spectrum()
        self.clean_spectrum_mask = None

        # =====================================================================
        # =====================================================================

    def clean_spectrum(self, templates, clean_ppxf=True, save_pp0=True,
                       start=[0, 100], mdegree=0, degree=12, info=True,
                       mask_emission=True):
        """ Firstly setup parameters needed for ppxf.
            Then use ppxf and auxillary functions to 'clean' the spectrum. Ie:
              - remove emission lines
              - remove sky lines
              - remove higly discrepant pixels (regions which cant be
                reconstructed using templates)
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get all the needed parameters from set_up_ppxf
        [loglam_gal, var, mask, goodpix, ns_clean, logflux_temp,
         loglam_temp, dv, velscale] = sph.set_up_ppxf(
            loglam_gal=self.loglam_gal, var=self.var, FWHM_inst=self.FWHM_inst,
            z=self.z, models=templates, mask_emission=mask_emission)

        logflux_gal = self.logflux_gal[mask]
        self.clean_spectrum_mask = mask

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if info:
            print('________________ Beginning PPXF analysis ________________')

        # ____________________________________________________________________#
        # ____ The first iteration of ppxf is to determine the noise level ___#
        # ____________________________________________________________________#
        pp0 = ppxf(logflux_temp, logflux_gal, ns_clean, velscale, start,
                   clean=False, moments=2, quiet=True, vsyst=dv,
                   mdegree=mdegree, degree=degree, goodpixels=goodpix,
                   lam=loglam_gal)

        if save_pp0:
            setattr(self, 'pp0', pp0)
            if clean_ppxf:
                sph.clean_ppxf_object(self.pp0)

        if info:
            print('____________ First ppxf iteration complete _______________')

        # Using the chi2 value from pp0 (or more specifically, the square root
        # of the reduced chi2),rescale the noise spectrum to obtain a
        # reduced_chi2 of 1.
        ns_scaled   = ns_clean * math.sqrt(pp0.chi2)

        # ___________________________________________________________________#
        # Second ppxf iteration uses the 1st iterations noise determination  #
        # ___________________________________________________________________#
        self.pp1 = ppxf(logflux_temp, logflux_gal, ns_scaled, velscale,
                        start=pp0.sol, clean=True, moments=2, quiet=True,
                        vsyst=dv, mdegree=mdegree, degree=degree,
                        goodpixels=pp0.goodpixels, lam=loglam_gal)
        if info:
            print('___________ Second ppxf iteration complete _______________')

        # delete from the ppxf all the useless attributes
        if clean_ppxf:
            sph.clean_ppxf_object(self.pp1)

        # ____________________________________________________________________#
        # Use ppxf's bestfitting solution to replace highly discrepent pixels #
        # ____________________________________________________________________#

        # for each pixel what is the normalised difference between the
        # spectrum and the bestfit
        diff = (logflux_gal / np.nanmedian(logflux_gal)
                - self.pp1.bestfit / np.median(self.pp1.bestfit))
        # define limit of acceptable difference = 5* Root-Median-Squared
        diff_val = 5. * math.sqrt(np.nanmedian(diff**2))

        # The 4 conditions for excluding pixels
        # condition 1 is if difference is greater than chosen limit
        # condition 2 is if outside of max goodpix
        # condition 3 is if outside of min goodpix
        # condition 4 ois if skyline at 5557 A (observed)
        cond1 = (np.abs(diff) > diff_val)
        cond2 = (loglam_gal > loglam_gal[self.pp1.goodpixels].max())
        cond3 = (loglam_gal < loglam_gal[self.pp1.goodpixels].min())
        cond4 = ((loglam_gal * (1 + self.z) > 5565)
                 & (loglam_gal * (1 + self.z) < 5590))

        m0 = np.where(cond1 | cond2 | cond3 | cond4)[0]

        # Currently, w doesn't always include the wings of emission lines.
        # Want to also replace ~4-5 pixels on each side of these regions
        # depending on velocity dispersion.

        # Velocity dispersion = self.pp1.sol[1] {km/s}, and the velocity scale
        # per pixel = velscale {km/s/pixel}. Hence self.pp1.sol[1] / velscale
        # gives the broadening in number of pixels. Add an extra 2 pixels for
        # good measure
        npix = math.ceil(self.pp1.sol[1] / velscale) + 2

        # for every pixel in m0, include the neibouring npix pixels, exluding
        # negative pixel indices, and of course only take unique pixel indices
        m = np.unique([i + n for n in range(-npix, npix + 1)
                       for i in m0
                       if i + n >= 0 and i + n < logflux_gal.size])

        # create a galaxy spectrum with these values found above replaced with
        # the bestfit spectrum (normalised to fit into the galaxy spectum
        # properly). note that 'temp' means 'temporary'
        temp = np.copy(logflux_gal)
        if len(m) > 0:
            temp[m] = (self.pp1.bestfit[m] * np.nanmedian(logflux_gal)
                       / np.nanmedian(self.pp1.bestfit))

        # Save the new cleaned, final (log binned) galaxy spectrum
        self.log_galaxy_final = np.copy(temp)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Make this into a new method?
        # convert the galaxy spectrum into linearly binned

        linlam_gal = np.linspace(loglam_gal[0], loglam_gal[-1],
                                 len(loglam_gal))
        self.galaxy_final = np.interp(linlam_gal, loglam_gal,
                                      self.log_galaxy_final)

        # Use the peculiar velocity determined by ppxf to finalise the
        # rest-frame galaxy wavelength. extra redshift equation from
        # Eq. 8 of Cappellari 2017
        self.extra_redshift  = np.exp(self.pp1.sol[0] / c) - 1
        self.lam_gal_rest    = loglam_gal / (self.extra_redshift + 1)
        self.linlam_gal_rest = linlam_gal   / (self.extra_redshift + 1)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        vel_disp             = self.pp1.sol[1].clip(min=0., max=300.)

        # Need to interpolate the logarithmically binned FWHM_inst to a linear
        # scale.
        FWHM_inst_lin        = np.interp(self.linlam_gal_rest, loglam_gal,
                                         self.FWHM_inst[mask])

        # sig_gal  -> Dispersion of the galaxy spectrum (wavelength Angstroms)
        # sig_inst -> Instrumental dispersion in wavelength Angstrom untis
        # sig_tot  -> Total spectral resolution in wavelength Angstrom units
        self.sig_inst        = ((FWHM_inst_lin / math.sqrt(8 * np.log(2)))
                                / (self.extra_redshift + 1))
        self.sig_gal         = (vel_disp / c) * self.linlam_gal_rest
        self.sig_tot         = np.sqrt(self.sig_inst**2 + self.sig_gal**2)

        # =====================================================================
        # =====================================================================

    def measure_lick_indices(self, flux=None, wavelength=None, sigma_tot=None,
                             noise=None, broaden=True, info=True,
                             index_table=None):
        """ ____________________  Extract LICK indices  ____________________"""

        assert hasattr(self, 'galaxy_final') is True, ("clean the spectrum"
                                                       "first!")
        if index_table is None:
            # get the absolute path to this file
            path_to_this_file = os.path.dirname(os.path.abspath(__file__))
            index_table = (path_to_this_file + '/../../data/Lick_and_models/'
                           'Lick_index_definitions.dat')

        indices = ['HdeltaA', 'HdeltaF', 'HgammaA', 'HgammaF', 'Hbeta', 'Mgb',
                   'Fe4383', 'Fe4668', 'Fe5015', 'Fe5270', 'Fe5335', 'CN1',
                   'CN2', 'Ca4227', 'G4300', 'Mg1', 'Mg2', 'Fe4531', 'Fe5406',
                   'Ca4455']

        if flux is None:
            flux = self.galaxy_final
        if wavelength is None:
            wavelength = self.linlam_gal_rest
        if sigma_tot is None:
            sigma_tot = self.sig_tot
        if noise is None:
            noise = self.pp1.noise

        # load in the file containing all the LICK indices information
        index_defs   = ap_ascii.read(index_table)
        index_names  = index_defs['Index']

        # need to extract the required resolution at which each index must be
        # measured to conform with the original LICK system definition
        res = np.zeros(len(indices))
        for i in range(len(indices)):
            res[i] = index_defs[np.where(index_defs['Index']
                                         == indices[i])[0]]['Res'][0]

        def measure_index(j):

            # need the passband for the specific index, in order to know the
            # resolution at that part of the spectrum (have an array of
            # resolutions)
            row = np.where(indices[j] == index_names)[0]

            # Firstly, check that both the index passband and the
            # pseudo-continuum bands are fully in the spectrum. Else, the
            # index cannot be measured.
            ind_range = [index_defs[row]['BC_lower'][0],
                         index_defs[row]['RC_upper'][0]]

            if (self.linlam_gal_rest[0] > ind_range[0]
                    or self.linlam_gal_rest[-1] < ind_range[-1]):
                if info:
                    print(f"{indices[j]} couldn't be measured: outside "
                          "spectral range")
                return [np.nan, np.nan]

            # find the array indices that are relevent to this particular LICK
            # index, i.e. everything between [I_lower, I_upper]
            relevant_inds  = np.where(
                (self.linlam_gal_rest > index_defs[row]['I_lower'][0])
                & (self.linlam_gal_rest < index_defs[row]['I_upper'][0]))[0]
            # relavent total sigma value
            sig_relevant   = np.nanmedian(self.sig_tot[relevant_inds])

            # the defined sigma of the LICK index
            sig_ind   = res[j] / math.sqrt(8 * np.log(2))
            # the difference in wavelength between pixels
            dlambda   = wavelength[1] - wavelength[0]

            # if the spectrum broadening is better than the index broadening,
            # need to broaden the spectrum to the defined index resolution
            if broaden is True:
                if sig_relevant < sig_ind:
                    # determine how much to broaden by
                    broadening = np.sqrt(
                        (sig_ind**2 - sig_relevant**2).clip(0)) / dlambda
                    # broaden spectrum
                    lick_res_spectrum = pu.gaussian_filter1d(flux, broadening)
                    # broaden noise
                    lick_res_var = pu.gaussian_filter1d(noise**2, broadening)
                    index_j = sph.measure_ind(wavelength, lick_res_spectrum,
                                              indices[j], index_defs)
                    err_j   = sph.mc_ind_errors(wavelength, lick_res_spectrum,
                                                lick_res_var, indices[j],
                                                index_defs)

                # else if the spectrum resolution is worse than index
                # resolution, need to correct the lick indices for this extra
                # dispersion
                elif sig_relevant >= sig_ind:
                    if info:
                        print("spectrum resolution worse than index"
                              "resolution")
                    # vdisp is amount of dispersion that needs to be corrected
                    # for (in km/s). NOTE:
                    # sigma [km/s] =  sigma [A] * c [km/s] / wavelength [A]
                    vdisp_squish = (np.sqrt(sig_relevant**2 - sig_ind**2) * c
                                    / np.median(
                                        self.linlam_gal_rest[relevant_inds]))
                    broadening        = np.nan
                    lick_res_spectrum = flux
                    lick_res_var      = noise**2
                    width   = sph.measure_ind(wavelength, lick_res_spectrum,
                                              indices[j], index_defs)
                    err_j   = sph.mc_ind_errors(wavelength, lick_res_spectrum,
                                                lick_res_var, indices[j],
                                                index_defs)
                    # Now correct width calculated above for exessive velocity
                    # dispersion broadening
                    index_j = sph.vdisp_correct(vdisp_squish, width,
                                                indices[j])

            elif broaden is False:
                broadening        = 0
                lick_res_spectrum = np.copy(flux)
                lick_res_var      = noise**2
                index_j = sph.measure_ind(wavelength, lick_res_spectrum,
                                          indices[j], index_defs)
                err_j   = sph.mc_ind_errors(wavelength, lick_res_spectrum,
                                            lick_res_var, indices[j],
                                            index_defs)

            return [index_j, err_j]

        results  = np.array([measure_index(i) for i in range(len(indices))])

        self.lick_indices = dict([(indices[i], [results[i, 0], results[i, 1]])
                                  for i in range(len(indices))])

        # =====================================================================
        # =====================================================================

    def measure_sp_parameters(self, model='tmj', n_models='all',
                              model_file=None, params=None):
        """ Using the measured Lick indices, compare to the models by Schiavon
            (2007), or Thomas, Maraston, Johansson (2011). NOTE files are
            originally from Nic Scott and have been interpolated to a finer
            scale than the standard model. """

        assert hasattr(self, 'lick_indices') is True, ("you havent measured "
                                                       "lick indices yet")

        # define the path to this file so can find relative data dir
        path_to_this_file = os.path.dirname(os.path.abspath(__file__))
        data_dir = path_to_this_file + '/../../data/Lick_and_models/'

        if model == 's07' and model_file is None:
            model_file = (data_dir
                          + 'alpha_models_ti_schiavon_reinterpolated_'
                          + 'log_age_nolick.fits')
            params = ['age', 'Z_H', 'alpha_Fe']

        if model == 'tmj' and model_file is None:
            model_file = (data_dir
                          + 'TMB_TMK_interpolated_alpha_model.fits')
            params = ['age', 'Z_H', 'alpha_Fe']

        if model == 'miles' and model_file is None:
            model_file = (data_dir
                          + 'Miles_Indicies_BaSTI_Chabrier_interpolated.fits')
            params = ['age', 'M_H']  # no alpha/Fe for these models

        # Read in the table of models
        with fits.open(model_file, memmap=False) as file:
            if type(n_models) == int:
                ssp_table = Table(file[1].data)[:n_models]
            elif n_models   == 'all':
                ssp_table = Table(file[1].data)

        model_indices = ssp_table.colnames[3:]
        self.n_models = len(ssp_table)

        # find the indices common between what was measured, and what is in
        # the model table available indicies
        av_indices = list(np.intersect1d(model_indices, [*self.lick_indices]))

        # Define a function to calculate the chi2 for each index and model
        def chi2_val(index, extra_args):
            n_models, index_dict, ssp_table = extra_args
            chi2_arr = np.zeros(n_models)
            for j in range(n_models):
                # calculate the chi2 value normalised by the uncertanty on the
                # index measure
                chi2_arr[j] = ((index_dict[index][0] - ssp_table[index][j])
                               / index_dict[index][1])**2
            return chi2_arr

        extra_args = (len(ssp_table), self.lick_indices, ssp_table)
        chi2_2Darr = np.array([chi2_val(ind, extra_args)
                               for ind in av_indices])

        # sum the chi2 values to get a single value for each model,
        # which then gives the bestfit SP parameters.
        chi2 = np.nansum(chi2_2Darr, axis=0)
        w    = np.where(chi2 == np.min(chi2))
        v    = np.where(chi2 < np.min(chi2) + 3.5)

        # Add the values to the table
        if params is None:
            params  = ['age', 'Z_H', 'alpha_Fe']

        sp_dict = dict([(p, [ssp_table[p][w][0], np.max(ssp_table[p][v]),
                             np.min(ssp_table[p][v])]) for p in params])
        setattr(self, 'lick_sp_' + model, sp_dict)

        # =====================================================================
        # =====================================================================

    def full_spectral_fitting(
            self, models, gas=False, tag='', regul=0, reg_ord=1, mdegree=10,
            degree=-1, clean_ppxf=False, rescale_noise=True,
            use_cleaned_spectrum=False, tie_balmer=True, limit_doublets=True,
            stellar_pops=True, **kwargs):

        """ A single method that can handle fitting either with masked
            emission lines or with fitting emission lines
        """
        # There are two options: can either use the spectrum cleaned using
        # clean_spectrum, or can use the spectrum as is.

        if use_cleaned_spectrum:
            logflux_gal = self.log_galaxy_final
            loglam_gal  = self.loglam_gal[self.clean_spectrum_mask]
            var         = self.var[self.clean_spectrum_mask]
            FWHM_inst   = self.FWHM_inst[self.clean_spectrum_mask]

        else:
            logflux_gal = self.logflux_gal
            loglam_gal  = self.loglam_gal
            var         = self.var
            FWHM_inst   = self.FWHM_inst

        # Get all the needed parameters from set_up_ppxf
        [loglam_gal, var, mask, goodpix, ns_clean, logflux_temp,
         loglam_temp, dv, velscale] = sph.set_up_ppxf(
            loglam_gal=loglam_gal, var=var, FWHM_inst=FWHM_inst, z=self.z,
            models=models, mask_emission=not gas)

        # check to see whether clean_spectrum() has been run. If not, run it
        if hasattr(self, 'pp0') is False:
            print("You haven't run clean_spectrum yet. Doing it now.....")
            self.clean_spectrum(models)
            print("Finished clean_spectrum")

        if rescale_noise is True:
            ns_clean = ns_clean * math.sqrt(self.pp0.chi2)

        # the demension of the star templates (in case of regularisation, which
        # is only preformed over the stellar template components, not the
        # emission templates)
        reg_dim = logflux_temp.shape[1:]

        # The total number of stellar templates
        n_temps = np.prod(logflux_temp.shape[1:])

        # reshape the star templates for resasons beyond my recollection
        logflux_temp = logflux_temp.reshape(logflux_temp.shape[0], -1)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #           Assign method dependent (w / wo emisison) variables
        if gas:
            gas_templates, gas_names, line_wave = pu.emission_lines(
                ln_lam_temp=np.log(loglam_temp),
                lam_range_gal=[loglam_gal[0], loglam_gal[-1]],
                FWHM_gal=np.median(FWHM_inst[mask]),
                tie_balmer=tie_balmer, limit_doublets=limit_doublets)

            logflux_temp  = np.column_stack([logflux_temp, gas_templates])

            # balmer line names always start with an 'H', so count these
            # forbidden lines contain "[*]"
            n_forbidden   = np.sum(["[" in a for a in gas_names])
            n_balmer      = len(gas_names) - n_forbidden
            #  component  = [0] * n_temps + [1] * n_balmer + [2] * n_forbidden
            component = [0] * n_temps + list(range(1, n_balmer + 1)) + \
                list(range(n_balmer + 1, n_balmer + n_forbidden + 1))
            gas_component = np.array(component) > 0  # gas_comp Tru 4 gas temps
            n_components  = len(set(component))  # num differnt components
            start         = [self.pp1.sol for _ in range(n_components)]
            moments       = [2 for _ in range(n_components)]
            gas_reddening = kwargs.pop('gas_reddening', 0)

        else:
            gas_names     = None
            component     = [0] * n_temps
            gas_component = None
            start         = self.pp1.sol
            moments       = 2
            gas_reddening = kwargs.pop('gas_reddening', None)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #                                Run pPXF
        pp = ppxf(logflux_temp, logflux_gal[mask], ns_clean, velscale,
                  start=start, lam=loglam_gal, goodpixels=goodpix,
                  degree=degree, mdegree=mdegree, vsyst=dv, reg_dim=reg_dim,
                  regul=regul, reg_ord=reg_ord, moments=moments,
                  component=component, gas_component=gas_component,
                  gas_names=gas_names, gas_reddening=gas_reddening, **kwargs)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #                              Save attributes

        # delete from the ppxf all the useless attributes
        if clean_ppxf:
            sph.clean_ppxf_object(pp)

        # Reshape the weights array from a 1D array of length
        # (n_ages * n_metals * n_order) to a normalised 2D array of shape
        # (n_ages, n_metals, [n_order]), and exclude weights of gas templates

        # if the tag doesn't start with an underscore, insert one
        if len(tag) > 0 and tag[0] != '_':
            tag = '_' + tag

        weights = pp.weights[~pp.gas_component]
        weights = weights.reshape(pp.reg_dim) / weights.sum()

        if stellar_pops:
            sp      = sph.mean_age_metal(weights=weights, models=models)
            sp_dict = {'age': sp[0], '[Z/H]': sp[1]}
            setattr(self, 'sp' + tag, sp_dict)
            setattr(self, 'weights' + tag, weights)

        setattr(self, 'pp' + tag, pp)
        setattr(getattr(self, 'pp' + tag),
                'weighting', models.weighting)

        # =====================================================================
        # =====================================================================

    def save(self, output_dir='', output_fn=None):
        """ A method to save all the stuffs """

        if output_fn is None:
            output_fn = self.filename.split('.')[0] + '.pckl'

        # ensure directory name correct
        if output_dir != '':
            if output_dir[-1] != '/':
                output_dir += '/'  # if last symbol not a slash, add one
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)

        with open(output_dir + output_fn, 'wb') as file:
            pickle.dump(self, file)

        # return the full output path + filename
        return output_dir + output_fn
        # =====================================================================
        # =====================================================================

    @classmethod
    def load(cls, pckl_fname):
        with open(pckl_fname, 'rb') as file:
            obj = pickle.load(file)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__} instance")
        return obj


"""
Functions for loading 1D galaxy spectra from various survey formats into the
standard (z, loglam_gal, logflux_gal, FWHM_inst, logvar) tuple expected by
Galaxy.__init__(). Most of these were written for specific projects, but in
general the get_from_file_generic_txt function can be used for any spectrum
with minimal preparation.

All loaders return:
    z           : cosmological redshift
    loglam_gal  : log-rebinned wavelength array in Angstroms
    logflux_gal : log-rebinned flux array
    FWHM_inst   : instrumental resolution as FWHM in Angstroms
    logvar      : log-rebinned variance array (noise = sqrt(logvar))

Loaders
-------
get_from_file_generic_txt       : generic 5-column ASCII file (ALF format)
JWST_2d_get_from_file           : JWST NIRSpec 2D IFU cube
MOSFIRE_get_from_file           : Keck/MOSFIRE (5-column ASCII, Y or J band)
GTC_OSIRIS_get_from_file        : GTC/OSIRIS 1D reduced spectrum (Pypeit)
GTC_OSIRIS_get_from_file_spec2d : GTC/OSIRIS 2D Pypeit output
KCWI_get_from_file              : Keck/KCWI IFU cube with aperture extraction
XSHOOTER_get_from_file          : VLT/X-Shooter 1D spectrum
SAMI_get_from_file              : SAMI aperture spectrum (blue+red combined)
SDSS_get_from_file              : SDSS Legacy/BOSS coadded spectrum
LEGAC_get_from_file             : LEGA-C 1D slit spectrum
MAGPI_get_from_file             : MAGPI/MUSE 1D spectrum
"""

import numpy as np
import ppxf.ppxf_util as ppxf_util
import astropy.wcs as wcs
import warnings
import scipy.ndimage
from . import sp_helper_functions as sph
from astropy.table import Table
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.constants import c


def get_from_file_generic_txt(input_fn, z_guess=0.0, R=None, vacuum=False):
    # this function is to read in any spectral data which has been prepared in
    # the following ascii file format:
    # wave [angstroms], flux [erg/s/cm2/AA], noise [erg/s/cm2/AA], weight, inst_res [sigma km/s].
    # Note that the flux and noise should be normalised so np.median(flux) ~ 1.
    tbl = Table.read(
        input_fn, names=["wave", "flux", "uncert", "weight", "inst_res"], format="ascii"
    )
    linflux_gal = tbl["flux"]
    linvar_gal = tbl["uncert"] ** 2
    linlam_gal = tbl["wave"]

    # logarithmically bin the data
    logflux_gal, loglam_gal, velscale = ppxf_util.log_rebin(lam=linlam_gal, spec=linflux_gal)

    # logaritmically bin the variance
    logvar = ppxf_util.log_rebin(lam=linlam_gal, spec=linvar_gal)[0]
    loglam_gal = np.exp(loglam_gal)

    # if the spectrum is in VACUUM wavelengths, need to convert this to air
    if vacuum is True:
        loglam_gal = sph.vacuum_to_air(loglam_gal)

    if R is None:
        # first need to logarithmically rebin the tbl['inst_res']
        sigma_inst_kms = tbl["inst_res"]
        logsigma_inst_kms = ppxf_util.log_rebin(lam=linlam_gal, spec=sigma_inst_kms)[0]
        R = c.to("km/s").value / (logsigma_inst_kms * 2 * np.sqrt(2 * np.log(2)))
    FWHM_inst = loglam_gal / R

    return (z_guess, loglam_gal, logflux_gal, FWHM_inst, logvar)


# ____________________________________________________________________________#


def JWST_2d_get_from_file(input_fn, x_pix=31, y_pix=34, z_guess=1.1):

    with fits.open(input_fn) as hdul:
        linflux_gal = hdul["SCI"].data[:, x_pix, y_pix]
        linvar_gal = (hdul["ERR"].data[:, x_pix, y_pix]) ** 2

        h = hdul["SCI"].header
        # convert the wavelength from micron to angstroms,
        # and convert to rest frame
        linlam_gal = h["CRVAL3"] + h["CDELT3"] * np.arange(h["NAXIS3"])
        linlam_gal = linlam_gal * 1e4

    # The flux and wavelength come linearly binned in units of Angstroms.
    # Need to logarithmically rebin.
    # Note however that log_rebin returns the wavelength array in units of
    # ln[AA], so need to to np.exp(loglam_gal) to get in units of AA
    logflux_gal, loglam_gal, velscale = ppxf_util.log_rebin(
        lam=[linlam_gal.min(), linlam_gal.max()], spec=linflux_gal
    )

    logvar = ppxf_util.log_rebin(lam=[linlam_gal.min(), linlam_gal.max()], spec=linvar_gal)[0]

    loglam_gal = np.exp(loglam_gal)

    # pPXF requires no NaNs in the flux array
    nan_mask = np.isnan(logflux_gal)
    logflux_gal[nan_mask] = 0.0
    logvar[nan_mask] = 0.0

    R = 1000
    FWHM_inst = loglam_gal / R

    return (z_guess, loglam_gal, logflux_gal, FWHM_inst, logvar)


# ____________________________________________________________________________#


def MOSFIRE_get_from_file(input_fn, band, z_guess=0):

    # inst res based on:
    # https://www2.keck.hawaii.edu/inst/mosfire/grating.html
    if band == "Y":
        R = 3388
    elif band == "J":
        R = 3318
    else:
        raise ValueError(f"band must be 'Y' or 'J', got {band!r}")

    # this function is to read in MOSFIRE data for a specific project. The data
    # is held in ascii files.
    tbl = Table.read(
        input_fn, names=["wave", "flux", "uncert", "weight", "inst_res"], format="ascii"
    )
    linflux_gal = tbl["flux"]
    linvar_gal = tbl["uncert"] ** 2
    linlam_gal = tbl["wave"]

    # logarithmically bin the data
    logflux_gal, loglam_gal, velscale = ppxf_util.log_rebin(lam=linlam_gal, spec=linflux_gal)
    # logaritmically bin the variance
    logvar = ppxf_util.log_rebin(lam=linlam_gal, spec=linvar_gal)[0]

    loglam_gal = np.exp(loglam_gal)
    FWHM_inst = loglam_gal / R

    return (z_guess, loglam_gal, logflux_gal, FWHM_inst, logvar)


# ____________________________________________________________________________#


def GTC_OSIRIS_get_from_file(input_fn, z_guess=0.722, grating="R2500R"):
    with fits.open(input_fn) as hdul:
        data_table = Table(hdul[1].data)

    # remove the first 3 and last 1 element from the table because they are bad
    data_table.remove_rows([0, 1, 2, -1])

    linflux_gal = data_table["flux"]
    linlam_gal = data_table["wave"]
    lin_ivar = data_table["ivar"]
    linvar = 1 / lin_ivar

    # The data is really messily binned, plus needs to be in log scale.
    logflux_gal, loglam_gal, velscale = ppxf_util.log_rebin(lam=linlam_gal, spec=linflux_gal)

    logvar = ppxf_util.log_rebin(lam=linlam_gal, spec=linvar, velscale=velscale)[0]

    # instrumental resolution taken from
    # http://www.gtc.iac.es/instruments/osiris/
    if grating == "R2500R":
        R = 2475
    elif grating == "R2000B":
        R = 2165
    else:
        raise ValueError(f"grating must be 'R2500R' or 'R2000B', got {grating!r}")

    loglam_gal = np.exp(loglam_gal)
    FWHM_inst = loglam_gal / R

    # The Pypeit pipeline puts wavelengths in vacuum, but the MILES ones are in
    # air. So convert
    loglam_gal = sph.vacuum_to_air(loglam_gal)

    return (z_guess, loglam_gal, logflux_gal, FWHM_inst, logvar)


# ____________________________________________________________________________#


def GTC_OSIRIS_get_from_file_spec2d(
    input_fn, z_guess=0.722, grating="R2500R", spaxel=225, detector=2
):
    with fits.open(input_fn) as hdul:
        linflux_gal = (hdul[f"DET0{detector}-SCIIMG"].data[:, spaxel]
                       - hdul[f"DET0{detector}-SKYMODEL"].data[:, spaxel])
        linlam_gal = hdul[f"DET0{detector}-WAVEIMG"].data[:, spaxel]
        lin_ivar = hdul[f"DET0{detector}-IVARRAW"].data[:, spaxel]

    linvar = 1 / lin_ivar

    # The data is really messily binned, plus needs to be in log scale.
    logflux_gal, loglam_gal, velscale = ppxf_util.log_rebin(lam=linlam_gal, spec=linflux_gal)

    logvar = ppxf_util.log_rebin(lam=linlam_gal, spec=linvar)[0]

    # instrumental resolution taken from
    # http://www.gtc.iac.es/instruments/osiris/
    if grating == "R2500R":
        R = 2475
    elif grating == "R2000B":
        R = 2165
    else:
        raise ValueError(f"grating must be 'R2500R' or 'R2000B', got {grating!r}")

    loglam_gal = np.exp(loglam_gal)
    FWHM_inst = loglam_gal / R

    # The Pypeit pipeline puts wavelengths in vacuum, but the MILES ones are in
    # air. So convert
    loglam_gal = sph.vacuum_to_air(loglam_gal)

    return (z_guess, loglam_gal, logflux_gal, FWHM_inst, logvar)


# ____________________________________________________________________________#


def KCWI_get_from_file(input_fn, x_pix=31, y_pix=34, r=1, z_guess=0.722, wave_min_obs=None):
    with fits.open(input_fn) as hdu:
        data = hdu[0].data
        h = hdu[0].header
        linlam_gal = h["CRVAL3"] + h["CDELT3"] * np.arange(h["NAXIS3"])
    if wave_min_obs is None:
        range_of_interest = np.arange(len(linlam_gal))

    else:
        # can choose to e.g. take everything above the lyman alpha line
        range_of_interest = np.where((linlam_gal >= wave_min_obs))[0]
        linlam_gal = linlam_gal[range_of_interest]

    with fits.open(input_fn.replace(".fits", "_var.fits")) as hdu:
        var = hdu[0].data

    source_xy = np.array([x_pix, y_pix]) - 1

    # sum the flux in a circular aperture around that point
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[2])

    mask = (y[np.newaxis, :] - source_xy[0]) ** 2 + (x[:, np.newaxis] - source_xy[1]) ** 2 < r**2

    linflux_gal = np.sum(data[:, mask], axis=1)[range_of_interest]
    linvar_gal = np.sum(var[:, mask], axis=1)[range_of_interest]

    # The flux and wavelength come linearly binned in units of Angstroms.
    # Need to logarithmically rebin.
    # Note however that log_rebin returns the wavelength array in units of
    # ln[AA], so need to to np.exp(loglam_gal) to get in units of AA
    logflux_gal, loglam_gal, velscale = ppxf_util.log_rebin(
        lam_range=[linlam_gal.min(), linlam_gal.max()], spec=linflux_gal
    )
    logvar = ppxf_util.log_rebin(
        lam_range=[linlam_gal.min(), linlam_gal.max()],
        spec=linvar_gal,
        velscale=velscale,
    )[0]

    loglam_gal = np.exp(loglam_gal)

    # pPXF requires no NaNs in the flux array
    nan_mask = np.isnan(logflux_gal)
    logflux_gal[nan_mask] = 0.0
    logvar[nan_mask] = 0.0

    # instrumental resolution R=1800
    R = 1800
    FWHM_inst = loglam_gal / R

    return (z_guess, loglam_gal, logflux_gal, FWHM_inst, logvar)


# ____________________________________________________________________________#


def XSHOOTER_get_from_file(input_fn, z_guess=0.49):
    with fits.open(input_fn) as hdu:
        h = hdu[1].header
        linflux_gal = hdu[1].data
        linlam_gal = (h["CRVAL1"] + h["CDELT1"] * np.arange(h["NAXIS1"])) * 1e4
        linvar = hdu[4].data

    # The flux and wavelength come linearly binned in units of Angstroms.
    # Need to logarithmically rebin.
    # Note however that log_rebin returns the wavelength array in units of
    # ln[AA], so need to to np.exp(loglam_gal) to get in units of AA
    logflux_gal, loglam_gal, velscale = ppxf_util.log_rebin(
        lam_range=[linlam_gal.min(), linlam_gal.max()], spec=linflux_gal
    )
    logvar = ppxf_util.log_rebin(
        lam_range=[linlam_gal.min(), linlam_gal.max()], spec=linvar, velscale=velscale
    )[0]

    loglam_gal = np.exp(loglam_gal)

    # pPXF requires no NaNs in the flux array
    nan_mask = np.isnan(logflux_gal)
    logflux_gal[nan_mask] = 0.0
    logvar[nan_mask] = 0.0

    R = 8900  # VIS arm spectral resolution (Vernet et al. 2011, A&A, 536, A105)
    FWHM_inst = loglam_gal / R

    return (z_guess, loglam_gal, logflux_gal, FWHM_inst, logvar)


# ____________________________________________________________________________#


def SAMI_get_from_file(input_fn, table_name, extension="RE_MGE"):
    # want the RE_MGE extension

    with fits.open(input_fn) as hdu:
        h = hdu[extension].header
        linflux_gal = hdu[extension].data
        linlam_gal = (
            np.arange(
                start=h["CRVAL1"],
                stop=h["CRVAL1"] + h["CDELT1"] * h["NAXIS1"],
                step=h["CDELT1"],
            )
            - h["CRPIX1"] * h["CDELT1"]
        )
        linvar = hdu[extension + "_VAR"].data

    # The flux and wavelength come linearly binned in units of Angstroms.
    # Need to logarithmically rebin.
    # Note however that log_rebin returns the wavelength array in units of
    # ln[AA], so need to to np.exp(loglam_gal) to get in units of AA
    logflux_gal, loglam_gal, velscale = ppxf_util.log_rebin(
        lam_range=[linlam_gal.min(), linlam_gal.max()], spec=linflux_gal
    )
    logvar = ppxf_util.log_rebin(
        lam_range=[linlam_gal.min(), linlam_gal.max()], spec=linvar, velscale=velscale
    )[0]

    loglam_gal = np.exp(loglam_gal)
    # Get redshift from table
    filename = input_fn.split("/")[-1]
    d = Table.read(table_name)
    fname_split = filename.split("_", 3)
    cubeid = f"{fname_split[0]}_{fname_split[3].replace('_apspec.fits', '')}"
    ind = np.where(d["CUBEID"] == cubeid)[0][0]
    z = d["z_spec"][ind]

    # pPXF requires no NaNs in the flux array
    nan_mask = np.isnan(logflux_gal)
    logflux_gal[nan_mask] = 0.0
    logvar[nan_mask] = 0.0

    # Jesse's blue-red combined aperture spectra degrades the red arm to be the
    # same resolution as the blue end. Therefore the specta now have a constant
    # R = 1730 (Croom et al. 2012). R is defined as Lam / dLam; therefore
    # Lam / dLam is constant. Also dLam, by the definition of R, is expressed
    # as the FWHM in units of A, which is exactly what we need. Therefore do:
    R = 1730  # spectral resolution (Croom et al. 2012)
    FWHM_inst = loglam_gal / R

    return (z, loglam_gal, logflux_gal, FWHM_inst, logvar)


# ____________________________________________________________________________#


def SDSS_get_from_file(input_fn):
    with fits.open(input_fn, memmap=False) as hdu:
        hdu_names = [hdu_name[1] for hdu_name in hdu.info(output="")]
        if "SPECOBJ" in hdu_names:
            z = hdu["SPECOBJ"].data["Z"][0]
        else:
            # If no SPECOBJ hdu, it means the input spectrum is from the
            # BOSS spectrograph not the original LEGACY spectrograph, so the
            # redshift is stored instead in an extention called "SPALL"
            z = hdu["SPALL"].data["Z"][0]

        loglam_gal = 10 ** hdu["COADD"].data["loglam"]
        logflux_gal = hdu["COADD"].data["flux"]
        # wdisp is the standard deviation (Wavelength dispersion i.e
        # sigma of fitted Gaussian) in units of number of pixel per pixel.
        # Will need to convert to FWHM in units of Angstroms.
        wdisp = hdu["COADD"].data["wdisp"]

        # Use context manager to momentarily ignore RuntimeWarnings when
        # dividing by zero
        with np.errstate(divide="ignore"):
            var = 1 / hdu["COADD"].data["ivar"]

    # SDSS wavelengths are in vacuum, but the MILES ones are in air. So convert
    loglam_gal = sph.vacuum_to_air(loglam_gal)

    # Want to know what the size of every pixel (in Angstroms) would be if the
    # spectrum were *linearly* binned. To convolve with the linearly binned
    # templates before they are log binbed. But the spectrum is currently
    # logarithmically binned. SO:
    # using the Equation dLam / Lam = dV / c = constant when logarithmically
    # binned. (if linearly binned, then dLam = constant so dLam / Lam not
    # constant)
    # I want to find dLam (here called dlam_gal).
    # dLam / Lam = constant = ((loglam1 - loglam0) / loglam0)
    # Therefore dLam = ((loglam1 - loglam0) / loglam0) * loglam
    # Constant lambda fraction per pixel so just get the fist
    dlam_gal = (loglam_gal[1] / loglam_gal[0] - 1) * loglam_gal

    # The wdisp is wavelength dispersion in units of pixels. Sigma in AA units:
    # sigma = wdisp * dlam_gal, where dlam_gal is size of each pixel in AA.
    # So Observed instrumental dispersion in units of AA:
    sigma_inst = wdisp * dlam_gal

    # Assume the instrumental line spread function is gaussian, so
    # FWHM = sigma * sqrt(8 * ln2). is the FWHM in units of Angstroms.
    FWHM_inst = sigma_inst * np.sqrt(8 * np.log(2))

    # If the spectra are from BOSS, the (log) wavelength spacing isn't quite
    # exact, so rebin so that it is on a more precise wavelength scale.
    # This has recently also become a problem for LEGACY spectra so do it for
    # both. FWHM_inst is rebinned here too so all returned arrays share the
    # same wavelength grid.
    logflux_gal_rbin, loglam_gal_rbin, velscale = ppxf_util.log_rebin(
        lam=loglam_gal, spec=logflux_gal
    )
    var_rbin = ppxf_util.log_rebin(lam=loglam_gal, spec=var)[0]
    FWHM_inst_rbin = ppxf_util.log_rebin(lam=loglam_gal, spec=FWHM_inst)[0]

    loglam_gal = np.exp(loglam_gal_rbin)
    logflux_gal = logflux_gal_rbin
    var = var_rbin
    FWHM_inst = FWHM_inst_rbin

    return (z, loglam_gal, logflux_gal, FWHM_inst, var)


# ____________________________________________________________________________#


def LEGAC_get_from_file(input_fn, table_name):
    # LEGA-C Spectra are already in AIR wavelengths, so do not do the
    # converstion from vacuum to air that was needed for SDSS spectra.

    with fits.open(input_fn, memmap=False) as f:
        linflux_gal = f[0].data

        # Use scipy's WCS function which takes as input the fits header
        # and uses that to extract the (linearly binned) wavelength
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)  # silence warning
            _wcs_ = wcs.WCS(f[0].header)
            linlam_gal = np.squeeze(_wcs_.all_pix2world(np.arange(_wcs_._naxis[0]), 0, 0)[0])
            # delete the _wcs_ object
            del _wcs_

    # Open the table to get the redshift
    d = Table.read(table_name)

    filename = input_fn.split("/")[-1]
    mask_ID = int(filename.split("_")[1].replace("M", ""))
    gal_ID = int(filename.split("_")[4].replace(".fits", ""))
    ind = np.where((d["id"] == gal_ID) & (d["mask"] == mask_ID))[0][0]
    # Get the redshift from the table
    z = d["z_spec"][ind]

    # The variance is stored in a separate file. Use context manager to
    # momentarily ignore RuntimeWarnings when dividing by zero or the warnings
    # will drive you mad
    with np.errstate(divide="ignore"):
        with fits.open(input_fn.replace("spec1d", "wht1d")) as f:
            linvar = 1 / f[0].data

    # Get rid of the invalid pixels (zeros and nans)
    valid_pixels = _get_valid_slit_spectral_pixels(linflux_gal)
    linflux_gal = linflux_gal[valid_pixels]
    linlam_gal = linlam_gal[valid_pixels]
    linvar = linvar[valid_pixels]

    # The flux and wavelength come linearly binned in units of Angstroms.
    # Need to logarithmically rebin.
    logflux_gal, loglam_gal, velscale = ppxf_util.log_rebin(
        lam_range=[linlam_gal.min(), linlam_gal.max()], spec=linflux_gal
    )

    logvar = ppxf_util.log_rebin(lam_range=[linlam_gal.min(), linlam_gal.max()], spec=linvar)[0]

    # Note that log_rebin returns the wavelength array in units of ln[AA],
    # so need to to np.exp(loglam_gal) to get in units of AA
    loglam_gal = np.exp(loglam_gal)

    # Use the spectral resolution to convert to a FWHM in units of Angstroms
    R = 3500  # spectral resolution (Straatman + 2018 section 3)
    FWHM_inst = loglam_gal / R

    return (z, loglam_gal, logflux_gal, FWHM_inst, logvar)


# ____________________________________________________________________________#


def MAGPI_get_from_file(input_fn, table_name):
    with fits.open(input_fn, memmap=False) as f:
        filename = input_fn.split("/")[-1]
        parts = filename.replace("magpi", "").split("_")[0::2]
        spec_name = f"{parts[0]}_{parts[1]}"

        d = Table.read(table_name)
        ind = np.where(d["spec_name"] == spec_name)[0][0]
        h = f[1].header

        linflux_gal = f[1].data
        linlam_gal = np.array([h["CRVAL1"] + i * h["CDELT1"] for i in range(h["NAXIS1"])])
        linvar = f[2].data

        # The flux and wavelength come linearly binned in units of Angstroms.
        # Need to logarithmically rebin.
        # Note however that log_rebin returns the wavelength array in units of
        # ln[AA], so need to to np.exp(loglam_gal) to get in units of AA
        logflux_gal, loglam_gal, velscale = ppxf_util.log_rebin(
            lam_range=[linlam_gal.min(), linlam_gal.max()], spec=linflux_gal
        )
        logvar = ppxf_util.log_rebin(
            lam_range=[linlam_gal.min(), linlam_gal.max()],
            spec=linvar,
            velscale=velscale,
        )[0]

        loglam_gal = np.exp(loglam_gal)
        # MUSE WFM-AO (GALACSI) spectral LSF: AO corrects spatial PSF only,
        # so spectral resolution matches WFM-NOAO. Quadratic parameterisation
        # from Bacon et al. 2017 (A&A, 608, A1); λ in Å, FWHM in Å.
        FWHM_inst = 5.866e-8 * loglam_gal**2 - 9.187e-4 * loglam_gal + 6.040
        z = d["Z"][ind]

    return (z, loglam_gal, logflux_gal, FWHM_inst, logvar)


# ____________________________________________________________________________#


def _get_valid_slit_spectral_pixels(slit):
    # LEGA-C spectra may often have zeros at the beginning/end. This is to
    # remove them (written by Francesco) :
    _oned_spec = slit
    _oned_spec_mask = np.isclose(_oned_spec, 0) | (_oned_spec < 0) | (~np.isfinite(_oned_spec))

    labels, n_interv = scipy.ndimage.label(_oned_spec_mask)

    start_ind = -1 if labels[0] == 0 else labels[0]
    end_ind = -1 if labels[-1] == 0 else labels[-1]

    valid_pixs = np.arange(len(_oned_spec), dtype=int)
    valid_pixs = valid_pixs[(labels != start_ind) & (labels != end_ind)]

    return valid_pixs


# ____________________________________________________________________________#

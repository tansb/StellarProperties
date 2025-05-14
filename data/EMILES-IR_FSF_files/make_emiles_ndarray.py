import numpy as np
import glob
from os import path
from astropy.io import fits

""" Extract the E-MILES spectra into a ndarray to make it faster to import """


def get_age_metal(filename):
    """Extract age and metallicity from filename"""
    s = path.basename(filename)
    t = s.find("T")
    metal = s[s.find("Z") + 1 : t]
    age = float(s[t + 1 : t + 8])
    if "m" in metal:
        metal = -float(metal[1:])
    elif "p" in metal:
        metal = float(metal[1:])
    else:
        raise ValueError("This is not a standard E-MILES filename")
    return (age, metal)


# 636 filesnames, each file a different spectrum
filenames = np.array(glob.glob("EMILES_BaSTI_bi_baseFe_infrared/Ebi*.fits"))
all_age_met = [get_age_metal(fname) for fname in filenames]
all_ages, all_mets = np.array(all_age_met).T

unique_ages = np.unique(all_ages)
unique_mets = np.unique(all_mets)
n_ages = len(unique_ages)
n_mets = len(unique_mets)

# 11112 = npix in each spectrum
star_temps = np.zeros((11112, n_ages, n_mets))
age_grid1 = np.zeros((n_ages, n_mets))
met_grid1 = np.zeros((n_ages, n_mets))

age_grid = np.array([unique_ages for _ in range(n_mets)]).T  # Gyrs
met_grid = np.array([unique_mets for _ in range(n_ages)])  # [Z/H]

for a, age in enumerate(unique_ages):
    for m, metal in enumerate(unique_mets):
        ind = all_age_met.index((age, metal))
        spec = fits.open(filenames[ind])[0].data
        star_temps[:, a, m] = spec

# Get the wavelength
h = fits.open(filenames[ind])[0].header
wave = np.array([h["CRVAL1"] + i * h["CDELT1"] for i in range(h["NAXIS1"])])

# Normalise all tempaltes as a whole to have a median of 1 (maintaining the
# difference /between/ templates, so they are still luminosity weighted)
star_temps /= np.median(star_temps)

# save everything
np.save("EMILES_all_data_3D", star_temps)
np.save("EMILES_age_grid", age_grid)
np.save("EMILES_met_grid", met_grid)

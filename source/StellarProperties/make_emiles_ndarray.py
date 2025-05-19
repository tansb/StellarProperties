import numpy as np
import glob
import os
from astropy.io import fits
import argparse
import shutil

""" Extract the E-MILES spectra into a ndarray to make it faster to import """


def main():
    parser = argparse.ArgumentParser(
        description="Extract the individual E-MILES spectra files and save"
        "into a single ndarray to make importing faster.")

    parser.add_argument("data_dir", help="Path to the data directory")

    args = parser.parse_args()
    data_dir = args.data_dir

    print(f"Loading data from: {data_dir}")

    # write a convinient function which takes a single file and extracts the
    # spectrum as well as the age and metallicity from the filename.
    def get_age_metal(filename):
        """Extract age and metallicity from filename"""
        s = os.path.basename(filename)
        t = s.find("T")
        metal = s[s.find("Z") + 1: t]
        age = float(s[t + 1: t + 8])
        if "m" in metal:
            metal = -float(metal[1:])
        elif "p" in metal:
            metal = float(metal[1:])
        else:
            raise ValueError("This is not a standard E-MILES filename")
        return (age, metal)

    # Each file a different spectrum
    # go through each file and extract the age and metallicity from the title
    # to know the age and metallicity range of the library.
    filenames = np.array(glob.glob(f"{data_dir}/*.fits"))
    all_age_met = [get_age_metal(fname) for fname in filenames]
    all_ages, all_mets = np.array(all_age_met).T

    unique_ages = np.unique(all_ages)
    unique_mets = np.unique(all_mets)
    n_ages = len(unique_ages)
    n_mets = len(unique_mets)

    age_grid = np.array([unique_ages for _ in range(n_mets)]).T  # Gyrs
    met_grid = np.array([unique_mets for _ in range(n_ages)])  # [Z/H]

    # 53689 = npix in each spectrum
    templates = [[None for _ in range(n_ages)] for _ in range(n_mets)]

    for a, age in enumerate(unique_ages):
        for m, metal in enumerate(unique_mets):
            ind = all_age_met.index((age, metal))
            spec = fits.open(filenames[ind])[0].data
            templates[m][a] = spec

    # transpose the array so it's in the shape that the rest of the codebase
    # expects
    templates = np.array(templates).T

    # Normalise all tempaltes as a whole to have a median of 1 (maintaining the
    # difference /between/ templates, so they are still luminosity weighted)
    templates /= np.median(templates)

    # Get the wavelength
    h = fits.open(filenames[ind])[0].header
    wave = np.array([h["CRVAL1"] + i * h["CDELT1"]
                     for i in range(h["NAXIS1"])])

    template_name = data_dir.split('/')[-2].replace('_FITS', '')

    # make the output directory if it doesn't already exist
    output_dir = f'../../data/Synthetic_Templates/{template_name}/'
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # copy the input fits dir into the new data dir
    shutil.copytree(data_dir,
                    output_dir + template_name + '_FITS', dirs_exist_ok=True)

    print("Saving to file: ")
    np.save(f"{output_dir}{template_name}_all_data_3D", templates)
    np.save(f"{output_dir}{template_name}_age_grid", age_grid)
    np.save(f"{output_dir}{template_name}_met_grid", met_grid)
    np.save(f"{output_dir}{template_name}_wave_array", wave)


if __name__ == "__main__":
    main()

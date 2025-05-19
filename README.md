# Stellar Population Synthesis

A Python package for analyzing galaxy spectra and performing stellar population synthesis using various spectroscopic templates and models.

The full spectral fitting aspect of the code is is built around Michele Cappellari's [pPXF](https://pypi.org/project/ppxf/).

## Features

- Support for multiple survey data formats
- Spectrum cleaning and preprocessing
- Lick indices measurement
- Stellar population measurements from Lick indices.
- Stellar population and kinematics measurement from full spectral fitting using various (MILES stars, E-MILES, CaT, BPASS)
- Full spectral fitting with both light-weighted or mass-weighted templates
- Handy visualization tools


## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- Astropy
- Scipy
- pPXF

If using the provided docker files, all requirements will be automatically installed. Otherwise, see the requirements.txt file for the specific versions.

## Installation Steps

### Pip

Start by creating a new conda environment
```bash
conda create --name StellarProperties python=3.10.12
conda activate StellarProperties
```
Then download the git repository and install using pip.
```bash
# Clone the repository
git clone https://github.com/tansb/StellarProperties.git

# Change to project directory
cd StellarProperties

# There a bunch of large template files that you could get from their origins, but for simplicity I've included a bunch here. They are too big for git so you need git's large file storage (lfs) tool.
git lfs install
git lfs pull

# Install the package
pip install .
```

### Docker
If you're familiar with docker, the easiest way to use this code is to just build the docker image and container described in the dockerfiles. The steps are as follows:

1. Download the dockerfile and the docker-compose.yml files

2. Set the MY_MOUNTED_DATA_DIR. Docker containers are designed to operate in isolation, but given you'll want a way to pass your data in and out of the container for and after analysis, define a directory that you want mounted. E.g. for simplicity you can just mount your entire home directory.

```bash
export MY_MOUNTED_DATA_DIR=<your_data_dir>
```

3. Build the docker image and container. I've written a docker-compose file to make it straightforward so all you need to do is:

```bash
docker compose up -d
```

```bash
docker exec -it stellar_properties_container bash
```

The first command builds the image and creates a (-d detatched) container with the name stellar_properties_container. The second line opens a bash shell in the stellar_properties_container in interactive mode. In the container, you should be able to open an ipython terminal and do:

```python
from Galaxy import Galaxy
from Tempaltes import Stellar_Templates

t = Stellar_Templates(model='MILES')
```

## Usage

### Lick Indices Example

```python
from Galaxy import Galaxy
from Templates import Stellar_Templates

# Initialize templates
templates = Stellar_Templates()

# Load and analyze a galaxy spectrum
galaxy = Galaxy(input_fn="path/to/spectrum.fits", spec_type="SDSS")

# Clean the spectrum
galaxy.clean_spectrum(templates)

# Measure spectral indices
galaxy.measure_lick_indices()

# Estimate stellar population parameters
galaxy.measure_sp_parameters(model="tmj")

# save the galaxy object to as a python pickle object.
galaxy.save(output_dir="results")
```

### Full Spectral Fitting Usage

This is essentially a wrapper around Michele Cappellari's pPXF. See the pPXF documentation for information about all the keyword arguments for the different use cases. you can pass any/all of ppxf's keyword arguemnts into the galaxy.full_spectral_fitting method.

```python
from Galaxy import Galaxy
from Templates import Synthetic_Templates

# Initialize templates
light_weighted_templates = Synthetic_Templates(light_weighted=True)
mass_weighted_templates = Synthetic_Templates(light_weighted=False)

# Load galaxy
galaxy = Galaxy(input_fn="path/to/spectrum.fits", spec_type="SDSS")

# Perform full spectral fitting
galaxy.full_spectral_fitting(light_weighted_templates, mdegree=10, tag="_LW")
galaxy.full_spectral_fitting(mass_weighted_templates, mdegree=10, tag="_MW")

# Save results
galaxy.save(output_dir="results")
```

## Input Data Sources

The package supports spectral data from:
- SDSS (Sloan Digital Sky Survey)
- SAMI (Sydney-AAO Multi-object Integral field spectrograph)
- LEGA-C (Large Early Galaxy Astrophysics Census)
- JWST (NIRSpec)
- Gran Telescopio CANARIAS/OSIRIS spectrograph
- KECK/KCWI
- VLT/XShooter

## Template data directory structure
To avoid you spending ages gathering the various model files and getting them into the correct format, I've provided them with this repo using git's large file storage tool. Of course I haven't included all combinations of models available, so you'll likely want to include the models you want to use. At the moment, this will require you going into the Templates.py file and including an additional "if else" statement for the model keyword and including the path to the files. Pretty well all the templates included here are from the [MILES team](https://research.iac.es/proyecto/miles/), who provide access to their models on their website.

Currently included models are:

#### Lick Index models
* S07: ([Schiavon 2007](https://ui.adsabs.harvard.edu/abs/2007ApJS..171..146S/abstract)
* TMJ: [Thomas, Maraston, Johansson 2011](https://ui.adsabs.harvard.edu/abs/2011MNRAS.412.2183T/abstract)
* MILES: BaSTI isochrones, Chabrier 2003 initial mass function

#### Stellar Templates
* CaT: Calcium Triplet library [CaT library v9.1](http://research.iac.es/proyecto/miles/pages/stellar-libraries/cat-library.php)
* MILES: [MILES v9.1](https://research.iac.es/proyecto/miles/pages/stellar-libraries/miles-library.php)

#### Synthetic Templates

* MILES: BaSTI isochrones, Chabrier (2003) initial mass function, baseFe $\alpha$-element abundance.

* EMILES: Padova isochrones, Chabrier (2003) initial mass function, base $\alpha$-element abundance.

* EMILES-IR: E-MILES models with just the infra-red part. BaSTI isochrones, bi-model initial mass function, baseFe $\alpha$-element abundance.

To speed up reading in the synthetic template libraries, rather than reading in each template from it's individual file, I saved all the templates for a particular library into an n-dimensial .npy file which the code then reads in. Similarly, the age and metallicity of each template in the .npy file are also saved in age_grid.npy and met_grid.npy files. The code also reads in a single template file to get the wavelength array of the templates. Therefore, if you want to use new templates you'll need to:

1. Include the directory of templates,
2. Generate the all_data_3D.npy, age_grid.npy and met_grid.npy files
3. Add the paths under another if/elif statement in the Templates.py file

[TO DO: track down the file I used to make them]

You can chose whether you want your stellar population analysis to be light-weighted or
mass-weighted by using the ```light_weighted=True``` keyword when initialising the templates.

## Reference

I first wrote this code to use in [Barone et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...898...62B), but it has grown considerably since.

```
@ARTICLE{2020ApJ...898...62B,
       author = {{Barone}, Tania M. and {D'Eugenio}, Francesco and {Colless}, Matthew and {Scott}, Nicholas},
        title = "{Gravitational Potential and Surface Density Drive Stellar Populations. II. Star-forming Galaxies}",
      journal = {\apj},
     keywords = {Scaling relations, Galaxy stellar content, Galaxy ages, Galaxy abundances, Extragalactic astronomy, 2031, 621, 576, 574, 506, Astrophysics - Astrophysics of Galaxies},
         year = 2020,
        month = jul,
       volume = {898},
       number = {1},
          eid = {62},
        pages = {62},
          doi = {10.3847/1538-4357/ab9951},
archivePrefix = {arXiv},
       eprint = {2006.00720},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020ApJ...898...62B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

## License

Copyright (c) <2025> <Barone>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# StellarPopulation_Synthesis

StellarPopulation_Synthesis is a python module for analysing galaxy optical spectra to measure various stellar population parameters.

## Installation
via github

## Data Files
To avoid having to specify the path to all input data files, the following directory structure is
needed in the Data directory:

* BPASS_FSF_files
	* BPASS_chab100_age_grid.npy
	* BPASS_chab100_all_data_normalised_4D.npy
	* BPASS_chab100_met_grid.npy

* MILES_FSF_files
	* MILES_age_grid.npy
	* MILES_all_data_normalised_3D.npy
	* MILES_met_grid.npy

* Miles_spectral_templates
	* s0001.fits
	* ...
	* s0985.fits
	
* velocity_dispersion_correction
	* sigcorr_icpt.dat
	* sigcorr_slope.dat
	
* Lick_and_models
	* Lick_index_definitions.dat
	* TMB_TMK_interpolated_alpha_model.fits
	* alpha_models_ti_schiavon_reinterpolated_log_age_nolick.fits
        * Miles_Indicies_BaSTI_Chabrier_interpolated.fits          
* MOSES_catalogue.fits
* spec-0266-51630-0146.fits

The Miles empirical spectral templates (fits format) can be downloaded from
[here](http://www.iac.es/proyecto/miles/pages/stellar-libraries/miles-library.php "Miles library")

The Miles synthetic templates can be downloaded from [here](ftp://ftp.iac.es/MILES).
The pipeline currently uses the templates with BASTI isochrones, Chabrier IMF, and base [alpha/Fe].



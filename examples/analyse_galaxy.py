#from source.StellarPopulation_Synthesis import *
from Galaxy import Galaxy
from Templates import Stellar_Templates, Synthetic_Templates
import sp_helper_functions as sph
import matplotlib.pyplot as plt


spec_filename = '../data/spec-0266-51630-0146.fits' # spectrum file

t0   = Stellar_Templates() 					# empirical templates for the clean_spectrum method
t_LW = Synthetic_Templates(light_weighted=True)  # light-weighted synthetic templates (MILES)
t_MW = Synthetic_Templates(light_weighted=False) # mass-weighted  synthetic templates (MILES)


g = Galaxy(spec_filename)  # initialise the galaxy object
g.clean_spectrum(t0)       # clean the spectrum using the empirical stellar templates

g.measure_lick_indices() # measure Lick indices
# using the Lick indices either the Thomas, Maraston & Johanson models or the Schiavon 2007 models
g.measure_sp_parameters(model='tmj') 
g.measure_sp_parameters(model='s07')

# do the full spectral fitting with the light-weighted and the mass-weighted templates
g.full_spectral_fitting(t_LW, tag='_LW')
g.full_spectral_fitting(t_MW, tag='_MW')

print(f"Light weighted age: {g.sp_MILES_LW['age']} Gyrs, [Z/H]: {g.sp_MILES_LW['[Z/H]']}")
print(f"Mass  weighted age: {g.sp_MILES_MW['age']} Gyrs, [Z/H]: {g.sp_MILES_MW['[Z/H]']}")

# make a plot of the weights from each fit
fig, ax = plt.subplots()

# need to specify the location of the age and metallicity grids
age_grid_path = '../data/MILES_FSF_files/MILES_age_grid.npy'
met_grid_path =  '../data/MILES_FSF_files/MILES_met_grid.npy'

sph.plot_ppxf_weights(g.pp_MILES_LW, 'light weighted', model='MILES', tag='', fig=fig, ax=ax,
					   age_grid_path=age_grid_path, met_grid_path=met_grid_path,
					   save=True, output_fn='example_output_weights_plot.png', output_dir='')


# save the output .pckl file in the same directory
g.save(output_fn='example_output.pckl') 


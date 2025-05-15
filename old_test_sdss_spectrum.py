import numpy as np
from Galaxy import Galaxy
from Templates import Stellar_Templates, Synthetic_Templates
import matplotlib.pyplot as plt
plt.ion()


legac = 'example_data/LEGAC/legac_M101_v3.8_spec1d_126153.fits'
sdss  = '../../data/spec-0266-51630-0146.fits'

#sami  = 'example_data/SAMI/' + \
#        '486957_blue_red_11_Y14SAR3_P001_15T076_2014_04_24-2014_05_04_apspec.fits'

sami  = 'example_data/SAMI/' + \
        '7139_blue_red_7_Y18SAR3_P002_12T122_2018_05_07-2018_05_16_apspec.fits'

legac_tname = 'example_data/LEGAC/legac_SP_TMB_SP_v3.fits'
sami_tname  = 'example_data/SAMI/sami_SP.fits'

t0       = Stellar_Templates()
t_LW     = Synthetic_Templates(light_weighted=True)
t_MW     = Synthetic_Templates(light_weighted=False)

g_legac = Galaxy(input_fn=legac, spec_type='LEGA-C', table_name=legac_tname)
g_sdss  = Galaxy(input_fn=sdss, spec_type='SDSS')
g_sami  = Galaxy(input_fn=sami, spec_type='SAMI', table_name=sami_tname)

for g in [g_legac, g_sdss, g_sami]:
    g.clean_spectrum(t0)
    g.measure_lick_indices()
    g.measure_sp_parameters(model='tmj')
    g.measure_sp_parameters(model='s07')
    g.measure_sp_parameters(model='miles')
    g.FSF(t_LW, tag='LW', gas=False, use_cleaned_spectrum=True)
    g.FSF(t_LW, tag='LW_gas', gas=True)
    g.FSF(t_LW, tag='LW_not_masked')


"""
def test_SDSS_galaxy():
    g = Galaxy(input_fn=sdss, spec_type='SDSS')
    g.clean_spectrum(t0)
    g.measure_lick_indices()
    g.measure_sp_parameters(model='tmj')
    g.measure_sp_parameters(model='s07')
    g.measure_sp_parameters(model='miles')


def test_SAMI_galaxy():
    g = Galaxy(input_fn=sami, spec_type='SAMI', table_name=sami_tname)
    g.clean_spectrum(t0)
    g.measure_lick_indices()
    g.measure_sp_parameters(model='tmj')
    g.measure_sp_parameters(model='s07')
    g.measure_sp_parameters(model='miles')


def test_LEGAC_galaxy():
    g = Galaxy(input_fn=legac, spec_type='LEGA-C', table_name=legac_tname)
    g.clean_spectrum(t0)
    g.measure_lick_indices()
    g.measure_sp_parameters(model='tmj')
    g.measure_sp_parameters(model='s07')
    g.measure_sp_parameters(model='miles')



g.clean_spectrum(t0)

g.measure_lick_indices()
g.measure_sp_parameters(model='tmj')
g.measure_sp_parameters(model='s07')
g.measure_sp_parameters(model='miles')

# LIGHT WEIGHTED
g.full_spectral_fitting(t_LW, mdegree=10, tag='_LW')
# MASS WEIGHTED
g.full_spectral_fitting(t_MW, mdegree=10, tag='_MW')


def test_galaxy_object():
    assert np.isfinite(g.z), 'Galaxy has non finite redshift given'

def test_template_objects():
    assert len(t_LW.templates.shape) >= 3 and len(t_MW.templates.shape) >= 3,\
        'Synthetic Templates wrong shape'
    assert len(t0.templates.shape) == 2,\
        'Stellar Templates wrong shape'

def test_clean_spectrum():
    assert g.pp1.chi2 > 0.5 and g.pp1.chi2 < 1.5, 'chi2 of pp1 not around 1'


def test_lick_indices():
    for k in g.lick_indices:
        if g.lick_indices[k][0] != np.nan:
            assert g.lick_indices[k][1] != np.nan, \
                'index has defined value with undefined uncertainties'


def test_lick_models():
    for model in ['tmj', 's07']:
        attr = getattr(g, f'lick_sp_{model}')
        assert (np.array_equal(np.isfinite(attr['age']),
                               np.array([True, True, True]))
                and np.array_equal(np.isfinite(attr['Z_H']),
                                   np.array([True, True, True]))
                and np.array_equal(np.isfinite(attr['alpha_Fe']),
                                   np.array([True, True, True]))), \
            f'Invalid value in lick + {model}'


def test_full_spectral_fitting():
    assert np.array_equal(t_LW.templates_convolved,
                          t_MW.templates_convolved) is False, \
        'mass and light weighted templates are the same'

    assert np.array_equal(g.pp_MILES_LW.weights,
                          g.pp_MILES_MW.weights) is False, \
        'mass and light weighted weights are the same'


def test_save():
    g  = Galaxy(input_fn=filename)
    g.save(output_dir='')
"""

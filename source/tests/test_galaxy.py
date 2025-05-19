import pytest
import os
import numpy as np
from StellarProperties import Galaxy
from StellarProperties.Templates import Stellar_Templates, Synthetic_Templates

testfile_path = os.path.dirname(__file__)


class TestGalaxy:

    @pytest.fixture
    def stellar_templates(self):
        """Create a template fixture for tests."""
        return Stellar_Templates(model='MILES')

    @pytest.fixture
    def synthetic_templates(self):
        """Create a template fixture for tests."""
        return Synthetic_Templates(model='EMILES')

    @pytest.fixture
    def sample_galaxy(self):
        """Create a sample galaxy for testing."""
        # Create a minimal galaxy object for testing
        # This might use a small test FITS file stored in tests/data/
        file_path = testfile_path + "/test_data/SDSS/spec-0266-51630-0146.fits"
        return Galaxy(input_fn=file_path, spec_type="SDSS")

    def test_initialization(self, sample_galaxy):
        """Test that a Galaxy object initializes properly."""
        assert sample_galaxy is not None
        assert hasattr(sample_galaxy, 'loglam_gal')
        assert hasattr(sample_galaxy, 'logflux_gal')

    def test_clean_spectrum_stellar_tempaltes(
            self, sample_galaxy, stellar_templates):
        """Test spectrum cleaning functionality."""
        # Save the original flux
        original_flux = sample_galaxy.logflux_gal.copy()

        # Perform cleaning
        sample_galaxy.clean_spectrum(stellar_templates)

        # Check that cleaning modified the flux
        assert not np.array_equal(original_flux,
                                  sample_galaxy.log_galaxy_final)

        # Check that the resulting spectrum has expected properties
        # check for both the log_binned spectrum and the linearly interpolated
        # spectrum
        assert not np.isnan(sample_galaxy.log_galaxy_final).any()
        assert not np.isinf(sample_galaxy.log_galaxy_final).any()
        assert not np.isnan(sample_galaxy.galaxy_final).any()
        assert not np.isinf(sample_galaxy.galaxy_final).any()

    def test_clean_spectrum_synthetic_tempaltes(
            self, sample_galaxy, synthetic_templates):
        """Test spectrum cleaning functionality."""
        # Perform cleaning
        sample_galaxy.clean_spectrum(synthetic_templates)

        # Check that cleaning modified the flux
        assert not np.array_equal(sample_galaxy.logflux_gal,
                                  sample_galaxy.log_galaxy_final)

        # Check that the resulting spectrum has expected properties
        # check for both the log_binned spectrum and the linearly interpolated
        # spectrum
        assert not np.isnan(sample_galaxy.log_galaxy_final).any()
        assert not np.isinf(sample_galaxy.log_galaxy_final).any()
        assert not np.isnan(sample_galaxy.galaxy_final).any()
        assert not np.isinf(sample_galaxy.galaxy_final).any()

    def test_measure_lick_indices(self, sample_galaxy, stellar_templates):
        """Test the measurement of Lick indices."""
        sample_galaxy.clean_spectrum(stellar_templates)
        sample_galaxy.measure_lick_indices()

        # Check that indices were calculated
        assert hasattr(sample_galaxy, 'lick_indices')
        assert len(sample_galaxy.lick_indices) > 0

    def test_measure_sp_parameters(self, sample_galaxy, stellar_templates):
        """Test estimation of stellar population parameters."""
        # This might require a previously measured lick indices
        sample_galaxy.clean_spectrum(stellar_templates)
        sample_galaxy.measure_lick_indices()

        # Test with the three models provided
        sample_galaxy.measure_sp_parameters(model="tmj")
        sample_galaxy.measure_sp_parameters(model="s07")
        sample_galaxy.measure_sp_parameters(model='miles')

        # Check that parameters were calculated
        for model in ['tmj', 's07', 'miles']:
            assert hasattr(sample_galaxy, f'lick_sp_{model}')
            sp_params = getattr(sample_galaxy, f'lick_sp_{model}')
            assert 'age' in sp_params

    def test_full_spectral_fitting(self, sample_galaxy, synthetic_templates):
        """Test the full spectral fitting method."""
        # Test fitting with some parameters
        sample_galaxy.full_spectral_fitting(synthetic_templates,
                                            mdegree=10, tag='_test')
        # Check results
        assert hasattr(sample_galaxy, 'pp_test')
        assert hasattr(sample_galaxy.pp_test, 'sol')

    def test_save_and_load(self, sample_galaxy, tmpdir):
        """Test saving and loading functionality."""
        # Create a temporary directory for testing
        output_dir = str(tmpdir)

        # Save the galaxy object
        filename = sample_galaxy.save(output_dir=output_dir)

        # Load the galaxy object
        loaded_galaxy = Galaxy.load(filename)

        # Check that the loaded object has the same properties
        assert hasattr(loaded_galaxy, 'loglam_gal')
        assert hasattr(loaded_galaxy, 'logflux_gal')
        assert np.array_equal(loaded_galaxy.loglam_gal,
                              sample_galaxy.loglam_gal)
        assert np.array_equal(loaded_galaxy.logflux_gal,
                              sample_galaxy.logflux_gal)

import pytest
from StellarProperties.Templates import Stellar_Templates, Synthetic_Templates


class TestTemplates:

    def test_stellar_templates_initialization(self):
        """Test initialization of Stellar_Templates."""
        templates = Stellar_Templates(model='MILES')

        # Check that templates were loaded
        assert templates is not None
        assert hasattr(templates, 'templates')
        assert hasattr(templates, 'linlam')

    def test_different_stellar_models(self):
        """Test initialization with different models."""
        # Test a few different models
        models = ['MILES', 'CaT']

        for model in models:
            templates = Stellar_Templates(model=model)
            assert templates is not None
            assert hasattr(templates, 'templates')

    def test_different_synthetic_models(self):
        """Test initialization with different models."""
        # Test a few different models
        models = ['MILES', 'EMILES', 'EMILES-IR']

        for model in models:
            templates = Synthetic_Templates(model=model)
            assert templates is not None
            assert hasattr(templates, 'templates')

    def test_synthetic_templates_lw(self):
        """Test light-weighted synthetic templates."""
        templates = Synthetic_Templates(light_weighted=True)

        # Check that templates were loaded
        assert templates is not None
        assert hasattr(templates, 'templates')
        assert hasattr(templates, 'linlam')

        # Check specific properties of light-weighted templates
        # (whatever those might be in your implementation)

    def test_synthetic_templates_mw(self):
        """Test mass-weighted synthetic templates."""
        templates = Synthetic_Templates(light_weighted=False)

        # Check that templates were loaded
        assert templates is not None
        assert hasattr(templates, 'templates')
        assert hasattr(templates, 'linlam')

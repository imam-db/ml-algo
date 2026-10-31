import importlib.util
import sys
from pathlib import Path
import pytest
import numpy as np

# Load data_generator.py from the 06_playground directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "06_playground" / "data_generator.py"

spec = importlib.util.spec_from_file_location(
    "data_generator", MODULE_PATH
)
data_generator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data_generator
spec.loader.exec_module(data_generator)

DataGenerator = data_generator.DataGenerator

class TestDataGenerator:
    def test_generate_linear_output_shapes(self):
        generator = DataGenerator()

        X, y = generator.generate_linear(n_samples=40)

        assert X.shape == (40, 1)
        assert y.shape == (40,)

    def test_generate_linear_is_reproducible(self):
        generator = DataGenerator()

        X1, y1 = generator.generate_linear(n_samples=40)
        X2, y2 = generator.generate_linear(n_samples=40)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_generate_clusters_output_shapes(self):
        generator = DataGenerator()

        X, y = generator.generate_clusters(
            n_samples=50,
            n_clusters=3,
            n_features=4,
        )

        assert X.shape == (50, 4)
        assert y.shape == (50,)

    def test_generate_clusters_is_reproducible(self):
        generator = DataGenerator()

        X1, y1 = generator.generate_clusters(n_samples=50)
        X2, y2 = generator.generate_clusters(n_samples=50)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_generate_dataset_moons_output_shapes(self):
        generator = DataGenerator()

        X, y = generator.generate_dataset("moons", n_samples=30)

        assert X.shape == (30, 2)
        assert y.shape == (30,)
    def test_generate_dataset_rejects_unknown_type(self):
        generator = DataGenerator()

        with pytest.raises(ValueError, match="Unknown data type"):
            generator.generate_dataset("invalid_type")
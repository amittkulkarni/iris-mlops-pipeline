import pytest
import pandas as pd
from data_loader import load_iris_dataset

class TestDataValidation:

    def test_load_iris_dataset(self):
        """Test IRIS dataset loading"""
        df = load_iris_dataset()
        assert not df.empty
        assert len(df) > 0
        assert 'species' in df.columns

    def test_iris_data_structure(self):
        """Test IRIS dataset structure"""[1]
        df = load_iris_dataset()

        # Check required columns
        required_columns = ['sepal_length', 'sepal_width',
                            'petal_length', 'petal_width']
        for col in required_columns:
            assert col in df.columns

        # Check data types
        for col in required_columns:
            assert pd.api.types.is_numeric_dtype(df[col])

    def test_no_missing_values(self):
        """Test that dataset has no missing values"""[6]
        df = load_iris_dataset()
        assert not df.isnull().values.any(), "Dataset contains missing values"

    def test_no_duplicates(self):
        """Test that dataset has no duplicate records"""[6]
        df = load_iris_dataset()
        feature_cols = ['sepal_length', 'sepal_width',
                        'petal_length', 'petal_width']
        assert not df[feature_cols].duplicated().any(), "Dataset contains duplicate records"

    def test_data_ranges(self):
        """Test data value ranges"""
        df = load_iris_dataset()

        # All measurements should be positive
        feature_cols = ['sepal_length', 'sepal_width',
                        'petal_length', 'petal_width']
        for col in feature_cols:
            assert (df[col] > 0).all(), f"Negative values found in {col}"

    def test_target_classes(self):
        """Test target classes are valid"""
        df = load_iris_dataset()
        unique_targets = df['species'].unique()
        expected_targets = ['setosa', 'versicolor', 'virginica']
        assert all(target in expected_targets for target in unique_targets)

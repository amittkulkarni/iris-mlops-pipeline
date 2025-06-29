import pytest
import pandas as pd
from data_loader import load_iris_dataset

class TestDataValidation:

    def test_load_iris_dataset(self):
        """Test IRIS dataset loading"""
        df = load_iris_dataset()
        assert df is not None, "load_iris_dataset returned None"
        assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
        assert not df.empty, "DataFrame is empty"
        assert len(df) > 0, "DataFrame has no rows"
        # Fixed: Check for 'species' column, not 'target'
        assert 'species' in df.columns, "Missing species column"

    def test_iris_data_structure(self):
        """Test IRIS dataset structure"""
        df = load_iris_dataset()

        assert df is not None and not df.empty, "Invalid DataFrame"

        # Check required columns
        required_columns = ['sepal_length', 'sepal_width',
                            'petal_length', 'petal_width']
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Check data types
        for col in required_columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric"

    def test_no_missing_values(self):
        """Test that dataset has no missing values"""
        df = load_iris_dataset()
        assert df is not None and not df.empty, "Invalid DataFrame"
        assert not df.isnull().values.any(), "Dataset contains missing values"

    def test_data_ranges(self):
        """Test data value ranges"""
        df = load_iris_dataset()
        assert df is not None and not df.empty, "Invalid DataFrame"

        # Fixed typo: removed space in 'petal_ length'
        feature_cols = ['sepal_length', 'sepal_width',
                        'petal_length', 'petal_width']
        for col in feature_cols:
            assert (df[col] > 0).all(), f"Negative values found in {col}"

    def test_target_classes(self):
        """Test target classes are valid"""
        df = load_iris_dataset()
        assert df is not None and not df.empty, "Invalid DataFrame"

        unique_targets = df['species'].unique()
        expected_targets = ['setosa', 'versicolor', 'virginica']
        assert all(target in expected_targets for target in unique_targets)        
import pytest
from data_loader import load_iris_dataset
from preprocessing import preprocess_iris_data
from model import IrisModel
from evaluation import evaluate_model_performance, validate_model_performance

class TestIrisModel:

    @pytest.fixture
    def sample_data(self):
        """Fixture to provide sample IRIS data"""[6]
        df = load_iris_dataset()
        processed_df, scaler = preprocess_iris_data(df)

        feature_cols = ['sepal_length', 'sepal_width',
                        'petal_length', 'petal_width']
        X = processed_df[feature_cols]
        y = processed_df['target']

        return X, y

    def test_model_training(self, sample_data):
        """Test model training process"""[6]
        X, y = sample_data
        model = IrisModel()

        results = model.train(X, y)

        assert model.is_trained
        assert 'train_accuracy' in results
        assert 'test_accuracy' in results
        assert results['train_accuracy'] > 0.8
        assert results['test_accuracy'] > 0.8

    def test_model_predictions(self, sample_data):
        """Test model prediction functionality"""
        X, y = sample_data
        model = IrisModel()

        # Train model
        model.train(X, y)

        # Test predictions
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_untrained_model_error(self, sample_data):
        """Test that untrained model raises error"""
        X, y = sample_data
        model = IrisModel()

        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(X[:5])

    def test_model_performance_validation(self, sample_data):
        """Test model performance meets requirements"""[6]
        X, y = sample_data
        model = IrisModel()

        results = model.train(X, y)
        metrics = evaluate_model_performance(results['y_test'], results['y_pred'])

        # Should not raise exception if performance is good
        validate_model_performance(metrics, min_accuracy=0.85)

        assert metrics['accuracy'] >= 0.85

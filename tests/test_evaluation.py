import pytest
import numpy as np
from evaluation import evaluate_model_performance, validate_model_performance, sanity_test_predictions
from model import IrisModel
from data_loader import load_iris_dataset
from preprocessing import preprocess_iris_data

class TestEvaluation:

    def test_evaluate_model_performance(self):
        """Test model performance evaluation"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])  # One wrong prediction

        metrics = evaluate_model_performance(y_true, y_pred)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics

        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1

    def test_performance_validation_pass(self):
        """Test performance validation with good metrics"""
        metrics = {
            'accuracy': 0.90,
            'f1_score': 0.85
        }

        # Should not raise exception
        assert validate_model_performance(metrics) == True

    def test_performance_validation_fail_accuracy(self):
        """Test performance validation fails with low accuracy"""
        metrics = {
            'accuracy': 0.70,  # Below minimum
            'f1_score': 0.85
        }

        with pytest.raises(ValueError, match="accuracy.*below minimum"):
            validate_model_performance(metrics)

    def test_sanity_test_predictions(self):
        """Test sanity testing of predictions"""
        # Load and prepare data
        df = load_iris_dataset()
        processed_df, scaler = preprocess_iris_data(df)

        feature_cols = ['sepal length', 'sepal width',
                        'petal length', 'petal width']
        X = processed_df[feature_cols]
        y = processed_df['target']

        # Train model
        model = IrisModel()
        model.train(X, y)

        # Test sanity predictions
        sample_data = X[:5]
        predictions = sanity_test_predictions(model, sample_data)

        assert len(predictions) == 5
        assert all(pred in [0, 1, 2] for pred in predictions)

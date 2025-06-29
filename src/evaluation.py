from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model_performance(y_true, y_pred):
    """Comprehensive model evaluation"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }

    # Add confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics

def validate_model_performance(metrics, min_accuracy=0.85):
    """Validate that model meets minimum performance criteria"""
    if metrics['accuracy'] < min_accuracy:
        raise ValueError(f"Model accuracy {metrics['accuracy']:.3f} below minimum {min_accuracy}")

    if metrics['f1_score'] < 0.80:
        raise ValueError(f"Model F1-score {metrics['f1_score']:.3f} below minimum 0.80")

    return True

def sanity_test_predictions(model, sample_data):
    """Run sanity tests on model predictions"""
    if len(sample_data) == 0:
        raise ValueError("Sample data cannot be empty")

    predictions = model.predict(sample_data)

    # Check prediction range (should be 0, 1, or 2 for IRIS)
    if not all(pred in [0, 1, 2] for pred in predictions):
        raise ValueError("Invalid prediction values detected")

    return predictions

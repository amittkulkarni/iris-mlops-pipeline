import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class IrisModel:
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=5
        )
        self.random_state = random_state
        self.is_trained = False

    def train(self, X, y, test_size=0.2):
        """Train the IRIS classification model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Return training metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        return {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': test_pred
        }

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def save_model(self, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        
import pandas as pd
from sklearn.datasets import load_iris
import os

def load_iris_dataset(file_path = os.path.join('..', 'data', 'iris.csv')):
    """Load IRIS dataset from file or sklearn"""
    if file_path and os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading from file {file_path}: {e}")
    if file_path is None:
        try:
            if '__file__' in globals():
                current_dir = os.path.dirname(__file__)
                file_path = os.path.join(current_dir, '..', 'data', 'iris.csv')
                file_path = os.path.abspath(file_path)

                if os.path.exists(file_path):
                    return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error with file path construction: {e}")

def save_iris_dataset(df, file_path):
    """Save IRIS dataset to CSV"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    return file_path
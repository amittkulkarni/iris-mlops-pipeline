import pandas as pd
from sklearn.datasets import load_iris
import os

def load_iris_dataset(file_path = os.path.join('..', 'data', 'iris.csv')):
    """Load IRIS dataset from file or sklearn"""
    if file_path and os.path.exists(file_path):
        return pd.read_csv(file_path)

def save_iris_dataset(df, file_path):
    """Save IRIS dataset to CSV"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    return file_path
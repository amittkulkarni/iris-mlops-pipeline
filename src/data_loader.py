import pandas as pd
from sklearn.datasets import load_iris
import os

def load_iris_dataset(file_path='../data/iris.csv'):
    """Load IRIS dataset from file or sklearn"""
    if file_path and os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        # Load from sklearn if file doesn't exist
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df

def save_iris_dataset(df, file_path):
    """Save IRIS dataset to CSV"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    return file_path
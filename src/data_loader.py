import pandas as pd
import os

//hi
def load_iris_dataset(file_path=None):
    """Load IRIS dataset from file"""

    # Set default file path if none provided
    if file_path is None:
        try:
            # Try to get relative path from current file location
            if '__file__' in globals():
                current_dir = os.path.dirname(__file__)
                file_path = os.path.join(current_dir, '..', 'data', 'iris.csv')
                file_path = os.path.abspath(file_path)
            else:
                # Fallback for when __file__ is not available (like in pytest)
                file_path = os.path.abspath(os.path.join('..', 'data', 'iris.csv'))
        except Exception as e:
            print(f"Error constructing file path: {e}")
            file_path = os.path.join('..', 'data', 'iris.csv')

    print(f"Attempting to load from: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"Current working directory: {os.getcwd()}")

    # Try to load the file
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded CSV with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading from file {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")
        # List files in parent directory to debug
        try:
            parent_dir = os.path.dirname(os.path.abspath('.'))
            if os.path.exists(parent_dir):
                print(f"Files in parent directory: {os.listdir(parent_dir)}")
            if os.path.exists('data'):
                print(f"Files in data directory: {os.listdir('data')}")
            if os.path.exists('../data'):
                print(f"Files in ../data directory: {os.listdir('../data')}")
        except Exception as e:
            print(f"Error listing directories: {e}")

    # Return None if file not found (don't use sklearn since you don't want it)
    print("ERROR: Could not load IRIS dataset from file")
    return None

def save_iris_dataset(df, file_path):
    """Save IRIS dataset to CSV"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        return file_path
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return None
        

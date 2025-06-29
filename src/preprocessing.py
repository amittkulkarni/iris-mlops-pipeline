import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def validate_iris_data(df):
    """Validate IRIS dataset structure and content"""
    required_columns = ['sepal_length', 'sepal_width',
                        'petal_length', 'petal_width']

    # Check if required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    # Check data types
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} should be numeric")

    return True

def preprocess_iris_data(df):
    """Preprocess IRIS dataset"""
    # Validate data first
    validate_iris_data(df)

    # Create a copy to avoid modifying original
    processed_df = df.copy()

    # Extract features and target
    feature_columns = ['sepal_length', 'sepal_width',
                       'petal_length', 'petal_width']

    # Scale features
    scaler = StandardScaler()
    processed_df[feature_columns] = scaler.fit_transform(processed_df[feature_columns])

    # Encode target if it's string
    if 'species' in processed_df.columns:
        le = LabelEncoder()
        processed_df['target'] = le.fit_transform(processed_df['species'])

    return processed_df, scaler

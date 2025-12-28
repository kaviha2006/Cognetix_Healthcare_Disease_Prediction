import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def split_features_target(df: pd.DataFrame, target_col: str):
    """
    Separates features (X) and target (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def apply_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Applies StandardScaler to features.
    Returns scaled arrays and the scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for readability if needed, but arrays are standard for sklearn
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled_df, X_test_scaled_df, scaler

def handle_imbalance(X, y):
    """
    Applies SMOTE to handle class imbalance.
    Returns resampled X and y.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Original dataset shape: {y.value_counts().to_dict()}")
    print(f"Resampled dataset shape: {y_resampled.value_counts().to_dict()}")
    return X_resampled, y_resampled

def perform_split(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

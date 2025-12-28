import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads proper dataset from csv file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data Loaded Successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles invalid zero values in Glucose, BloodPressure, BMI, Insulin.
    Replaces 0 with median of the respective column.
    """
    # Columns where 0 is invalid
    cols_to_fix = ['Glucose', 'BloodPressure', 'BMI', 'Insulin']
    
    # Replace 0 with NaN first to correctly calculate median (ignoring 0s if they are actual missing values placeholder)
    # The prompt says 0 is invalid.
    df_clean = df.copy()
    
    for col in cols_to_fix:
        # Count zeros
        zero_count = (df_clean[col] == 0).sum()
        if zero_count > 0:
            print(f"Column {col}: Found {zero_count} zero values. Replacing with median.")
            # Calculate median excluding zeros (or treat 0 as NaN for median calc)
            # Standard approach: replace 0 with NaN, then fill with median
            df_clean[col] = df_clean[col].replace(0, np.nan)
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
        else:
            print(f"Column {col}: No zero values found.")
            
    return df_clean

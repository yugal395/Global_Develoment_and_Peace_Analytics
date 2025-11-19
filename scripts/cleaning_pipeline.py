import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load the dataset"""
    df = pd.read_csv(file_path)
    print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_structure(df):
    """Fix column names and remove duplicates"""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("%", "percent")
    )
    df.drop_duplicates(inplace=True)
    print("Column names cleaned and duplicates removed.")
    return df

def handle_missing_values(df):
    """Handle missing values"""
    threshold = len(df) * 0.7
    df = df.dropna(axis=1, thresh=threshold)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"Missing values in numerical column '{col}' filled with median.")
    return df

def fix_data_types(df):
    """Correct data types"""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.year
        print("Date column converted to year.")
    else:
        print("No 'date' column found, skipping type conversion.")
    return df  # <-- return must be here, outside the if

def scale_numeric(df):
    """Scale numerical values between 0 and 1 (except year)"""
    scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if 'date' in num_cols:
        num_cols = num_cols.drop('date')
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("Numerical columns scaled between 0 and 1.")
    return df  # <-- return must also be here, outside the if

def save_cleaned_data(df, output_path):
    """Save cleaned data to a new CSV"""
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned data saved to {output_path}")

def main():
    raw_path = "../data/raw/world_governance_indicators.csv"
    save_path = "../data/processed/world_governance_cleaned.csv"

    df = load_data(raw_path)
    df = clean_structure(df)
    df = handle_missing_values(df)
    df = fix_data_types(df)
    df = scale_numeric(df)
    save_cleaned_data(df, save_path)
    print("🎯 Cleaning pipeline completed successfully.")

if __name__ == "__main__":
    main()

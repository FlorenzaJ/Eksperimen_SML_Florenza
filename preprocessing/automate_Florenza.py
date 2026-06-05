import pandas as pd
import os

def run_preprocessing():
    """
    Automated preprocessing function to return execution-ready data.
    """
    print("Initializing data preprocessing pipeline...")
    
    raw_data_path = os.path.join("..", "credit_card_fraud_raw", "raw_credit_card_fraud_2025.csv")
    
    try:
        df_full = pd.read_csv(raw_data_path)
        
        df = df_full.sample(n=20000, random_state=42).reset_index(drop=True)
        print(f"Successfully loaded a sample of {len(df)} rows.")
        
    except FileNotFoundError:
        print(f"Error: Raw file not found at {raw_data_path}")
        return None

    print("Executing data cleaning...")
    df = df.dropna()
    df = df.drop_duplicates()
    
    irrelevant_cols = ['Time', 'id', 'ID', 'Transaction_ID', 'Unnamed: 0']
    cols_to_drop = [col for col in irrelevant_cols if col in df.columns]
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")
    
    processed_data_path = "clean_dataset.csv"
    
    df.to_csv(processed_data_path, index=False)
    print(f"Data cleaning complete. Processed data saved to: {processed_data_path}")
    
    return df

if __name__ == "__main__":
    run_preprocessing()
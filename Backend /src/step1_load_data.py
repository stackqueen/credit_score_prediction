import pandas as pd

def load_data():
    dev_df = pd.read_csv("data/raw/Dev_data_to_be_shared.csv")
    val_df = pd.read_csv("data/raw/validation_data_to_be_shared.csv")

    print(f"Dev shape: {dev_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"\nClass balance:\n{dev_df['bad_flag'].value_counts(normalize=True).round(3)}")
    print(f"\nFirst 10 columns: {list(dev_df.columns[:10])}")

    return dev_df, val_df

if __name__ == "__main__":
    dev_df, val_df = load_data()
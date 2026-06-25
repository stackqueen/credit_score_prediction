import pandas as pd
import numpy as np
import gc

# First pass: find columns to drop and medians
print("First pass: scanning dev data...")
dev_df = pd.read_csv("data/raw/Dev_data_to_be_shared.csv")

missing_pct = dev_df.isnull().mean()
cols_to_drop = missing_pct[missing_pct > 0.8].index.tolist()
cols_to_drop += ['account_number']
print(f"Dropping {len(cols_to_drop)} columns")

# Compute medians only for kept columns
keep_cols = [c for c in dev_df.columns if c not in cols_to_drop and c != 'bad_flag']
medians = dev_df[keep_cols].median()
y = dev_df['bad_flag'].values
del dev_df
gc.collect()

# Second pass: load only needed columns
print("Second pass: loading clean columns...")
use_cols = keep_cols + ['bad_flag']
dev_df = pd.read_csv("data/raw/Dev_data_to_be_shared.csv", usecols=use_cols)
for col in keep_cols:
    dev_df[col].fillna(medians[col], inplace=True)
dev_df.to_csv("data/processed/dev_cleaned.csv", index=False)
print(f"Dev saved: {dev_df.shape}")
del dev_df
gc.collect()

# Val
print("Processing val data...")
val_df = pd.read_csv("data/raw/validation_data_to_be_shared.csv", usecols=[c for c in keep_cols if c in pd.read_csv("data/raw/validation_data_to_be_shared.csv", nrows=1).columns] + ['account_number'])
for col in keep_cols:
    if col in val_df.columns:
        val_df[col].fillna(medians[col], inplace=True)
val_df.to_csv("data/processed/val_cleaned.csv", index=False)
print(f"Val saved: {val_df.shape}")
print("Done!")

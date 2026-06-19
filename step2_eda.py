import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dev_df = pd.read_csv("data/raw/Dev_data_to_be_shared.csv")

# 1. Missing values
missing = dev_df.isnull().sum()
missing_pct = (missing / len(dev_df) * 100).sort_values(ascending=False)
print("Top 20 columns with missing values:")
print(missing_pct[missing_pct > 0].head(20))

# 2. Class balance plot
plt.figure(figsize=(5,4))
dev_df['bad_flag'].value_counts().plot(kind='bar', color=['steelblue','crimson'])
plt.title('Class Balance (bad_flag)')
plt.xticks([0,1], ['Good (0)','Bad (1)'], rotation=0)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('reports/class_balance.png')
print("\nClass balance plot saved.")

# 3. Basic stats
print("\nNumerical summary (first 5 cols):")
print(dev_df.iloc[:, 2:7].describe().round(2))

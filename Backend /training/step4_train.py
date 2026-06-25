import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

print("Loading data...")
dev_df = pd.read_csv("data/processed/dev_cleaned.csv")

X = dev_df.drop(columns=['bad_flag'])
y = dev_df['bad_flag']

# Fill NaNs column by column
print("Filling NaNs...")
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

# Select top 100 features by correlation with target
print("Selecting top 100 features...")
correlations = X.apply(lambda col: col.corr(y)).abs()
top_features = correlations.nlargest(100).index.tolist()
X = X[top_features]
print(f"Shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
print("Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

# Train
print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=70,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_sm, y_train_sm)

# Evaluate
y_pred = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_pred)
gini = 2 * auc - 1
print(f"\nAUC:  {auc:.4f}")
print(f"GINI: {gini:.4f}")

# Save
joblib.dump(model, "models/xgb_model.pkl")
pd.Series(top_features).to_csv("models/feature_list.csv", index=False)
print("Model saved!")

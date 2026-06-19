import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from cryptography.fernet import Fernet
import joblib

print("Loading data...")
dev_df = pd.read_csv("data/processed/dev_cleaned.csv")
val_df = pd.read_csv("data/processed/val_cleaned.csv")

# Fill NaNs
for col in dev_df.columns:
    if dev_df[col].isnull().any():
        dev_df[col] = dev_df[col].fillna(dev_df[col].median())

# Top 100 features by correlation
X = dev_df.drop(columns=['bad_flag'])
y = dev_df['bad_flag']
correlations = X.apply(lambda col: col.corr(y)).abs()
top_features = correlations.nlargest(100).index.tolist()
X = X[top_features]

# Split + SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Train
print("Training XGBoost...")
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                      scale_pos_weight=70, eval_metric='auc', random_state=42, n_jobs=-1)
model.fit(X_train_sm, y_train_sm)

# Evaluate
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print(f"AUC: {auc:.4f} | GINI: {2*auc-1:.4f}")

# Predict on val using SAME features
val_ids = val_df['account_number']
val_X = pd.DataFrame()
for f in top_features:
    val_X[f] = val_df[f].fillna(0) if f in val_df.columns else 0
val_X = val_X.copy()

probs = model.predict_proba(val_X)[:,1]
submission = pd.DataFrame({'account_number': val_ids.values, 'predicted_probability': probs})
submission.to_csv("reports/submission.csv", index=False)
print(f"Submission saved! Shape: {submission.shape}")
print(submission.head())

# Encrypt model
joblib.dump(model, "models/xgb_model.pkl")
key = Fernet.generate_key()
with open("security/model.key", "wb") as f: f.write(key)
with open("models/xgb_model.pkl", "rb") as f: data = f.read()
with open("security/xgb_model_encrypted.pkl", "wb") as f: f.write(Fernet(key).encrypt(data))
print("Model encrypted! Done!")

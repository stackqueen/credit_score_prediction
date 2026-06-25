import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

print("Loading data...")
dev_df = pd.read_csv("data/processed/dev_cleaned.csv")
X = dev_df.drop(columns=['bad_flag'])
y = dev_df['bad_flag']

for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

correlations = X.apply(lambda col: col.corr(y)).abs()
top_features = correlations.nlargest(100).index.tolist()
X = X[top_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = joblib.load("models/xgb_model.pkl")

y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_prob)
gini = 2 * auc - 1
accuracy = (y_pred == y_test).mean()
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0f1117')

# 1. Confusion Matrix
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax1,
            xticklabels=['Predicted Good','Predicted Bad'],
            yticklabels=['Actual Good','Actual Bad'],
            linewidths=2, linecolor='#0f1117', annot_kws={"size": 16, "weight": "bold"})
ax1.set_title('Confusion Matrix', color='white', fontsize=14, pad=15)
ax1.tick_params(colors='white')
ax1.set_facecolor('#1a1d2e')
for text in ax1.texts:
    text.set_color('black')

# 2. Metrics Bar
ax2 = axes[1]
ax2.set_facecolor('#1a1d2e')
metrics = ['Accuracy', 'AUC', 'GINI']
values = [accuracy, auc, gini]
colors = ['#6c63ff', '#3b82f6', '#51cf66']
bars = ax2.bar(metrics, values, color=colors, width=0.5, edgecolor='none')
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.1%}', ha='center', va='bottom', color='white', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.set_title('Model Performance Metrics', color='white', fontsize=14, pad=15)
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('#444')
ax2.spines['left'].set_color('#444')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_facecolor('#1a1d2e')
ax2.yaxis.set_tick_params(labelcolor='white')
ax2.xaxis.set_tick_params(labelcolor='white')

# 3. Actual vs Predicted correlation
ax3 = axes[2]
ax3.set_facecolor('#1a1d2e')
good_probs = y_pred_prob[y_test == 0]
bad_probs = y_pred_prob[y_test == 1]
ax3.hist(good_probs, bins=30, alpha=0.7, color='#51cf66', label='Actual Good (0)', density=True)
ax3.hist(bad_probs, bins=30, alpha=0.7, color='#ff6b6b', label='Actual Bad (1)', density=True)
ax3.set_title('Predicted Score by Actual Class', color='white', fontsize=14, pad=15)
ax3.set_xlabel('Predicted Probability', color='white')
ax3.set_ylabel('Density', color='white')
ax3.tick_params(colors='white')
ax3.spines['bottom'].set_color('#444')
ax3.spines['left'].set_color('#444')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(facecolor='#2a2d3e', labelcolor='white', fontsize=10)

plt.suptitle(f'Credit Risk Model Evaluation   |   Accuracy: {accuracy:.1%}   |   GINI: {gini:.4f}', 
             color='white', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('reports/model_evaluation.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("Saved to reports/model_evaluation.png")
plt.show()

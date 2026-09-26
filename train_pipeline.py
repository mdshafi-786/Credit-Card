import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

print("="*60)
print("PHASE 1: MODEL BUILDING & EDA PIPELINE")
print("="*60)

# Paths
DATA_PATH = "transactions.parquet"
EDA_DIR = "notebooks/eda_outputs"
MODELS_DIR = "models"
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load Data
print("\n[1] Loading dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df)} transactions with {len(df.columns)} columns.")

# 2. EDA Analysis & Visualizations
print("\n[2] Performing Exploratory Data Analysis (EDA)...")

# 2.1 Class Imbalance
fraud_counts = df['Fraud_Status'].value_counts()
fraud_rate = (fraud_counts.get('Fraud', 0) / len(df)) * 100
print(f"Class Distribution: {fraud_counts.to_dict()} (Fraud Rate: {fraud_rate:.2f}%)")

plt.figure(figsize=(7, 5))
colors = ['#10B981', '#EF4444']
sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette=colors)
plt.title(f"Class Distribution: Fraud Status (Fraud Rate: {fraud_rate:.2f}%)", fontsize=13, fontweight='bold')
plt.xlabel("Fraud Status", fontsize=11)
plt.ylabel("Number of Transactions", fontsize=11)
for i, v in enumerate(fraud_counts.values):
    plt.text(i, v + 300, f"{v:,} ({v/len(df)*100:.1f}%)", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_1_class_imbalance.png", dpi=200)
plt.close()

# 2.2 Transaction Amount vs Fraud
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='Fraud_Status', y='Transaction_Amount', palette=['#10B981', '#EF4444'], showfliers=False)
plt.title("Transaction Amount by Fraud Status (without outliers)", fontsize=13, fontweight='bold')
plt.xlabel("Fraud Status", fontsize=11)
plt.ylabel("Transaction Amount (INR)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_2_amount_vs_fraud.png", dpi=200)
plt.close()

# 2.3 Transaction Hour vs Fraud Rate
hour_df = df.groupby('Transaction_Hour')['Fraud_Status'].apply(lambda s: (s == 'Fraud').mean() * 100).reset_index()
hour_df.columns = ['Hour', 'Fraud_Rate_Pct']
plt.figure(figsize=(10, 5))
sns.barplot(data=hour_df, x='Hour', y='Fraud_Rate_Pct', color='#3B82F6')
plt.axhline(fraud_rate, color='#EF4444', linestyle='--', label=f'Avg Fraud Rate ({fraud_rate:.1f}%)')
plt.title("Fraud Rate (%) by Transaction Hour", fontsize=13, fontweight='bold')
plt.xlabel("Hour of Day (0 - 23)", fontsize=11)
plt.ylabel("Fraud Rate (%)", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_3_hour_vs_fraud.png", dpi=200)
plt.close()

# 2.4 Distance From Home vs Fraud
plt.figure(figsize=(9, 5))
sns.kdeplot(data=df[df['Fraud_Status'] == 'Genuine']['Distance_From_Home_km'], label='Genuine', fill=True, color='#10B981', alpha=0.3)
sns.kdeplot(data=df[df['Fraud_Status'] == 'Fraud']['Distance_From_Home_km'], label='Fraud', fill=True, color='#EF4444', alpha=0.3)
plt.title("Distance From Home (km) Distribution by Fraud Status", fontsize=13, fontweight='bold')
plt.xlabel("Distance From Home (km)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_4_distance_vs_fraud.png", dpi=200)
plt.close()

# 2.5 Location Mismatch vs Fraud
loc_df = df.groupby('Location_Mismatch')['Fraud_Status'].apply(lambda s: (s == 'Fraud').mean() * 100).reset_index()
loc_df.columns = ['Location_Mismatch', 'Fraud_Rate_Pct']
plt.figure(figsize=(6, 5))
sns.barplot(data=loc_df, x='Location_Mismatch', y='Fraud_Rate_Pct', palette=['#3B82F6', '#F59E0B'])
plt.title("Fraud Rate (%) by Location Mismatch", fontsize=13, fontweight='bold')
plt.xlabel("Home vs Transaction City Mismatch", fontsize=11)
plt.ylabel("Fraud Rate (%)", fontsize=11)
for i, row in loc_df.iterrows():
    plt.text(i, row['Fraud_Rate_Pct'] + 0.3, f"{row['Fraud_Rate_Pct']:.2f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_5_location_mismatch_vs_fraud.png", dpi=200)
plt.close()

# 2.6 Failed Attempts Last 24h vs Fraud
fail_df = df.groupby('Failed_Attempts_Last_24h')['Fraud_Status'].apply(lambda s: (s == 'Fraud').mean() * 100).reset_index()
fail_df.columns = ['Failed_Attempts', 'Fraud_Rate_Pct']
plt.figure(figsize=(8, 5))
sns.barplot(data=fail_df, x='Failed_Attempts', y='Fraud_Rate_Pct', color='#8B5CF6')
plt.title("Fraud Rate (%) by Failed Attempts in Last 24h", fontsize=13, fontweight='bold')
plt.xlabel("Failed Attempts Last 24h", fontsize=11)
plt.ylabel("Fraud Rate (%)", fontsize=11)
for i, row in fail_df.iterrows():
    plt.text(i, row['Fraud_Rate_Pct'] + 0.5, f"{row['Fraud_Rate_Pct']:.2f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_6_failed_attempts_vs_fraud.png", dpi=200)
plt.close()

# 2.7 Correlation Heatmap of Numeric Features vs Fraud_Risk_Score
numeric_cols = [
    'Transaction_Amount', 'Transaction_Hour', 'Customer_Age', 'Customer_Income_Annual',
    'Customer_Tenure_Years', 'Transactions_Last_30_Days', 'Customer_Avg_Transaction_Amount',
    'Distance_From_Home_km', 'Transactions_Last_24h', 'Failed_Attempts_Last_24h', 'Fraud_Risk_Score'
]
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
plt.title("Correlation Heatmap: Numeric Features vs Fraud Risk Score", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_7_correlation_heatmap.png", dpi=200)
plt.close()

# 2.8 Fraud Risk Score Distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x='Fraud_Risk_Score', hue='Fraud_Status', bins=40, kde=True, palette=['#10B981', '#EF4444'])
plt.axvline(60.0, color='black', linestyle='--', label='Suspicious Threshold (60.0)')
plt.title("Fraud Risk Score Distribution by Status", fontsize=13, fontweight='bold')
plt.xlabel("Fraud Risk Score (0 - 100)", fontsize=11)
plt.ylabel("Count", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_8_risk_score_distribution.png", dpi=200)
plt.close()
print(f"EDA charts successfully saved to {EDA_DIR}/")

# 3. Preprocessing & Feature Engineering
print("\n[3] Preprocessing and Feature Engineering...")

# Parse DateTime
df['Transaction_DateTime'] = pd.to_datetime(df['Transaction_DateTime'])
df['day_of_week'] = df['Transaction_DateTime'].dt.dayofweek
df['is_night_transaction'] = df['Transaction_Hour'].apply(lambda h: 1 if 0 <= h <= 5 else 0)

# Feature engineering
df['amount_vs_avg_ratio'] = df['Transaction_Amount'] / (df['Customer_Avg_Transaction_Amount'] + 1e-5)
df['velocity_score'] = df['Transactions_Last_24h'] + df['Failed_Attempts_Last_24h']

# Categorical encodings
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['Customer_Risk_Profile_Ord'] = df['Customer_Risk_Profile'].map(risk_map)

df['International_Transaction_Bin'] = (df['International_Transaction'] == 'Yes').astype(int)
df['Location_Mismatch_Bin'] = (df['Location_Mismatch'] == 'Yes').astype(int)

# One-hot encode nominal categoricals
one_hot_cols = ['Merchant_Category', 'Device_Type', 'Payment_Method', 'Customer_Home_Location', 'Transaction_Location']
df_encoded = pd.get_dummies(df, columns=one_hot_cols, drop_first=False)

# Target variables
y_class = (df['Fraud_Status'] == 'Fraud').astype(int)
y_reg = df['Fraud_Risk_Score']

# Collect engineered feature names
exclude_cols = [
    'Transaction_ID', 'Transaction_DateTime', 'Fraud_Status', 'Fraud_Risk_Score',
    'Suspicious_Transaction', 'Customer_Risk_Profile', 'International_Transaction',
    'Location_Mismatch'
]
feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]
print(f"Total features: {len(feature_cols)}")

X = df_encoded[feature_cols].copy()

# Save metadata for feature columns and categories
encoders_metadata = {
    'risk_map': risk_map,
    'one_hot_categories': {col: sorted(df[col].unique().tolist()) for col in one_hot_cols},
    'one_hot_cols': one_hot_cols,
    'numeric_cols_to_scale': [
        'Transaction_Amount', 'Transaction_Hour', 'Customer_Age', 'Customer_Income_Annual',
        'Customer_Tenure_Years', 'Transactions_Last_30_Days', 'Customer_Avg_Transaction_Amount',
        'Distance_From_Home_km', 'Transactions_Last_24h', 'Failed_Attempts_Last_24h',
        'day_of_week', 'amount_vs_avg_ratio', 'velocity_score'
    ]
}

# 4. Train / Test Split (Stratified 80/20)
print("\n[4] Train/Test Split (Stratified 80/20)...")
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
)
print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train fraud rate: {y_class_train.mean():.4f}, Test fraud rate: {y_class_test.mean():.4f}")

# Scale numeric features
scaler = StandardScaler()
cols_to_scale = encoders_metadata['numeric_cols_to_scale']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# Save scaler and feature columns
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
joblib.dump(encoders_metadata, f"{MODELS_DIR}/encoders.pkl")
with open(f"{MODELS_DIR}/feature_columns.json", 'w') as f:
    json.dump(feature_cols, f, indent=2)

# 5. Model 1: Fraud Classification
print("\n[5] Training Model 1: Fraud Classification...")

classifiers = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.08,
        scale_pos_weight=(len(y_class_train) - sum(y_class_train)) / sum(y_class_train),
        random_state=42, eval_metric='logloss', n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.08,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
    )
}

clf_metrics = {}
best_clf_name = None
best_clf_score = -1.0
best_clf_model = None

for name, clf in classifiers.items():
    print(f"--- Training {name} ---")
    clf.fit(X_train_scaled, y_class_train)
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    prec = precision_score(y_class_test, y_pred)
    rec = recall_score(y_class_test, y_pred)
    f1 = f1_score(y_class_test, y_pred)
    roc_auc = roc_auc_score(y_class_test, y_proba)
    pr_auc = average_precision_score(y_class_test, y_proba)
    cm = confusion_matrix(y_class_test, y_pred).tolist()
    
    clf_metrics[name] = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm
    }
    print(f"{name} -> Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    
    # Prioritizing Recall & PR-AUC / F1
    combined_score = 0.5 * rec + 0.3 * pr_auc + 0.2 * f1
    if combined_score > best_clf_score:
        best_clf_score = combined_score
        best_clf_name = name
        best_clf_model = clf

print(f"\n>>> Best Classifier Selected: {best_clf_name} (Combined score: {best_clf_score:.4f})")
joblib.dump(best_clf_model, f"{MODELS_DIR}/model_fraud_classifier.pkl")

# 6. Model 2: Fraud Risk Score Regressor
print("\n[6] Training Model 2: Fraud Risk Score Regressor...")
regressors = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1),
    "LightGBM Regressor": LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1)
}

reg_metrics = {}
best_reg_name = None
best_reg_r2 = -999.0
best_reg_model = None

for name, reg in regressors.items():
    print(f"--- Training {name} ---")
    reg.fit(X_train_scaled, y_reg_train)
    y_reg_pred = reg.predict(X_test_scaled)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2 = r2_score(y_reg_test, y_reg_pred)
    
    reg_metrics[name] = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }
    print(f"{name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    if r2 > best_reg_r2:
        best_reg_r2 = r2
        best_reg_name = name
        best_reg_model = reg

print(f"\n>>> Best Regressor Selected: {best_reg_name} (R²: {best_reg_r2:.4f})")
joblib.dump(best_reg_model, f"{MODELS_DIR}/model_risk_score.pkl")

# 7. Suspicious Transaction Flag Logic & Threshold Configuration
print("\n[7] Defining Thresholds and Suspicious Transaction Logic...")
threshold_config = {
    "fraud_probability_threshold": 0.50,
    "risk_score_suspicious_threshold": 60.0,
    "risk_tiers": {
        "Low": [0.0, 39.9],
        "Medium": [40.0, 69.9],
        "High": [70.0, 100.0]
    },
    "heuristic_rules": {
        "high_velocity": {"min_velocity": 4, "weight": 15},
        "location_mismatch_far": {"min_distance_km": 150.0, "weight": 20},
        "night_high_amount": {"min_amount_ratio": 2.5, "is_night": 1, "weight": 25},
        "multiple_failed_attempts": {"min_failed": 2, "weight": 20}
    },
    "suspicious_logic_description": "A transaction is marked suspicious if: predicted risk_score >= 60.0 OR classifier fraud_probability >= 0.50 OR (Location_Mismatch=Yes AND Distance_From_Home_km > 150 AND Failed_Attempts_Last_24h >= 2)"
}

with open(f"{MODELS_DIR}/threshold_config.json", 'w') as f:
    json.dump(threshold_config, f, indent=2)

# 8. Feature Importances / Explainability
print("\n[8] Generating Feature Importance / Explainability dictionary...")
if hasattr(best_clf_model, 'feature_importances_'):
    importances = best_clf_model.feature_importances_
elif hasattr(best_clf_model, 'coef_'):
    importances = np.abs(best_clf_model.coef_[0])
else:
    importances = np.ones(len(feature_cols))

feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
top_feats = {k: float(v) for k, v in feat_imp[:15]}
print("Top 10 Important Features:", list(top_feats.items())[:10])

with open(f"{MODELS_DIR}/feature_importances.json", 'w') as f:
    json.dump(top_feats, f, indent=2)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
top_items = list(top_feats.items())[:12]
sns.barplot(x=[v for k, v in top_items], y=[k for k, v in top_items], palette='viridis')
plt.title(f"Top 12 Feature Importances ({best_clf_name})", fontsize=13, fontweight='bold')
plt.xlabel("Importance", fontsize=11)
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_9_feature_importances.png", dpi=200)
plt.close()

# 9. Save metrics summary
metrics_summary = {
    "classification_models": clf_metrics,
    "regression_models": reg_metrics,
    "best_classifier": best_clf_name,
    "best_regressor": best_reg_name,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "total_features": len(feature_cols)
}
with open(f"{MODELS_DIR}/metrics_summary.json", 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print("\n" + "="*60)
print("PHASE 1 COMPLETE: All models and artifacts successfully generated!")
print("="*60)

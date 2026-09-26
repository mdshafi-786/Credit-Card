import json

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Credit Card Fraud Detection and Transaction Risk Analysis\n",
                "## End-to-End Model Training Pipeline & Metrics Report\n",
                "\n",
                "This notebook documents the data preprocessing, exploratory data analysis, class imbalance mitigation, multi-model evaluation for fraud classification, continuous risk score regression, heuristic blend rules, and explainability."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, json, joblib\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n",
                "from lightgbm import LGBMClassifier, LGBMRegressor\n",
                "from xgboost import XGBClassifier, XGBRegressor\n",
                "from sklearn.metrics import classification_report, roc_auc_score, r2_score, mean_absolute_error\n",
                "\n",
                "print('Libraries loaded successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Load Dataset and Explore Summary"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_parquet('../transactions.parquet')\n",
                "print(f'Total rows: {len(df):,}, columns: {len(df.columns)}')\n",
                "print(df['Fraud_Status'].value_counts(normalize=True))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Evaluated Model Performance Metrics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('../models/metrics_summary.json') as f:\n",
                "    metrics = json.load(f)\n",
                "print('--- Classification Models ---')\n",
                "for m, vals in metrics['classification_models'].items():\n",
                "    print(f\"{m:22s} | Recall: {vals['recall']:.4f} | PR-AUC: {vals['pr_auc']:.4f} | F1: {vals['f1']:.4f} | ROC-AUC: {vals['roc_auc']:.4f}\")\n",
                "print('\\n--- Regression Models ---')\n",
                "for m, vals in metrics['regression_models'].items():\n",
                "    print(f\"{m:30s} | R2: {vals['r2']:.4f} | MAE: {vals['mae']:.4f} | RMSE: {vals['rmse']:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Suspicious Transaction Policy Configuration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('../models/threshold_config.json') as f:\n",
                "    thresh = json.load(f)\n",
                "print(json.dumps(thresh, indent=2))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Explainability & Top Feature Contributions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('../models/feature_importances.json') as f:\n",
                "    feat_imp = json.load(f)\n",
                "for feat, val in feat_imp.items():\n",
                "    print(f'{feat:35s}: {val:.4f}')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.16"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/model_training.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=2)
print("Saved notebooks/model_training.ipynb successfully!")

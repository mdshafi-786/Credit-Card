# FraudShield AI - Streamlit Cloud Deployment Guide

## 🚀 Deploy to Streamlit Community Cloud

### Prerequisites
1. Push your project to a **GitHub repository**
2. A free [Streamlit Community Cloud](https://share.streamlit.io) account (sign in with GitHub)

### Step 1: Push to GitHub

```powershell
# Initialize git (if not already)
cd "Credit Card fraud detection and transition risk analysis"
git init
git add .
git commit -m "FraudShield AI - Streamlit App"

# Create a GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/fraudshield-ai.git
git branch -M main
git push -u origin main
```

> **Important**: Make sure these files are pushed:
> - `streamlit_app.py` (main app)
> - `requirements.txt` (dependencies)
> - `.streamlit/config.toml` (theme config)
> - `models/` directory (all `.pkl`, `.json` files)
> - `backend/transactions.db` (database)
> - `notebooks/eda_outputs/` (EDA charts)

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repository
4. Set the **Main file path** to: `streamlit_app.py`
5. Click **"Deploy!"**

### Step 3: Configuration (if needed)

Streamlit Cloud auto-detects:
- `requirements.txt` → installs Python dependencies
- `.streamlit/config.toml` → applies the dark theme

### 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure all packages are in `requirements.txt` |
| Database not found | Make sure `backend/transactions.db` is committed to git |
| Models not loading | Verify `models/` directory is pushed (not in `.gitignore`) |
| Large file error | Use Git LFS for `transactions.db` if > 100MB |

### 📦 Files Required for Deployment

```
├── streamlit_app.py          ← Main Streamlit app
├── requirements.txt          ← Python dependencies  
├── .streamlit/config.toml    ← Theme & server config
├── models/
│   ├── model_fraud_classifier.pkl
│   ├── model_risk_score.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_columns.json
│   ├── feature_importances.json
│   ├── metrics_summary.json
│   └── threshold_config.json
├── backend/
│   └── transactions.db       ← SQLite database (25K records)
└── notebooks/
    └── eda_outputs/          ← 9 EDA charts (PNG)
```

### 🌐 App URL

The live application is hosted at:
```
https://credit-card-fraud-sheild-ai.streamlit.app/
```


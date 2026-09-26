"""
FraudShield AI — Streamlit Application
Credit Card Fraud Detection & Transaction Risk Intelligence
"""

import streamlit as st
import json
import uuid
import sqlite3
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
import io
import os

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Paths & Constants
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "backend" / "transactions.db"
EDA_DIR = BASE_DIR / "notebooks" / "eda_outputs"
FRAUD_PROB_THRESHOLD = 0.50
RISK_SCORE_THRESHOLD = 60.0

CITIES = ["Bengaluru", "Mumbai", "Hyderabad", "Jaipur", "Ahmedabad",
          "Kolkata", "Chennai", "Visakhapatnam", "Pune", "Delhi"]
MERCHANT_CATEGORIES = ["Fuel", "Online Shopping", "Healthcare", "Restaurants",
                       "Clothing", "Travel", "Jewelry", "Grocery", "Electronics", "Entertainment"]
DEVICE_TYPES = ["Mobile", "Laptop", "Tablet", "POS Terminal"]
PAYMENT_METHODS = ["Online", "Contactless", "Chip & PIN", "Mobile Wallet"]
RISK_PROFILES = ["Low", "Medium", "High"]

# ─────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ───── Import Google Font ───── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ───── Root Variables ───── */
    :root {
        --bg-primary: #0a0e1a;
        --bg-card: rgba(15, 23, 42, 0.85);
        --bg-card-hover: rgba(20, 30, 55, 0.95);
        --border-glow: rgba(99, 102, 241, 0.25);
        --accent-indigo: #6366f1;
        --accent-cyan: #22d3ee;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --accent-violet: #8b5cf6;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --gradient-1: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        --gradient-2: linear-gradient(135deg, #06b6d4, #22d3ee);
        --gradient-fraud: linear-gradient(135deg, #ef4444, #f43f5e);
        --gradient-safe: linear-gradient(135deg, #10b981, #34d399);
    }

    /* ───── Global Overrides ───── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ───── Sidebar styling ───── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* ───── KPI Metric Cards ───── */
    .kpi-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 27, 75, 0.7));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .kpi-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0 4px;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-icon {
        font-size: 1.8rem;
        margin-bottom: 4px;
    }

    /* ───── Risk Gauge ───── */
    .risk-gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
    }

    /* ───── Status Badge ───── */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    .badge-fraud {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(244, 63, 94, 0.15));
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-genuine {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.15));
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-suspicious {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(251, 191, 36, 0.15));
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* ───── Reason Card ───── */
    .reason-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        color: #cbd5e1;
        font-size: 0.9rem;
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }
    .reason-card .reason-icon {
        color: #f59e0b;
        font-size: 1rem;
        flex-shrink: 0;
    }

    /* ───── Section Headers ───── */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-sub {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }

    /* ───── Hero Section ───── */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 20px;
        padding: 40px 48px;
        margin-bottom: 32px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e0e7ff, #c7d2fe, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        position: relative;
    }
    .hero-subtitle {
        color: #a5b4fc;
        font-size: 1.05rem;
        font-weight: 400;
        position: relative;
    }

    /* ───── Preset buttons ───── */
    .preset-btn {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
        width: 100%;
    }
    .preset-btn:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
    }

    /* ───── Table Styling ───── */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ───── Footer ───── */
    .footer {
        text-align: center;
        padding: 20px;
        color: #475569;
        font-size: 0.8rem;
        border-top: 1px solid rgba(99, 102, 241, 0.1);
        margin-top: 40px;
    }

    /* Fix tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    /* Hide default Streamlit elements for cleaner look, but keep sidebar arrow button visible */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {
        background-color: transparent !important;
    }
    
    /* Ensure the sidebar toggle arrow button is always clearly visible and styled */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        color: #a5b4fc !important;
        background: rgba(15, 23, 42, 0.95) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 8px !important;
        padding: 4px 8px !important;
        margin: 8px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        z-index: 1000 !important;
    }
    [data-testid="collapsedControl"]:hover {
        background: rgba(99, 102, 241, 0.3) !important;
        border-color: #818cf8 !important;
        transform: scale(1.05) !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        color: #a5b4fc !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Database Helpers (standalone, no backend imports)
# ─────────────────────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def db_exists():
    return DB_PATH.exists()


# ─────────────────────────────────────────────────────────────
# ML Engine (standalone, mirrors backend/app/ml/engine.py)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_engine():
    """Load all ML artifacts once and cache."""
    engine = {}
    try:
        engine["classifier"] = joblib.load(MODELS_DIR / "model_fraud_classifier.pkl")
        engine["regressor"] = joblib.load(MODELS_DIR / "model_risk_score.pkl")
        engine["scaler"] = joblib.load(MODELS_DIR / "scaler.pkl")
        engine["encoders"] = joblib.load(MODELS_DIR / "encoders.pkl")
        with open(MODELS_DIR / "feature_columns.json", "r") as f:
            engine["feature_columns"] = json.load(f)
        with open(MODELS_DIR / "feature_importances.json", "r") as f:
            engine["feature_importances"] = json.load(f)
        with open(MODELS_DIR / "threshold_config.json", "r") as f:
            engine["threshold_config"] = json.load(f)
        with open(MODELS_DIR / "metrics_summary.json", "r") as f:
            engine["metrics_summary"] = json.load(f)
        engine["loaded"] = True
    except Exception as e:
        engine["loaded"] = False
        engine["error"] = str(e)
    return engine


def prepare_features(engine, txn_dict):
    """Prepare feature vector from transaction dict."""
    dt = pd.to_datetime(txn_dict["transaction_datetime"])
    hour = dt.hour
    day_of_week = dt.dayofweek
    is_night = 1 if 0 <= hour <= 5 else 0

    home_loc = txn_dict["customer_home_location"]
    txn_loc = txn_dict["transaction_location"]
    inferred_mismatch = "Yes" if home_loc != txn_loc else "No"
    location_mismatch = txn_dict.get("location_mismatch") or inferred_mismatch
    loc_mismatch_bin = 1 if location_mismatch == "Yes" else 0

    distance = txn_dict.get("distance_from_home_km")
    if distance is None:
        distance = 0.0 if home_loc == txn_loc else 350.0

    avg_amt = txn_dict.get("customer_avg_transaction_amount")
    if avg_amt is None or avg_amt <= 0:
        avg_amt = float(txn_dict["transaction_amount"])

    amt = float(txn_dict["transaction_amount"])
    amount_vs_avg_ratio = amt / (avg_amt + 1e-5)

    tx_24h = int(txn_dict.get("transactions_last_24h") or 1)
    fails_24h = int(txn_dict.get("failed_attempts_last_24h") or 0)
    velocity_score = tx_24h + fails_24h

    risk_profile = txn_dict.get("customer_risk_profile", "Medium")
    risk_map = engine["encoders"].get("risk_map", {"Low": 0, "Medium": 1, "High": 2})
    risk_ord = risk_map.get(risk_profile, 1)

    intl_bin = 1 if txn_dict.get("international_transaction") == "Yes" else 0

    row = {col: 0.0 for col in engine["feature_columns"]}
    row["Transaction_Amount"] = amt
    row["Transaction_Hour"] = float(hour)
    row["Customer_Age"] = float(txn_dict.get("customer_age", 35))
    row["Customer_Income_Annual"] = float(txn_dict.get("customer_income_annual", 60000))
    row["Customer_Tenure_Years"] = float(txn_dict.get("customer_tenure_years", 3.0))
    row["Transactions_Last_30_Days"] = float(txn_dict.get("transactions_last_30_days", 15))
    row["Customer_Avg_Transaction_Amount"] = float(avg_amt)
    row["Distance_From_Home_km"] = float(distance)
    row["Transactions_Last_24h"] = float(tx_24h)
    row["Failed_Attempts_Last_24h"] = float(fails_24h)
    row["day_of_week"] = float(day_of_week)
    row["is_night_transaction"] = float(is_night)
    row["amount_vs_avg_ratio"] = float(amount_vs_avg_ratio)
    row["velocity_score"] = float(velocity_score)
    row["Customer_Risk_Profile_Ord"] = float(risk_ord)
    row["International_Transaction_Bin"] = float(intl_bin)
    row["Location_Mismatch_Bin"] = float(loc_mismatch_bin)

    one_hot_mappings = [
        (txn_dict.get("merchant_category", "Online Shopping"), "Merchant_Category_"),
        (txn_dict.get("device_type", "Mobile"), "Device_Type_"),
        (txn_dict.get("payment_method", "Online"), "Payment_Method_"),
        (home_loc, "Customer_Home_Location_"),
        (txn_loc, "Transaction_Location_")
    ]
    for val, prefix in one_hot_mappings:
        col_name = f"{prefix}{val}"
        if col_name in row:
            row[col_name] = 1.0

    df_features = pd.DataFrame([row])[engine["feature_columns"]]
    cols_to_scale = engine["encoders"]["numeric_cols_to_scale"]
    df_scaled = df_features.copy()
    df_scaled[cols_to_scale] = engine["scaler"].transform(df_features[cols_to_scale])

    augmented_info = {
        "hour": hour, "day_of_week": day_of_week, "is_night": is_night,
        "location_mismatch": location_mismatch, "loc_mismatch_bin": loc_mismatch_bin,
        "distance": distance, "avg_amt": avg_amt,
        "amount_vs_avg_ratio": amount_vs_avg_ratio, "velocity_score": velocity_score,
        "tx_24h": tx_24h, "fails_24h": fails_24h
    }
    return df_scaled, augmented_info


def generate_reasons(txn_dict, aug, prob, score):
    """Generate explainability reasons."""
    reasons = []
    if aug["amount_vs_avg_ratio"] >= 2.5:
        reasons.append(f"🔺 Spending Surge: Transaction amount (₹{txn_dict['transaction_amount']:,.2f}) is {aug['amount_vs_avg_ratio']:.1f}x higher than customer average (₹{aug['avg_amt']:,.2f})")
    elif aug["amount_vs_avg_ratio"] >= 1.5:
        reasons.append(f"📈 Elevated Ticket Size: Amount is {aug['amount_vs_avg_ratio']:.1f}x higher than typical ticket size")

    if aug["fails_24h"] >= 2:
        reasons.append(f"🔐 Authentication Warnings: {aug['fails_24h']} consecutive failed PIN/password attempts in the last 24 hours")
    elif aug["fails_24h"] == 1:
        reasons.append("🔐 Authentication Warning: 1 failed transaction attempt in the last 24 hours")

    if aug["loc_mismatch_bin"] == 1:
        reasons.append(f"📍 Geo-Spatial Discrepancy: Transaction in {txn_dict['transaction_location']}, but customer home is {txn_dict['customer_home_location']} ({aug['distance']:.0f} km away)")

    if aug["is_night"] == 1:
        reasons.append("🌙 Unusual Timing: Transaction during high-risk late night hours (12:00 AM - 05:59 AM)")

    if aug["velocity_score"] >= 4:
        reasons.append(f"⚡ High Velocity: {aug['velocity_score']} cumulative attempts & transactions within 24 hours")

    if txn_dict.get("international_transaction") == "Yes":
        reasons.append("🌍 Cross-Border Risk: International cross-border payment detected")

    if txn_dict.get("customer_risk_profile") == "High":
        reasons.append("⚠️ Customer Profile: Account marked under High Risk tier")

    if not reasons:
        if score < 40.0 and prob < 0.30:
            reasons.append("✅ Low Risk: Activity aligns with historical customer baseline and verified geographical profile")
        else:
            reasons.append("📊 Statistical Variance: Deviations detected across multiple multidimensional risk factors")

    return reasons[:4]


def predict_one(engine, txn_dict):
    """Run inference on a single transaction."""
    df_scaled, aug = prepare_features(engine, txn_dict)

    fraud_prob = float(engine["classifier"].predict_proba(df_scaled)[0, 1])
    fraud_threshold = engine["threshold_config"].get("fraud_probability_threshold", FRAUD_PROB_THRESHOLD)
    fraud_status = "Fraud" if fraud_prob >= fraud_threshold else "Genuine"

    risk_score_raw = float(engine["regressor"].predict(df_scaled)[0])
    risk_score = float(np.clip(risk_score_raw, 0.0, 100.0))

    susp_threshold = engine["threshold_config"].get("risk_score_suspicious_threshold", RISK_SCORE_THRESHOLD)
    rule_hit = (aug["loc_mismatch_bin"] == 1 and aug["distance"] > 150.0 and aug["fails_24h"] >= 2)
    is_suspicious = bool(risk_score >= susp_threshold or fraud_prob >= fraud_threshold or rule_hit)

    reasons = generate_reasons(txn_dict, aug, fraud_prob, risk_score)

    return {
        "fraud_status": fraud_status,
        "fraud_probability": round(fraud_prob, 4),
        "risk_score": round(risk_score, 2),
        "is_suspicious": is_suspicious,
        "top_reasons": reasons,
        "augmented_info": aug
    }


# ─────────────────────────────────────────────────────────────
# Feature Catalog & Configuration (All 18 Model Features)
# ─────────────────────────────────────────────────────────────
FEATURE_CONFIG = {
    "merchant_category": {
        "col": "merchant_category",
        "label": "🏪 Merchant Category",
        "type": "cat",
        "desc": "Category of merchant terminal"
    },
    "transaction_location": {
        "col": "transaction_location",
        "label": "📍 Transaction City / Location",
        "type": "cat",
        "desc": "City where card was swiped / charged"
    },
    "customer_home_location": {
        "col": "customer_home_location",
        "label": "🏠 Customer Home City",
        "type": "cat",
        "desc": "Cardholder registered home residence"
    },
    "device_type": {
        "col": "device_type",
        "label": "📱 Device Type",
        "type": "cat",
        "desc": "Device used (Mobile, Laptop, Tablet, POS)"
    },
    "payment_method": {
        "col": "payment_method",
        "label": "💳 Payment Method",
        "type": "cat",
        "desc": "Payment mechanism (Online, Contactless, Chip & PIN, Mobile Wallet)"
    },
    "customer_risk_profile": {
        "col": "customer_risk_profile",
        "label": "⚠️ Customer Risk Profile",
        "type": "cat",
        "desc": "Pre-assigned historical risk tier (Low, Medium, High)"
    },
    "international_transaction": {
        "col": "international_transaction",
        "label": "🌍 International Transaction",
        "type": "cat",
        "desc": "Cross-border payment flag"
    },
    "location_mismatch": {
        "col": "location_mismatch",
        "label": "🗺️ Location Mismatch",
        "type": "cat",
        "desc": "Home city vs Transaction city mismatch"
    },
    "transaction_hour": {
        "col": "transaction_hour",
        "label": "⏰ Transaction Hour (0-23)",
        "type": "cat",
        "desc": "Time of day (0 to 23)"
    },
    "transaction_amount": {
        "col": "transaction_amount",
        "label": "💰 Transaction Amount (₹)",
        "type": "num",
        "desc": "Monetary value of transaction in INR",
        "cases": [
            "WHEN transaction_amount < 2500 THEN '< ₹2,500'",
            "WHEN transaction_amount < 7500 THEN '₹2,500 - 7,500'",
            "WHEN transaction_amount < 15000 THEN '₹7,500 - 15,000'",
            "WHEN transaction_amount < 30000 THEN '₹15,000 - 30,000'",
            "ELSE '> ₹30,000'"
        ]
    },
    "customer_age": {
        "col": "customer_age",
        "label": "👤 Customer Age",
        "type": "num",
        "desc": "Age of the customer",
        "cases": [
            "WHEN customer_age < 26 THEN '18 - 25 yrs'",
            "WHEN customer_age < 36 THEN '26 - 35 yrs'",
            "WHEN customer_age < 51 THEN '36 - 50 yrs'",
            "WHEN customer_age < 66 THEN '51 - 65 yrs'",
            "ELSE '65+ yrs'"
        ]
    },
    "customer_income_annual": {
        "col": "customer_income_annual",
        "label": "💼 Annual Income (₹)",
        "type": "num",
        "desc": "Annual income reported by customer",
        "cases": [
            "WHEN customer_income_annual < 300000 THEN '< ₹3 Lakh'",
            "WHEN customer_income_annual < 600000 THEN '₹3L - 6L'",
            "WHEN customer_income_annual < 1000000 THEN '₹6L - 10L'",
            "WHEN customer_income_annual < 1500000 THEN '₹10L - 15L'",
            "ELSE '> ₹15 Lakh'"
        ]
    },
    "distance_from_home_km": {
        "col": "distance_from_home_km",
        "label": "📏 Distance From Home (km)",
        "type": "num",
        "desc": "Calculated distance between home and terminal",
        "cases": [
            "WHEN distance_from_home_km = 0 THEN '0 km (Home City)'",
            "WHEN distance_from_home_km < 100 THEN '1 - 100 km'",
            "WHEN distance_from_home_km < 500 THEN '100 - 500 km'",
            "WHEN distance_from_home_km < 1000 THEN '500 - 1,000 km'",
            "ELSE '> 1,000 km'"
        ]
    },
    "customer_avg_transaction_amount": {
        "col": "customer_avg_transaction_amount",
        "label": "💵 Customer Avg Ticket Size (₹)",
        "type": "num",
        "desc": "Baseline average spending per transaction",
        "cases": [
            "WHEN customer_avg_transaction_amount < 2500 THEN '< ₹2,500'",
            "WHEN customer_avg_transaction_amount < 5000 THEN '₹2,500 - 5,000'",
            "WHEN customer_avg_transaction_amount < 10000 THEN '₹5,000 - 10,000'",
            "ELSE '> ₹10,000'"
        ]
    },
    "transactions_last_24h": {
        "col": "transactions_last_24h",
        "label": "⚡ Velocity: Transactions Last 24h",
        "type": "cat",
        "desc": "Number of card uses in past 24 hours"
    },
    "failed_attempts_last_24h": {
        "col": "failed_attempts_last_24h",
        "label": "🔐 Security: Failed Attempts Last 24h",
        "type": "cat",
        "desc": "Declined or incorrect PIN/OTP entries"
    },
    "customer_tenure_years": {
        "col": "customer_tenure_years",
        "label": "⏳ Customer Relationship Tenure",
        "type": "num",
        "desc": "Years as an active bank customer",
        "cases": [
            "WHEN customer_tenure_years < 1 THEN '< 1 year'",
            "WHEN customer_tenure_years < 3 THEN '1 - 3 years'",
            "WHEN customer_tenure_years < 5 THEN '3 - 5 years'",
            "ELSE '5+ years'"
        ]
    },
    "transactions_last_30_days": {
        "col": "transactions_last_30_days",
        "label": "📊 Monthly Frequency (Last 30 Days)",
        "type": "num",
        "desc": "Total count of transactions in past month",
        "cases": [
            "WHEN transactions_last_30_days < 10 THEN '0 - 9'",
            "WHEN transactions_last_30_days < 20 THEN '10 - 19'",
            "WHEN transactions_last_30_days < 30 THEN '20 - 29'",
            "ELSE '30+'"
        ]
    }
}


def build_filter_conditions(filters):
    """Build standardized SQL WHERE conditions and params from filters dictionary."""
    conditions = []
    params = []
    if not filters:
        return conditions, params

    if filters.get("merchant_category") and filters["merchant_category"] != "All":
        conditions.append("merchant_category = ?")
        params.append(filters["merchant_category"])
    if filters.get("location") and filters["location"] != "All":
        conditions.append("transaction_location = ?")
        params.append(filters["location"])
    if filters.get("home_location") and filters["home_location"] != "All":
        conditions.append("customer_home_location = ?")
        params.append(filters["home_location"])
    if filters.get("device_type") and filters["device_type"] != "All":
        conditions.append("device_type = ?")
        params.append(filters["device_type"])
    if filters.get("payment_method") and filters["payment_method"] != "All":
        conditions.append("payment_method = ?")
        params.append(filters["payment_method"])
    if filters.get("risk_profile") and filters["risk_profile"] != "All":
        conditions.append("customer_risk_profile = ?")
        params.append(filters["risk_profile"])
    if filters.get("international_transaction") and filters["international_transaction"] != "All":
        conditions.append("international_transaction = ?")
        params.append(filters["international_transaction"])
    if filters.get("location_mismatch") and filters["location_mismatch"] != "All":
        conditions.append("location_mismatch = ?")
        params.append(filters["location_mismatch"])
    if filters.get("fraud_status") and filters["fraud_status"] != "All":
        conditions.append("fraud_status = ?")
        params.append(filters["fraud_status"])
    if filters.get("is_suspicious") is not None:
        conditions.append("is_suspicious = ?")
        params.append(1 if filters["is_suspicious"] else 0)
    if filters.get("min_amount") is not None:
        conditions.append("transaction_amount >= ?")
        params.append(float(filters["min_amount"]))
    if filters.get("max_amount") is not None:
        conditions.append("transaction_amount <= ?")
        params.append(float(filters["max_amount"]))
    if filters.get("min_age") is not None:
        conditions.append("customer_age >= ?")
        params.append(int(filters["min_age"]))
    if filters.get("max_age") is not None:
        conditions.append("customer_age <= ?")
        params.append(int(filters["max_age"]))
    if filters.get("min_hour") is not None:
        conditions.append("transaction_hour >= ?")
        params.append(int(filters["min_hour"]))
    if filters.get("max_hour") is not None:
        conditions.append("transaction_hour <= ?")
        params.append(int(filters["max_hour"]))
    if filters.get("search"):
        conditions.append("(transaction_id LIKE ? OR transaction_location LIKE ? OR customer_home_location LIKE ?)")
        term = f"%{filters['search']}%"
        params.extend([term, term, term])
    if filters.get("risk_level") and filters["risk_level"] != "All":
        rl = filters["risk_level"]
        if rl == "Low":
            conditions.append("fraud_risk_score < 40.0")
        elif rl == "Medium":
            conditions.append("fraud_risk_score >= 40.0 AND fraud_risk_score < 70.0")
        elif rl == "High":
            conditions.append("fraud_risk_score >= 70.0")

    return conditions, params


# ─────────────────────────────────────────────────────────────
# Dashboard Data Queries
# ─────────────────────────────────────────────────────────────
def get_dashboard_summary(filters=None):
    """Get aggregated dashboard summary from SQLite."""
    if not db_exists():
        return None

    conditions, params = build_filter_conditions(filters)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count,
                SUM(CASE WHEN fraud_status = 'Genuine' THEN 1 ELSE 0 END) as genuine_count,
                AVG(transaction_amount) as avg_amount,
                AVG(fraud_risk_score) as avg_risk_score,
                SUM(CASE WHEN is_suspicious = 1 THEN 1 ELSE 0 END) as susp_count
            FROM transactions {where_clause}
        """, params)
        agg = cursor.fetchone()
        total = agg["total"] or 0
        fraud_count = agg["fraud_count"] or 0
        genuine_count = agg["genuine_count"] or 0
        avg_amount = float(agg["avg_amount"] or 0.0)
        avg_risk_score = float(agg["avg_risk_score"] or 0.0)
        susp_count = agg["susp_count"] or 0

        # Breakdown by Category
        cursor.execute(f"""
            SELECT merchant_category as category, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count,
                   AVG(fraud_risk_score) as avg_risk_score
            FROM transactions {where_clause}
            GROUP BY merchant_category ORDER BY count DESC
        """, params)
        cat_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in cat_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Breakdown by Location Mismatch
        cursor.execute(f"""
            SELECT location_mismatch, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY location_mismatch
        """, params)
        loc_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in loc_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Breakdown by Device
        cursor.execute(f"""
            SELECT device_type, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY device_type
        """, params)
        dev_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in dev_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Breakdown by Payment Method
        cursor.execute(f"""
            SELECT payment_method, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY payment_method
        """, params)
        pay_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in pay_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Breakdown by Hour
        cursor.execute(f"""
            SELECT transaction_hour as hour, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY transaction_hour ORDER BY transaction_hour ASC
        """, params)
        hour_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in hour_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Breakdown by Risk Profile
        cursor.execute(f"""
            SELECT customer_risk_profile as risk_profile, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY customer_risk_profile
        """, params)
        profile_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in profile_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Risk score distribution
        cursor.execute(f"""
            SELECT 
                CASE 
                    WHEN fraud_risk_score < 20 THEN '0-20'
                    WHEN fraud_risk_score < 40 THEN '20-40'
                    WHEN fraud_risk_score < 60 THEN '40-60'
                    WHEN fraud_risk_score < 80 THEN '60-80'
                    ELSE '80-100'
                END as bin,
                COUNT(*) as count,
                SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY bin ORDER BY bin ASC
        """, params)
        risk_dist = [dict(r) for r in cursor.fetchall()]

    fraud_rate_pct = round((fraud_count / total * 100), 2) if total > 0 else 0.0
    susp_rate_pct = round((susp_count / total * 100), 2) if total > 0 else 0.0

    return {
        "total_transactions": total,
        "fraud_transactions": fraud_count,
        "genuine_transactions": genuine_count,
        "fraud_rate_pct": fraud_rate_pct,
        "average_transaction_amount": round(avg_amount, 2),
        "average_fraud_risk_score": round(avg_risk_score, 2),
        "suspicious_transactions": susp_count,
        "suspicious_rate_pct": susp_rate_pct,
        "breakdown_by_category": cat_breakdown,
        "breakdown_by_location_mismatch": loc_breakdown,
        "breakdown_by_device": dev_breakdown,
        "breakdown_by_payment_method": pay_breakdown,
        "breakdown_by_hour": hour_breakdown,
        "breakdown_by_risk_profile": profile_breakdown,
        "risk_score_distribution": risk_dist
    }


def get_feature_breakdown(feature_key, filters=None):
    """Aggregate breakdown and metrics for any model feature from SQLite."""
    if not db_exists():
        return None
    cfg = FEATURE_CONFIG.get(feature_key)
    if not cfg:
        return None

    conditions, params = build_filter_conditions(filters)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    if cfg["type"] == "cat":
        expr = cfg["col"]
        order = f"{cfg['col']} ASC" if cfg["col"] in ["transaction_hour", "transactions_last_24h", "failed_attempts_last_24h"] else "count DESC"
    else:
        expr = "CASE " + " ".join(cfg["cases"]) + " END"
        order = "count DESC"

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT {expr} as bucket,
                   COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count,
                   AVG(fraud_risk_score) as avg_risk,
                   AVG(transaction_amount) as avg_amount
            FROM transactions {where_clause}
            GROUP BY bucket
            ORDER BY {order}
        """, params)
        rows = [dict(r) for r in cursor.fetchall()]
        for r in rows:
            r["fraud_rate"] = round(r["fraud_count"] / r["count"] * 100, 2) if r["count"] > 0 else 0.0
            r["avg_risk"] = round(float(r["avg_risk"] or 0.0), 1)
            r["avg_amount"] = round(float(r["avg_amount"] or 0.0), 2)
            r["genuine_count"] = r["count"] - r["fraud_count"]
        return rows


def get_transactions_df(page=1, page_size=50, filters=None):
    """Fetch paginated transaction data as DataFrame with all model feature filters."""
    if not db_exists():
        return pd.DataFrame(), 0

    conditions, params = build_filter_conditions(filters)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM transactions {where_clause}", params)
        total = cursor.fetchone()[0]

        offset = (page - 1) * page_size
        cursor.execute(f"""
            SELECT * FROM transactions {where_clause}
            ORDER BY transaction_datetime DESC
            LIMIT ? OFFSET ?
        """, params + [page_size, offset])
        rows = cursor.fetchall()
        if rows:
            df = pd.DataFrame([dict(r) for r in rows])
        else:
            df = pd.DataFrame()

    return df, total


# ─────────────────────────────────────────────────────────────
# Plotly Theme Helpers
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.08)"),
    yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.08)"),
)

COLOR_PALETTE = ["#6366f1", "#22d3ee", "#f43f5e", "#10b981", "#f59e0b",
                  "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#3b82f6"]


# ─────────────────────────────────────────────────────────────
# Render Risk Score Gauge (Plotly)
# ─────────────────────────────────────────────────────────────
def render_risk_gauge(score, fraud_prob, fraud_status, is_suspicious):
    """Render a beautiful risk score gauge."""
    if score < 40:
        gauge_color = "#10b981"
        bar_color = "rgba(16, 185, 129, 0.2)"
    elif score < 70:
        gauge_color = "#f59e0b"
        bar_color = "rgba(245, 158, 11, 0.2)"
    else:
        gauge_color = "#ef4444"
        bar_color = "rgba(239, 68, 68, 0.2)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(
            font=dict(size=48, color=gauge_color, family="Inter"),
            suffix="/100"
        ),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#475569", tickfont=dict(color="#64748b")),
            bar=dict(color=gauge_color, thickness=0.25),
            bgcolor=bar_color,
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color="rgba(16, 185, 129, 0.06)"),
                dict(range=[40, 70], color="rgba(245, 158, 11, 0.06)"),
                dict(range=[70, 100], color="rgba(239, 68, 68, 0.06)"),
            ],
            threshold=dict(
                line=dict(color="#f43f5e", width=3),
                thickness=0.8,
                value=60
            ),
        ),
        title=dict(
            text="Fraud Risk Score",
            font=dict(size=16, color="#94a3b8")
        ),
    ))
    fig.update_layout(
        height=280,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ['xaxis', 'yaxis', 'margin']},
        margin=dict(l=30, r=30, t=60, b=20)
    )
    return fig


PAGES = [
    "🏠 Dashboard",
    "🔍 Transaction Check",
    "📦 Batch Upload",
    "📋 Transaction Explorer",
    "🧠 Model Insights"
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = PAGES[0]

# ─────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 16px 0 24px;">
        <div style="font-size: 2.5rem;">🛡️</div>
        <div style="font-size: 1.3rem; font-weight: 800; 
            background: linear-gradient(135deg, #a5b4fc, #818cf8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-top: 4px;">FraudShield AI</div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
            Transaction Risk Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    sidebar_current = st.session_state.get("current_page", PAGES[0])
    if sidebar_current not in PAGES:
        sidebar_current = PAGES[0]

    selected_sidebar = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(sidebar_current),
        label_visibility="collapsed"
    )
    if selected_sidebar != sidebar_current:
        st.session_state["current_page"] = selected_sidebar
        st.rerun()

    st.markdown("---")

    # Model status
    engine = load_ml_engine()
    if engine.get("loaded"):
        st.success("✅ ML Engine Loaded")
    else:
        st.error(f"❌ ML Engine Error: {engine.get('error', 'Unknown')}")

    if db_exists():
        with get_db() as conn:
            count = conn.cursor().execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        st.info(f"💾 Database: {count:,} records")
    else:
        st.warning("⚠️ No database found")

    st.markdown("""
    <div style="position: fixed; bottom: 12px; left: 16px; right: 16px;
        color: #475569; font-size: 0.7rem; text-align: center;">
        Built with Streamlit • FraudShield AI v2.0
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Top App Navigation Bar with Quick Arrow Switch Buttons
# ─────────────────────────────────────────────────────────────
current_page = st.session_state.get("current_page", PAGES[0])
if current_page not in PAGES:
    current_page = PAGES[0]

curr_idx = PAGES.index(current_page)
prev_idx = (curr_idx - 1) % len(PAGES)
next_idx = (curr_idx + 1) % len(PAGES)

top_c1, top_c2, top_c3 = st.columns([2.5, 7, 2.5])
with top_c1:
    if st.button(f"⬅️ {PAGES[prev_idx]}", key="top_prev_btn", use_container_width=True, help=f"Switch to {PAGES[prev_idx]}"):
        st.session_state["current_page"] = PAGES[prev_idx]
        st.rerun()

with top_c2:
    selected_top = st.segmented_control(
        "Page Navigation",
        PAGES,
        default=current_page,
        label_visibility="collapsed"
    )
    if selected_top and selected_top != current_page:
        st.session_state["current_page"] = selected_top
        st.rerun()

with top_c3:
    if st.button(f"{PAGES[next_idx]} ➡️", key="top_next_btn", use_container_width=True, help=f"Switch to {PAGES[next_idx]}"):
        st.session_state["current_page"] = PAGES[next_idx]
        st.rerun()

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
page = st.session_state["current_page"]


# ─────────────────────────────────────────────────────────────
# PAGE 1: Dashboard
# ─────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Fraud Analytics Dashboard</div>
        <div class="hero-subtitle">
            Real-time intelligence across 25,000+ transactions • Dual ML architecture • 
            Logistic Regression + LightGBM + Behavioral Heuristics
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filters - All Model Features
    with st.expander("🎛️ Global Filters (Filter by All Model Features)", expanded=False):
        st.markdown("<div style='color: #94a3b8; font-size: 0.85rem; margin-bottom: 8px;'>Filter the entire dashboard intelligence across all features used by the fraud detection model.</div>", unsafe_allow_html=True)
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            f_cat = st.selectbox("🏪 Merchant Category", ["All"] + MERCHANT_CATEGORIES, key="dash_cat")
            f_pay = st.selectbox("💳 Payment Method", ["All"] + PAYMENT_METHODS, key="dash_pay")
            f_status = st.selectbox("🚨 Fraud Status", ["All", "Fraud", "Genuine"], key="dash_status")
        with fcol2:
            f_loc = st.selectbox("📍 Transaction City", ["All"] + CITIES, key="dash_loc")
            f_risk = st.selectbox("⚠️ Risk Profile", ["All"] + RISK_PROFILES, key="dash_risk")
            f_susp = st.selectbox("⚡ Suspicious Status", ["All", "Suspicious", "Normal"], key="dash_susp")
        with fcol3:
            f_home = st.selectbox("🏠 Customer Home City", ["All"] + CITIES, key="dash_home")
            f_intl = st.selectbox("🌍 International?", ["All", "Yes", "No"], key="dash_intl")
            f_hour_range = st.slider("⏰ Transaction Hour", 0, 23, (0, 23), key="dash_hour")
        with fcol4:
            f_dev = st.selectbox("📱 Device Type", ["All"] + DEVICE_TYPES, key="dash_dev")
            f_mismatch = st.selectbox("🗺️ Location Mismatch?", ["All", "Yes", "No"], key="dash_mismatch")
            f_amt_range = st.slider("💰 Amount Range (₹)", 0, 100000, (0, 100000), step=1000, key="dash_amt")

    filters = {}
    if f_cat != "All":
        filters["merchant_category"] = f_cat
    if f_loc != "All":
        filters["location"] = f_loc
    if f_home != "All":
        filters["home_location"] = f_home
    if f_dev != "All":
        filters["device_type"] = f_dev
    if f_pay != "All":
        filters["payment_method"] = f_pay
    if f_risk != "All":
        filters["risk_profile"] = f_risk
    if f_intl != "All":
        filters["international_transaction"] = f_intl
    if f_mismatch != "All":
        filters["location_mismatch"] = f_mismatch
    if f_status != "All":
        filters["fraud_status"] = f_status
    if f_susp != "All":
        filters["is_suspicious"] = (f_susp == "Suspicious")
    if f_amt_range != (0, 100000):
        filters["min_amount"] = f_amt_range[0]
        filters["max_amount"] = f_amt_range[1]
    if f_hour_range != (0, 23):
        filters["min_hour"] = f_hour_range[0]
        filters["max_hour"] = f_hour_range[1]

    summary = get_dashboard_summary(filters or None)

    if summary:
        # KPI Row
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpis = [
            (k1, "📊", f"{summary['total_transactions']:,}", "Total Transactions"),
            (k2, "🚨", f"{summary['fraud_transactions']:,}", f"Fraud ({summary['fraud_rate_pct']}%)"),
            (k3, "✅", f"{summary['genuine_transactions']:,}", "Genuine"),
            (k4, "💰", f"₹{summary['average_transaction_amount']:,.0f}", "Avg Amount"),
            (k5, "🎯", f"{summary['average_fraud_risk_score']:.1f}", "Avg Risk Score"),
            (k6, "⚡", f"{summary['suspicious_transactions']:,}", f"Suspicious ({summary['suspicious_rate_pct']}%)"),
        ]
        for col, icon, value, label in kpis:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 1
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-header">📊 Fraud Rate by Merchant Category</div>', unsafe_allow_html=True)
            cat_df = pd.DataFrame(summary["breakdown_by_category"])
            if not cat_df.empty:
                fig = px.bar(
                    cat_df, x="category", y="fraud_rate",
                    color="fraud_rate",
                    color_continuous_scale=["#6366f1", "#f43f5e"],
                    labels={"fraud_rate": "Fraud Rate (%)", "category": "Category"},
                    text="fraud_rate"
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">⏰ Hourly Fraud Rate Trend</div>', unsafe_allow_html=True)
            hour_df = pd.DataFrame(summary["breakdown_by_hour"])
            if not hour_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hour_df["hour"], y=hour_df["fraud_rate"],
                    mode="lines+markers",
                    line=dict(color="#22d3ee", width=3, shape="spline"),
                    marker=dict(size=7, color="#22d3ee", line=dict(color="#0e7490", width=1)),
                    fill="tozeroy",
                    fillcolor="rgba(34, 211, 238, 0.08)",
                    name="Fraud Rate %"
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT, height=400,
                    xaxis_title="Hour of Day",
                    yaxis_title="Fraud Rate (%)"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Charts Row 2
        col_left2, col_right2 = st.columns(2)

        with col_left2:
            st.markdown('<div class="section-header">📱 Fraud by Device Type</div>', unsafe_allow_html=True)
            dev_df = pd.DataFrame(summary["breakdown_by_device"])
            if not dev_df.empty:
                fig = px.bar(
                    dev_df, x="device_type", y=["count", "fraud_count"],
                    barmode="group",
                    color_discrete_sequence=["#6366f1", "#f43f5e"],
                    labels={"value": "Count", "device_type": "Device"}
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=380)
                st.plotly_chart(fig, use_container_width=True)

        with col_right2:
            st.markdown('<div class="section-header">🎯 Risk Score Distribution</div>', unsafe_allow_html=True)
            risk_df = pd.DataFrame(summary["risk_score_distribution"])
            if not risk_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=risk_df["bin"], y=risk_df["count"],
                    name="Total",
                    marker_color="#6366f1",
                    marker_line=dict(color="#818cf8", width=1)
                ))
                fig.add_trace(go.Bar(
                    x=risk_df["bin"], y=risk_df["fraud_count"],
                    name="Fraud",
                    marker_color="#f43f5e",
                    marker_line=dict(color="#fb7185", width=1)
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=380, barmode="overlay")
                fig.update_traces(opacity=0.8, selector=dict(name="Total"))
                st.plotly_chart(fig, use_container_width=True)

        # Charts Row 3
        col_left3, col_right3 = st.columns(2)

        with col_left3:
            st.markdown('<div class="section-header">📍 Location Mismatch Impact</div>', unsafe_allow_html=True)
            loc_df = pd.DataFrame(summary["breakdown_by_location_mismatch"])
            if not loc_df.empty:
                fig = px.pie(
                    loc_df, names="location_mismatch", values="fraud_count",
                    color_discrete_sequence=["#10b981", "#f43f5e"],
                    hole=0.5
                )
                fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ['xaxis', 'yaxis']}, height=380)
                fig.update_traces(textinfo="label+percent", textfont_size=13)
                st.plotly_chart(fig, use_container_width=True)

        with col_right3:
            st.markdown('<div class="section-header">👤 Customer Risk Profile vs Fraud</div>', unsafe_allow_html=True)
            prof_df = pd.DataFrame(summary["breakdown_by_risk_profile"])
            if not prof_df.empty:
                fig = px.bar(
                    prof_df, x="risk_profile", y="fraud_rate",
                    color="risk_profile",
                    color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                    labels={"fraud_rate": "Fraud Rate (%)", "risk_profile": "Profile"},
                    text="fraud_rate"
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Payment Method
        st.markdown('<div class="section-header">💳 Fraud Rate by Payment Method</div>', unsafe_allow_html=True)
        pay_df = pd.DataFrame(summary["breakdown_by_payment_method"])
        if not pay_df.empty:
            fig = px.bar(
                pay_df, x="payment_method", y="fraud_rate",
                color="fraud_rate",
                color_continuous_scale=["#8b5cf6", "#f43f5e"],
                labels={"fraud_rate": "Fraud Rate (%)", "payment_method": "Payment Method"},
                text="fraud_rate"
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # ─────────────────────────────────────────────────────────────
        # Section: Interactive Feature Analyzer (Analyze All Model Features)
        # ─────────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="section-header">🔬 Model Feature Deep Dive & Interactive Analyzer</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 16px;">
            Select <b>any feature in the ML model</b> below to inspect its detailed distribution, 
            fraud rate correlation, and risk metrics across all records.
        </div>
        """, unsafe_allow_html=True)

        feature_options = {
            "merchant_category": "🏪 Merchant Category",
            "transaction_location": "📍 Transaction City / Location",
            "customer_home_location": "🏠 Customer Home City",
            "device_type": "📱 Device Type",
            "payment_method": "💳 Payment Method",
            "customer_risk_profile": "⚠️ Customer Risk Profile Tier",
            "international_transaction": "🌍 International Transaction (Cross-Border)",
            "location_mismatch": "🗺️ Location Mismatch (Home vs Txn City)",
            "transaction_hour": "⏰ Transaction Hour (00:00 - 23:00)",
            "transaction_amount": "💰 Transaction Amount Bins (INR)",
            "customer_age": "👤 Customer Age Groups",
            "customer_income_annual": "💼 Annual Income Bins",
            "distance_from_home_km": "📏 Distance From Home Bins (km)",
            "customer_avg_transaction_amount": "💵 Customer Avg Ticket Size",
            "transactions_last_24h": "⚡ Transactions Velocity (Last 24h)",
            "failed_attempts_last_24h": "🔐 Failed Authentication Attempts (Last 24h)",
            "customer_tenure_years": "⏳ Customer Relationship Tenure (Years)",
            "transactions_last_30_days": "📊 Monthly Frequency (Last 30 Days)"
        }

        sel_feature_col = st.selectbox(
            "👉 Select Any Feature in the Model to Analyze:",
            list(feature_options.keys()),
            format_func=lambda k: feature_options[k],
            key="dash_feature_selector"
        )

        feat_data = get_feature_breakdown(sel_feature_col, filters or None)
        if feat_data:
            df_feat = pd.DataFrame(feat_data)

            # Show summary KPIs for this selected feature
            fk1, fk2, fk3, fk4 = st.columns(4)
            highest_fraud_row = df_feat.loc[df_feat["fraud_rate"].idxmax()]
            lowest_fraud_row = df_feat.loc[df_feat["fraud_rate"].idxmin()]
            most_common_row = df_feat.loc[df_feat["count"].idxmax()]

            with fk1:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px;">
                    <div style="font-size: 0.75rem; color: #94a3b8;">HIGHEST FRAUD RATE</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #f43f5e; margin-top: 4px;">{highest_fraud_row['bucket']}</div>
                    <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">{highest_fraud_row['fraud_rate']}% fraud ({highest_fraud_row['fraud_count']:,} txns)</div>
                </div>
                """, unsafe_allow_html=True)
            with fk2:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px;">
                    <div style="font-size: 0.75rem; color: #94a3b8;">SAFEST CATEGORY / BUCKET</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #10b981; margin-top: 4px;">{lowest_fraud_row['bucket']}</div>
                    <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">{lowest_fraud_row['fraud_rate']}% fraud ({lowest_fraud_row['fraud_count']:,} txns)</div>
                </div>
                """, unsafe_allow_html=True)
            with fk3:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px;">
                    <div style="font-size: 0.75rem; color: #94a3b8;">HIGHEST VOLUME</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #22d3ee; margin-top: 4px;">{most_common_row['bucket']}</div>
                    <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">{most_common_row['count']:,} total transactions</div>
                </div>
                """, unsafe_allow_html=True)
            with fk4:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px;">
                    <div style="font-size: 0.75rem; color: #94a3b8;">HIGHEST AVG RISK SCORE</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #f59e0b; margin-top: 4px;">{df_feat.loc[df_feat['avg_risk'].idxmax()]['bucket']}</div>
                    <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">Score: {df_feat['avg_risk'].max():.1f} / 100</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Dual-Axis Plotly Chart
            fig_feature = make_subplots(specs=[[{"secondary_y": True}]])
            fig_feature.add_trace(
                go.Bar(
                    name="Genuine Transactions",
                    x=df_feat["bucket"],
                    y=df_feat["genuine_count"],
                    marker_color="#3b82f6",
                    opacity=0.85
                ),
                secondary_y=False
            )
            fig_feature.add_trace(
                go.Bar(
                    name="Fraud Transactions",
                    x=df_feat["bucket"],
                    y=df_feat["fraud_count"],
                    marker_color="#ef4444",
                    opacity=0.9
                ),
                secondary_y=False
            )
            fig_feature.add_trace(
                go.Scatter(
                    name="Fraud Rate (%)",
                    x=df_feat["bucket"],
                    y=df_feat["fraud_rate"],
                    mode="lines+markers+text",
                    text=[f"{r:.1f}%" for r in df_feat["fraud_rate"]],
                    textposition="top center",
                    textfont=dict(color="#f87171", size=11),
                    line=dict(color="#f43f5e", width=3),
                    marker=dict(size=8, color="#f43f5e")
                ),
                secondary_y=True
            )
            feat_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"}
            fig_feature.update_layout(
                **feat_layout,
                height=450,
                barmode="stack",
                title=f"Volume & Fraud Rate Breakdown by {feature_options[sel_feature_col]}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_feature.update_yaxes(title_text="Transaction Count", secondary_y=False, gridcolor="rgba(99,102,241,0.08)")
            fig_feature.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True, gridcolor="rgba(244,63,94,0.08)")
            st.plotly_chart(fig_feature, use_container_width=True)

            # Detailed Data Breakdown Table
            with st.expander(f"📋 View Full Data Table for {feature_options[sel_feature_col]}", expanded=False):
                display_df = df_feat.copy()
                display_df = display_df.rename(columns={
                    "bucket": "Feature Value / Bucket",
                    "count": "Total Volume",
                    "genuine_count": "Genuine Count",
                    "fraud_count": "Fraud Count",
                    "fraud_rate": "Fraud Rate (%)",
                    "avg_risk": "Avg Risk Score (0-100)",
                    "avg_amount": "Avg Amount (₹)"
                })
                display_df["Avg Amount (₹)"] = display_df["Avg Amount (₹)"].apply(lambda x: f"₹{x:,.2f}")
                display_df["Fraud Rate (%)"] = display_df["Fraud Rate (%)"].apply(lambda x: f"{x:.2f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No database available. Please seed the database first.")


# ─────────────────────────────────────────────────────────────
# PAGE 2: Transaction Check
# ─────────────────────────────────────────────────────────────
elif page == "🔍 Transaction Check":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔍 Real-Time Transaction Check</div>
        <div class="hero-subtitle">
            Score any transaction instantly using dual ML models • Get fraud probability, 
            risk score, and AI-driven explainability reasons
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Presets
    st.markdown('<div class="section-header">⚡ Quick Test Presets</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        if st.button("✅ Safe Transaction", use_container_width=True, type="secondary"):
            st.session_state["preset"] = {
                "amount": 2500.0, "avg_amount": 2400.0, "category": "Grocery", "age": 32,
                "income": 750000, "risk": "Low", "home": "Mumbai",
                "txn_loc": "Mumbai", "distance": 0.0, "loc_mismatch": "No",
                "device": "Mobile", "payment": "Online",
                "intl": "No", "fails": 0, "tx24h": 2, "tenure": 4.0, "tx30d": 18
            }
    with pc2:
        if st.button("⚠️ Medium Risk", use_container_width=True, type="secondary"):
            st.session_state["preset"] = {
                "amount": 15000.0, "avg_amount": 5000.0, "category": "Electronics", "age": 28,
                "income": 500000, "risk": "Medium", "home": "Delhi",
                "txn_loc": "Jaipur", "distance": 280.0, "loc_mismatch": "Yes",
                "device": "Laptop", "payment": "Contactless",
                "intl": "No", "fails": 1, "tx24h": 4, "tenure": 2.0, "tx30d": 22
            }
    with pc3:
        if st.button("🚨 High Risk / Fraud", use_container_width=True, type="primary"):
            st.session_state["preset"] = {
                "amount": 48000.0, "avg_amount": 3500.0, "category": "Jewelry", "age": 45,
                "income": 400000, "risk": "High", "home": "Chennai",
                "txn_loc": "Delhi", "distance": 1750.0, "loc_mismatch": "Yes",
                "device": "Tablet", "payment": "Mobile Wallet",
                "intl": "Yes", "fails": 3, "tx24h": 7, "tenure": 1.0, "tx30d": 35
            }

    preset = st.session_state.get("preset", {})

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form - All Model Features
    with st.form("txn_form", clear_on_submit=False):
        st.markdown('<div class="section-header">📝 Transaction Details (All Model Features)</div>', unsafe_allow_html=True)

        # Section 1: Financial & Core Transaction Attributes
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #a5b4fc; margin-bottom: 8px;'>💰 FINANCIAL & TRANSACTION BASICS</div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            amount = st.number_input("Transaction Amount (₹)", min_value=1.0,
                                     value=float(preset.get("amount", 5000.0)), step=100.0)
        with c2:
            avg_amount = st.number_input("Customer Avg Ticket (₹)", min_value=1.0,
                                         value=float(preset.get("avg_amount", 5000.0)), step=100.0)
        with c3:
            category = st.selectbox("Merchant Category", MERCHANT_CATEGORIES,
                                    index=MERCHANT_CATEGORIES.index(preset.get("category", "Online Shopping")))
        with c4:
            txn_date = st.date_input("Transaction Date", value=datetime.now().date())
        with c5:
            txn_time = st.time_input("Transaction Time", value=datetime.now().time())

        # Section 2: Geospatial & Distance Attributes
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #a5b4fc; margin: 16px 0 8px;'>📍 GEOSPATIAL & LOCATION ATTRIBUTES</div>", unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            home_loc = st.selectbox("🏠 Customer Home City", CITIES,
                                    index=CITIES.index(preset.get("home", "Mumbai")))
        with g2:
            txn_loc = st.selectbox("📍 Transaction City", CITIES,
                                   index=CITIES.index(preset.get("txn_loc", "Mumbai")))
        with g3:
            default_dist = preset.get("distance", 0.0 if home_loc == txn_loc else 350.0)
            distance = st.number_input("📏 Distance From Home (km)", min_value=0.0,
                                       value=float(default_dist), step=25.0)
        with g4:
            mismatch_opts = ["Auto-Detect", "Yes", "No"]
            preset_mismatch = preset.get("loc_mismatch", "Auto-Detect")
            loc_mismatch_choice = st.selectbox("🗺️ Location Mismatch?", mismatch_opts,
                                              index=mismatch_opts.index(preset_mismatch) if preset_mismatch in mismatch_opts else 0)

        # Section 3: Customer & Account Profile
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #a5b4fc; margin: 16px 0 8px;'>👤 CUSTOMER PROFILE & BEHAVIORAL HISTORY</div>", unsafe_allow_html=True)
        cp1, cp2, cp3, cp4, cp5 = st.columns(5)
        with cp1:
            age = st.number_input("Customer Age", min_value=18, max_value=100,
                                  value=int(preset.get("age", 35)))
        with cp2:
            income = st.number_input("Annual Income (₹)", min_value=0,
                                     value=int(preset.get("income", 600000)), step=50000)
        with cp3:
            risk_profile = st.selectbox("Risk Profile Tier", RISK_PROFILES,
                                        index=RISK_PROFILES.index(preset.get("risk", "Medium")))
        with cp4:
            tenure = st.number_input("Customer Tenure (years)", min_value=0.0,
                                     value=float(preset.get("tenure", 3.0)), step=0.5)
        with cp5:
            txns_30d = st.number_input("Txns Last 30 Days", min_value=0,
                                       value=int(preset.get("tx30d", 15)))

        # Section 4: Channel, Device & Security Risk
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #a5b4fc; margin: 16px 0 8px;'>📱 CHANNEL, DEVICE & VELOCITY ATTRIBUTES</div>", unsafe_allow_html=True)
        cd1, cd2, cd3, cd4, cd5 = st.columns(5)
        with cd1:
            device = st.selectbox("Device Type", DEVICE_TYPES,
                                  index=DEVICE_TYPES.index(preset.get("device", "Mobile")))
        with cd2:
            payment = st.selectbox("Payment Method", PAYMENT_METHODS,
                                   index=PAYMENT_METHODS.index(preset.get("payment", "Online")))
        with cd3:
            intl = st.selectbox("International?", ["No", "Yes"],
                                index=1 if preset.get("intl") == "Yes" else 0)
        with cd4:
            txns_24h = st.number_input("Transactions (Last 24h)", min_value=0,
                                       value=int(preset.get("tx24h", 1)))
        with cd5:
            failed_attempts = st.number_input("Failed Attempts (Last 24h)", min_value=0,
                                              value=int(preset.get("fails", 0)))

        # Live Derived Features Indicator
        calc_ratio = amount / (avg_amount + 1e-5)
        calc_velocity = txns_24h + failed_attempts
        calc_night = "Yes 🌙" if 0 <= txn_time.hour <= 5 else "No ☀️"
        calc_mismatch = (loc_mismatch_choice if loc_mismatch_choice != "Auto-Detect" else ("Yes ⚠️" if home_loc != txn_loc else "No ✅"))

        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px dashed rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 10px 16px; margin: 14px 0;">
            <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">📐 DERIVED MODEL FEATURES PREVIEW:</span>&nbsp;&nbsp;
            <span style="font-size: 0.8rem; color: #cbd5e1;">Spending Ratio: <b>{calc_ratio:.2f}x</b></span> &nbsp;•&nbsp;
            <span style="font-size: 0.8rem; color: #cbd5e1;">Velocity Score: <b>{calc_velocity}</b></span> &nbsp;•&nbsp;
            <span style="font-size: 0.8rem; color: #cbd5e1;">Late Night Flag: <b>{calc_night}</b></span> &nbsp;•&nbsp;
            <span style="font-size: 0.8rem; color: #cbd5e1;">Location Mismatch: <b>{calc_mismatch}</b></span>
        </div>
        """, unsafe_allow_html=True)

        submitted = st.form_submit_button("🚀 Analyze Transaction with All Features", use_container_width=True, type="primary")

    if submitted and engine.get("loaded"):
        txn_datetime = datetime.combine(txn_date, txn_time).strftime("%Y-%m-%d %H:%M:%S")
        txn_dict = {
            "transaction_amount": amount,
            "customer_avg_transaction_amount": avg_amount,
            "merchant_category": category,
            "transaction_datetime": txn_datetime,
            "customer_age": age,
            "customer_income_annual": income,
            "customer_risk_profile": risk_profile,
            "customer_home_location": home_loc,
            "transaction_location": txn_loc,
            "distance_from_home_km": distance,
            "location_mismatch": None if loc_mismatch_choice == "Auto-Detect" else loc_mismatch_choice,
            "device_type": device,
            "payment_method": payment,
            "international_transaction": intl,
            "customer_tenure_years": tenure,
            "transactions_last_30_days": txns_30d,
            "failed_attempts_last_24h": failed_attempts,
            "transactions_last_24h": txns_24h,
        }

        with st.spinner("🔄 Running ML inference..."):
            result = predict_one(engine, txn_dict)

        st.markdown("---")
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)

        # Results layout
        res_left, res_mid, res_right = st.columns([1, 1, 1])

        with res_left:
            gauge_fig = render_risk_gauge(
                result["risk_score"], result["fraud_probability"],
                result["fraud_status"], result["is_suspicious"]
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with res_mid:
            st.markdown("<br>", unsafe_allow_html=True)

            # Fraud Status Badge
            badge_class = "badge-fraud" if result["fraud_status"] == "Fraud" else "badge-genuine"
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 16px;">
                <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 6px;">CLASSIFICATION</div>
                <span class="status-badge {badge_class}" style="font-size: 1.1rem;">
                    {'🚨' if result['fraud_status'] == 'Fraud' else '✅'} {result['fraud_status']}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Suspicious Badge
            if result["is_suspicious"]:
                st.markdown("""
                <div style="text-align: center; margin-bottom: 16px;">
                    <span class="status-badge badge-suspicious">⚡ SUSPICIOUS</span>
                </div>
                """, unsafe_allow_html=True)

            # Probability
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #94a3b8; font-size: 0.8rem;">FRAUD PROBABILITY</div>
                <div style="font-size: 2rem; font-weight: 700; color: {'#f43f5e' if result['fraud_probability'] >= 0.5 else '#10b981'};">
                    {result['fraud_probability']*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_right:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px;">🧠 AI EXPLAINABILITY DRIVERS</div>', unsafe_allow_html=True)
            for reason in result["top_reasons"]:
                st.markdown(f"""
                <div class="reason-card">
                    <div>{reason}</div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3: Batch Upload
# ─────────────────────────────────────────────────────────────
elif page == "📦 Batch Upload":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📦 Batch CSV Scoring</div>
        <div class="hero-subtitle">
            Upload a CSV file with multiple transactions for bulk fraud scoring • 
            Download results with risk scores and explainability
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample CSV Template
    with st.expander("📥 Download Sample CSV Template"):
        sample_data = {
            "transaction_amount": [2500, 15000, 48000],
            "merchant_category": ["Grocery", "Electronics", "Jewelry"],
            "transaction_datetime": ["2025-06-15 10:30:00", "2025-06-15 14:00:00", "2025-06-16 02:15:00"],
            "customer_age": [32, 28, 45],
            "customer_income_annual": [750000, 500000, 400000],
            "customer_risk_profile": ["Low", "Medium", "High"],
            "customer_home_location": ["Mumbai", "Delhi", "Chennai"],
            "transaction_location": ["Mumbai", "Jaipur", "Delhi"],
            "device_type": ["Mobile", "Laptop", "Tablet"],
            "payment_method": ["Online", "Contactless", "Mobile Wallet"],
            "international_transaction": ["No", "No", "Yes"],
            "failed_attempts_last_24h": [0, 1, 3],
            "transactions_last_24h": [2, 4, 7],
        }
        sample_df = pd.DataFrame(sample_data)
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Template CSV",
            csv_buffer.getvalue(),
            "fraud_check_template.csv",
            "text/csv",
            use_container_width=True
        )
        st.dataframe(sample_df, use_container_width=True)

    # File Upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV with transaction data for batch scoring"
    )

    if uploaded_file is not None and engine.get("loaded"):
        df_upload = pd.read_csv(uploaded_file)
        st.markdown(f"**Uploaded:** {len(df_upload)} transactions")

        if st.button("🚀 Score All Transactions", type="primary", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, row in df_upload.iterrows():
                txn_dict = row.to_dict()
                # Ensure required fields have defaults
                for field in ["device_type", "payment_method", "international_transaction"]:
                    if field not in txn_dict or pd.isna(txn_dict.get(field)):
                        if field == "device_type":
                            txn_dict[field] = "Mobile"
                        elif field == "payment_method":
                            txn_dict[field] = "Online"
                        elif field == "international_transaction":
                            txn_dict[field] = "No"

                for field in ["failed_attempts_last_24h", "transactions_last_24h"]:
                    if field not in txn_dict or pd.isna(txn_dict.get(field)):
                        txn_dict[field] = 0 if "failed" in field else 1

                try:
                    pred = predict_one(engine, txn_dict)
                    results.append({
                        "transaction_id": txn_dict.get("transaction_id", f"BATCH_{idx+1:04d}"),
                        "transaction_amount": txn_dict["transaction_amount"],
                        "merchant_category": txn_dict.get("merchant_category", ""),
                        "fraud_status": pred["fraud_status"],
                        "fraud_probability": pred["fraud_probability"],
                        "risk_score": pred["risk_score"],
                        "is_suspicious": pred["is_suspicious"],
                        "top_reason": pred["top_reasons"][0] if pred["top_reasons"] else "",
                    })
                except Exception as e:
                    results.append({
                        "transaction_id": f"BATCH_{idx+1:04d}",
                        "transaction_amount": txn_dict.get("transaction_amount", 0),
                        "merchant_category": txn_dict.get("merchant_category", ""),
                        "fraud_status": "ERROR",
                        "fraud_probability": 0,
                        "risk_score": 0,
                        "is_suspicious": False,
                        "top_reason": str(e),
                    })

                progress_bar.progress((idx + 1) / len(df_upload))
                status_text.text(f"Processing {idx+1}/{len(df_upload)}...")

            status_text.text("✅ Batch scoring complete!")
            results_df = pd.DataFrame(results)

            # Summary KPIs
            st.markdown("---")
            fraud_count = len(results_df[results_df["fraud_status"] == "Fraud"])
            susp_count = len(results_df[results_df["is_suspicious"] == True])
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Total Processed", len(results_df))
            with bc2:
                st.metric("🚨 Fraud Detected", fraud_count)
            with bc3:
                st.metric("⚡ Suspicious", susp_count)

            # Results Table
            st.markdown('<div class="section-header">📊 Scored Results</div>', unsafe_allow_html=True)
            st.dataframe(
                results_df.style.applymap(
                    lambda v: "background-color: rgba(239,68,68,0.15); color: #f87171;"
                    if v == "Fraud" else ("background-color: rgba(16,185,129,0.15); color: #34d399;" if v == "Genuine" else ""),
                    subset=["fraud_status"]
                ),
                use_container_width=True,
                height=400
            )

            # Download
            csv_out = io.StringIO()
            results_df.to_csv(csv_out, index=False)
            st.download_button(
                "⬇️ Download Scored CSV",
                csv_out.getvalue(),
                "batch_scored_results.csv",
                "text/csv",
                use_container_width=True,
                type="primary"
            )


# ─────────────────────────────────────────────────────────────
# PAGE 4: Transaction Explorer
# ─────────────────────────────────────────────────────────────
elif page == "📋 Transaction Explorer":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📋 Transaction Explorer</div>
        <div class="hero-subtitle">
            Search, filter, and explore all 25,000+ scored transactions in the database • 
            Drill-down into individual records
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not db_exists():
        st.warning("⚠️ Database not found. Please seed the database first.")
    else:
        # Filters
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        with fc1:
            search = st.text_input("🔍 Search ID/Location", "")
        with fc2:
            fs = st.selectbox("Fraud Status", ["All", "Fraud", "Genuine"], key="exp_fs")
        with fc3:
            mc = st.selectbox("Category", ["All"] + MERCHANT_CATEGORIES, key="exp_mc")
        with fc4:
            rl = st.selectbox("Risk Level", ["All", "Low", "Medium", "High"], key="exp_rl")
        with fc5:
            susp = st.selectbox("Suspicious", ["All", "Yes", "No"], key="exp_susp")

        with st.expander("🔧 Additional Feature Filters", expanded=False):
            ef1, ef2, ef3, ef4 = st.columns(4)
            with ef1:
                dev = st.selectbox("Device Type", ["All"] + DEVICE_TYPES, key="exp_dev")
            with ef2:
                pay = st.selectbox("Payment Method", ["All"] + PAYMENT_METHODS, key="exp_pay")
            with ef3:
                intl = st.selectbox("International?", ["All", "Yes", "No"], key="exp_intl")
            with ef4:
                mismatch = st.selectbox("Location Mismatch?", ["All", "Yes", "No"], key="exp_mismatch")

        filters = {}
        if search:
            filters["search"] = search
        if fs != "All":
            filters["fraud_status"] = fs
        if mc != "All":
            filters["merchant_category"] = mc
        if rl != "All":
            filters["risk_level"] = rl
        if susp != "All":
            filters["is_suspicious"] = susp == "Yes"
        if dev != "All":
            filters["device_type"] = dev
        if pay != "All":
            filters["payment_method"] = pay
        if intl != "All":
            filters["international_transaction"] = intl
        if mismatch != "All":
            filters["location_mismatch"] = mismatch

        # Pagination
        page_size = 50
        if "explorer_page" not in st.session_state:
            st.session_state["explorer_page"] = 1

        df_txns, total = get_transactions_df(
            page=st.session_state["explorer_page"],
            page_size=page_size,
            filters=filters or None
        )

        total_pages = max(1, (total + page_size - 1) // page_size)

        st.markdown(f"**Showing {len(df_txns)} of {total:,} transactions** | Page {st.session_state['explorer_page']} of {total_pages}")

        # Navigation
        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("⬅️ Previous", disabled=st.session_state["explorer_page"] <= 1, use_container_width=True):
                st.session_state["explorer_page"] -= 1
                st.rerun()
        with nav3:
            if st.button("➡️ Next", disabled=st.session_state["explorer_page"] >= total_pages, use_container_width=True):
                st.session_state["explorer_page"] += 1
                st.rerun()

        if not df_txns.empty:
            # Select columns to display
            display_cols = [
                "transaction_id", "transaction_amount", "merchant_category",
                "transaction_datetime", "transaction_location", "fraud_status",
                "fraud_probability", "fraud_risk_score", "is_suspicious",
                "customer_risk_profile", "device_type", "payment_method"
            ]
            available_cols = [c for c in display_cols if c in df_txns.columns]
            st.dataframe(
                df_txns[available_cols],
                use_container_width=True,
                height=500
            )

            # Detail View
            st.markdown("---")
            st.markdown('<div class="section-header">🔎 Transaction Detail View</div>', unsafe_allow_html=True)
            selected_id = st.selectbox(
                "Select Transaction ID for details",
                df_txns["transaction_id"].tolist() if "transaction_id" in df_txns.columns else []
            )

            if selected_id:
                detail = df_txns[df_txns["transaction_id"] == selected_id].iloc[0]
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.markdown("**Transaction Info**")
                    st.write(f"**ID:** {detail.get('transaction_id', 'N/A')}")
                    st.write(f"**Amount:** ₹{detail.get('transaction_amount', 0):,.2f}")
                    st.write(f"**Category:** {detail.get('merchant_category', 'N/A')}")
                    st.write(f"**Date:** {detail.get('transaction_datetime', 'N/A')}")
                    st.write(f"**Location:** {detail.get('transaction_location', 'N/A')}")
                with dc2:
                    st.markdown("**Customer Info**")
                    st.write(f"**Age:** {detail.get('customer_age', 'N/A')}")
                    st.write(f"**Income:** ₹{detail.get('customer_income_annual', 0):,}")
                    st.write(f"**Home:** {detail.get('customer_home_location', 'N/A')}")
                    st.write(f"**Risk Profile:** {detail.get('customer_risk_profile', 'N/A')}")
                    st.write(f"**Device:** {detail.get('device_type', 'N/A')}")
                with dc3:
                    st.markdown("**ML Results**")
                    badge = "🚨 Fraud" if detail.get("fraud_status") == "Fraud" else "✅ Genuine"
                    st.write(f"**Status:** {badge}")
                    st.write(f"**Fraud Probability:** {detail.get('fraud_probability', 0)*100:.1f}%")
                    st.write(f"**Risk Score:** {detail.get('fraud_risk_score', 0):.1f}/100")
                    st.write(f"**Suspicious:** {'⚡ Yes' if detail.get('is_suspicious') else 'No'}")
                    reasons = detail.get("top_reasons", "[]")
                    if isinstance(reasons, str):
                        try:
                            reasons = json.loads(reasons)
                        except Exception:
                            reasons = []
                    if reasons:
                        st.markdown("**Reasons:**")
                        for r in reasons:
                            st.write(f"• {r}")


# ─────────────────────────────────────────────────────────────
# PAGE 5: Model Insights
# ─────────────────────────────────────────────────────────────
elif page == "🧠 Model Insights":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🧠 Model Insights & Explainability</div>
        <div class="hero-subtitle">
            Comprehensive model performance metrics, feature importances, EDA visualizations, 
            and threshold configurations
        </div>
    </div>
    """, unsafe_allow_html=True)

    if engine.get("loaded"):
        metrics = engine["metrics_summary"]

        # Model Cards
        st.markdown('<div class="section-header">🏆 Best Models Selected</div>', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)

        with mc1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-value">Logistic Regression</div>
                <div class="kpi-label">Best Fraud Classifier</div>
                <div style="margin-top: 12px; color: #94a3b8; font-size: 0.85rem;">
                    Recall: 64.71% • ROC-AUC: 0.7148 • PR-AUC: 0.2029
                </div>
            </div>
            """, unsafe_allow_html=True)

        with mc2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📈</div>
                <div class="kpi-value">LightGBM Regressor</div>
                <div class="kpi-label">Best Risk Score Model</div>
                <div style="margin-top: 12px; color: #94a3b8; font-size: 0.85rem;">
                    R²: 0.8008 • MAE: 5.13 • RMSE: 6.90
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Classifier Comparison
        st.markdown('<div class="section-header">📊 Classification Model Comparison</div>', unsafe_allow_html=True)
        clf_data = metrics["classification_models"]
        clf_df = pd.DataFrame([
            {"Model": name, "Precision": m["precision"], "Recall": m["recall"],
             "F1": m["f1"], "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"]}
            for name, m in clf_data.items()
        ])

        fig = go.Figure()
        for metric_name in ["Recall", "ROC-AUC", "PR-AUC", "Precision"]:
            fig.add_trace(go.Bar(
                name=metric_name,
                x=clf_df["Model"],
                y=clf_df[metric_name],
                text=[f"{v:.3f}" for v in clf_df[metric_name]],
                textposition="outside",
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=420, barmode="group",
            title="Classification Metrics Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Regressor Comparison
        st.markdown('<div class="section-header">📈 Regression Model Comparison</div>', unsafe_allow_html=True)
        reg_data = metrics["regression_models"]
        reg_df = pd.DataFrame([
            {"Model": name, "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"]}
            for name, m in reg_data.items()
        ])

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            name="R²", x=reg_df["Model"], y=reg_df["R²"],
            marker_color="#6366f1", text=[f"{v:.4f}" for v in reg_df["R²"]],
            textposition="outside"
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            name="MAE", x=reg_df["Model"], y=reg_df["MAE"],
            mode="lines+markers", marker=dict(color="#f43f5e", size=10),
            line=dict(color="#f43f5e", width=2)
        ), secondary_y=True)
        fig.update_layout(**PLOTLY_LAYOUT, height=400, title="Regression Metrics (R² vs MAE)")
        fig.update_yaxes(title_text="R²", secondary_y=False, gridcolor="rgba(99,102,241,0.08)")
        fig.update_yaxes(title_text="MAE", secondary_y=True, gridcolor="rgba(99,102,241,0.08)")
        st.plotly_chart(fig, use_container_width=True)

        # Feature Importances
        st.markdown('<div class="section-header">🔬 Feature Importances & Model Weights</div>', unsafe_allow_html=True)
        st.markdown("Analyze the relative importance and coefficients for all 55 features utilized by the model (including one-hot categories and engineered behavioral features).")
        
        fi = engine["feature_importances"]
        fi_all_df = pd.DataFrame([
            {"Feature": k, "Importance": v}
            for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)
        ])
        
        f_ctrl1, f_ctrl2 = st.columns([1, 2])
        with f_ctrl1:
            fi_scope = st.radio("Display Scope:", ["Top 15 Primary Drivers", "Top 25 Important Features", "All 55 Features in Model"], horizontal=True, key="fi_scope_radio")
        with f_ctrl2:
            fi_search = st.text_input("🔍 Search Any Feature by Name", "", key="fi_search_box")

        plot_df = fi_all_df.copy()
        if fi_search:
            plot_df = plot_df[plot_df["Feature"].str.contains(fi_search, case=False)]
        elif fi_scope == "Top 15 Primary Drivers":
            plot_df = plot_df.head(15)
        elif fi_scope == "Top 25 Important Features":
            plot_df = plot_df.head(25)

        chart_height = max(420, min(1200, len(plot_df) * 22))
        fig = px.bar(
            plot_df, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=["#6366f1", "#22d3ee", "#f43f5e"],
            labels={"Importance": "Model Weight / Absolute Importance", "Feature": "Feature Name"}
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=chart_height, showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed", gridcolor="rgba(99,102,241,0.08)"))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 View Complete 55-Feature Catalog with Classifications & Weights", expanded=False):
            catalog_rows = []
            for _, r in fi_all_df.iterrows():
                fname = r["Feature"]
                imp = r["Importance"]
                if "Merchant_Category" in fname:
                    cat = "Merchant Category (One-Hot)"
                elif "Device_Type" in fname:
                    cat = "Device Type (One-Hot)"
                elif "Payment_Method" in fname:
                    cat = "Payment Method (One-Hot)"
                elif "Customer_Home_Location" in fname:
                    cat = "Customer Home City (One-Hot)"
                elif "Transaction_Location" in fname:
                    cat = "Transaction City (One-Hot)"
                elif fname in ["is_night_transaction", "amount_vs_avg_ratio", "velocity_score", "day_of_week"]:
                    cat = "Engineered Behavioral Feature"
                elif fname in ["Location_Mismatch_Bin", "International_Transaction_Bin", "Customer_Risk_Profile_Ord"]:
                    cat = "Encoded Ordinal / Binary Attribute"
                else:
                    cat = "Normalized Numerical Feature"
                catalog_rows.append({"Feature Name": fname, "Feature Category": cat, "Absolute Importance Score": f"{imp:.4f}"})
            st.dataframe(pd.DataFrame(catalog_rows), use_container_width=True, hide_index=True)

        # Threshold Config
        st.markdown('<div class="section-header">⚙️ Threshold Configuration</div>', unsafe_allow_html=True)
        thresh = engine["threshold_config"]
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Fraud Probability Threshold</div>
                <div class="kpi-value">{thresh['fraud_probability_threshold']*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with tc2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Risk Score Suspicious Threshold</div>
                <div class="kpi-value">{thresh['risk_score_suspicious_threshold']:.0f}/100</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Suspicious Logic:** {thresh.get('suspicious_logic_description', 'N/A')}")

        # Training info
        st.markdown('<div class="section-header">📦 Training Summary</div>', unsafe_allow_html=True)
        ti1, ti2, ti3 = st.columns(3)
        with ti1:
            st.metric("Training Samples", f"{metrics['train_samples']:,}")
        with ti2:
            st.metric("Test Samples", f"{metrics['test_samples']:,}")
        with ti3:
            st.metric("Total Features", metrics["total_features"])

        # EDA Outputs
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Exploratory Data Analysis (EDA) Outputs</div>', unsafe_allow_html=True)

        eda_files = sorted(EDA_DIR.glob("*.png"))
        if eda_files:
            eda_names = {
                "eda_1": "Class Imbalance Analysis",
                "eda_2": "Transaction Amount vs Fraud",
                "eda_3": "Hourly Fraud Concentration",
                "eda_4": "Distance vs Fraud Density",
                "eda_5": "Location Mismatch vs Fraud",
                "eda_6": "Failed Attempts vs Fraud",
                "eda_7": "Correlation Heatmap",
                "eda_8": "Risk Score Distribution",
                "eda_9": "Feature Importances"
            }

            # Display in 3-column grid
            for i in range(0, len(eda_files), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(eda_files):
                        eda_file = eda_files[i + j]
                        prefix = eda_file.stem.rsplit("_", 1)[0] if "_" in eda_file.stem else eda_file.stem
                        # Get the eda_N prefix
                        parts = eda_file.stem.split("_")
                        key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else eda_file.stem
                        title = eda_names.get(key, eda_file.stem)
                        with col:
                            st.markdown(f"**{title}**")
                            st.image(str(eda_file), use_container_width=True)
        else:
            st.warning("No EDA outputs found. Run `train_pipeline.py` first.")
    else:
        st.error(f"ML Engine failed to load: {engine.get('error', 'Unknown error')}")


# ─────────────────────────────────────────────────────────────
# Bottom Arrow Navigation (Switch Between Options)
# ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
bot_c1, bot_c2, bot_c3 = st.columns([3, 4, 3])
with bot_c1:
    if st.button(f"⬅️ Previous: {PAGES[prev_idx]}", key="bot_arrow_prev", use_container_width=True):
        st.session_state["current_page"] = PAGES[prev_idx]
        st.rerun()
with bot_c2:
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding-top: 8px;">
        Viewing Option <b>{curr_idx + 1} of {len(PAGES)}</b>: <span style="color: #a5b4fc; font-weight: 600;">{current_page}</span>
    </div>
    """, unsafe_allow_html=True)
with bot_c3:
    if st.button(f"Next: {PAGES[next_idx]} ➡️", key="bot_arrow_next", use_container_width=True, type="primary"):
        st.session_state["current_page"] = PAGES[next_idx]
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🛡️ FraudShield AI v2.0 — Credit Card Fraud Detection & Transaction Risk Intelligence<br>
    Dual ML Architecture: Logistic Regression + LightGBM + Behavioral Heuristics<br>
    Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)

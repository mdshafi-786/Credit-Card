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

    /* Hide default Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
# Dashboard Data Queries
# ─────────────────────────────────────────────────────────────
def get_dashboard_summary(filters=None):
    """Get aggregated dashboard summary from SQLite."""
    if not db_exists():
        return None

    conditions = []
    params = []
    if filters:
        if filters.get("merchant_category"):
            conditions.append("merchant_category = ?")
            params.append(filters["merchant_category"])
        if filters.get("location"):
            conditions.append("transaction_location = ?")
            params.append(filters["location"])
        if filters.get("device_type"):
            conditions.append("device_type = ?")
            params.append(filters["device_type"])
        if filters.get("risk_profile"):
            conditions.append("customer_risk_profile = ?")
            params.append(filters["risk_profile"])

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


def get_transactions_df(page=1, page_size=50, filters=None):
    """Fetch paginated transaction data as DataFrame."""
    if not db_exists():
        return pd.DataFrame(), 0

    conditions = []
    params = []
    if filters:
        if filters.get("search"):
            conditions.append("(transaction_id LIKE ? OR transaction_location LIKE ?)")
            term = f"%{filters['search']}%"
            params.extend([term, term])
        if filters.get("fraud_status"):
            conditions.append("fraud_status = ?")
            params.append(filters["fraud_status"])
        if filters.get("merchant_category"):
            conditions.append("merchant_category = ?")
            params.append(filters["merchant_category"])
        if filters.get("is_suspicious") is not None:
            conditions.append("is_suspicious = ?")
            params.append(1 if filters["is_suspicious"] else 0)
        if filters.get("risk_level"):
            rl = filters["risk_level"]
            if rl == "Low":
                conditions.append("fraud_risk_score < 40.0")
            elif rl == "Medium":
                conditions.append("fraud_risk_score >= 40.0 AND fraud_risk_score < 70.0")
            elif rl == "High":
                conditions.append("fraud_risk_score >= 70.0")

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
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ['xaxis', 'yaxis']},
        margin=dict(l=30, r=30, t=60, b=20)
    )
    return fig


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

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Transaction Check", "📦 Batch Upload",
         "📋 Transaction Explorer", "🧠 Model Insights"],
        label_visibility="collapsed"
    )

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

    # Filters
    with st.expander("🎛️ Global Filters", expanded=False):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            f_cat = st.selectbox("Merchant Category", ["All"] + MERCHANT_CATEGORIES, key="dash_cat")
        with fcol2:
            f_loc = st.selectbox("Location", ["All"] + CITIES, key="dash_loc")
        with fcol3:
            f_dev = st.selectbox("Device Type", ["All"] + DEVICE_TYPES, key="dash_dev")
        with fcol4:
            f_risk = st.selectbox("Risk Profile", ["All"] + RISK_PROFILES, key="dash_risk")

    filters = {}
    if f_cat != "All":
        filters["merchant_category"] = f_cat
    if f_loc != "All":
        filters["location"] = f_loc
    if f_dev != "All":
        filters["device_type"] = f_dev
    if f_risk != "All":
        filters["risk_profile"] = f_risk

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
                "amount": 2500.0, "category": "Grocery", "age": 32,
                "income": 750000, "risk": "Low", "home": "Mumbai",
                "txn_loc": "Mumbai", "device": "Mobile", "payment": "Online",
                "intl": "No", "fails": 0, "tx24h": 2
            }
    with pc2:
        if st.button("⚠️ Medium Risk", use_container_width=True, type="secondary"):
            st.session_state["preset"] = {
                "amount": 15000.0, "category": "Electronics", "age": 28,
                "income": 500000, "risk": "Medium", "home": "Delhi",
                "txn_loc": "Jaipur", "device": "Laptop", "payment": "Contactless",
                "intl": "No", "fails": 1, "tx24h": 4
            }
    with pc3:
        if st.button("🚨 High Risk / Fraud", use_container_width=True, type="primary"):
            st.session_state["preset"] = {
                "amount": 48000.0, "category": "Jewelry", "age": 45,
                "income": 400000, "risk": "High", "home": "Chennai",
                "txn_loc": "Delhi", "device": "Tablet", "payment": "Mobile Wallet",
                "intl": "Yes", "fails": 3, "tx24h": 7
            }

    preset = st.session_state.get("preset", {})

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form
    with st.form("txn_form", clear_on_submit=False):
        st.markdown('<div class="section-header">📝 Transaction Details</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            amount = st.number_input("💰 Transaction Amount (₹)", min_value=1.0,
                                     value=preset.get("amount", 5000.0), step=100.0)
        with c2:
            category = st.selectbox("🏪 Merchant Category", MERCHANT_CATEGORIES,
                                    index=MERCHANT_CATEGORIES.index(preset.get("category", "Online Shopping")))
        with c3:
            txn_date = st.date_input("📅 Transaction Date", value=datetime.now().date())
        with c4:
            txn_time = st.time_input("🕐 Transaction Time", value=datetime.now().time())

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            age = st.number_input("👤 Customer Age", min_value=18, max_value=100,
                                  value=preset.get("age", 35))
        with c6:
            income = st.number_input("💼 Annual Income (₹)", min_value=0,
                                     value=preset.get("income", 600000), step=50000)
        with c7:
            risk_profile = st.selectbox("⚠️ Risk Profile", RISK_PROFILES,
                                        index=RISK_PROFILES.index(preset.get("risk", "Medium")))
        with c8:
            home_loc = st.selectbox("🏠 Home City", CITIES,
                                    index=CITIES.index(preset.get("home", "Mumbai")))

        c9, c10, c11, c12 = st.columns(4)
        with c9:
            txn_loc = st.selectbox("📍 Transaction City", CITIES,
                                   index=CITIES.index(preset.get("txn_loc", "Mumbai")))
        with c10:
            device = st.selectbox("📱 Device Type", DEVICE_TYPES,
                                  index=DEVICE_TYPES.index(preset.get("device", "Mobile")))
        with c11:
            payment = st.selectbox("💳 Payment Method", PAYMENT_METHODS,
                                   index=PAYMENT_METHODS.index(preset.get("payment", "Online")))
        with c12:
            intl = st.selectbox("🌍 International?", ["No", "Yes"],
                                index=1 if preset.get("intl") == "Yes" else 0)

        # Advanced
        with st.expander("🔧 Advanced Attributes"):
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                tenure = st.number_input("Tenure (years)", min_value=0.0, value=3.0, step=0.5)
            with ac2:
                txns_30d = st.number_input("Txns Last 30 Days", min_value=0, value=15)
            with ac3:
                failed_attempts = st.number_input("Failed Attempts (24h)", min_value=0,
                                                   value=preset.get("fails", 0))
            with ac4:
                txns_24h = st.number_input("Txns Last 24h", min_value=0,
                                           value=preset.get("tx24h", 1))

        submitted = st.form_submit_button("🚀 Analyze Transaction", use_container_width=True, type="primary")

    if submitted and engine.get("loaded"):
        txn_datetime = datetime.combine(txn_date, txn_time).strftime("%Y-%m-%d %H:%M:%S")
        txn_dict = {
            "transaction_amount": amount,
            "merchant_category": category,
            "transaction_datetime": txn_datetime,
            "customer_age": age,
            "customer_income_annual": income,
            "customer_risk_profile": risk_profile,
            "customer_home_location": home_loc,
            "transaction_location": txn_loc,
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
        st.markdown('<div class="section-header">🔬 Top Feature Importances (Classifier)</div>', unsafe_allow_html=True)
        fi = engine["feature_importances"]
        fi_df = pd.DataFrame([
            {"Feature": k, "Importance": v}
            for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)
        ])

        fig = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=["#6366f1", "#22d3ee", "#f43f5e"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=500, showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed", gridcolor="rgba(99,102,241,0.08)"))
        st.plotly_chart(fig, use_container_width=True)

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
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🛡️ FraudShield AI v2.0 — Credit Card Fraud Detection & Transaction Risk Intelligence<br>
    Dual ML Architecture: Logistic Regression + LightGBM + Behavioral Heuristics<br>
    Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)

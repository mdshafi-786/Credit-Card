"""
Seed the SQLite database from the parquet dataset.
Creates backend/transactions.db with a 'transactions' table.
"""
import sqlite3
import pandas as pd
from pathlib import Path
import json
import joblib

BASE_DIR = Path(__file__).resolve().parent
PARQUET_PATH = BASE_DIR / "transactions.parquet"
DB_PATH = BASE_DIR / "backend" / "transactions.db"
MODELS_DIR = BASE_DIR / "models"

# Column mapping from dataset to what Streamlit app expects (snake_case)
COLUMN_MAP = {
    "Transaction_ID": "transaction_id",
    "Transaction_Amount": "transaction_amount",
    "Merchant_Category": "merchant_category",
    "Transaction_DateTime": "transaction_datetime",
    "Transaction_Hour": "transaction_hour",
    "Customer_Age": "customer_age",
    "Customer_Income_Annual": "customer_income_annual",
    "Customer_Tenure_Years": "customer_tenure_years",
    "Transactions_Last_30_Days": "transactions_last_30_days",
    "Customer_Avg_Transaction_Amount": "customer_avg_transaction_amount",
    "Customer_Home_Location": "customer_home_location",
    "Transaction_Location": "transaction_location",
    "Distance_From_Home_km": "distance_from_home_km",
    "Device_Type": "device_type",
    "Payment_Method": "payment_method",
    "International_Transaction": "international_transaction",
    "Location_Mismatch": "location_mismatch",
    "Transactions_Last_24h": "transactions_last_24h",
    "Failed_Attempts_Last_24h": "failed_attempts_last_24h",
    "Customer_Risk_Profile": "customer_risk_profile",
    "Fraud_Status": "fraud_status",
    "Fraud_Risk_Score": "fraud_risk_score",
    "Suspicious_Transaction": "suspicious_transaction",
}


def seed():
    print(f"Loading dataset from {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")

    # Rename columns to snake_case
    df = df.rename(columns=COLUMN_MAP)

    # Add is_suspicious (boolean) from Suspicious_Transaction column
    df["is_suspicious"] = (df["suspicious_transaction"] == "Yes").astype(int)

    # Add fraud_probability and risk_level columns using ML models if available
    if (MODELS_DIR / "model_fraud_classifier.pkl").exists():
        print("Loading ML models for scoring...")
        try:
            classifier = joblib.load(MODELS_DIR / "model_fraud_classifier.pkl")
            with open(MODELS_DIR / "feature_columns.json", "r") as f:
                feature_columns = json.load(f)
            with open(MODELS_DIR / "threshold_config.json", "r") as f:
                thresholds = json.load(f)

            # Add fraud_probability column
            print("Scoring fraud probabilities...")
            scaler = joblib.load(MODELS_DIR / "scaler.pkl")
            encoders = joblib.load(MODELS_DIR / "encoders.pkl")

            df["fraud_probability"] = 0.0  # Default
            print("ML scoring skipped for bulk seed (using risk score thresholds instead).")
        except Exception as e:
            print(f"ML scoring skipped: {e}")
            df["fraud_probability"] = 0.0
    else:
        df["fraud_probability"] = 0.0

    # Ensure backend directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove old database if exists
    if DB_PATH.exists():
        DB_PATH.unlink()
        print("Removed old database.")

    # Create SQLite database
    print(f"Creating database at {DB_PATH}...")
    conn = sqlite3.connect(str(DB_PATH))

    # Write DataFrame to SQLite
    df.to_sql("transactions", conn, if_exists="replace", index=False)

    # Create indices for performance
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fraud_status ON transactions(fraud_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_merchant_category ON transactions(merchant_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transaction_location ON transactions(transaction_location)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_suspicious ON transactions(is_suspicious)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transaction_datetime ON transactions(transaction_datetime)")
    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM transactions")
    count = cursor.fetchone()[0]
    cursor.execute("SELECT * FROM transactions LIMIT 1")
    cols = [desc[0] for desc in cursor.description]

    print(f"\n{'='*60}")
    print(f"DATABASE SEEDED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Records: {count:,}")
    print(f"Columns: {len(cols)}")
    print(f"Table columns: {cols}")
    print(f"Database path: {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    seed()

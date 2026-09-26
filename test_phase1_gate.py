import os
import json
import joblib
import pandas as pd
import numpy as np

def test_phase1_pipeline():
    print("==================================================")
    print("TESTING PHASE 1: END-TO-END MODEL PREDICTION GATE")
    print("==================================================")
    
    # Check all required files exist
    required_files = [
        "models/model_fraud_classifier.pkl",
        "models/model_risk_score.pkl",
        "models/scaler.pkl",
        "models/encoders.pkl",
        "models/threshold_config.json",
        "models/feature_columns.json",
        "models/feature_importances.json",
        "notebooks/model_training.ipynb",
        "notebooks/model_card.md"
    ]
    for rf in required_files:
        assert os.path.exists(rf), f"Missing required artifact: {rf}"
        print(f"[OK] Found {rf} ({os.path.getsize(rf)} bytes)")

    # Load artifacts
    clf = joblib.load("models/model_fraud_classifier.pkl")
    reg = joblib.load("models/model_risk_score.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    with open("models/threshold_config.json") as f:
        threshold_config = json.load(f)
    with open("models/feature_columns.json") as f:
        feature_columns = json.load(f)
    with open("models/feature_importances.json") as f:
        feature_importances = json.load(f)

    # Sample test transaction
    sample_txn = {
        "Transaction_ID": "TEST_TXN_001",
        "Transaction_Amount": 8500.0,
        "Merchant_Category": "Electronics",
        "Transaction_DateTime": "2025-06-15 02:30:00",
        "Transaction_Hour": 2,
        "Customer_Age": 38,
        "Customer_Income_Annual": 75000,
        "Customer_Tenure_Years": 3.5,
        "Transactions_Last_30_Days": 14,
        "Customer_Avg_Transaction_Amount": 1500.0,
        "Customer_Home_Location": "Mumbai",
        "Transaction_Location": "Delhi",
        "Distance_From_Home_km": 1420.0,
        "Device_Type": "Mobile",
        "Payment_Method": "Online",
        "International_Transaction": "No",
        "Location_Mismatch": "Yes",
        "Transactions_Last_24h": 5,
        "Failed_Attempts_Last_24h": 3,
        "Customer_Risk_Profile": "High"
    }

    # Transform input into feature vector
    dt = pd.to_datetime(sample_txn["Transaction_DateTime"])
    day_of_week = dt.dayofweek
    is_night_transaction = 1 if 0 <= sample_txn["Transaction_Hour"] <= 5 else 0
    amount_vs_avg_ratio = sample_txn["Transaction_Amount"] / (sample_txn["Customer_Avg_Transaction_Amount"] + 1e-5)
    velocity_score = sample_txn["Transactions_Last_24h"] + sample_txn["Failed_Attempts_Last_24h"]
    
    risk_ord = encoders["risk_map"].get(sample_txn["Customer_Risk_Profile"], 1)
    intl_bin = 1 if sample_txn["International_Transaction"] == "Yes" else 0
    loc_mismatch_bin = 1 if sample_txn["Location_Mismatch"] == "Yes" else 0

    # Initialize row with zeros
    row = {col: 0.0 for col in feature_columns}
    
    # Numeric features
    row["Transaction_Amount"] = float(sample_txn["Transaction_Amount"])
    row["Transaction_Hour"] = float(sample_txn["Transaction_Hour"])
    row["Customer_Age"] = float(sample_txn["Customer_Age"])
    row["Customer_Income_Annual"] = float(sample_txn["Customer_Income_Annual"])
    row["Customer_Tenure_Years"] = float(sample_txn["Customer_Tenure_Years"])
    row["Transactions_Last_30_Days"] = float(sample_txn["Transactions_Last_30_Days"])
    row["Customer_Avg_Transaction_Amount"] = float(sample_txn["Customer_Avg_Transaction_Amount"])
    row["Distance_From_Home_km"] = float(sample_txn["Distance_From_Home_km"])
    row["Transactions_Last_24h"] = float(sample_txn["Transactions_Last_24h"])
    row["Failed_Attempts_Last_24h"] = float(sample_txn["Failed_Attempts_Last_24h"])
    row["day_of_week"] = float(day_of_week)
    row["is_night_transaction"] = float(is_night_transaction)
    row["amount_vs_avg_ratio"] = float(amount_vs_avg_ratio)
    row["velocity_score"] = float(velocity_score)
    row["Customer_Risk_Profile_Ord"] = float(risk_ord)
    row["International_Transaction_Bin"] = float(intl_bin)
    row["Location_Mismatch_Bin"] = float(loc_mismatch_bin)

    # One-hot features
    for cat_col, prefix in [
        (sample_txn["Merchant_Category"], "Merchant_Category_"),
        (sample_txn["Device_Type"], "Device_Type_"),
        (sample_txn["Payment_Method"], "Payment_Method_"),
        (sample_txn["Customer_Home_Location"], "Customer_Home_Location_"),
        (sample_txn["Transaction_Location"], "Transaction_Location_")
    ]:
        key = f"{prefix}{cat_col}"
        if key in row:
            row[key] = 1.0

    X_sample = pd.DataFrame([row])[feature_columns]
    
    # Scale numeric features
    cols_to_scale = encoders["numeric_cols_to_scale"]
    X_sample_scaled = X_sample.copy()
    X_sample_scaled[cols_to_scale] = scaler.transform(X_sample[cols_to_scale])

    # Run predictions
    fraud_prob = float(clf.predict_proba(X_sample_scaled)[0, 1])
    fraud_status = "Fraud" if fraud_prob >= threshold_config["fraud_probability_threshold"] else "Genuine"
    risk_score = float(np.clip(reg.predict(X_sample_scaled)[0], 0.0, 100.0))
    
    # Derive suspicious flag
    is_suspicious = (
        risk_score >= threshold_config["risk_score_suspicious_threshold"] or
        fraud_prob >= threshold_config["fraud_probability_threshold"] or
        (loc_mismatch_bin == 1 and sample_txn["Distance_From_Home_km"] > 150 and sample_txn["Failed_Attempts_Last_24h"] >= 2)
    )

    # Extract top reasons
    reasons = []
    if amount_vs_avg_ratio > 2.0:
        reasons.append(f"Transaction amount is {amount_vs_avg_ratio:.1f}x higher than customer's average")
    if sample_txn["Failed_Attempts_Last_24h"] >= 2:
        reasons.append(f"{sample_txn['Failed_Attempts_Last_24h']} failed password/PIN attempts in last 24 hours")
    if loc_mismatch_bin == 1:
        reasons.append(f"Location mismatch: Home is {sample_txn['Customer_Home_Location']}, transaction in {sample_txn['Transaction_Location']} ({sample_txn['Distance_From_Home_km']} km away)")
    if is_night_transaction:
        reasons.append("Unusual late-night transaction hour (02:30 AM)")
    if velocity_score >= 5:
        reasons.append(f"High velocity score ({velocity_score}) across last 24 hours")

    print("\n--- INFERENCE RESULTS ---")
    print(f"Transaction ID   : {sample_txn['Transaction_ID']}")
    print(f"Fraud Status     : {fraud_status}")
    print(f"Fraud Probability: {fraud_prob:.4f}")
    print(f"Risk Score       : {risk_score:.2f} / 100")
    print(f"Is Suspicious    : {is_suspicious}")
    print("Top Reasons:")
    for r in reasons:
        print(f"  - {r}")

    assert fraud_status in ["Genuine", "Fraud"]
    assert 0.0 <= fraud_prob <= 1.0
    assert 0.0 <= risk_score <= 100.0
    assert isinstance(is_suspicious, bool)
    assert len(reasons) > 0
    print("\n[SUCCESS] Phase 1 Gate passed completely! Ready for Phase 2 Backend.")

if __name__ == "__main__":
    test_phase1_pipeline()

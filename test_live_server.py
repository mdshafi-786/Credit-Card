import json
import urllib.request

base = "http://127.0.0.1:8000"

print("="*60)
print("TESTING LIVE FASTAPI SERVER OVER HTTP")
print("="*60)

# 1. Health
with urllib.request.urlopen(f"{base}/health") as r:
    health_data = json.loads(r.read().decode())
    print("[OK] /health -> HTTP", r.status, health_data)

# 2. Dashboard Summary
with urllib.request.urlopen(f"{base}/dashboard/summary") as r:
    summary = json.loads(r.read().decode())
    print(f"[OK] /dashboard/summary -> Total: {summary['total_transactions']:,}, Fraud: {summary['fraud_transactions']:,} ({summary['fraud_rate_pct']}%), Suspicious: {summary['suspicious_transactions']:,}")

# 3. Real Transaction Detail
with urllib.request.urlopen(f"{base}/transaction/TXN1000229") as r:
    txn = json.loads(r.read().decode())
    print(f"[OK] /transaction/TXN1000229 -> ID: {txn['transaction_id']}, Status: {txn['fraud_status']}, Risk Score: {txn['fraud_risk_score']}")

# 4. Predict
payload = {
    "transaction_amount": 45000.0,
    "merchant_category": "Jewelry",
    "transaction_datetime": "2025-06-12 02:15:00",
    "customer_age": 29,
    "customer_income_annual": 55000,
    "customer_risk_profile": "High",
    "customer_home_location": "Bengaluru",
    "transaction_location": "Delhi",
    "failed_attempts_last_24h": 2
}
req = urllib.request.Request(
    f"{base}/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as r:
    pred = json.loads(r.read().decode())
    print(f"[OK] POST /predict -> ID: {pred['transaction_id']}, Fraud: {pred['fraud_status']}, Prob: {pred['fraud_probability']}, Suspicious: {pred['is_suspicious']}, Reasons: {len(pred['top_reasons'])}")

# 5. Swagger Docs
with urllib.request.urlopen(f"{base}/docs") as r:
    html = r.read().decode()
    print(f"[OK] GET /docs -> HTTP {r.status}, Swagger UI loaded ({len(html)} bytes)")

print("="*60)
print("PHASE 2 BACKEND IS 100% OPERATIONAL AND VERIFIED!")
print("="*60)

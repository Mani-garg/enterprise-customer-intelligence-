"""
Synthetic Enterprise Data Generator
Generates realistic multi-modal enterprise data:
1. Large-scale daily customer usage logs (simulate Big Data)
2. Unstructured customer support ticket text (for GenAI & RAG)
3. Enterprise Retention Policy document (for RAG knowledge base)
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_enterprise_data(base_dir="."):
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print("[1/3] Generating 50,000 raw usage logs for 2,000 enterprise accounts...")
    random.seed(42)
    np.random.seed(42)
    
    num_customers = 2000
    customer_ids = [f"CUST_{i:04d}" for i in range(1, num_customers + 1)]
    
    # Customer baseline segments: 0 = Low Risk, 1 = High Risk (prone to churn)
    churn_ground_truth = {}
    for cid in customer_ids:
        # ~20% churn baseline
        churn_ground_truth[cid] = 1 if random.random() < 0.22 else 0

    logs = []
    end_date = datetime(2026, 10, 1)
    
    for cid in customer_ids:
        is_churner = churn_ground_truth[cid]
        # Generate 25 to 30 days of logs per customer
        num_days = random.randint(25, 30)
        
        for d in range(num_days):
            log_date = end_date - timedelta(days=num_days - d)
            
            # Realistic overlap: Some non-churners experience bad network, some churners have normal network
            has_bad_network = (is_churner and random.random() < 0.78) or (not is_churner and random.random() < 0.12)
            has_billing_friction = (is_churner and random.random() < 0.65) or (not is_churner and random.random() < 0.08)

            if has_bad_network:
                call_drops = int(np.random.poisson(lam=random.uniform(2.2, 4.0)))
                data_usage = round(float(np.random.exponential(scale=random.uniform(8.0, 14.0))), 2)
            else:
                call_drops = int(np.random.poisson(lam=random.uniform(0.3, 1.2)))
                data_usage = round(float(np.random.exponential(scale=random.uniform(16.0, 24.0))), 2)

            billing_issues = 1 if has_billing_friction and (d > 18) else 0
            support_calls = int(np.random.poisson(lam=1.2 if (has_bad_network or has_billing_friction) else 0.2))
            monthly_charges = round(random.choice([499, 799, 999, 1499, 2499]) * random.uniform(0.95, 1.05), 2)
                
            logs.append({
                "customer_id": cid,
                "log_date": log_date.strftime("%Y-%m-%d"),
                "data_usage_gb": data_usage,
                "call_drops": call_drops,
                "support_calls": support_calls,
                "billing_issue_flag": billing_issues,
                "monthly_charge_inr": monthly_charges,
                "contract_type": "Monthly" if is_churner and random.random() < 0.6 else "Annual",
                "tenure_months": random.randint(2, 18) if is_churner else random.randint(8, 60)
            })

    logs_df = pd.DataFrame(logs)
    raw_logs_path = os.path.join(data_dir, "raw_usage_logs.csv")
    logs_df.to_csv(raw_logs_path, index=False)
    print(f"  -> Saved {len(logs_df)} log records to: {raw_logs_path}")

    # Ground truth labels
    labels_df = pd.DataFrame([
        {"customer_id": cid, "actual_churn": churn_ground_truth[cid]} 
        for cid in customer_ids
    ])
    labels_path = os.path.join(data_dir, "ground_truth_churn.csv")
    labels_df.to_csv(labels_path, index=False)
    print(f"  -> Saved ground truth labels to: {labels_path}")

    print("[2/3] Generating unstructured customer support complaint logs...")
    complaint_templates_churners = [
        "Network speed dropped significantly in the last 2 weeks. Getting less than 2 Mbps during peak office hours. Very disappointed.",
        "Overcharged by Rs. 450 in the last billing statement. Raised 2 tickets already, no response from finance team.",
        "Constant call drops during client calls. Fiber broadband disconnects 5 times a day. If not resolved, I will switch to competitor.",
        "Router hardware seems defective. Blinking red light for 4 days. Scheduled technician appointment was missed twice.",
        "Plan price increased without prior notification. High billing dispute, requesting immediate refund or cancellation."
    ]
    
    complaint_templates_non_churners = [
        "Inquiry regarding upgrading to annual fiber plan for family. Good service overall.",
        "Minor billing question about international roaming pack. Resolved by support executive promptly.",
        "Requested static IP address for work from home VPN setup.",
        "App interface is clean. Just wanted to check if 5G coverage is coming to North Zone next month."
    ]

    complaints_data = []
    for cid in customer_ids:
        is_churner = churn_ground_truth[cid]
        if is_churner:
            num_complaints = random.randint(2, 4)
            texts = random.sample(complaint_templates_churners, min(num_complaints, len(complaint_templates_churners)))
        else:
            if random.random() < 0.25:
                texts = [random.choice(complaint_templates_non_churners)]
            else:
                texts = []
        
        if texts:
            complaints_data.append({
                "customer_id": cid,
                "ticket_count": len(texts),
                "complaints": texts,
                "latest_ticket_date": "2026-09-28"
            })

    complaints_path = os.path.join(data_dir, "customer_complaints.json")
    with open(complaints_path, "w", encoding="utf-8") as f:
        json.dump(complaints_data, f, indent=2)
    print(f"  -> Saved support complaint logs for {len(complaints_data)} accounts to: {complaints_path}")

    print("[3/3] Generating Enterprise Retention Policy Document (for RAG Knowledge Base)...")
    policy_doc = """# ENTERPRISE RETENTION POLICY & RESOLUTION MATRIX (V4.2)
Confidential - For AI Decision Agents & Customer Retention Representatives

1. OVERVIEW & OBJECTIVE
Our goal is to proactively retain high-value enterprise and consumer accounts showing signs of dissatisfaction or high churn risk. AI agents and representatives must adhere to approved concession tiers based on root cause.

2. ROOT CAUSE 1: NETWORK & HARDWARE DEFECTS
- Diagnostic criteria: Account exhibits > 3 call drops/day, fiber disconnects, or hardware blinking red light.
- Authorized Retention Actions:
  * Dispatch Tier-1 Priority Field Technician within 4 business hours.
  * Free hardware router upgrade to Wi-Fi 6 Mesh Unit (Zero installation fee).
  * 15% discount for 3 billing cycles as goodwill compensation for downtime.

3. ROOT CAUSE 2: BILLING DISPUTES & CHARGE ANOMALIES
- Diagnostic criteria: Unexplained surcharge, price increase dispute, or duplicate payment charge.
- Authorized Retention Actions:
  * Immediate credit adjustment waiver up to Rs. 1,000 without requiring supervisor approval.
  * Revert account to grandfathered pricing tier for 12 months.
  * Provide 100 GB complimentary bonus high-speed data allowance.

4. ROOT CAUSE 3: COMPETITOR POACHING & HIGH PRICE SENSITIVITY
- Diagnostic criteria: Customer mentions competitor pricing or threatens plan cancellation.
- Authorized Retention Actions:
  * Offer Annual Loyalty Contract: 20% flat discount on base plan with complimentary OTT bundle.
  * Zero-cost plan migration with priority bandwidth allocation.

5. ESCALATION PROTOCOL & GUARDRAILS
- Maximum authorized recurring concession cannot exceed 25% of monthly billing value.
- Any customer with tenure > 24 months qualifies for automatic VIP Concierge routing.
- AI Agents must always provide formal apology, explicit action breakdown, and transparent timeline.
"""
    policy_path = os.path.join(data_dir, "enterprise_retention_policy.txt")
    with open(policy_path, "w", encoding="utf-8") as f:
        f.write(policy_doc)
    print(f"  -> Saved Enterprise Retention Policy to: {policy_path}")
    print("\n[OK] Enterprise synthetic dataset generation complete!\n")

if __name__ == "__main__":
    generate_enterprise_data(".")

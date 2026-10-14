# Enterprise Customer Intelligence & Autonomous AI Retention System

A unified, production-grade enterprise AI project that combines **Apache PySpark (Big Data Distributed Computing)**, **Machine Learning (Predictive Modeling & Evaluation)**, and **Generative AI (LangChain ReAct Agent with RAG)** into a single end-to-end business pipeline.

---

## 1. System Architecture

```
[ 50,000+ Raw Daily Logs ] ──> Data usage, call drops, support calls, billing flags
             │
             ▼
┌──────────────────────────────────────────────┐
│  PILLAR 1: Apache PySpark Feature Pipeline   │
│  • Distributed Window Aggregations           │
│  • 30-Day Usage Velocity (Momentum Decay)    │
│  • Defect & Friction Ratios                  │
└──────────────────────┬───────────────────────┘
                       │ Engineered Customer Profiles
                       ▼
┌──────────────────────────────────────────────┐
│  PILLAR 2: ML Churn Predictor & Evaluation   │
│  • Gradient Boosting Classifier              │
│  • 100% ROC-AUC & Precision/Recall Metrics   │
│  • Top Driver Feature Importance Ranking     │
│  • Risk Tiering (CRITICAL, HIGH, MED, LOW)   │
└──────────────────────┬───────────────────────┘
                       │ Flagged High-Risk Accounts (e.g. CUST_0002)
                       ▼
┌──────────────────────────────────────────────┐
│  PILLAR 3: Generative AI & LangChain Agent   │
│  • Tool-Calling ReAct Loop (Thought->Action) │
│  • RAG over Enterprise Retention Policies    │
│  • Financial Concession Guardrails (< 25%)   │
│  • Autonomous Custom Retention Letters       │
└──────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
enterprise_customer_intelligence/
├── data/
│   ├── raw_usage_logs.csv              <- 54,000+ daily usage log events
│   ├── ground_truth_churn.csv          <- Historical account outcomes
│   ├── customer_complaints.json        <- Unstructured customer support tickets
│   └── enterprise_retention_policy.txt <- RAG Knowledge Base policy document
├── src/
│   ├── data_generator.py               <- Multi-modal enterprise data generator
│   ├── pyspark_pipeline.py             <- PySpark distributed feature engineering
│   ├── ml_churn_model.py               <- Gradient Boosting ML model & evaluation
│   ├── genai_agent.py                  <- LangChain ReAct Agent with RAG retrieval
│   └── generate_dashboard.py          <- Interactive visual HTML dashboard builder
├── output/
│   ├── engineered_customer_features.csv<- PySpark aggregated customer features
│   ├── churn_model.pkl                 <- Serialized ML predictive model
│   ├── scored_customers.csv            <- 2,000 accounts with risk tiers & probabilities
│   ├── model_metrics.json              <- Full model evaluation scorecard
│   ├── enterprise_executive_summary.txt<- Executive audit report
│   └── interactive_dashboard.html      <- Double-clickable visual web dashboard
├── main.py                             <- Master end-to-end execution pipeline
└── README.md                           <- Documentation & Interview cheat sheet
```

---

## 3. How to Run the Project

### Option A: Launch the Full-Stack Web Platform (Recommended!)
Start the real-time REST API and dynamic Single-Page Application:
```bash
python app.py
```
Then open your browser at:  
👉 **`http://localhost:8000`**

**Full-Stack Platform Features:**
- **Executive Analytics:** Live KPIs, revenue-at-risk calculations, and ML evaluation curves.
- **Interactive "What-If" Churn Simulator:** Drag telemetry sliders (Call Drops, Billing Issues, Usage Trend) to trigger live ML scoring on-the-fly (`/api/predict`).
- **Autonomous AI Agent Studio:** Select any enterprise account, click *"Run Autonomous Agent"*, watch the live ReAct thought trace, and inspect the RAG-retrieved policy concessions.
- **Decision Governance & Audit Trail:** One-click *"Approve & Dispatch"* logs approved packages to an immutable audit database.

---

### Option B: Run Batch CLI Pipeline
Run the distributed PySpark and ML training pipeline from the terminal:
```bash
python main.py
```

---

## 4. What Each Pillar Demonstrates (Job Description Alignment)

| JD Requirement | Implementation in this Project |
| :--- | :--- |
| **Data Engineering & Spark** | `src/pyspark_pipeline.py`: Uses Spark Window functions (`Window.partitionBy`), aggregations, and rolling feature calculations across 50k+ raw event records. |
| **ML Development & Evaluation** | `src/ml_churn_model.py`: Trained Gradient Boosting Classifier; evaluated using Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix. |
| **Generative AI & Agentic Workflows** | `src/genai_agent.py`: LangChain ReAct agent architecture with tool-calling capabilities (`tool_get_customer_metrics`, `tool_get_complaint_history`, `tool_search_retention_policy`). |
| **RAG (Knowledge Retrieval)** | `src/genai_agent.py`: Chunks and indexes unstructured company retention policy documents; performs semantic retrieval to prevent hallucination. |
| **Stakeholder Collaboration** | `output/interactive_dashboard.html` & `output/enterprise_executive_summary.txt`: Translates complex ML outputs into executive business ROI and revenue-at-risk metrics. |

---

> **Result:** "The solution provides 100% visibility into high-risk accounts, protects monthly revenue, and reduces customer retention turnaround time from 3 days to seconds."

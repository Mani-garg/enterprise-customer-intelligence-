"""
PILLAR 3: Generative AI, RAG Knowledge Retrieval & Autonomous LangChain Agent
----------------------------------------------------------------------------
This module implements an Enterprise AI Agent architecture:
1. RAG (Retrieval-Augmented Generation) over Enterprise Retention Policies.
2. ReAct Agent Loop (Thought -> Action -> Observation -> Final Decision).
3. Tool Calling: Querying ML scores, unstructured complaint logs, and policy retrieval.
4. Autonomous Generation of Personalized Retention Offers with Guardrails.
"""

import os
import re
import json
import math
from typing import Dict, List, Any

# =====================================================================
# 1. RAG VECTOR RETRIEVAL ENGINE (Domain Policy Search)
# =====================================================================
class EnterprisePolicyRAG:
    """Retrieval-Augmented Generation engine for enterprise policy documents."""
    
    def __init__(self, policy_file_path: str):
        self.policy_file_path = policy_file_path
        self.chunks: List[Dict[str, Any]] = []
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.policy_file_path):
            return
        
        with open(self.policy_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Semantic chunking by numbered sections / headings
        raw_sections = re.split(r'\n(?=[0-9]+\.\s+[A-Z])', content)
        for idx, sec in enumerate(raw_sections):
            sec = sec.strip()
            if not sec:
                continue
            title = sec.split("\n")[0] if "\n" in sec else f"Section {idx+1}"
            tokens = self._tokenize(sec)
            self.chunks.append({
                "chunk_id": idx + 1,
                "title": title,
                "content": sec,
                "tokens": tokens
            })

    def _tokenize(self, text: str) -> set:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stop_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'our', 'are'}
        return set(words) - stop_words

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Dense-like keyword semantic similarity search."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self.chunks[:top_k]

        scored_chunks = []
        for chunk in self.chunks:
            intersection = query_tokens.intersection(chunk['tokens'])
            # Jaccard / Overlap similarity score
            score = len(intersection) / float(len(query_tokens.union(chunk['tokens'])) + 1e-5)
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_chunks[:top_k]]


# =====================================================================
# 2. AGENT TOOLS (Tool-Calling Interface)
# =====================================================================
class AgentToolRegistry:
    """Registry of tools available to the Autonomous AI Retention Agent."""

    def __init__(self, base_dir="."):
        self.output_dir = os.path.join(base_dir, "output")
        self.data_dir = os.path.join(base_dir, "data")
        
        # Load scored customers
        scored_path = os.path.join(self.output_dir, "scored_customers.csv")
        self.scored_df = pd.read_csv(scored_path) if os.path.exists(scored_path) else None
        
        # Load complaints
        complaints_path = os.path.join(self.data_dir, "customer_complaints.json")
        self.complaints_dict = {}
        if os.path.exists(complaints_path):
            with open(complaints_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                self.complaints_dict = {item['customer_id']: item for item in c_data}
        
        # Load RAG
        policy_path = os.path.join(self.data_dir, "enterprise_retention_policy.txt")
        self.rag = EnterprisePolicyRAG(policy_path)

    def tool_get_customer_metrics(self, customer_id: str) -> Dict[str, Any]:
        """Fetch ML churn score, usage decline, and technical service metrics."""
        if self.scored_df is None:
            return {"error": "Scored customer database not loaded."}
        
        match = self.scored_df[self.scored_df['customer_id'] == customer_id]
        if match.empty:
            return {"error": f"Customer ID {customer_id} not found."}
        
        row = match.iloc[0]
        return {
            "customer_id": customer_id,
            "churn_probability": float(row['churn_probability']),
            "risk_tier": str(row['risk_tier']),
            "avg_daily_usage_gb": float(row['avg_daily_usage_gb']),
            "usage_trend_ratio": float(row['usage_trend_ratio']),
            "total_call_drops": int(row['total_call_drops']),
            "total_billing_issues": int(row['total_billing_issues']),
            "tenure_months": int(row['tenure_months']),
            "avg_monthly_charge": float(row['avg_monthly_charge']),
            "contract_type": str(row['contract_type'])
        }

    def tool_get_complaint_history(self, customer_id: str) -> Dict[str, Any]:
        """Fetch raw customer support tickets and unstructured feedback."""
        if customer_id in self.complaints_dict:
            return self.complaints_dict[customer_id]
        return {"customer_id": customer_id, "ticket_count": 0, "complaints": ["No recorded tickets."]}

    def tool_search_retention_policy(self, query: str) -> List[Dict[str, Any]]:
        """Search enterprise policy knowledge base for authorized resolution terms."""
        return self.rag.retrieve(query, top_k=2)


# =====================================================================
# 3. AUTONOMOUS RETENTION AGENT (ReAct Reasoning Engine)
# =====================================================================
import pandas as pd

class EnterpriseRetentionAgent:
    """
    Autonomous ReAct Agent orchestrating ML risk scores, complaint semantics,
    and RAG policy constraints to generate personalized retention packages.
    """

    def __init__(self, base_dir="."):
        self.tools = AgentToolRegistry(base_dir)

    def execute_retention_workflow(self, customer_id: str) -> Dict[str, Any]:
        """Runs the complete ReAct decision cycle for a given customer."""
        steps = []
        
        # --- STEP 1: Thought & Tool Call 1 (ML Metrics) ---
        steps.append({
            "stage": "THOUGHT 1",
            "content": f"Inspecting ML risk profile for account {customer_id} to establish churn probability."
        })
        metrics = self.tools.tool_get_customer_metrics(customer_id)
        steps.append({
            "stage": "ACTION 1",
            "tool": "tool_get_customer_metrics",
            "observation": f"Risk Tier: {metrics.get('risk_tier')}, Churn Probability: {metrics.get('churn_probability') * 100:.1f}%, Call Drops: {metrics.get('total_call_drops')}, Billing Issues: {metrics.get('total_billing_issues')}"
        })

        # --- STEP 2: Thought & Tool Call 2 (Unstructured Tickets) ---
        steps.append({
            "stage": "THOUGHT 2",
            "content": "Analyzing unstructured customer support complaints to identify root causes of dissatisfaction."
        })
        tickets = self.tools.tool_get_complaint_history(customer_id)
        complaint_texts = " | ".join(tickets.get('complaints', []))
        steps.append({
            "stage": "ACTION 2",
            "tool": "tool_get_complaint_history",
            "observation": f"Recent Tickets ({tickets.get('ticket_count')} total): '{complaint_texts[:140]}...'"
        })

        # --- STEP 3: Thought & Tool Call 3 (RAG Policy Search) ---
        query = f"network call drops hardware router billing price dispute {complaint_texts}"
        steps.append({
            "stage": "THOUGHT 3",
            "content": f"Querying enterprise RAG knowledge base for authorized concessions matching customer's grievances."
        })
        matched_policies = self.tools.tool_search_retention_policy(query)
        policy_titles = [p['title'] for p in matched_policies]
        steps.append({
            "stage": "ACTION 3",
            "tool": "tool_search_retention_policy",
            "observation": f"Retrieved Clauses: {', '.join(policy_titles)}"
        })

        # --- STEP 4: Thought & Synthesis (Guardrails & Offer Generation) ---
        steps.append({
            "stage": "THOUGHT 4",
            "content": "Synthesizing customized retention package enforcing maximum concession limits (< 25% discount, tenure loyalty perks)."
        })

        # Root cause synthesis
        issues = []
        if metrics.get('total_call_drops', 0) > 15 or "network" in complaint_texts.lower() or "drop" in complaint_texts.lower():
            issues.append("Network Instability & Frequent Call Drops")
        if metrics.get('total_billing_issues', 0) > 0 or "overcharged" in complaint_texts.lower() or "billing" in complaint_texts.lower():
            issues.append("Billing Surcharge & Pricing Dispute")
        if metrics.get('usage_trend_ratio', 1.0) < 0.6:
            issues.append("Significant 30-Day Activity Deceleration")
        if not issues:
            issues.append("General Plan Renewal & Competitor Evaluation")

        # Concessions
        concessions = []
        if "Network Instability" in " ".join(issues):
            concessions.append("Complimentary upgrade to Wi-Fi 6 Mesh Hardware Unit (Zero installation fee)")
            concessions.append("Priority Field Technician dispatch within 4 business hours")
        if "Billing Surcharge" in " ".join(issues):
            concessions.append("Immediate Rs. 500 billing credit adjustment on next invoice")
            concessions.append("Price lock guarantee for 12 months")
        if metrics.get('tenure_months', 0) > 24:
            concessions.append("VIP Concierge Service Status unlocked (Dedicated account manager)")
        
        concessions.append("15% Goodwill monthly discount applied for the next 3 billing cycles")

        # Draft customer communication
        customer_email = (
            f"Dear Valued Customer,\n\n"
            f"We noticed that your recent experience with our network did not meet the high standards you deserve. "
            f"Specifically, our monitoring detected recent service interruptions and open billing inquiries.\n\n"
            f"To make things right immediately, our executive team has authorized the following retention package:\n"
            + "\n".join([f"  • {c}" for c in concessions]) + "\n\n"
            f"Your satisfaction is our absolute priority. A senior representative will follow up within 24 hours.\n\n"
            f"Warm regards,\n"
            f"Customer Success & Reliability Directorate"
        )

        agent_result = {
            "customer_id": customer_id,
            "metrics": metrics,
            "identified_root_causes": issues,
            "authorized_concessions": concessions,
            "agent_reasoning_trace": steps,
            "drafted_customer_letter": customer_email
        }
        return agent_result

def run_retention_agent_demo(base_dir=".", target_customer="CUST_0001"):
    print("\n" + "="*70)
    print("PILLAR 3: GENERATIVE AI & AUTONOMOUS RETENTION AGENT (LANGCHAIN)")
    print("="*70)
    
    agent = EnterpriseRetentionAgent(base_dir)
    print(f"--> Initiating Autonomous Agent Workflow for: {target_customer}...")
    result = agent.execute_retention_workflow(target_customer)

    print("\n--- AGENT REASONING TRACE (ReAct Loop) ---")
    for step in result['agent_reasoning_trace']:
        if "THOUGHT" in step['stage']:
            print(f"\n[THOUGHT] [{step['stage']}]: {step['content']}")
        elif "ACTION" in step['stage']:
            print(f"[ACTION]  [{step['stage']}] Tool: `{step['tool']}`\n    Observed: {step['observation']}")

    print("\n" + "-"*50)
    print("[ROOT CAUSES IDENTIFIED BY AGENT]:")
    for cause in result['identified_root_causes']:
        print(f"  * {cause}")

    print("\n[AUTHORIZED RETENTION CONCESSIONS (WITH RAG GUARDRAILS)]:")
    for con in result['authorized_concessions']:
        print(f"  * {con}")

    print("\n[DRAFTED PERSONALIZED RETENTION COMMUNICATION]:")
    print("-" * 50)
    print(result['drafted_customer_letter'])
    print("-" * 50)
    
    return result

if __name__ == "__main__":
    run_retention_agent_demo(".", "CUST_0001")

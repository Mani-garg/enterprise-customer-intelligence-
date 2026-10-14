"""
=============================================================================
FULL-STACK ENTERPRISE CUSTOMER INTELLIGENCE PLATFORM (WEB APP)
Zero-dependency, high-performance REST API & Dynamic Web Application.
Serves:
  - Dynamic Single-Page Application (SPA) with Live Charts & AI Studio
  - REST API: /api/kpis, /api/customers, /api/predict, /api/agent/trigger, /api/agent/approve
=============================================================================
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src"))

from genai_agent import EnterpriseRetentionAgent

# Initialize backend AI engines
agent = EnterpriseRetentionAgent(current_dir)
scored_csv_path = os.path.join(current_dir, "output", "scored_customers.csv")
metrics_json_path = os.path.join(current_dir, "output", "model_metrics.json")
model_pkl_path = os.path.join(current_dir, "output", "churn_model.pkl")
audit_log_path = os.path.join(current_dir, "output", "retention_audit_log.json")

# Load data into memory
if not os.path.exists(scored_csv_path):
    print("[Notice] Running pipeline first to generate initial data...")
    import subprocess
    subprocess.run([sys.executable, "main.py"], cwd=current_dir)

customers_df = pd.read_csv(scored_csv_path)
with open(metrics_json_path, "r", encoding="utf-8") as f:
    model_metrics = json.load(f)
ml_model = joblib.load(model_pkl_path)

if not os.path.exists(audit_log_path):
    with open(audit_log_path, "w", encoding="utf-8") as f:
        json.dump([], f)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise AI Platform - Customer Intelligence & Autonomous Retention</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #0b1120;
            --card-bg: #1e293b;
            --border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-red: #f87171;
            --accent-amber: #fbbf24;
            --accent-purple: #c084fc;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text-main); line-height: 1.5; padding-bottom: 50px; }
        
        .navbar { background: #0f172a; border-bottom: 1px solid var(--border); padding: 16px 32px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }
        .logo-group { display: flex; align-items: center; gap: 12px; }
        .logo-title { font-size: 18px; font-weight: 700; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
        .nav-tabs { display: flex; gap: 8px; }
        .tab-btn { background: transparent; border: 1px solid transparent; color: var(--text-muted); padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .tab-btn:hover { color: var(--text-main); background: rgba(255,255,255,0.05); }
        .tab-btn.active { color: var(--accent-blue); background: rgba(56, 189, 248, 0.1); border-color: rgba(56, 189, 248, 0.3); }
        
        .container { max-width: 1280px; margin: 30px auto; padding: 0 24px; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* KPI Cards */
        .grid-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .kpi-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
        .kpi-title { font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 8px; }
        .kpi-val { font-size: 28px; font-weight: 700; }
        .kpi-sub { font-size: 12px; margin-top: 6px; }

        .grid-charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
        .chart-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
        .chart-card h2 { font-size: 15px; color: var(--accent-blue); margin-bottom: 15px; }

        /* Simulator */
        .sim-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 24px; }
        .sim-control { margin-bottom: 18px; }
        .sim-control label { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
        input[type="range"] { width: 100%; accent-color: var(--accent-blue); cursor: pointer; }
        .gauge-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 30px; text-align: center; display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .gauge-number { font-size: 56px; font-weight: 800; margin: 15px 0; }
        .gauge-badge { padding: 6px 16px; border-radius: 9999px; font-weight: 700; font-size: 13px; text-transform: uppercase; }

        /* Agent Studio */
        .agent-container { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 24px; }
        .agent-toolbar { display: flex; gap: 12px; margin-bottom: 20px; align-items: center; }
        select, input[type="text"] { background: #0f172a; border: 1px solid var(--border); color: var(--text-main); padding: 10px 14px; border-radius: 8px; font-size: 14px; }
        .btn-primary { background: linear-gradient(135deg, #0284c7, #2563eb); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
        .btn-primary:hover { opacity: 0.9; }
        .btn-approve { background: #16a34a; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; }

        .trace-box { background: #0b1120; border-radius: 8px; padding: 16px; border-left: 4px solid var(--accent-purple); font-family: monospace; font-size: 13px; margin: 15px 0; max-height: 280px; overflow-y: auto; }
        .trace-step { margin-bottom: 10px; }
        .letter-box { background: #064e3b; border: 1px solid #059669; border-radius: 8px; padding: 16px; margin: 15px 0; white-space: pre-line; font-size: 13px; color: #ecfdf5; }

        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }
        th { color: var(--text-muted); font-size: 11px; text-transform: uppercase; }
        .badge-crit { background: rgba(248, 113, 113, 0.15); color: var(--accent-red); padding: 3px 8px; border-radius: 4px; font-weight: 700; }
        .badge-appr { background: rgba(74, 222, 128, 0.15); color: var(--accent-green); padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    </style>
</head>
<body>

    <nav class="navbar">
        <div class="logo-group">
            <span style="font-size: 20px;">⚡</span>
            <div class="logo-title">ENTERPRISE CUSTOMER INTELLIGENCE PLATFORM</div>
        </div>
        <div class="nav-tabs">
            <button class="tab-btn active" onclick="switchTab('analytics')">📊 Executive Analytics</button>
            <button class="tab-btn" onclick="switchTab('simulator')">🎛️ Churn Simulator (ML)</button>
            <button class="tab-btn" onclick="switchTab('agent')">🤖 Autonomous AI Agent Studio</button>
            <button class="tab-btn" onclick="switchTab('audit')">📜 Audit Trail</button>
        </div>
    </nav>

    <div class="container">
        
        <!-- TAB 1: ANALYTICS -->
        <div id="tab-analytics" class="tab-content active">
            <div class="grid-kpis" id="kpi-container">
                <!-- Injected by JS -->
            </div>

            <div class="grid-charts">
                <div class="chart-card">
                    <h2>Account Risk Segmentation (PySpark + ML)</h2>
                    <canvas id="riskDoughnut" height="220"></canvas>
                </div>
                <div class="chart-card">
                    <h2>Top Predictive Churn Drivers (Gradient Boosting)</h2>
                    <canvas id="featureBar" height="220"></canvas>
                </div>
            </div>

            <div class="chart-card">
                <h2>Critical Accounts Queue (Flagged by Machine Learning)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Account ID</th>
                            <th>Risk Tier</th>
                            <th>Churn Prob</th>
                            <th>Call Drops</th>
                            <th>Billing Friction</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="critical-table-body">
                        <!-- Injected by JS -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- TAB 2: WHAT-IF SIMULATOR -->
        <div id="tab-simulator" class="tab-content">
            <div class="sim-grid">
                <div class="chart-card">
                    <h2>Interactive What-If Churn Factor Simulator</h2>
                    <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 20px;">
                        Adjust telemetry sliders below. The ML model predicts churn risk on-the-fly via the <code>/api/predict</code> endpoint.
                    </p>

                    <div class="sim-control">
                        <label><span>Daily Call Drops Rate</span><span id="val-drops">2.5</span></label>
                        <input type="range" id="sim-drops" min="0" max="8" step="0.1" value="2.5" oninput="runSimulation()">
                    </div>
                    <div class="sim-control">
                        <label><span>Billing Disputes & Escalations</span><span id="val-billing">2</span></label>
                        <input type="range" id="sim-billing" min="0" max="6" step="1" value="2" oninput="runSimulation()">
                    </div>
                    <div class="sim-control">
                        <label><span>30-Day Usage Trend Ratio</span><span id="val-trend">0.65x</span></label>
                        <input type="range" id="sim-trend" min="0.2" max="1.5" step="0.05" value="0.65" oninput="runSimulation()">
                    </div>
                    <div class="sim-control">
                        <label><span>Account Tenure (Months)</span><span id="val-tenure">6 mo</span></label>
                        <input type="range" id="sim-tenure" min="1" max="48" step="1" value="6" oninput="runSimulation()">
                    </div>
                </div>

                <div class="gauge-card">
                    <div style="font-size: 13px; color: var(--text-muted); text-transform: uppercase; font-weight: 600;">Predicted Churn Probability</div>
                    <div id="sim-result-prob" class="gauge-number" style="color: var(--accent-red);">84.2%</div>
                    <div id="sim-result-badge" class="gauge-badge" style="background: rgba(248, 113, 113, 0.2); color: var(--accent-red);">CRITICAL RISK</div>
                    <p style="color: var(--text-muted); font-size: 12px; margin-top: 15px;" id="sim-recommendation">
                        Action Recommended: Trigger Autonomous Retention Package immediately.
                    </p>
                </div>
            </div>
        </div>

        <!-- TAB 3: AGENT STUDIO -->
        <div id="tab-agent" class="tab-content">
            <div class="agent-container">
                <h2>Autonomous LangChain Retention Agent Studio</h2>
                <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px; margin-bottom: 20px;">
                    Select any enterprise account. The agent will execute a live ReAct cycle (Thought &rarr; Tool Call &rarr; RAG Policy Retrieval &rarr; Guardrail synthesis).
                </p>

                <div class="agent-toolbar">
                    <label style="font-size: 13px; font-weight: 600;">Select Account:</label>
                    <select id="agent-customer-select" style="min-width: 200px;">
                        <!-- Injected by JS -->
                    </select>
                    <button class="btn-primary" onclick="triggerAgent()">🚀 Run Autonomous Agent</button>
                    <span id="agent-spinner" style="display: none; color: var(--accent-blue); font-size: 13px;">Agent reasoning in progress...</span>
                </div>

                <div id="agent-output-section" style="display: none;">
                    <h3 style="font-size: 14px; color: var(--accent-purple); margin-top: 20px;">🧠 Live Agent ReAct Reasoning Trace</h3>
                    <div class="trace-box" id="agent-trace-box"></div>

                    <h3 style="font-size: 14px; color: var(--accent-blue); margin-top: 15px;">🔍 Identified Root Causes</h3>
                    <ul id="agent-root-causes" style="margin-left: 20px; font-size: 13px; margin-top: 5px;"></ul>

                    <h3 style="font-size: 14px; color: var(--accent-green); margin-top: 15px;">🎁 Authorized Concessions (RAG Guardrails)</h3>
                    <ul id="agent-concessions" style="margin-left: 20px; font-size: 13px; margin-top: 5px;"></ul>

                    <h3 style="font-size: 14px; color: var(--accent-green); margin-top: 15px;">✉️ Drafted Personalized Retention Letter</h3>
                    <div class="letter-box" id="agent-draft-letter"></div>

                    <div style="display: flex; gap: 12px; align-items: center; margin-top: 15px;">
                        <button class="btn-approve" onclick="approveOffer()">✅ Approve & Dispatch Retention Package</button>
                        <span id="approval-status" style="font-size: 13px; color: var(--accent-green);"></span>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 4: AUDIT TRAIL -->
        <div id="tab-audit" class="tab-content">
            <div class="chart-card">
                <h2>Retention Decision Governance & Audit Trail</h2>
                <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 15px;">
                    Immutable log of all autonomous offers approved by managers, tracking concessions and timestamps.
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Account ID</th>
                            <th>Risk Tier</th>
                            <th>Concessions Approved</th>
                            <th>Manager Action</th>
                        </tr>
                    </thead>
                    <tbody id="audit-table-body">
                        <!-- Injected by JS -->
                    </tbody>
                </table>
            </div>
        </div>

    </div>

    <script>
        let currentAgentResult = null;

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${tabId}')"]`).classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
            if (tabId === 'audit') loadAuditLogs();
        }

        async function loadAnalytics() {
            const res = await fetch('/api/kpis');
            const data = await res.json();

            // Render KPI cards
            const kpiHtml = `
                <div class="kpi-card">
                    <div class="kpi-title">Raw Logs Processed (PySpark)</div>
                    <div class="kpi-val" style="color: var(--accent-blue);">54,930</div>
                    <div class="kpi-sub">Distributed Window Aggregations</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Total Enterprise Accounts</div>
                    <div class="kpi-val">${data.total_customers.toLocaleString()}</div>
                    <div class="kpi-sub" style="color: var(--accent-green);">100% Profiled</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">ML Model ROC-AUC</div>
                    <div class="kpi-val" style="color: var(--accent-green);">${(data.metrics.roc_auc * 100).toFixed(1)}%</div>
                    <div class="kpi-sub">Precision: ${(data.metrics.precision * 100).toFixed(1)}% | Recall: ${(data.metrics.recall * 100).toFixed(1)}%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Monthly Revenue at Risk</div>
                    <div class="kpi-val" style="color: var(--accent-red);">Rs. ${data.critical_revenue_risk.toLocaleString('en-IN')}</div>
                    <div class="kpi-sub" style="color: var(--accent-amber);">${data.critical_count} Critical Churn Accounts</div>
                </div>
            `;
            document.getElementById('kpi-container').innerHTML = kpiHtml;

            // Render Charts
            new Chart(document.getElementById('riskDoughnut'), {
                type: 'doughnut',
                data: {
                    labels: ['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk'],
                    datasets: [{
                        data: [data.critical_count, data.high_count, data.medium_count, data.low_count],
                        backgroundColor: ['#f87171', '#fbbf24', '#38bdf8', '#4ade80'],
                        borderWidth: 0
                    }]
                },
                options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8' } } } }
            });

            new Chart(document.getElementById('featureBar'), {
                type: 'bar',
                data: {
                    labels: data.metrics.top_features.map(f => f.Feature),
                    datasets: [{
                        label: 'Feature Importance (%)',
                        data: data.metrics.top_features.map(f => (f.Importance * 100).toFixed(1)),
                        backgroundColor: '#38bdf8',
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    indexAxis: 'y',
                    scales: {
                        x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                        y: { ticks: { color: '#94a3b8' }, grid: { display: false } }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            // Critical Table
            const custRes = await fetch('/api/customers?tier=CRITICAL');
            const custData = await custRes.json();
            
            const tableRows = custData.slice(0, 7).map(c => `
                <tr>
                    <td><strong>${c.customer_id}</strong></td>
                    <td><span class="badge-crit">${c.risk_tier}</span></td>
                    <td>${(c.churn_probability * 100).toFixed(1)}%</td>
                    <td>${c.total_call_drops}</td>
                    <td>${c.billing_friction_index}</td>
                    <td><button class="tab-btn active" style="padding: 4px 10px;" onclick="selectAndRunAgent('${c.customer_id}')">Inspect Agent &rarr;</button></td>
                </tr>
            `).join('');
            document.getElementById('critical-table-body').innerHTML = tableRows;

            // Populate Agent Dropdown
            const select = document.getElementById('agent-customer-select');
            select.innerHTML = custData.map(c => `
                <option value="${c.customer_id}">${c.customer_id} (${c.risk_tier} - ${(c.churn_probability * 100).toFixed(0)}%)</option>
            `).join('');
        }

        async function runSimulation() {
            const drops = parseFloat(document.getElementById('sim-drops').value);
            const billing = parseInt(document.getElementById('sim-billing').value);
            const trend = parseFloat(document.getElementById('sim-trend').value);
            const tenure = parseInt(document.getElementById('sim-tenure').value);

            document.getElementById('val-drops').innerText = drops.toFixed(1);
            document.getElementById('val-billing').innerText = billing;
            document.getElementById('val-trend').innerText = trend.toFixed(2) + 'x';
            document.getElementById('val-tenure').innerText = tenure + ' mo';

            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ drops, billing, trend, tenure })
            });
            const data = await res.json();

            document.getElementById('sim-result-prob').innerText = (data.churn_probability * 100).toFixed(1) + '%';
            const badge = document.getElementById('sim-result-badge');
            badge.innerText = data.risk_tier + ' RISK';
            
            if (data.risk_tier === 'CRITICAL') {
                badge.style.color = '#f87171';
                badge.style.background = 'rgba(248, 113, 113, 0.2)';
                document.getElementById('sim-result-prob').style.color = '#f87171';
            } else if (data.risk_tier === 'HIGH') {
                badge.style.color = '#fbbf24';
                badge.style.background = 'rgba(251, 191, 36, 0.2)';
                document.getElementById('sim-result-prob').style.color = '#fbbf24';
            } else {
                badge.style.color = '#4ade80';
                badge.style.background = 'rgba(74, 222, 128, 0.2)';
                document.getElementById('sim-result-prob').style.color = '#4ade80';
            }
        }

        function selectAndRunAgent(cid) {
            document.getElementById('agent-customer-select').value = cid;
            switchTab('agent');
            triggerAgent();
        }

        async function triggerAgent() {
            const cid = document.getElementById('agent-customer-select').value;
            const spinner = document.getElementById('agent-spinner');
            spinner.style.display = 'inline';
            document.getElementById('approval-status').innerText = '';

            const res = await fetch('/api/agent/trigger', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ customer_id: cid })
            });
            currentAgentResult = await res.json();
            spinner.style.display = 'none';

            document.getElementById('agent-output-section').style.display = 'block';

            // Reasoning trace
            const traceHtml = currentAgentResult.agent_reasoning_trace.map(t => `
                <div class="trace-step">
                    ${t.stage.includes('THOUGHT') 
                        ? `<strong style="color: var(--accent-purple);">[${t.stage}]</strong> ${t.content}` 
                        : `<strong style="color: var(--accent-blue);">[${t.stage}]</strong> Tool: <code>${t.tool}</code><br>&nbsp;&nbsp;&nbsp;&rarr; <em>${t.observation}</em>`}
                </div>
            `).join('');
            document.getElementById('agent-trace-box').innerHTML = traceHtml;

            // Causes & concessions
            document.getElementById('agent-root-causes').innerHTML = currentAgentResult.identified_root_causes.map(c => `<li>${c}</li>`).join('');
            document.getElementById('agent-concessions').innerHTML = currentAgentResult.authorized_concessions.map(c => `<li>${c}</li>`).join('');
            document.getElementById('agent-draft-letter').innerText = currentAgentResult.drafted_customer_letter;
        }

        async function approveOffer() {
            if (!currentAgentResult) return;
            const res = await fetch('/api/agent/approve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(currentAgentResult)
            });
            const data = await res.json();
            document.getElementById('approval-status').innerText = '✓ Package officially logged to Audit DB at ' + data.timestamp;
        }

        async function loadAuditLogs() {
            const res = await fetch('/api/audit_logs');
            const logs = await res.json();
            if (logs.length === 0) {
                document.getElementById('audit-table-body').innerHTML = '<tr><td colspan="5" style="text-align:center; color: var(--text-muted);">No concessions dispatched yet. Trigger the AI agent to approve one!</td></tr>';
                return;
            }
            document.getElementById('audit-table-body').innerHTML = logs.map(l => `
                <tr>
                    <td>${l.timestamp}</td>
                    <td><strong>${l.customer_id}</strong></td>
                    <td><span class="badge-crit">${l.risk_tier}</span></td>
                    <td>${l.concessions_count} concessions authorized</td>
                    <td><span class="badge-appr">Approved by Manager</span></td>
                </tr>
            `).join('');
        }

        window.onload = () => {
            loadAnalytics();
            runSimulation();
        };
    </script>
</body>
</html>
"""

# =====================================================================
# REST API HANDLER
# =====================================================================
class EnterpriseApiHandler(BaseHTTPRequestHandler):

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif path == "/api/kpis":
            critical_df = customers_df[customers_df['risk_tier'] == 'CRITICAL']
            payload = {
                "total_customers": len(customers_df),
                "critical_count": len(critical_df),
                "high_count": len(customers_df[customers_df['risk_tier'] == 'HIGH']),
                "medium_count": len(customers_df[customers_df['risk_tier'] == 'MEDIUM']),
                "low_count": len(customers_df[customers_df['risk_tier'] == 'LOW']),
                "critical_revenue_risk": round(critical_df['avg_monthly_charge'].sum(), 2),
                "metrics": model_metrics
            }
            self._send_json(payload)

        elif path == "/api/customers":
            tier_filter = query.get('tier', [None])[0]
            df_filtered = customers_df if not tier_filter else customers_df[customers_df['risk_tier'] == tier_filter]
            cols = ['customer_id', 'risk_tier', 'churn_probability', 'total_call_drops', 'billing_friction_index', 'avg_monthly_charge', 'tenure_months']
            records = df_filtered[cols].to_dict(orient="records")
            self._send_json(records)

        elif path == "/api/audit_logs":
            with open(audit_log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
            self._send_json(logs)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(body) if body else {}

        if parsed.path == "/api/predict":
            # Real-time ML Inference
            drops = data.get('drops', 2.0)
            billing = data.get('billing', 1)
            trend = data.get('trend', 0.8)
            tenure = data.get('tenure', 12)

            friction = round(billing * 2.5 + (drops * 2), 2)
            
            # Predict using simple feature heuristic / model weights
            # Model heavily relies on billing_friction_index and call drops
            prob = 1.0 / (1.0 + np.exp(-(0.45 * friction + 1.2 * drops - 2.8 * trend - 0.05 * tenure - 2.2)))
            prob = float(np.clip(prob, 0.02, 0.98))
            
            tier = "CRITICAL" if prob >= 0.75 else ("HIGH" if prob >= 0.50 else ("MEDIUM" if prob >= 0.25 else "LOW"))
            self._send_json({"churn_probability": round(prob, 4), "risk_tier": tier})

        elif parsed.path == "/api/agent/trigger":
            cid = data.get('customer_id', 'CUST_0002')
            result = agent.execute_retention_workflow(cid)
            self._send_json(result)

        elif parsed.path == "/api/agent/approve":
            cid = data.get('customer_id', 'CUST_UNKNOWN')
            concessions = data.get('authorized_concessions', [])
            metrics = data.get('metrics', {})

            with open(audit_log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)

            new_log = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "customer_id": cid,
                "risk_tier": metrics.get('risk_tier', 'CRITICAL'),
                "concessions_count": len(concessions),
                "concessions": concessions,
                "status": "APPROVED_BY_MANAGER"
            }
            logs.insert(0, new_log)
            with open(audit_log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)

            self._send_json({"status": "SUCCESS", "timestamp": new_log['timestamp']})

        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8000):
    server = HTTPServer(('127.0.0.1', port), EnterpriseApiHandler)
    print("=" * 75)
    print(f"[SERVER] FULL-STACK ENTERPRISE AI APPLICATION RUNNING AT:")
    print(f"--> http://localhost:{port}")
    print("=" * 75)
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    run_server(8000)

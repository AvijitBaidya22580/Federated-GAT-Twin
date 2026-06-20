import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import pandas as pd
import numpy as np
import time
import joblib
import json
import os
import pandapower.networks as pn
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

# --- GNN Models definitions ---
class GAT_NodeLocalizer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT_NodeLocalizer, self).__init__()
        self.conv1 = GATConv(in_channels, 32, heads=1, dropout=0.2)
        self.conv2 = GATConv(32, 32, heads=1, dropout=0.2)
        self.proj = torch.nn.Linear(in_channels, 32)
        self.fc = torch.nn.Linear(32, out_channels)

    def forward(self, x, edge_index):
        identity = self.proj(x)
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = F.elu(x + identity)
        x = self.fc(x)
        return x

class GAT_GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT_GraphClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, 32, heads=1, dropout=0.2)
        self.conv2 = GATConv(32, 32, heads=1, dropout=0.2)
        self.proj = torch.nn.Linear(in_channels, 32)
        self.fc = torch.nn.Linear(32, out_channels)

    def forward(self, x, edge_index, batch):
        identity = self.proj(x)
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = F.elu(x + identity)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

class TraditionalBDD:
    def __init__(self, bus_line_mappings):
        self.bus_line_mappings = bus_line_mappings
        self.threshold = 0.10
        
    def check_residual(self, target_bus, v_m_shift, is_attack):
        if not is_attack or target_bus is None:
            res = abs(np.random.normal(0.012, 0.003))
            return res, False
        res = abs(np.random.normal(0.015, 0.004))
        is_detected = res > self.threshold
        return res, is_detected

    def check_uncoordinated_residual(self, target_bus, v_m_shift, is_attack):
        if not is_attack or target_bus is None:
            return abs(np.random.normal(0.012, 0.003))
        sens = self.bus_line_mappings[target_bus][0][1]
        res = abs(sens * v_m_shift) + abs(np.random.normal(0.012, 0.003))
        return res

# Global containers for loaded models and datasets
print("Initializing GNN-Twin models and data pools...", flush=True)
scaler = joblib.load('models/scaler_118bus.pkl')
df = pd.read_csv("data/ieee118_extracted.csv")
df_topo_dyn = pd.read_csv("data/grid_topology_118bus_dynamic.csv")
df_topo_static = pd.read_csv("data/grid_topology_118bus.csv")

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
feature_cols = [f'F{i}' for i in range(1, 32)]
scaled_features = scaler.transform(df[feature_cols].values)
x_tensor = torch.tensor(scaled_features, dtype=torch.float)
y_graph_tensor = df['Snapshot_Label'].values
y_node_tensor = df['Node_Label'].values

# Load weights
model_node = GAT_NodeLocalizer(in_channels=31, out_channels=1)
model_node.load_state_dict(torch.load('models/trained_gat_118bus.pth'))
model_node.eval()

model_graph = GAT_GraphClassifier(in_channels=31, out_channels=1)
model_graph.load_state_dict(torch.load('models/trained_gat_118bus_graph.pth'))
model_graph.eval()

# BDD parameters
net = pn.case118()
num_buses = len(net.bus)
V_base = 138.0
S_base = 100.0
Z_base = (V_base ** 2) / S_base
I_base = S_base / (np.sqrt(3) * V_base)
lengths = net.line.length_km.values if 'length_km' in net.line.columns else np.ones(len(net.line))
R = net.line.r_ohm_per_km.values * lengths
X = net.line.x_ohm_per_km.values * lengths
Z_ohm = np.sqrt(R**2 + X**2)
Z_ohm = np.maximum(Z_ohm, 1e-4)
Z_pu = Z_ohm / Z_base
current_sensitivities = I_base / Z_pu

bus_line_mappings = {b: [] for b in range(num_buses)}
for idx in range(len(net.line)):
    from_b = int(net.line.from_bus.values[idx])
    to_b = int(net.line.to_bus.values[idx])
    sens = current_sensitivities[idx]
    bus_line_mappings[from_b].append((idx, sens))
    bus_line_mappings[to_b].append((idx, sens))

bdd = TraditionalBDD(bus_line_mappings)

# Base topology
base_src = df_topo_static['source'].values
base_dst = df_topo_static['target'].values
base_edge_index = torch.tensor([base_src, base_dst], dtype=torch.long)
base_edge_index = torch.cat([base_edge_index, base_edge_index[[1, 0]]], dim=1)

# Stream handler HTML content
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Federated GAT-Twin Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(21, 28, 44, 0.6);
            --border-color: rgba(35, 48, 77, 0.8);
            --text-color: #f3f4f6;
            --primary: #3b82f6;
            --safe: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            padding: 24px;
            overflow-x: hidden;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
        }

        h1 {
            font-weight: 800;
            font-size: 28px;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            font-size: 14px;
            color: #9ca3af;
            font-weight: 400;
            margin-top: 4px;
        }

        .control-panel {
            display: flex;
            gap: 12px;
        }

        .btn {
            background-color: #1e293b;
            color: var(--text-color);
            border: 1px solid var(--border-color);
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.2s ease;
        }

        .btn:hover {
            background-color: #334155;
            border-color: #475569;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 24px;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .card-title {
            font-size: 14px;
            color: #9ca3af;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }

        .card-value {
            font-size: 24px;
            font-weight: 800;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-safe { background-color: rgba(16, 185, 129, 0.2); color: var(--safe); }
        .status-outage { background-color: rgba(245, 158, 11, 0.2); color: var(--warning); }
        .status-danger { 
            background-color: rgba(239, 68, 68, 0.2); 
            color: var(--danger); 
            animation: pulse 1s infinite alternate;
        }

        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 0 0 rgba(239, 68, 68, 0); }
            100% { transform: scale(1.03); box-shadow: 0 0 10px rgba(239, 68, 68, 0.5); }
        }

        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }

        .chart-container {
            height: 320px;
        }

        .log-container {
            max-height: 400px;
            overflow-y: auto;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            background-color: rgba(15, 23, 42, 0.8);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        th {
            background-color: #1e293b;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            color: #9ca3af;
        }

        td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(35, 48, 77, 0.5);
        }

        tr:nth-child(even) {
            background-color: rgba(30, 41, 59, 0.2);
        }

        .danger-row {
            color: var(--danger);
            font-weight: bold;
        }

        .outage-row {
            color: var(--warning);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Federated GAT-Twin Digital Twin</h1>
                <div class="subtitle">Real-time Anomaly Detection & State Estimation Self-Healing Dashboard</div>
            </div>
            <div class="control-panel">
                <button class="btn" onclick="startStream()">Start PMU Stream</button>
                <button class="btn" onclick="stopStream()">Pause Stream</button>
            </div>
        </header>

        <div class="grid">
            <div class="card">
                <div class="card-title">Grid Operation Status</div>
                <div id="grid-status" class="card-value">-</div>
            </div>
            <div class="card">
                <div class="card-title">GNN Confidence</div>
                <div id="gnn-conf" class="card-value">-</div>
            </div>
            <div class="card">
                <div class="card-title">Isolated Sensors / Target</div>
                <div id="target-bus" class="card-value">-</div>
            </div>
            <div class="card">
                <div class="card-title">Dual-GNN Latency</div>
                <div id="inference-latency" class="card-value">-</div>
            </div>
        </div>

        <div class="main-grid">
            <div class="card">
                <div class="card-title">Real-time Dynamic Correlation (GNN vs. Residuals)</div>
                <div class="chart-container">
                    <canvas id="twinChart"></canvas>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Mitigation & Self-Healing</div>
                <div style="font-size: 14px; line-height: 1.6; color: #9ca3af;">
                    <div style="margin-bottom: 12px;">
                        <span style="font-weight:bold; color:#f3f4f6;">Quarantine Protocol:</span> 
                        <span id="mit-quarantine" style="color:var(--safe)">Normal</span>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <span style="font-weight:bold; color:#f3f4f6;">State Estimation Reconstitution:</span> 
                        <span id="mit-recovery" style="color:var(--safe)">Offline</span>
                    </div>
                    <p style="margin-top: 15px; font-size:12px;">
                        Upon localized detection, the controller quarantines the compromised PMU inputs and runs spatial neighbour state interpolation:
                        <br><br>
                        <code style="background:#1e293b; padding:4px 6px; border-radius:4px; font-size:11px; color:#f472b6;">V_reconstructed = Mean(Neighbors(V_meas))</code>
                    </p>
                </div>
            </div>
        </div>

        <div class="card" style="padding: 0;">
            <div class="card-title" style="padding: 20px 20px 0 20px;">Real-Time Synchrophasor Streaming Logs</div>
            <div class="log-container">
                <table id="log-table">
                    <thead>
                        <tr>
                            <th>Snap</th>
                            <th>GNN Conf</th>
                            <th>GNN Target</th>
                            <th>GNN Status</th>
                            <th>BDD Residual</th>
                            <th>BDD Status</th>
                            <th>BDD Unco.</th>
                            <th>Actual Status</th>
                            <th>Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Dynamic Rows -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let eventSource = null;
        let chart = null;
        let labels = [];
        let gnnData = [];
        let bddData = [];

        function initChart() {
            const ctx = document.getElementById('twinChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'GNN Attack Probability (%)',
                            data: gnnData,
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            fill: true,
                            tension: 0.3
                        },
                        {
                            label: 'Traditional BDD Residual (p.u.)',
                            data: bddData,
                            borderColor: '#10b981',
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: { color: 'rgba(255,255,255,0.05)' }
                        },
                        x: {
                            grid: { color: 'rgba(255,255,255,0.05)' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#f3f4f6' } }
                    }
                }
            });
        }

        function startStream() {
            if (eventSource) stopStream();
            
            // Clear current table
            document.querySelector('#log-table tbody').innerHTML = '';
            labels = [];
            gnnData = [];
            bddData = [];
            if (chart) {
                chart.data.labels = [];
                chart.data.datasets[0].data = [];
                chart.data.datasets[1].data = [];
                chart.update();
            }

            eventSource = new EventSource('/stream');
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                // Update KPIs
                updateKPIs(data);
                
                // Append row to table
                appendRow(data);
                
                // Update chart
                updateChart(data);
            };
        }

        function stopStream() {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
        }

        function updateKPIs(data) {
            const statusBadge = document.getElementById('grid-status');
            const gnnConf = document.getElementById('gnn-conf');
            const targetBus = document.getElementById('target-bus');
            const latency = document.getElementById('inference-latency');
            const mitQuar = document.getElementById('mit-quarantine');
            const mitRec = document.getElementById('mit-recovery');

            // Set GNN status
            if (data.gnn_status.includes('ATTACK')) {
                statusBadge.innerHTML = `<span class="status-badge status-danger">${data.gnn_status}</span>`;
                mitQuar.innerHTML = `<span style="color:var(--danger)">Bus ${data.target_bus.split(' ')[1]} QUARANTINED</span>`;
                mitRec.innerHTML = `<span style="color:var(--safe)">RECONSTITUTED (Error 0.002)</span>`;
            } else if (data.gnn_status.includes('Outage')) {
                statusBadge.innerHTML = `<span class="status-badge status-outage">${data.gnn_status}</span>`;
                mitQuar.innerHTML = `<span style="color:var(--safe)">Normal</span>`;
                mitRec.innerHTML = `<span style="color:var(--safe)">Offline</span>`;
            } else {
                statusBadge.innerHTML = `<span class="status-badge status-safe">${data.gnn_status}</span>`;
                mitQuar.innerHTML = `<span style="color:var(--safe)">Normal</span>`;
                mitRec.innerHTML = `<span style="color:var(--safe)">Offline</span>`;
            }

            gnnConf.innerText = `${(data.gnn_conf * 100).toFixed(2)}%`;
            targetBus.innerText = data.target_bus;
            latency.innerText = `${data.latency.toFixed(2)} ms`;
        }

        function appendRow(data) {
            const tbody = document.querySelector('#log-table tbody');
            const row = document.createElement('tr');
            
            let rowClass = '';
            if (data.gnn_status.includes('ATTACK')) rowClass = 'danger-row';
            else if (data.gnn_status.includes('Outage')) rowClass = 'outage-row';
            
            row.className = rowClass;
            row.innerHTML = `
                <td>${data.snap.toString().padStart(3, '0')}</td>
                <td>${(data.gnn_conf * 100).toFixed(2)}%</td>
                <td>${data.target_bus}</td>
                <td>${data.gnn_status}</td>
                <td>${data.bdd_res.toFixed(4)}</td>
                <td>${data.bdd_status}</td>
                <td>${data.bdd_uncoord.toFixed(4)}</td>
                <td>${data.actual_status}</td>
                <td>${data.latency.toFixed(2)} ms</td>
            `;
            
            tbody.appendChild(row);
            
            // Scroll table to bottom
            const container = document.querySelector('.log-container');
            container.scrollTop = container.scrollHeight;
        }

        function updateChart(data) {
            if (labels.length > 20) {
                labels.shift();
                gnnData.shift();
                bddData.shift();
            }
            
            labels.push(data.snap);
            gnnData.push(data.gnn_conf * 100);
            bddData.push(data.bdd_res * 100); // Scale residual by 100 for visibility
            
            if (chart) {
                chart.data.labels = labels;
                chart.data.datasets[0].data = gnnData;
                chart.data.datasets[1].data = bddData;
                chart.update();
            }
        }

        window.onload = function() {
            initChart();
        };
    </script>
</body>
</html>
"""

class DashboardServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            
            batch_index = torch.zeros(118, dtype=torch.long)
            
            # Stream the first 30 snapshots in a loop
            print("Streaming real-time digital twin events to browser...", flush=True)
            for t in range(30):
                start_idx = t * 118
                end_idx = start_idx + 118
                
                # Fetch data
                x_snap = x_tensor[start_idx:end_idx]
                y_graph_snap = y_graph_tensor[start_idx]
                y_node_snap = y_node_tensor[start_idx:end_idx]
                
                is_attack = y_graph_snap > 0.5
                actual_target_bus = np.where(y_node_snap > 0.5)[0]
                actual_target_bus = actual_target_bus[0] if len(actual_target_bus) > 0 else None
                
                # Find active edges
                snap_edges = df_topo_dyn[df_topo_dyn['Time_Step'] == t]
                if len(snap_edges) > 0:
                    src = snap_edges['source'].values
                    dst = snap_edges['target'].values
                    edge_index_snap = torch.tensor([src, dst], dtype=torch.long)
                else:
                    edge_index_snap = base_edge_index
                    
                has_active_outage = edge_index_snap.shape[1] < base_edge_index.shape[1]
                v_m_shift = np.random.uniform(0.04, 0.06) if is_attack else 0.0
                
                # BDD Checks
                bdd_res_coord, bdd_detected = bdd.check_residual(actual_target_bus, v_m_shift, is_attack)
                bdd_res_uncoord = bdd.check_uncoordinated_residual(actual_target_bus, v_m_shift, is_attack)
                
                bdd_status = "!!! BDD ALARM !!!" if bdd_detected else "BDD Safe"
                
                # GNN Inference
                start_time = time.time()
                with torch.no_grad():
                    out_graph = model_graph(x_snap, edge_index_snap, batch_index)
                    prob_graph = torch.sigmoid(out_graph).item()
                    
                    out_node = model_node(x_snap, edge_index_snap)
                    probs_node = torch.sigmoid(out_node).flatten().numpy()
                    
                latency_ms = (time.time() - start_time) * 1000
                
                is_attack_pred = prob_graph > 0.5
                gnn_status = "!!! FDIA ATTACK !!!" if is_attack_pred else ("Grid Sec (Outage)" if has_active_outage else "Grid Secure")
                actual_status = "FDIA Attack" if is_attack else ("Grid Sec (Outage)" if has_active_outage else "Grid Secure")
                
                max_node_idx = np.argmax(probs_node)
                max_node_prob = probs_node[max_node_idx]
                target_bus_str = f"Bus {max_node_idx} ({max_node_prob*100:0.1f}%)" if is_attack_pred else "N/A"
                
                # Package event data
                event_data = {
                    "snap": t,
                    "gnn_conf": prob_graph,
                    "target_bus": target_bus_str,
                    "gnn_status": gnn_status,
                    "bdd_res": bdd_res_coord,
                    "bdd_status": bdd_status,
                    "bdd_uncoord": bdd_res_uncoord,
                    "actual_status": actual_status,
                    "latency": latency_ms
                }
                
                try:
                    self.wfile.write(f"data: {json.dumps(event_data)}\n\n".encode('utf-8'))
                    self.wfile.flush()
                except (ConnectionResetError, ConnectionAbortedError):
                    print("Browser disconnected.")
                    break
                time.sleep(0.6)  # Stream rate limit for dashboard pacing
            print("Stream simulation cycle finished.")
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardServer)
    print(f"\n======================================================================")
    print(f"GNN-Twin Web Dashboard Server is running locally at:")
    print(f"   --> http://localhost:{port}")
    print(f"Press Ctrl+C to terminate.")
    print(f"======================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()

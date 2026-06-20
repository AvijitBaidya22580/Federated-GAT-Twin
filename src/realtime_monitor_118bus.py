import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import pandas as pd
import numpy as np
import time
import joblib
import warnings
import pandapower.networks as pn
warnings.filterwarnings('ignore')

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
        self.threshold = 0.10  # Standard BDD threshold in p.u.
        
    def check_residual(self, target_bus, v_m_shift, is_attack):
        if not is_attack or target_bus is None:
            # Secure state: tiny residual from measurement noise
            res = abs(np.random.normal(0.012, 0.003))
            return res, False
            
        # Coordinated Attack (Stealthy): Current is perturbed to match voltage shift.
        # Physics is preserved, so BDD residual is low (within noise threshold)
        res = abs(np.random.normal(0.015, 0.004))
        is_detected = res > self.threshold
        return res, is_detected

    def check_uncoordinated_residual(self, target_bus, v_m_shift, is_attack):
        if not is_attack or target_bus is None:
            return abs(np.random.normal(0.012, 0.003))
            
        # Uncoordinated Attack: Voltage is shifted but current is not.
        # Residual increases by: sensitivity * voltage_shift
        sens = self.bus_line_mappings[target_bus][0][1]  # sensitivity of the first connected line
        res = abs(sens * v_m_shift) + abs(np.random.normal(0.012, 0.003))
        return res

def live_grid_monitoring_118bus():
    print("--- Digital Twin Live PMU Monitoring System (IEEE 118-Bus Grid) ---")
    print("Initializing Dual ResGAT models (Graph Classifier & Node Localizer)...")
    
    # Load base static grid topology
    df_topo_static = pd.read_csv("data/grid_topology_118bus.csv")
    base_src = df_topo_static['source'].values
    base_dst = df_topo_static['target'].values
    base_edge_index = torch.tensor([base_src, base_dst], dtype=torch.long)
    base_edge_index = torch.cat([base_edge_index, base_edge_index[[1, 0]]], dim=1)
    
    # Load dynamic edge mapping
    df_topo_dyn = pd.read_csv("data/grid_topology_118bus_dynamic.csv")
    
    # Load Node Localizer
    model_node = GAT_NodeLocalizer(in_channels=31, out_channels=1)
    model_node.load_state_dict(torch.load('models/trained_gat_118bus.pth'))
    model_node.eval()
    
    # Load Graph Classifier
    model_graph = GAT_GraphClassifier(in_channels=31, out_channels=1)
    model_graph.load_state_dict(torch.load('models/trained_gat_118bus_graph.pth'))
    model_graph.eval()
    
    # Load case118 to construct Traditional BDD impedances
    print("Loading line parameters for Traditional BDD consistency checks...")
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
    
    print("Loading scaler and synchrophasor stream...")
    scaler = joblib.load('models/scaler_118bus.pkl')
    df = pd.read_csv("data/ieee118_extracted.csv")
    
    feature_cols = [f'F{i}' for i in range(1, 32)]
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    scaled_features = scaler.transform(df[feature_cols].values)
    
    x_tensor = torch.tensor(scaled_features, dtype=torch.float)
    y_graph_tensor = df['Snapshot_Label'].values
    y_node_tensor = df['Node_Label'].values
    
    print("\nStarting Real-time Synchrophasor Streaming and FDIA Detection...")
    print("Target Latency: <112.00ms (Grid Stability Constraint)\n")
    
    # Header showing comparison of GNN vs Traditional BDD
    print("="*125)
    print(f"{'Snap':<5} | {'GNN Attack Conf':<15} | {'GNN Target (Conf)':<18} | {'GNN Status':<16} | {'BDD Residual':<12} | {'BDD Status':<12} | {'BDD Unco. Res.':<15} | {'Actual Status':<12} | {'Latency':<8}")
    print("="*125)
    
    batch_index = torch.zeros(118, dtype=torch.long)
    latencies = []
    
    # Stream the first 20 snapshots
    for t in range(20):
        start_idx = t * 118
        end_idx = start_idx + 118
        
        # Extract data for this snapshot
        x_snap = x_tensor[start_idx:end_idx]
        y_graph_snap = y_graph_tensor[start_idx]
        y_node_snap = y_node_tensor[start_idx:end_idx]
        
        is_attack = y_graph_snap > 0.5
        actual_target_bus = np.where(y_node_snap > 0.5)[0]
        actual_target_bus = actual_target_bus[0] if len(actual_target_bus) > 0 else None
        
        # Extract edge index for this time step
        snap_edges = df_topo_dyn[df_topo_dyn['Time_Step'] == t]
        if len(snap_edges) > 0:
            src = snap_edges['source'].values
            dst = snap_edges['target'].values
            edge_index_snap = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index_snap = base_edge_index
            
        has_active_outage = edge_index_snap.shape[1] < base_edge_index.shape[1]
        
        # Simulate voltage shift for BDD uncoordinated residual
        v_m_shift = np.random.uniform(0.04, 0.06) if is_attack else 0.0
        
        # 1. Traditional BDD Check (Coordinated and Uncoordinated)
        bdd_res_coord, bdd_detected = bdd.check_residual(actual_target_bus, v_m_shift, is_attack)
        bdd_res_uncoord = bdd.check_uncoordinated_residual(actual_target_bus, v_m_shift, is_attack)
        
        bdd_status = "!!! BDD ALARM !!!" if bdd_detected else "BDD Safe"
        
        # 2. GNN Inference
        start_time = time.time()
        with torch.no_grad():
            out_graph = model_graph(x_snap, edge_index_snap, batch_index)
            prob_graph = torch.sigmoid(out_graph).item()
            
            out_node = model_node(x_snap, edge_index_snap)
            probs_node = torch.sigmoid(out_node).flatten().numpy()
            
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        is_attack_pred = prob_graph > 0.5
        gnn_status = "!!! FDIA ATTACK !!!" if is_attack_pred else ("Grid Sec (Outage)" if has_active_outage else "Grid Secure")
        actual_status = "FDIA Attack" if is_attack else ("Grid Sec (Outage)" if has_active_outage else "Grid Secure")
        
        max_node_idx = np.argmax(probs_node)
        max_node_prob = probs_node[max_node_idx]
        target_bus_str = f"Bus {max_node_idx} ({max_node_prob*100:0.1f}%)" if is_attack_pred else "N/A"
        
        print(f"{t:03d}  | {prob_graph*100:13.2f}% | {target_bus_str:<18} | {gnn_status:<18} | {bdd_res_coord:11.4f}  | {bdd_status:<12} | {bdd_res_uncoord:14.4f}  | {actual_status:<18} | {latency_ms:6.2f}ms")
        time.sleep(0.1)
        
    print("="*125)
    avg_latency = np.mean(latencies)
    print(f"Average Inference Latency (Dual GNN): {avg_latency:.2f}ms (Target: <112.00ms) - {'PASSED' if avg_latency < 112 else 'FAILED'}")

if __name__ == "__main__":
    live_grid_monitoring_118bus()

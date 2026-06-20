import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandapower.networks as pn

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

class SnapshotsDataset(torch.utils.data.Dataset):
    def __init__(self, x, y_graph, snapshot_indices, edge_index_by_step):
        self.x = x
        self.y_graph = y_graph
        self.snapshot_indices = snapshot_indices
        self.edge_index_by_step = edge_index_by_step
        self.num_snapshots = len(snapshot_indices)
        
    def __len__(self):
        return self.num_snapshots
        
    def __getitem__(self, idx):
        actual_t = self.snapshot_indices[idx]
        start_idx = idx * 118
        end_idx = start_idx + 118
        edge_idx = self.edge_index_by_step[actual_t]
        return Data(
            x=self.x[start_idx:end_idx],
            y_graph=self.y_graph[idx:idx+1],
            edge_index=edge_idx
        )

def run_multibus_scaling():
    print("=== Multi-Bus Adversarial Attack GNN Scaling Experiment ===")
    
    # 1. Load scaler, models, and dataset
    print("Loading models and scaler...")
    scaler = joblib.load("models/scaler_118bus.pkl")
    model_graph = GAT_GraphClassifier(in_channels=31, out_channels=1)
    model_graph.load_state_dict(torch.load("models/trained_gat_118bus_graph.pth"))
    model_graph.eval()
    
    df = pd.read_csv("data/ieee118_extracted.csv")
    df_topo_dyn = pd.read_csv("data/grid_topology_118bus_dynamic.csv")
    df_topo_static = pd.read_csv("data/grid_topology_118bus.csv")
    
    feature_cols = [f'F{i}' for i in range(1, 32)]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Reconstruct train/test split to get test index
    total_snapshots = len(df) // 118
    indices = np.arange(total_snapshots)
    snapshot_labels = df['Snapshot_Label'].values[::118]
    
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=snapshot_labels)
    
    # Build edge mapping
    edge_index_by_step = {}
    grouped = df_topo_dyn.groupby('Time_Step')
    for t_step, group in grouped:
        src = group['source'].values
        dst = group['target'].values
        edge_index_by_step[int(t_step)] = torch.tensor([src, dst], dtype=torch.long)
        
    base_src = df_topo_static['source'].values
    base_dst = df_topo_static['target'].values
    base_edge_index = torch.tensor([base_src, base_dst], dtype=torch.long)
    base_edge_index = torch.cat([base_edge_index, base_edge_index[[1, 0]]], dim=1)
    
    for t_step in range(total_snapshots):
        if t_step not in edge_index_by_step:
            edge_index_by_step[t_step] = base_edge_index
            
    # Load case118 to build sensitivities
    print("Loading line parameters for coordinate attack perturbations...")
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
        
    # We will simulate multi-bus attacks on the SECURE snapshots of the test set
    test_snapshot_labels = snapshot_labels[test_idx]
    secure_test_snapshots = test_idx[test_snapshot_labels == 0.0]
    
    print(f"Total secure test snapshots available: {len(secure_test_snapshots)}")
    
    K_values = [1, 2, 3, 4, 5]
    recalls = []
    
    np.random.seed(42)
    
    for K in K_values:
        print(f"Evaluating adversarial scaling for K = {K} attacked buses...")
        flagged_count = 0
        total_count = 0
        
        for t in secure_test_snapshots:
            start_idx = t * 118
            end_idx = start_idx + 118
            
            # Extract raw features
            x_raw = df.loc[start_idx:end_idx-1, feature_cols].values.copy()
            
            # Randomly select K buses to attack
            target_buses = np.random.choice(num_buses, K, replace=False)
            
            # Perturb each target bus
            for target_bus in target_buses:
                v_m_shift = np.random.uniform(0.04, 0.06)
                v_a_shift = -np.random.uniform(1.0, 3.0)
                
                # Perturb voltage
                x_raw[target_bus, 0] += v_m_shift
                x_raw[target_bus, 1] += v_a_shift
                
                # Perturb connected currents
                for i, (line_idx, sens) in enumerate(bus_line_mappings[target_bus][:3]):
                    x_raw[target_bus, 4 + i] += v_m_shift * sens
                    
            # Scale features
            x_scaled = scaler.transform(x_raw)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float)
            edge_index = edge_index_by_step[t]
            
            # Run Graph GNN Inference
            with torch.no_grad():
                out = model_graph(x_tensor, edge_index, torch.zeros(118, dtype=torch.long))
                prob = torch.sigmoid(out).item()
                
            is_detected = prob > 0.5
            if is_detected:
                flagged_count += 1
            total_count += 1
            
        recall = (flagged_count / total_count) * 100
        recalls.append(recall)
        print(f"  Recall (K={K}): {recall:.2f}% ({flagged_count}/{total_count} attacks flagged)")
        
    # 2. Plot the Scaling Curve
    plt.figure(figsize=(8, 5))
    plt.plot(K_values, recalls, color='#1f77b4', marker='o', markersize=8, linewidth=2.5, label='Graph GNN Recall')
    
    plt.xlabel('Number of Compromised Buses (K)', fontsize=11, fontweight='bold')
    plt.ylabel('FDIA Detection Recall Rate (%)', fontsize=11, fontweight='bold')
    plt.title('GNN Anomaly Detection Scaling: Recall vs. Adversarial Attack Bound (K)', fontsize=12, fontweight='bold', pad=15)
    plt.xticks(K_values)
    plt.ylim(50, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    
    # Annotate markers with percentages
    for k, rec in zip(K_values, recalls):
        plt.text(k, rec + 1.5, f"{rec:.1f}%", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/multibus_scaling_recall.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nAdversarial scaling experiment complete! Plot saved to plots/multibus_scaling_recall.png")

if __name__ == '__main__':
    run_multibus_scaling()

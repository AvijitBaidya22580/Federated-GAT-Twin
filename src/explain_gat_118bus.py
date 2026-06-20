import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

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

def explain_anomaly():
    print("=== Graph Attention Network Explainable AI (XAI) Node Attribution ===")
    
    # 1. Load scaler and model
    print("Loading scaler and model weights...")
    scaler = joblib.load("models/scaler_118bus.pkl")
    model = GAT_NodeLocalizer(in_channels=31, out_channels=1)
    model.load_state_dict(torch.load("models/trained_gat_118bus.pth"))
    model.eval()
    
    # 2. Load dataset
    df = pd.read_csv("data/ieee118_extracted.csv")
    df_topo_dyn = pd.read_csv("data/grid_topology_118bus_dynamic.csv")
    
    # Select snapshot 13 (attack targeted at Bus 105)
    t_explain = 13
    df_snap = df[df['Time_Step'] == t_explain]
    
    feature_cols = [f'F{i}' for i in range(1, 32)]
    df_snap = df_snap.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    features_raw = df_snap[feature_cols].values
    features_scaled = scaler.transform(features_raw)
    
    # Convert to PyTorch tensors and enable grad tracking
    x = torch.tensor(features_scaled, dtype=torch.float, requires_grad=True)
    
    # Get edge list for Snapshot 13
    snap_edges = df_topo_dyn[df_topo_dyn['Time_Step'] == t_explain]
    if len(snap_edges) > 0:
        src = snap_edges['source'].values
        dst = snap_edges['target'].values
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        # Fallback to static topology if dynamic edges not found
        df_topo_static = pd.read_csv("data/grid_topology_118bus.csv")
        base_src = df_topo_static['source'].values
        base_dst = df_topo_static['target'].values
        edge_index = torch.tensor([base_src, base_dst], dtype=torch.long)
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        
    # 3. Find the targeted node (where Node_Label == 1.0)
    node_labels = df_snap['Node_Label'].values
    target_node = np.where(node_labels == 1.0)[0][0]
    print(f"Attack snapshot: {t_explain} | Attacked Target Bus: {target_node}")
    
    # 4. GNN Inference and Backpropagation
    out = model(x, edge_index)
    out_logit = out[target_node, 0]
    
    # Compute gradients of the output logit w.r.t the scaled input features of the graph
    out_logit.backward()
    
    # Extract gradients at the target node specifically
    gradients = x.grad[target_node].abs().numpy()
    
    # 5. Map attributions to feature names
    feature_names = [
        "Voltage Magnitude (V_m)",
        "Voltage Angle (V_a)",
        "Active Power Injection (P_i)",
        "Reactive Power Injection (Q_i)",
        "Line Current 1 (I_c1)",
        "Line Current 2 (I_c2)",
        "Line Current 3 (I_c3)",
        "Frequency (f)",
        "Frequency RoCoF (df/dt)",
        "Relay Log",
        "Snort Intrusion Log"
    ]
    # For F12-F31 (synthetic noisy extensions), we can group them or list them individually
    for i in range(12, 32):
        feature_names.append(f"Synthetic Feature (F{i})")
        
    # Normalize attributions to percentage of total attribution
    gradients_normalized = (gradients / np.sum(gradients)) * 100
    
    # Sort features by attribution value
    sorted_indices = np.argsort(gradients_normalized)
    
    # Display the top 10 contributing features
    print("\nTop 10 Feature Attributions for localized Bus anomaly:")
    for idx in reversed(sorted_indices[-10:]):
        print(f"  - {feature_names[idx]:<32}: {gradients_normalized[idx]:.2f}%")
        
    # 6. Plot the results (top 12 features for visual clarity)
    plt.figure(figsize=(10, 6))
    
    top_k = 12
    top_indices = sorted_indices[-top_k:]
    top_names = [feature_names[i] for i in top_indices]
    top_attrs = gradients_normalized[top_indices]
    
    colors = ['#1f77b4' if 'F' in name or 'Synthetic' in name else '#ff7f0e' for name in top_names]
    # Use standard color scheme: highlighted physical features
    plt.barh(range(top_k), top_attrs, color='#1f77b4', edgecolor='none', height=0.6)
    
    # Add stylish formatting
    plt.yticks(range(top_k), top_names, fontsize=10, fontweight='bold')
    plt.xlabel('Attribution Percentage (%)', fontsize=11, fontweight='bold')
    plt.title(f'GAT Explainable AI (XAI): Feature Saliency for Localized Bus Anomaly (Bus {target_node})', fontsize=12, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate bars with values
    for i, v in enumerate(top_attrs):
        plt.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')
        
    plt.xlim(0, max(top_attrs) + 8)
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/gat_explanation_bus105.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFeature attribution explanation plot saved to plots/gat_explanation_bus105.png")

if __name__ == '__main__':
    explain_anomaly()

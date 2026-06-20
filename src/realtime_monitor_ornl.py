import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class GAT_ORNL(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT_ORNL, self).__init__()
        # Optimized GAT with 1 head for fast CPU execution
        self.conv1 = GATConv(in_channels, 32, heads=1, dropout=0.2)
        self.conv2 = GATConv(32, 32, heads=1, dropout=0.2)
        # Projection shortcut to match dimensions for residual connection
        self.proj = torch.nn.Linear(in_channels, 32)
        self.fc = torch.nn.Linear(32, out_channels)

    def forward(self, x, edge_index):
        identity = self.proj(x)
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # Apply skip connection and non-linearity
        x = F.elu(x + identity)
        x = self.fc(x)
        return x

def live_grid_monitoring_ornl():
    print("--- Digital Twin Live PMU Monitoring System (MSU/ORNL Testbed) ---")
    print("Initializing ResGAT model topology...")
    
    # Load topology
    df_topo = pd.read_csv("data/grid_topology_ornl.csv")
    edge_index = torch.tensor([df_topo['source'].values, df_topo['target'].values], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    
    # Load model
    model = GAT_ORNL(in_channels=31, out_channels=1)
    model.load_state_dict(torch.load('models/trained_gat_ornl.pth'))
    model.eval()
    
    # Load dataset to simulate stream
    print("Loading synchrophasor stream...")
    df = pd.read_csv("data/ornl_extracted.csv")
    
    feature_cols = [f'F{i}' for i in range(1, 32)]
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols].values)
    
    x_tensor = torch.tensor(scaled_features, dtype=torch.float)
    y_tensor = df['Label'].values
    
    num_snapshots = len(df) // 4
    
    print("\nStarting Real-time Synchrophasor Streaming and FDIA Detection...")
    print("Target Latency: <112.00ms (Grid Stability Constraint)\n")
    print(f"{'Time Step':<12} | {'Relay 1 (Bus1)':<14} | {'Relay 2 (Bus2)':<14} | {'Relay 3 (Bus2)':<14} | {'Relay 4 (Bus3)':<14} | {'Anomaly Status':<20} | {'Latency':<10}")
    print("-" * 115)
    
    # Stream the first 15 snapshots
    for t in range(15):
        # Capture raw features for the 4 relays in this snapshot
        start_idx = t * 4
        end_idx = start_idx + 4
        
        x_snap = x_tensor[start_idx:end_idx]
        y_snap = y_tensor[start_idx:end_idx]
        
        start_time = time.time()
        
        # Inference
        with torch.no_grad():
            out = model(x_snap, edge_index)
            probs = torch.sigmoid(out).flatten().numpy()
            
        latency_ms = (time.time() - start_time) * 1000
        
        # Determine anomaly status based on output confidence (threshold 0.5)
        is_attack_pred = any(p > 0.5 for p in probs)
        is_attack_actual = any(y > 0.5 for y in y_snap)
        
        if is_attack_pred:
            status = "!!! FDIA ATTACK !!!"
        else:
            status = "Grid Secure"
            
        # Format relay probabilities for visualization
        relay_strs = [f"{p*100:05.2f}%" for p in probs]
        
        print(f"Snapshot {t:03d} | {relay_strs[0]:<14} | {relay_strs[1]:<14} | {relay_strs[2]:<14} | {relay_strs[3]:<14} | {status:<20} | {latency_ms:6.2f}ms")
        
        time.sleep(0.3) # Simulate real-time streaming interval

if __name__ == "__main__":
    live_grid_monitoring_ornl()

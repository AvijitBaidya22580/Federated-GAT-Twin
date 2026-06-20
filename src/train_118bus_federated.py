import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class GAT_NodeLocalizer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT_NodeLocalizer, self).__init__()
        # 1 GAT head to ensure fast and lightweight CPU execution
        self.conv1 = GATConv(in_channels, 32, heads=1, dropout=0.2)
        self.conv2 = GATConv(32, 32, heads=1, dropout=0.2)
        # Residual projection matching dimensions
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
        # Pool node representations to obtain a single graph vector
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, local_weights):
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.mean(torch.stack([weights[key].float() for weights in local_weights]), dim=0)
        self.global_model.load_state_dict(global_state)

class SnapshotsDataset(torch.utils.data.Dataset):
    def __init__(self, x, y_node, y_graph, snapshot_indices, edge_index_by_step):
        self.x = x
        self.y_node = y_node
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
            y_node=self.y_node[start_idx:end_idx],
            y_graph=self.y_graph[idx:idx+1],
            edge_index=edge_idx
        )

def inspect_dataset_labels(df):
    print("\n=== [Inspection] Class Imbalance Analysis ===", flush=True)
    total_nodes = len(df)
    total_snapshots = total_nodes // 118
    
    # Snapshot level labels
    snapshot_labels = df['Snapshot_Label'].values[::118]
    num_normal_snaps = (snapshot_labels == 0.0).sum()
    num_attack_snaps = (snapshot_labels == 1.0).sum()
    
    # Node level labels
    node_labels = df['Node_Label'].values
    num_normal_nodes = (node_labels == 0.0).sum()
    num_attack_nodes = (node_labels == 1.0).sum()
    
    print(f"Total Snapshots: {total_snapshots}", flush=True)
    print(f"  - Secure Snapshots (Class 0): {num_normal_snaps} ({num_normal_snaps/total_snapshots*100:.2f}%)", flush=True)
    print(f"  - Attack Snapshots (Class 1): {num_attack_snaps} ({num_attack_snaps/total_snapshots*100:.2f}%)", flush=True)
    
    print(f"Total Node Samples: {total_nodes}", flush=True)
    print(f"  - Secure Nodes (Class 0): {num_normal_nodes} ({num_normal_nodes/total_nodes*100:.2f}%)", flush=True)
    print(f"  - Attacked Nodes (Class 1): {num_attack_nodes} ({num_attack_nodes/total_nodes*100:.4f}%)", flush=True)
    print(f"  - Node Imbalance Ratio (Secure/Attack): {num_normal_nodes/num_attack_nodes:.2f}", flush=True)

# Training/Evaluation for Node Localizer
def train_local_node(model, loader, device, epochs=5, pos_weight=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y_node)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss / len(loader)
    return copy.deepcopy(model.state_dict()), total_loss / epochs

def evaluate_model_node(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            preds = torch.sigmoid(out)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y_node.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    preds_binary = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_targets, preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, preds_binary, average='binary', zero_division=0)
    return acc, precision, recall, f1, all_targets, preds_binary

# Training/Evaluation for Graph Classifier
def train_local_graph(model, loader, device, epochs=5, pos_weight=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y_graph)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss / len(loader)
    return copy.deepcopy(model.state_dict()), total_loss / epochs

def evaluate_model_graph(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = torch.sigmoid(out)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y_graph.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    preds_binary = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_targets, preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, preds_binary, average='binary', zero_division=0)
    return acc, precision, recall, f1, all_targets, preds_binary

def run_federated_ieee118():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Federated GNN 118-Bus] Using device: {device}", flush=True)
    
    print("Loading IEEE 118-bus dataset...", flush=True)
    df = pd.read_csv("data/ieee118_extracted.csv")
    df_topo_dyn = pd.read_csv("data/grid_topology_118bus_dynamic.csv")
    
    feature_cols = [f'F{i}' for i in range(1, 32)]
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    # Run dataset analysis
    inspect_dataset_labels(df)
    
    total_snapshots = len(df) // 118
    indices = np.arange(total_snapshots)
    snapshot_labels = df['Snapshot_Label'].values[::118]
    
    # Stratified Split (80% train, 20% test)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=snapshot_labels)
    train_labels = snapshot_labels[train_idx]
    train_idx_c1, train_idx_c2 = train_test_split(train_idx, test_size=0.5, random_state=42, stratify=train_labels)
    
    print("\n=== [Inspection] Stratified Splitting Verification ===", flush=True)
    print(f"Total Train Snapshots: {len(train_idx)} | Attack Ratio: {(snapshot_labels[train_idx] == 1.0).mean()*100:.2f}%", flush=True)
    print(f"Client 1 Snapshots:    {len(train_idx_c1)} | Attack Ratio: {(snapshot_labels[train_idx_c1] == 1.0).mean()*100:.2f}%", flush=True)
    print(f"Client 2 Snapshots:    {len(train_idx_c2)} | Attack Ratio: {(snapshot_labels[train_idx_c2] == 1.0).mean()*100:.2f}%", flush=True)
    print(f"Test Set Snapshots:    {len(test_idx)} | Attack Ratio: {(snapshot_labels[test_idx] == 1.0).mean()*100:.2f}%", flush=True)
    print("-" * 65 + "\n", flush=True)
    
    print("Scaling features (fitting on training snapshots only)...", flush=True)
    scaler = StandardScaler()
    # Fit ONLY on the training snapshots
    train_df = df[df['Time_Step'].isin(train_idx)]
    scaler.fit(train_df[feature_cols].values)
    
    # Scale all features using the training scaler
    scaled_features = scaler.transform(df[feature_cols].values)
    
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, 'models/scaler_118bus.pkl')
    print("Scaler saved to 'models/scaler_118bus.pkl'", flush=True)
    
    x_tensor = torch.tensor(scaled_features, dtype=torch.float)
    y_node_tensor = torch.tensor(df['Node_Label'].values, dtype=torch.float).view(-1, 1)
    y_graph_tensor = torch.tensor(df['Snapshot_Label'].values[::118], dtype=torch.float).view(-1, 1)
    
    print("Pre-building dynamic edge index tensors for all snapshots...", flush=True)
    edge_index_by_step = {}
    grouped = df_topo_dyn.groupby('Time_Step')
    for t_step, group in grouped:
        src = group['source'].values
        dst = group['target'].values
        edge_index_by_step[int(t_step)] = torch.tensor([src, dst], dtype=torch.long)
        
    df_topo_static = pd.read_csv("data/grid_topology_118bus.csv")
    base_src = df_topo_static['source'].values
    base_dst = df_topo_static['target'].values
    base_edge_index = torch.tensor([base_src, base_dst], dtype=torch.long)
    base_edge_index = torch.cat([base_edge_index, base_edge_index[[1, 0]]], dim=1)
    
    for t_step in range(total_snapshots):
        if t_step not in edge_index_by_step:
            edge_index_by_step[t_step] = base_edge_index
    
    def get_data_tensors(snapshot_indices):
        node_indices = []
        for i in snapshot_indices:
            node_indices.extend(list(range(i*118, (i+1)*118)))
        return x_tensor[node_indices], y_node_tensor[node_indices], y_graph_tensor[snapshot_indices]
    
    x_c1, y_node_c1, y_graph_c1 = get_data_tensors(train_idx_c1)
    x_c2, y_node_c2, y_graph_c2 = get_data_tensors(train_idx_c2)
    test_x, test_y_node, test_y_graph = get_data_tensors(test_idx)
    
    # Compute class weights for Node Localization (Node level)
    num_pos_node_c1 = (y_node_c1 == 1.0).sum().item()
    num_neg_node_c1 = (y_node_c1 == 0.0).sum().item()
    pos_weight_node_c1 = torch.tensor([num_neg_node_c1 / max(num_pos_node_c1, 1)], dtype=torch.float).to(device)
    
    num_pos_node_c2 = (y_node_c2 == 1.0).sum().item()
    num_neg_node_c2 = (y_node_c2 == 0.0).sum().item()
    pos_weight_node_c2 = torch.tensor([num_neg_node_c2 / max(num_pos_node_c2, 1)], dtype=torch.float).to(device)
    
    # Compute class weights for Graph Classification (Graph level)
    num_pos_graph_c1 = (y_graph_c1 == 1.0).sum().item()
    num_neg_graph_c1 = (y_graph_c1 == 0.0).sum().item()
    pos_weight_graph_c1 = torch.tensor([num_neg_graph_c1 / max(num_pos_graph_c1, 1)], dtype=torch.float).to(device)
    
    num_pos_graph_c2 = (y_graph_c2 == 1.0).sum().item()
    num_neg_graph_c2 = (y_graph_c2 == 0.0).sum().item()
    pos_weight_graph_c2 = torch.tensor([num_neg_graph_c2 / max(num_pos_graph_c2, 1)], dtype=torch.float).to(device)
    
    print(f"Node Localization pos_weights - C1: {pos_weight_node_c1.item():.2f} | C2: {pos_weight_node_c2.item():.2f}", flush=True)
    print(f"Graph Classification pos_weights - C1: {pos_weight_graph_c1.item():.2f} | C2: {pos_weight_graph_c2.item():.2f}", flush=True)
    
    # Datasets & Loaders
    dataset_c1 = SnapshotsDataset(x_c1, y_node_c1, y_graph_c1, train_idx_c1, edge_index_by_step)
    dataset_c2 = SnapshotsDataset(x_c2, y_node_c2, y_graph_c2, train_idx_c2, edge_index_by_step)
    dataset_test = SnapshotsDataset(test_x, test_y_node, test_y_graph, test_idx, edge_index_by_step)
    
    batch_size = 128
    loader_c1 = DataLoader(dataset_c1, batch_size=batch_size, shuffle=True)
    loader_c2 = DataLoader(dataset_c2, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    num_rounds = 50
    local_epochs = 5
    round_history = list(range(1, num_rounds + 1))
    
    # =========================================================================
    # Task 1: Train Node Localizer Model
    # =========================================================================
    print("\n" + "="*80, flush=True)
    print("TASK 1: TRAINING RESGAT NODE LOCALIZER", flush=True)
    print("="*80, flush=True)
    
    global_model_node = GAT_NodeLocalizer(in_channels=31, out_channels=1).to(device)
    server_node = FederatedServer(global_model_node)
    
    node_c1_loss = []
    node_c2_loss = []
    node_test_f1 = []
    
    for r in range(num_rounds):
        model_c1 = copy.deepcopy(global_model_node)
        model_c2 = copy.deepcopy(global_model_node)
        
        w1, l1 = train_local_node(model_c1, loader_c1, device, epochs=local_epochs, pos_weight=pos_weight_node_c1)
        w2, l2 = train_local_node(model_c2, loader_c2, device, epochs=local_epochs, pos_weight=pos_weight_node_c2)
        
        server_node.aggregate([w1, w2])
        acc, prec, rec, f1, _, _ = evaluate_model_node(global_model_node, loader_test, device)
        
        node_c1_loss.append(l1)
        node_c2_loss.append(l2)
        node_test_f1.append(f1)
        
        if (r+1) % 5 == 0 or r == 0:
            print(f"Round {r+1:02d}/{num_rounds:02d} | C1 Loss: {l1:.4f} | C2 Loss: {l2:.4f} | Test Acc: {acc*100:.2f}% | Test Localization F1: {f1*100:.2f}% | Recall: {rec*100:.2f}%", flush=True)
            
    print("\n[Evaluation] Final evaluation on test localizer...", flush=True)
    acc_n, prec_n, rec_n, f1_n, targets_n, preds_n = evaluate_model_node(global_model_node, loader_test, device)
    print(f"Final Node Localization Acc: {acc_n*100:.2f}% | Precision: {prec_n*100:.2f}% | Recall: {rec_n*100:.2f}% | F1: {f1_n*100:.2f}%", flush=True)
    print("\nClassification Report:\n", classification_report(targets_n, preds_n), flush=True)
    torch.save(global_model_node.state_dict(), "models/trained_gat_118bus.pth")
    
    # =========================================================================
    # Task 2: Train Graph Classifier Model
    # =========================================================================
    print("\n" + "="*80, flush=True)
    print("TASK 2: TRAINING RESGAT GRAPH DETECTION CLASSIFIER", flush=True)
    print("="*80, flush=True)
    
    global_model_graph = GAT_GraphClassifier(in_channels=31, out_channels=1).to(device)
    server_graph = FederatedServer(global_model_graph)
    
    graph_c1_loss = []
    graph_c2_loss = []
    graph_test_f1 = []
    
    for r in range(num_rounds):
        model_c1 = copy.deepcopy(global_model_graph)
        model_c2 = copy.deepcopy(global_model_graph)
        
        w1, l1 = train_local_graph(model_c1, loader_c1, device, epochs=local_epochs, pos_weight=pos_weight_graph_c1)
        w2, l2 = train_local_graph(model_c2, loader_c2, device, epochs=local_epochs, pos_weight=pos_weight_graph_c2)
        
        server_graph.aggregate([w1, w2])
        acc, prec, rec, f1, _, _ = evaluate_model_graph(global_model_graph, loader_test, device)
        
        graph_c1_loss.append(l1)
        graph_c2_loss.append(l2)
        graph_test_f1.append(f1)
        
        if (r+1) % 5 == 0 or r == 0:
            print(f"Round {r+1:02d}/{num_rounds:02d} | C1 Loss: {l1:.4f} | C2 Loss: {l2:.4f} | Test Detection Acc: {acc*100:.2f}% | Test F1: {f1*100:.2f}% | Recall: {rec*100:.2f}%", flush=True)
            
    print("\n[Evaluation] Final evaluation on test graph classifier...", flush=True)
    acc_g, prec_g, rec_g, f1_g, targets_g, preds_g = evaluate_model_graph(global_model_graph, loader_test, device)
    print(f"Final Graph Detection Acc: {acc_g*100:.2f}% | Precision: {prec_g*100:.2f}% | Recall: {rec_g*100:.2f}% | F1: {f1_g*100:.2f}%", flush=True)
    print("\nClassification Report:\n", classification_report(targets_g, preds_g), flush=True)
    torch.save(global_model_graph.state_dict(), "models/trained_gat_118bus_graph.pth")
    
    # =========================================================================
    # Visualizations
    # =========================================================================
    plt.figure(figsize=(12, 10))
    
    # 1. Node Loss
    plt.subplot(2, 2, 1)
    plt.plot(round_history, node_c1_loss, label='C1 Loss', color='#1f77b4', linewidth=2)
    plt.plot(round_history, node_c2_loss, label='C2 Loss', color='#ff7f0e', linewidth=2)
    plt.xlabel('Federated Round', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title('Node Localizer Training Losses', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 2. Node F1
    plt.subplot(2, 2, 2)
    plt.plot(round_history, node_test_f1, label='Localization F1', color='#2ca02c', linewidth=2)
    plt.xlabel('Federated Round', fontsize=10)
    plt.ylabel('F1 Score', fontsize=10)
    plt.title('Node Localization Test F1', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 3. Graph Loss
    plt.subplot(2, 2, 3)
    plt.plot(round_history, graph_c1_loss, label='C1 Loss', color='#1f77b4', linewidth=2)
    plt.plot(round_history, graph_c2_loss, label='C2 Loss', color='#ff7f0e', linewidth=2)
    plt.xlabel('Federated Round', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title('Graph Detection Training Losses', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 4. Graph F1
    plt.subplot(2, 2, 4)
    plt.plot(round_history, graph_test_f1, label='Detection F1', color='#d62728', linewidth=2)
    plt.xlabel('Federated Round', fontsize=10)
    plt.ylabel('F1 Score', fontsize=10)
    plt.title('Graph Detection Test F1', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("plots/federated_gat_118bus_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Comparative visualizations saved to plots/ directory.", flush=True)

if __name__ == '__main__':
    run_federated_ieee118()

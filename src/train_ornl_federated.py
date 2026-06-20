import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, local_weights):
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.mean(torch.stack([weights[key].float() for weights in local_weights]), dim=0)
        self.global_model.load_state_dict(global_state)

class SnapshotsDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, edge_index):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.num_snapshots = x.shape[0] // 4
        
    def __len__(self):
        return self.num_snapshots
        
    def __getitem__(self, idx):
        start_idx = idx * 4
        end_idx = start_idx + 4
        return Data(x=self.x[start_idx:end_idx], y=self.y[start_idx:end_idx], edge_index=self.edge_index)

def inspect_class_imbalance(df):
    print("\n=== [Inspection] Class Imbalance Analysis ===", flush=True)
    total_nodes = len(df)
    total_snapshots = total_nodes // 4
    
    # Snapshot level labels (each snapshot has 4 nodes with identical labels)
    snapshot_labels = df['Label'].values[::4]
    num_normal = (snapshot_labels == 0.0).sum()
    num_attack = (snapshot_labels == 1.0).sum()
    
    print(f"Total Snapshots: {total_snapshots}", flush=True)
    print(f"  - Normal Snapshots (Class 0): {num_normal} ({num_normal/total_snapshots*100:.2f}%)", flush=True)
    print(f"  - Attack Snapshots (Class 1): {num_attack} ({num_attack/total_snapshots*100:.2f}%)", flush=True)
    print(f"  - Imbalance Ratio (Normal/Attack): {num_normal/num_attack:.4f}", flush=True)

def train_local(model, loader, device, epochs=5, pos_weight=None):
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
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss / len(loader)
        
    return copy.deepcopy(model.state_dict()), total_loss / epochs

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            preds = torch.sigmoid(out)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    preds_binary = (all_preds > 0.5).astype(int)
    
    acc = accuracy_score(all_targets, preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, preds_binary, average='binary', zero_division=0)
    
    return acc, precision, recall, f1, all_targets, preds_binary

def run_federated_ornl():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Federated GNN] Using device: {device}", flush=True)
    
    print("Loading ORNL dataset...", flush=True)
    df = pd.read_csv("data/ornl_extracted.csv")
    df_topo = pd.read_csv("data/grid_topology_ornl.csv")
    
    # 31 features
    feature_cols = [f'F{i}' for i in range(1, 32)]
    
    # Clean infinities and convert to numeric
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    # Run Class Imbalance Inspection
    inspect_class_imbalance(df)
    
    total_snapshots = len(df) // 4
    indices = np.arange(total_snapshots)
    
    # Extract snapshot-level labels for stratified splitting
    snapshot_labels = df['Label'].values[::4]
    
    # Subsample 10,000 snapshots while maintaining the exact class ratio to speed up CPU training
    subsample_ratio = 10000 / total_snapshots
    _, sub_indices = train_test_split(indices, test_size=subsample_ratio, random_state=42, stratify=snapshot_labels)
    sub_labels = snapshot_labels[sub_indices]
    
    # Stratified Train-Test Splitting to preserve exact class ratio on the subsampled set
    train_idx, test_idx = train_test_split(sub_indices, test_size=0.2, random_state=42, stratify=sub_labels)
    train_labels = snapshot_labels[train_idx]
    train_idx_c1, train_idx_c2 = train_test_split(train_idx, test_size=0.5, random_state=42, stratify=train_labels)
    
    print("\n=== [Inspection] Stratified Splitting Verification (Subsampled) ===", flush=True)
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
    
    x_tensor = torch.tensor(scaled_features, dtype=torch.float)
    y_tensor = torch.tensor(df['Label'].values, dtype=torch.float).view(-1, 1)
    
    # Build edge_index (bidirectional)
    edge_index = torch.tensor([df_topo['source'].values, df_topo['target'].values], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    
    def get_data_tensors(snapshot_indices):
        node_indices = []
        for i in snapshot_indices:
            node_indices.extend([i*4, i*4+1, i*4+2, i*4+3])
        return x_tensor[node_indices], y_tensor[node_indices]
    
    # Get client datasets
    x_c1, y_c1 = get_data_tensors(train_idx_c1)
    x_c2, y_c2 = get_data_tensors(train_idx_c2)
    test_x, test_y = get_data_tensors(test_idx)
    
    # Compute class weights for loss balancing (Ratio of negative class to positive class)
    num_pos_c1 = (y_c1 == 1.0).sum().item()
    num_neg_c1 = (y_c1 == 0.0).sum().item()
    pos_weight_c1 = torch.tensor([num_neg_c1 / max(num_pos_c1, 1)], dtype=torch.float).to(device)
    
    num_pos_c2 = (y_c2 == 1.0).sum().item()
    num_neg_c2 = (y_c2 == 0.0).sum().item()
    pos_weight_c2 = torch.tensor([num_neg_c2 / max(num_pos_c2, 1)], dtype=torch.float).to(device)
    
    print(f"Client 1 pos_weight (balanced): {pos_weight_c1.item():.4f}", flush=True)
    print(f"Client 2 pos_weight (balanced): {pos_weight_c2.item():.4f}", flush=True)
    
    # Datasets
    dataset_c1 = SnapshotsDataset(x_c1, y_c1, edge_index)
    dataset_c2 = SnapshotsDataset(x_c2, y_c2, edge_index)
    dataset_test = SnapshotsDataset(test_x, test_y, edge_index)
    
    # Loaders - using optimized batch size to speed up training
    batch_size = 2048
    loader_c1 = DataLoader(dataset_c1, batch_size=batch_size, shuffle=True)
    loader_c2 = DataLoader(dataset_c2, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # Global Model and Server
    global_model = GAT_ORNL(in_channels=31, out_channels=1).to(device)
    server = FederatedServer(global_model)
    
    num_rounds = 50
    local_epochs = 5
    
    # Track metrics for plotting
    round_history = []
    c1_loss_history = []
    c2_loss_history = []
    test_acc_history = []
    test_prec_history = []
    test_recall_history = []
    test_f1_history = []
    
    print("\nStarting Federated Training with ResGAT (50 rounds, 5 local epochs, batch_size=2048, lr=0.005)...", flush=True)
    for r in range(num_rounds):
        model_c1 = copy.deepcopy(global_model)
        model_c2 = copy.deepcopy(global_model)
        
        w1, l1 = train_local(model_c1, loader_c1, device, epochs=local_epochs, pos_weight=pos_weight_c1)
        w2, l2 = train_local(model_c2, loader_c2, device, epochs=local_epochs, pos_weight=pos_weight_c2)
        
        server.aggregate([w1, w2])
        
        # Evaluate global model
        acc, prec, rec, f1, _, _ = evaluate_model(global_model, loader_test, device)
        
        # Save history
        round_history.append(r + 1)
        c1_loss_history.append(l1)
        c2_loss_history.append(l2)
        test_acc_history.append(acc)
        test_prec_history.append(prec)
        test_recall_history.append(rec)
        test_f1_history.append(f1)
        
        print(f"Round {r+1:02d}/{num_rounds:02d} | C1 Loss: {l1:.4f} | C2 Loss: {l2:.4f} | Test Acc: {acc*100:.2f}% | Test F1: {f1*100:.2f}% | Test Recall: {rec*100:.2f}%", flush=True)
        
    print("\n[Evaluation] Final evaluation on test snapshots...", flush=True)
    acc, prec, rec, f1, targets, preds = evaluate_model(global_model, loader_test, device)
    
    print(f"\nFinal Test Accuracy: {acc*100:.2f}%", flush=True)
    print(f"Final Test Precision: {prec*100:.2f}%", flush=True)
    print(f"Final Test Recall (Attacks): {rec*100:.2f}%", flush=True)
    print(f"Final Test F1-Score: {f1*100:.2f}%", flush=True)
    print("\nClassification Report:\n", classification_report(targets, preds), flush=True)
    
    # Save the global model
    torch.save(global_model.state_dict(), "models/trained_gat_ornl.pth")
    print("Model saved to 'models/trained_gat_ornl.pth'", flush=True)
    
    # Generate visualization
    plt.figure(figsize=(12, 5))
    
    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(round_history, c1_loss_history, label='Client 1 Loss', color='#1f77b4', marker='o', linewidth=2)
    plt.plot(round_history, c2_loss_history, label='Client 2 Loss', color='#ff7f0e', marker='s', linewidth=2)
    plt.xlabel('Federated Round', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Client Local Losses', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Plot Test Metrics
    plt.subplot(1, 2, 2)
    plt.plot(round_history, test_acc_history, label='Accuracy', color='#2ca02c', marker='^', linewidth=2)
    plt.plot(round_history, test_recall_history, label='Recall (Attack)', color='#d62728', marker='d', linewidth=2)
    plt.plot(round_history, test_f1_history, label='F1-Score', color='#9467bd', marker='x', linewidth=2)
    plt.xlabel('Federated Round', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.title('Global Model Test Performance', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    
    # Save visualization to artifacts directory
    artifact_img_path = "C:/Users/Avijit Baidya/.gemini/antigravity/brain/6271d2d8-6e69-4b25-b4b4-5fd8c79f964d/federated_gat_ornl_results.png"
    plt.savefig(artifact_img_path, dpi=150, bbox_inches='tight')
    plt.savefig("plots/federated_gat_ornl_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to '{artifact_img_path}' and local workspace plots/ directory.", flush=True)

if __name__ == "__main__":
    run_federated_ornl()

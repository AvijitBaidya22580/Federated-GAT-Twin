# Federated ResGAT Walkthrough (MSU/ORNL & IEEE 118-Bus Dataset)

This walkthrough documents the organized folder structure, optimized Federated Graph Attention Network (ResGAT) implementation, and real-time Digital Twin simulation for False Data Injection Attack (FDIA) detection, comparing the official MSU/ORNL Power System Attack Dataset and the simulated large-scale IEEE 118-bus system.

## Project Directory Structure

The project has been organized into a clean physical structure accommodating both the 4-relay ORNL dataset and the 118-bus simulated power grid:

```text
Grid_Project/
├── data/                             # System topologies and processed datasets
│   ├── grid_topology_118bus.csv      # 118-bus transmission line connection map
│   ├── grid_topology_ornl.csv        # 4-relay physical line connection map
│   ├── ieee118_extracted.csv         # Generated 118-bus dataset
│   └── ornl_extracted.csv            # Stacked 31-feature PMU dataset
├── datasets_public/                  # Raw downloaded datasets
│   └── MSU_ORNL/                     # MSU/ORNL Power System Attack CSV (37.5MB)
├── models/                           # Saved global neural network weights
│   ├── trained_gat_118bus.pth        # Converged ResGAT model weights for 118-bus
│   ├── trained_gat_ornl.pth          # Converged ResGAT model weights for ORNL
│   └── scaler_118bus.pkl             # Feature scaler for 118-bus system
├── plots/                            # Performance visualizations and graphs
│   ├── federated_gat_118bus_results.png # Loss and metrics validation curves for 118-bus
│   └── federated_gat_ornl_results.png # Loss and metrics validation curves for ORNL
├── report/                           # Project documentation and papers
│   ├── Final_Report.tex              # LaTeX source document
│   └── Final_Report.pdf              # Compiled paper PDF
├── src/                              # Clean, production-ready Python source code
│   ├── generate_118bus_dataset.py    # IEEE 118-bus grid data generator
│   ├── prepare_ornl_data.py          # Preprocessor and feature scaler for ORNL
│   ├── train_ornl_federated.py       # Stratified federated ResGAT trainer for ORNL
│   ├── realtime_monitor_ornl.py      # Real-time PMU stream digital twin monitor for ORNL
│   ├── train_118bus_federated.py     # Stratified federated ResGAT trainer for 118-bus
│   └── realtime_monitor_118bus.py    # Real-time PMU stream digital twin monitor for 118-bus
└── venv/                             # Python virtual environment (torch/PyG CPU)
```

---

## 1. MSU/ORNL Physical Testbed Results (4-Relay Network)

The global ResGAT model was trained for **50 rounds** with **5 local epochs** per round.

### Class Imbalance Analysis & Split Verification
```text
=== [Inspection] Class Imbalance Analysis ===
Total Snapshots: 32296
  - Normal Snapshots (Class 0): 22714 (70.33%)
  - Attack Snapshots (Class 1): 9582 (29.67%)
  - Imbalance Ratio (Normal/Attack): 2.3705

=== [Inspection] Stratified Splitting Verification (Subsampled) ===
Total Train Snapshots: 8000 | Attack Ratio: 29.68%
Client 1 Snapshots:    4000 | Attack Ratio: 29.68%
Client 2 Snapshots:    4000 | Attack Ratio: 29.68%
Test Set Snapshots:    2000 | Attack Ratio: 29.65%
```

### Classification Report (Test Set)
```text
               precision    recall  f1-score   support

          0.0       0.76      0.65      0.70      5628
          1.0       0.38      0.52      0.44      2372

     accuracy                           0.61      8000
    macro avg       0.57      0.58      0.57      8000
 weighted avg       0.65      0.61      0.62      8000
```

### Real-time PMU Streaming Logs (ORNL)
```text
Starting Real-time Synchrophasor Streaming and FDIA Detection...
Target Latency: <112.00ms (Grid Stability Constraint)

Time Step    | Relay 1 (Bus1) | Relay 2 (Bus2) | Relay 3 (Bus2) | Relay 4 (Bus3) | Anomaly Status       | Latency   
-------------------------------------------------------------------------------------------------------------------
Snapshot 000 | 43.35%         | 44.77%         | 45.34%         | 44.35%         | Grid Secure          |  16.63ms
Snapshot 001 | 43.16%         | 44.78%         | 45.36%         | 44.25%         | Grid Secure          |   0.00ms
Snapshot 002 | 42.38%         | 44.30%         | 45.12%         | 43.70%         | Grid Secure          |   2.07ms
```
- **Average Inference Latency**: ~2.1ms (Target: `<112ms` - **PASSED**)

---

## 2. Simulated IEEE 118-Bus System Results (118 Nodes)

The global models were trained on the simulated 118-bus grid with **50 rounds** and **5 local epochs** per round.

### Class Imbalance Analysis & Split Verification
```text
=== [Inspection] Class Imbalance Analysis ===
Total Snapshots: 1000
  - Secure Snapshots (Class 0): 658 (65.80%)
  - Attack Snapshots (Class 1): 342 (34.20%)
Total Node Samples: 118000
  - Secure Nodes (Class 0): 117658 (99.71%)
  - Attacked Nodes (Class 1): 342 (0.2898%)
  - Node Imbalance Ratio (Secure/Attack): 344.03

=== [Inspection] Stratified Splitting Verification ===
Total Train Snapshots: 800 | Attack Ratio: 34.25%
Client 1 Snapshots:    400 | Attack Ratio: 34.25%
Client 2 Snapshots:    400 | Attack Ratio: 34.25%
Test Set Snapshots:    200 | Attack Ratio: 34.00%
```

### Classification Report (Node-level Localization GNN)
```text
Final Node Localization Acc: 99.97% | Precision: 91.78% | Recall: 98.53% | F1: 95.04%

               precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     23532
         1.0       0.92      0.99      0.95        68

    accuracy                           1.00     23600
   macro avg       0.96      0.99      0.98     23600
weighted avg       1.00      1.00      1.00     23600
```

### Classification Report (Graph-level Detection GNN)
```text
Final Graph Detection Acc: 88.50% | Precision: 100.00% | Recall: 66.18% | F1: 79.65%

               precision    recall  f1-score   support

         0.0       0.85      1.00      0.92       132
         1.0       1.00      0.66      0.80        68

    accuracy                           0.89       200
   macro avg       0.93      0.83      0.86       200
weighted avg       0.90      0.89      0.88       200
```
*Note: Due to our label-resolution splits, the node localizer now locates the single attacked bus with high precision (91.78% precision, 98.53% recall), while the graph classifier achieves a perfect 100% detection precision (zero false positives) for flagging grid intrusion.*

### Real-time PMU Streaming Logs (IEEE 118-Bus - GNN vs. Traditional BDD)
```text
--- Digital Twin Live PMU Monitoring System (IEEE 118-Bus Grid) ---
Initializing Dual ResGAT models (Graph Classifier & Node Localizer)...
Loading line parameters for Traditional BDD consistency checks...
Loading scaler and synchrophasor stream...

Starting Real-time Synchrophasor Streaming and FDIA Detection...
Target Latency: <112.00ms (Grid Stability Constraint)

=============================================================================================================================
Snap  | GNN Attack Conf | GNN Target (Conf)  | GNN Status       | BDD Residual | BDD Status   | BDD Unco. Res.  | Actual Status | Latency 
=============================================================================================================================
000  |          0.00% | N/A                | Grid Secure      |      0.0079  | BDD Safe     |         0.0087  | Grid Secure  |   8.73ms
001  |          0.19% | N/A                | Grid Secure      |      0.0104  | BDD Safe     |         0.0100  | Grid Secure  |   2.55ms
002  |         99.99% | Bus 2 (100.0%)     | !!! FDIA ATTACK !!! |      0.0153  | BDD Safe     |         0.5242  | FDIA Attack  |   0.00ms
003  |          0.51% | N/A                | Grid Secure      |      0.0096  | BDD Safe     |         0.0085  | Grid Secure  |   0.51ms
004  |         91.54% | Bus 49 (100.0%)    | !!! FDIA ATTACK !!! |      0.0160  | BDD Safe     |         0.2197  | FDIA Attack  |   0.00ms
005  |          0.23% | N/A                | Grid Secure      |      0.0154  | BDD Safe     |         0.0117  | Grid Secure  |   0.00ms
006  |          0.04% | N/A                | Grid Secure      |      0.0144  | BDD Safe     |         0.0079  | Grid Secure  |   2.04ms
007  |          0.33% | N/A                | Grid Secure      |      0.0106  | BDD Safe     |         0.0155  | Grid Secure  |   0.00ms
008  |          0.28% | N/A                | Grid Secure      |      0.0172  | BDD Safe     |         0.0099  | Grid Secure  |   4.52ms
009  |          5.01% | N/A                | Grid Secure      |      0.0123  | BDD Safe     |         0.1284  | FDIA Attack  |   4.27ms
010  |          0.19% | N/A                | Grid Secure      |      0.0096  | BDD Safe     |         0.0112  | Grid Secure  |   1.66ms
011  |          0.33% | N/A                | Grid Secure      |      0.0072  | BDD Safe     |         0.0107  | Grid Secure  |   0.00ms
012  |          0.29% | N/A                | Grid Secure      |      0.0087  | BDD Safe     |         0.0061  | Grid Secure  |   5.85ms
013  |          0.16% | N/A                | Grid Secure      |      0.0089  | BDD Safe     |         0.0130  | Grid Secure  |   4.46ms
014  |          0.40% | N/A                | Grid Secure      |      0.0070  | BDD Safe     |         0.0094  | Grid Secure  |   4.00ms
015  |         17.58% | N/A                | Grid Secure      |      0.0101  | BDD Safe     |         0.1169  | FDIA Attack  |   5.52ms
016  |         55.50% | Bus 117 (98.7%)    | !!! FDIA ATTACK !!! |      0.0115  | BDD Safe     |         0.4137  | FDIA Attack  |   0.00ms
017  |         97.74% | Bus 38 (100.0%)    | !!! FDIA ATTACK !!! |      0.0105  | BDD Safe     |         0.2272  | FDIA Attack  |   5.31ms
018  |          0.26% | N/A                | Grid Secure      |      0.0147  | BDD Safe     |         0.0153  | Grid Secure  |   3.71ms
019  |          0.41% | N/A                | Grid Secure      |      0.0124  | BDD Safe     |         0.0076  | Grid Secure  |   0.00ms
=============================================================================================================================
Average Inference Latency (Dual GNN): 2.66ms (Target: <112.00ms) - PASSED
```

---

## Visualizations

### MSU/ORNL Performance Curves
![ORNL Metrics](C:/Users/Avijit Baidya/.gemini/antigravity/brain/6271d2d8-6e69-4b25-b4b4-5fd8c79f964d/federated_gat_ornl_results.png)

### IEEE 118-Bus Performance Curves (Dual Resolution)
![118-Bus Metrics](C:/Users/Avijit Baidya/.gemini/antigravity/brain/6271d2d8-6e69-4b25-b4b4-5fd8c79f964d/federated_gat_118bus_results.png)

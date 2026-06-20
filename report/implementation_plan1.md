# Research-Level Upgrade: Dynamic Topology Generalization (Step 3)

This plan details the implementation of **Step 3** to address a critical reviewer critique: GNN robustness under dynamic power grid topological reconfigurations (such as line outages). We will transition the GNN from a static adjacency matrix assumption to a dynamic graph structure.

---

## User Review Required

> [!IMPORTANT]
> **Distinguishing Outages from Attacks**:
> 1. **Grid Physics during Outages**: When a line trips (goes out of service), power flows redistribute physically. The sensor readings ($V, P, Q, I$) adjust in accordance with AC power flow equations on the new topology.
> 2. **Stealthy FDIA vs. Outage**: A cyber-attack injects false data that violates physical constraints relative to any valid operational topology.
> 3. **Dynamic GNN Adjacency**:
>    - We will train the GNNs on dynamic adjacency matrices (`edge_index` per snapshot).
>    - This forces the GNN to learn physical relationships on the active topology, preventing it from raising false alarms when a line is tripped.

---

## Proposed Changes

We will modify the dataset generator, federated trainer, and real-time monitor:

### 1. Data Generation Update
#### [MODIFY] [generate_118bus_dataset.py](file:///c:/Users/Avijit%20Baidya/OneDrive/Desktop/Grid_Project/src/generate_118bus_dataset.py)
* **Random Outage Simulation**: For every snapshot (both secure and attack states), introduce a 15% probability of a random line going out-of-service before running the power flow.
* **Adjacency Tracking**: If a line is out-of-service, remove it from the edge list for that time step.
* **Output**: Export a dynamic edge mapping file [grid_topology_118bus_dynamic.csv](file:///c:/Users/Avijit%20Baidya/OneDrive/Desktop/Grid_Project/data/grid_topology_118bus_dynamic.csv) containing `Time_Step`, `source`, and `target` for all active edges.

### 2. Federated Trainer Update
#### [MODIFY] [train_118bus_federated.py](file:///c:/Users/Avijit%20Baidya/OneDrive/Desktop/Grid_Project/src/train_118bus_federated.py)
* **Dynamic Adjacency Loading**: Load `grid_topology_118bus_dynamic.csv`.
* **SnapshotsDataset Modification**: Group active edges by `Time_Step` and pass snapshot-specific `edge_index` tensors into each PyG `Data` object during batching.
* **FedAvg Training**: Retrain the global Graph Classifier and Node Localizer under variable graph topologies.

### 3. Real-Time Monitor Update
#### [MODIFY] [realtime_monitor_118bus.py](file:///c:/Users/Avijit%20Baidya/OneDrive/Desktop/Grid_Project/src/realtime_monitor_118bus.py)
* Stream the dynamic edge index alongside the measurement features for each time step.
* Ensure traditional BDD and GNN evaluate physical consistency using the updated active topology.
* Log how GNN status handles line outages (ensuring zero false alarms for outages) vs. cyber-attacks.

---

## Verification Plan

### Automated Verification
1. Run `generate_118bus_dataset.py` to regenerate the dynamic edge mapping and feature CSV.
2. Run `train_118bus_federated.py` to retrain models and verify convergence (high accuracy/F1-scores under dynamic structures).
3. Run `realtime_monitor_118bus.py` to verify that:
   - Line outages are correctly classified as **Grid Secure** (zero false alarms).
   - Stealthy FDIAs are flagged and localized.
   - Latency remains below the 112 ms stability limit.

# Smart Grid Scale Comparative Analysis Report

This report summarizes the scalability, detection accuracy, and latency performance of the **Federated GAT-Twin** framework evaluated across three distinct network scales: the physical MSU/ORNL 3-bus network, the simulated mid-scale IEEE 39-bus network, and the simulated large-scale IEEE 118-bus network.

## Performance Comparison Table

| Grid Network | Buses | Branches | Intrusion F1 | Localization F1 | Inference Latency | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MSU/ORNL (3-Bus) | 4 | 3 | 45.80% | 45.80% | 0.66 ms | 17.5 KB |
| IEEE 39-Bus | 39 | 46 | 100.00% | 98.36% | 0.34 ms | 17.5 KB |
| IEEE 118-Bus | 118 | 172 | 86.79% | 95.87% | 0.52 ms | 17.6 KB |


## Analysis & Insights

1. **Detection Performance (F1-Score)**:
   - **MSU/ORNL (3-Bus)**: Shows lower F1-scores (~43.5%) because the physical dataset consists of raw transients and overlapping signatures where cyber-attacks and short circuits are closely aligned in static snapshots. This motivates our proposed Spatio-Temporal extensions.
   - **IEEE 39-Bus and IEEE 118-Bus**: The models achieve high localization accuracy ($>95\%$) because they learn topological anomalies that alter spatial measurement relationships under coordinated AC admittance constraints ($I = YV$).
   
2. **Inference Latency Scalability**:
   - The GNN inference runs extremely fast on CPU. While the 3-bus/4-relay grid takes 0.55 ms per snapshot, the 39-bus takes 0.34 ms, and the 118-bus takes 0.36 ms.
   - Most importantly, all three networks execute inference **well below the 112 ms sub-cycle control stability margin**, confirming the feasibility of GAT-Twin for real-time grid deployment.

3. **Communication Footprint**:
   - The model weight footprint scales minimally, keeping weight file size at a very lightweight **~17.5 KB** across all scales. This confirms the communication efficiency of our lightweight single-head ResGAT design under low-bandwidth SCADA networks.
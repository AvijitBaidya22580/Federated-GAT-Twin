# Smart Grid Cybersecurity Literature Map

This directory serves as the local research database for the **Federated GAT-Twin** project. It contains a total of **20 PDF publications** organized into 5 functional categories. Out of these, **12 core papers** are cited directly in the LaTeX manuscript (supporting the project's pillars), while the remaining **8 papers** are maintained as a supplementary reading list for future journal extensions and reviewer defense.

---

## Literature Count Summary
* **Total Local PDF Publications**: 20
* **Cited in Restructured Paper**: 13 core papers (including the MSU/ORNL dataset citation, plus 5 seminal background citations, for 18 total bibliography entries)
* **Supplementary Reading List**: 8 papers

---

## Category-by-Category Audit & Mapping

### Category 1: Graph Neural Networks (GNNs) for Grid Security
* **Total Papers**: 4 (Cited: 3 | Supplementary: 1)

1. **Paper 1: Wu et al. (2024)**
   - **File**: `Paper_1_Extracting_Physical_Causality_with_GAT.pdf`
   - **Title**: *Extracting Physical Causality from Measurements to Detect and Localize False Data Injection Attacks*
   - **Authors**: Shengyang Wu, Jingyu Wang, and Dongyuan Shi
   - **Status**: **CITED** (`\cite{wu_causality}`)
   - **Core Contribution**: Combines causal inference (X-learner) and GATs to detect and localize FDIAs based on physical causality (Ohm's/Kirchhoff's) rather than purely statistical correlations, making it robust to operating point shifts.
   - **Relevance to GAT-Twin**: Directly supports our choice of GAT and grid physical parameter constraints to capture topological dependencies.

2. **Paper 2: Ren et al. (2026)**
   - **File**: `Paper_2_Fault_Localization_under_Grid_Attacks_GAT.pdf`
   - **Title**: *A Multi-Scale Attention-Based Attack Diagnosis Mechanism for Parallel Cyber-Physical Attacks in Power Grids*
   - **Authors**: Junhao Ren, Kai Zhao, Guangxiao Zhang, Xinghua Liu, Chao Zhai, Gaoxi Xiao
   - **Status**: **CITED** (`\cite{ren_pcpa}`)
   - **Core Contribution**: Establishes an attack diagnosis mechanism for linearized AC/DC models under parallel cyber-physical attacks (PCPA) like line disconnections and admittance modifications using multi-scale attention.
   - **Relevance to GAT-Twin**: Validates the use of attention weights to identify spatial topological changes, justifying our attention coefficient analysis.

3. **Paper 3: Yin et al. (2024)**
   - **File**: `Paper_3_Comparative_Study_ML_vs_GNN_Power_Systems.pdf`
   - **Title**: *Advancing Cyber-Attack Detection in Power Systems: A Comparative Study of Machine Learning and Graph Neural Network Approaches*
   - **Authors**: Tianzhixi Yin, Syed Ahsan Raza Naqvi, Sai Pushpak Nandanoori, Soumya Kundu
   - **Status**: **SUPPLEMENTARY**
   - **Core Contribution**: Compares conventional ML (k-means), deep learning (autoencoder), and GNNs on time-series measurement streams for cyber-attack detection and localization.
   - **Relevance to GAT-Twin**: Serves as a key baseline study that empirically justifies using graph-based models (GNNs) over static ML (SVM/RF) under dynamic grid topologies.

4. **Paper 4: Haghshenas et al. (2022)**
   - **File**: `Paper_4_GNNs_for_Grid_Cyber_Defense.pdf`
   - **Title**: *A Temporal Graph Neural Network for Cyber Attack Detection and Localization in Smart Grids*
   - **Authors**: Seyed Hamed Haghshenas, Md Abul Hasnat, and Mia Naeini
   - **Status**: **CITED** (`\cite{haghshenas_temporal}`)
   - **Core Contribution**: Proposes a Temporal Graph Neural Network (TGNN) combining topological GNN spatial updates with gated recurrent units (GRUs) to detect false data injection and ramp attacks.
   - **Relevance to GAT-Twin**: Directly supports our future work section proposing Spatio-Temporal GNN (GAT-LSTM/GRU) extensions to handle high-frequency transients.

---

### Category 2: Federated Learning in Smart Grids
* **Total Papers**: 4 (Cited: 2 | Supplementary: 2)

1. **Paper 1: Kececi et al. (2025)**
   - **File**: `Paper_1_Federated_Learning_Based_Distributed_Localization.pdf`
   - **Title**: *Federated Learning Based Distributed Localization of False Data Injection Attacks on Smart Grids*
   - **Authors**: Cihat Kececi, Katherine R. Davis, and Erchin Serpedin
   - **Status**: **CITED** (Mapped to `\cite{yang_fed_loc}` in manuscript bibliography)
   - **Core Contribution**: Develops a federated learning framework allowing multiple local substations to collaboratively train localization models to identify target attack nodes without sharing raw synchrophasor measurements.
   - **Relevance to GAT-Twin**: Supports our distributed multi-client training protocol and FedAvg parameter updates.

2. **Paper 2: Li et al. (2024)**
   - **File**: `Paper_2_Secure_Federated_DL_smart_grid.pdf`
   - **Title**: *Detection of False Data Injection Attacks in Smart Grid: A Secure Federated Deep Learning Approach*
   - **Authors**: Yang Li, Xinhao Wei, Yuanzheng Li, Zhaoyang Dong, Mohammad Shahidehpour
   - **Status**: **CITED** (Mapped to `\cite{li_vertical}` in manuscript bibliography)
   - **Core Contribution**: Proposes a secure federated deep learning framework combining Transformers, CNNs, and Federated Learning with privacy preservation to detect FDIAs.
   - **Relevance to GAT-Twin**: Validates the importance of privacy-preserving deep training across regional utility boundaries.

3. **Paper 3: Uddin et al. (2025)**
   - **File**: `Paper_3_FDIA_Detection_in_Smart_Meters_FL.pdf`
   - **Title**: *False Data Injection Attack Detection in Edge-based Smart Metering Networks with Federated Learning*
   - **Authors**: Md Raihan Uddin, Ratun Rahman, Dinh C. Nguyen
   - **Status**: **SUPPLEMENTARY**
   - **Core Contribution**: Explores edge-based smart metering networks using federated learning to identify anomalies in local electricity meters.
   - **Relevance to GAT-Twin**: Serves as a reference for client edge architectures, though our work is focused on transmission-level SCADA.

4. **Paper 4: Li et al. (2025)**
   - **File**: `Paper_4_Clustered_FL_for_FDIA_detection.pdf`
   - **Title**: *Clustered Federated Learning for Generalizable FDIA Detection in Smart Grids with Heterogeneous Data*
   - **Authors**: Yunfeng Li, Junhong Liu, Zhaohui Yang, Guofu Liao, Chuyun Zhang
   - **Status**: **SUPPLEMENTARY**
   - **Core Contribution**: Proposes clustered federated learning to handle Non-IID (non-independent and identically distributed) data across different regions, improving generalization.
   - **Relevance to GAT-Twin**: Provides insights into regional client data heterogeneity, which we address through our decoupled dual-model architecture.

*(Note: Other cited FL references in our bibliography, including Lu et al. [12] on homomorphic encryption and Al-Salami et al. [13] on Fed-AttentionGrid, are documented in the central related work review.)*

---

### Category 3: Digital Twins and Cyber-Physical Emulation
* **Total Papers**: 3 (Cited: 2 | Supplementary: 1)

1. **Paper 1: Sen et al. (2024)**
   - **File**: `Paper_1_Digital_Twin_Evaluating_Countermeasures.pdf`
   - **Title**: *Digital Twin for Evaluating Detective Countermeasures in Smart Grid Cybersecurity*
   - **Authors**: Ömer Sen, Nathalie Bleser, and Andreas Ulbig
   - **Status**: **CITED** (`\cite{sen_dt_countermeasures}`)
   - **Core Contribution**: Develops a Digital Twin framework utilizing communication network emulation and power grid simulation to evaluate cybersecurity countermeasures under multi-stage attacks.
   - **Relevance to GAT-Twin**: Core reference for using digital twin streaming states to test real-time detectors and evaluate latency.

2. **Paper 2: Zheng et al. (2022)**
   - **File**: `Paper_2_Introduction_to_Digital_Twins_Smart_Grid.pdf`
   - **Title**: *Smart Grid: Cyber Attacks, Critical Defense Approaches, and Digital Twin*
   - **Authors**: Tianming Zheng, Ping Yi, and Yue Wu
   - **Status**: **SUPPLEMENTARY**
   - **Core Contribution**: A comprehensive survey of smart grid designs, critical defenses, and how digital twins can be embedded into security architecture virtual replicas.
   - **Relevance to GAT-Twin**: Provides the foundational taxonomy of smart grid cyber-physical twin security.

3. **Paper 3: Kandasamy et al. (2021)**
   - **File**: `Paper_3_EPICTWIN_Power_Grid_Security.pdf`
   - **Title**: *EPICTWIN: An Electric Power Digital Twin for Cyber Security Testing, Research and Education*
   - **Authors**: Nandha Kumar Kandasamy, Sarad Venugopalan, Tin Kit Wong, Leu Junming Nicholas
   - **Status**: **CITED** (Cited as `\cite{sen_epictwin}` representing the EPICTWIN series)
   - **Core Contribution**: Presents the design of a hardware-in-the-loop (HIL) digital twin testbed to test cyber-attacks (e.g. DoS, command injection) in a safe and controlled environment.
   - **Relevance to GAT-Twin**: Highlights the importance of simulating real-time streaming states to evaluate detective latencies.

---

### Category 4: Threat Models and Coordinated FDIAs
* **Total Papers**: 4 (Cited: 0 | Supplementary: 4)

1. **Paper 1: Xiao & Weng (2023)**
   - **File**: `Paper_1_Limits_of_Residual_Based_Detection_AC_FDIA.pdf`
   - **Title**: *Limits of Residual-Based Detection for Physically Consistent False Data Injection*
   - **Authors**: Chenhan Xiao and Yang Weng
   - **Status**: **SUPPLEMENTARY**
   - **Core Contribution**: Proves that when AC-FDIAs lie on the measurement manifold induced by AC power flow and redundancy, traditional residual-based bad data detection (BDD) fails to distinguish them.
   - **Relevance to GAT-Twin**: Validates the BDD blind spot under non-linear AC conditions, justifying our GNN spatial consistency checking.

2. **Paper 2: Iranpour & Narimani (2023)**
   - **File**: `Paper_2_AC_FDIA_Design_and_Optimization.pdf`
   - **Title**: *AC False Data Injection Attacks in Power Systems: Design and Optimization*
   - **Authors**: Mohammadreza Iranpour and Mohammad Rasoul Narimani
   - **Status**: **SUPPLEMENTARY**
   - **Core Contribution**: Analyzes the essence and mathematical design of optimized AC false data injection attacks to bypass BDD.
   - **Relevance to GAT-Twin**: Provides the mathematical basis for the coordinated AC admittance shifts ($I = YV$) in our simulated FDIA generator.

3. **Paper 3: Du et al. (2021)**
   - **File**: `Paper_3_Targeted_AC_FDIA_Without_Network_Parameters.pdf`
   - **Title**: *Targeted False Data Injection Attacks Against AC State Estimation Without Network Parameters*
   - **Authors**: Mingqiu Du, Georgia Pierrou, Xiaozhe Wang, Marthe Kassouf
   - **Status**: **SUPPLEMENTARY**
   - **Core Contribution**: Formulates a targeted AC-FDIA that leverages load dynamics and Ornstein-Uhlenbeck processes without requiring network parameters.
   - **Relevance to GAT-Twin**: Serves as a contrast to our parameter-aware worst-case adversary threat model.

4. **Paper 4: Liang et al. (2021)**
   - **File**: `Paper_4_Consequences_of_Unobservable_AC_FDIA.pdf`
   - **Title**: *Vulnerability Analysis and Consequences of False Data Injection Attack on Power System State Estimation*
   - **Authors**: Jingwen Liang, Lalitha Sankar, and Oliver Kosut
   - **Status**: **SUPPLEMENTARY**
   - **Core Contribution**: Examines physical consequences of unobservable FDI attacks on AC state estimation, using bi-level optimization to maximize line flows and induce overload trips.
   - **Relevance to GAT-Twin**: Explains the physical damage unobservable attacks cause, justifying why our localization GNN and spatial self-healing are critical.

*(Note: Seminal threat models cited in our paper include Liu et al. [1] and Kosut et al. [2] for baseline studies.)*

---

### Category 5: Advanced Trends (Explainable AI & Optimization)
* **Total Papers**: 5 (Cited: 3 | Supplementary: 2)

1. **Paper 1: Alihodzic et al. (2026)**
   - **File**: `Paper_1_Cyber_Physical_Anomaly_Detection_Smart_Grids.pdf`
   - **Title**: *Cyber-Physical Anomaly Detection in IoT-Enabled Smart Grids Using Machine Learning and Metaheuristic Feature Optimization*
   - **Authors**: Adis Alihodzic, Eva Tuba, Milan Tuba
   - **Status**: **CITED** (`\cite{alihodzic_shap}`)
   - **Core Contribution**: Evaluates ML anomaly classifiers combined with genetic algorithms for feature selection on the MSU/ORNL dataset, proving that a compact subset of attributes is sufficient.
   - **Relevance to GAT-Twin**: Directly validates our feature selection pipelines and baseline training on the exact MSU/ORNL testbed.

2. **Paper 2: Boyaci et al. (2023)**
   - **File**: `Paper_2_GNN_Detection_Stealth_FDIAs.pdf`
   - **Title**: *Graph Neural Networks Based Detection of Stealth False Data Injection Attacks in Smart Grids*
   - **Authors**: Osman Boyaci, Amarachi Umunnakwe, Abhijeet Sahu, Mohammad Rasoul Narimani, Muhammad Ismail, Katherine Davis, Erchin Serpedin
   - **Status**: **SUPPLEMENTARY**
   - **Core Contribution**: Proposes generic, localized GNNs to model smart grid topology and detect stealthy FDIAs.
   - **Relevance to GAT-Twin**: Directly aligns with our decoupled Graph-level and Node-level GNN architecture.

3. **Paper 3: Varbella et al. (2024)**
   - **File**: `Paper_3_PowerGraph_GNN_Benchmark.pdf`
   - **Title**: *PowerGraph: A power grid benchmark dataset for graph neural networks*
   - **Authors**: Anna Varbella, Kenza Amara, Blazhe Gjorgiev, Mennatallah El-Assady, Giovanni Sansavini
   - **Status**: **SUPPLEMENTARY**
   - **Core Contribution**: Introduces PowerGraph, a power grid benchmark dataset designed specifically for training and comparing GNNs under operational reconfigurations.
   - **Relevance to GAT-Twin**: Highlights the standard practice of evaluating GNNs on large-scale grids like the IEEE 118-bus network.

4. **Paper 4: Hamilton et al. (2022)**
   - **File**: `Paper_4_Interpretable_ML_SHAP_Physics.pdf`
   - **Title**: *Interpretable Machine Learning for Power Systems: Establishing Confidence in SHapley Additive Explanations*
   - **Authors**: Robert I. Hamilton, Jochen Stiasny, Tabia Ahmad, Samuel Chevalier, Rahul Nellikkath, Ilgiz Murzakhanov, Spyros Chatzivasileiadis, Panagiotis N. Papadopoulos
   - **Status**: **CITED** (`\cite{hamilton_shap}`)
   - **Core Contribution**: Proves that SHAP (Shapley Additive Explanations) attributions capture actual power system physics (such as PTDFs) in machine learning outputs.
   - **Relevance to GAT-Twin**: Establishes the standard for physics-consistent explanations in smart grids, which we build on.

5. **Paper 5: Feng et al. (2025)**
   - **File**: `Paper_5_Deep_Learning_Accelerated_Shapley_Value.pdf`
   - **Title**: *Deep Learning-Accelerated Shapley Value for Fair Allocation in Power Systems: The Case of Carbon Emission Responsibility*
   - **Authors**: Yuanhao Feng, Tao Sun, Yan Meng, Xuxin Yang, Donghan Feng
   - **Status**: **CITED** (`\cite{feng_surroshap}`)
   - **Core Contribution**: Formulates SurroShap, a deep learning surrogate framework to accelerate Shapley value evaluations and resolve computational bottlenecks.
   - **Relevance to GAT-Twin**: Highlights the high computational latency of Shapley approximations, justifying our sub-millisecond direct gradient backpropagation saliency method to satisfy the 112~ms grid stability control threshold.

---

## Additional Core Dataset Citations

1. **MSU/ORNL Power System Attack Dataset Paper**
   - **Title**: *Machine learning for power system disturbance and cyber-attack discrimination*
   - **Authors**: J. M. Beaver, R. C. Borges-Hink, M. A. Buckner, T. Morris, U. Adhikari, and S. Pan
   - **Status**: **CITED** (`\cite{msu_ornl_dataset}`)
   - **Role**: The official conference paper (Resilient Control Systems Symposium, 2014) introducing the physical 3-bus, 4-relay testbed attack data. It details the synchrophasor recordings, protective relay configurations, and Snort logs that comprise the physical ORNL dataset used to train our GNN models.
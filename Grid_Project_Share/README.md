# Federated GAT-Twin Smart Grid Datasets and Code Package

This package contains the dataset generation and preprocessing scripts used in the Federated GAT-Twin grid security project.

## Directory Structure
*   `src/`
    *   `generate_118bus_dataset.py`: Loads the standard IEEE 118-bus grid topology, simulates AC power flows under load variations, injects physics-conforming False Data Injection Attacks (FDIAs), and outputs the 118-bus dataset.
    *   `prepare_ornl_data.py`: Preprocesses the raw MSU/ORNL 4-relay physical testbed dataset, extracting and mapping the 31 synchrophasor/cyber features.
*   `data/`
    *   `grid_topology_118bus.csv`: Edge connections (transmission lines) for the IEEE 118-bus grid.
    *   `grid_topology_ornl.csv`: Edge connections for the MSU/ORNL 4-relay testbed.

## How to Generate/Preprocess the Datasets
To build or extract the datasets locally, execute the following commands in your Python environment:

1.  **For the simulated IEEE 118-bus grid**:
    ```bash
    python src/generate_118bus_dataset.py
    ```
    This script runs the AC power flows and saves the processed dataset to `data/ieee118_extracted.csv`.

2.  **For the physical MSU/ORNL testbed**:
    Ensure the raw CSV `data_injection_and_normal_events_dataset.csv` is downloaded from the UAH ICS datasets repository and placed in `datasets_public/MSU_ORNL/`. Then run:
    ```bash
    python src/prepare_ornl_data.py
    ```
    This will extract the features and save the processed dataset to `data/ornl_extracted.csv`.

## Feature Descriptions (`F1` to `F31`)
The extracted datasets contain 31 synchronized feature columns at each node (bus/relay) for each time step:
*   `F1`: Voltage Magnitude ($V_m$) in per-unit (p.u.) values.
*   `F2`: Voltage Phase Angle ($V_a$) in degrees.
*   `F3`: Real Active Power Injection ($P$) in MW.
*   `F4`: Reactive Power Injection ($Q$) in MVar.
*   `F5` – `F7`: Line Current Magnities ($I_1, I_2, I_3$) for up to 3 connected lines.
*   `F8`: System Frequency ($f$) in Hz.
*   `F9`: Rate of Change of Frequency ($df/dt$).
*   `F10`: Relay Breaker Log ($1.0$ if physically tripped, $0.0$ otherwise).
*   `F11`: Snort IDS Log ($1.0$ if cyber intrusion alert triggers, $0.0$ otherwise).
*   `F12` – `F31`: Apparent Impedance parameters and measurement noise variations.

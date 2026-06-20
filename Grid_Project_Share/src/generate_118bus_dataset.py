import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def generate_ieee118_dataset():
    print("Initializing IEEE 118-bus network...")
    # Load case118
    net = pn.case118()
    
    # Save the topology
    print("Saving 118-bus topology...")
    topo_df = pd.DataFrame({
        'source': net.line.from_bus.values,
        'target': net.line.to_bus.values
    })
    os.makedirs('data', exist_ok=True)
    topo_df.to_csv('data/grid_topology_118bus.csv', index=False)
    print(f"Topology saved with {len(topo_df)} lines.")
    
    num_buses = len(net.bus)
    num_snapshots = 1000  # Generate 1,000 snapshots (118,000 node samples)
    np.random.seed(42)
    
    # Pre-extract from_bus and to_bus arrays to avoid slow iterrows() inside loop
    from_buses = net.line.from_bus.values.astype(int)
    to_buses = net.line.to_bus.values.astype(int)
    
    # Pre-calculate line impedances and current sensitivities for coordinate attacks
    V_base = 138.0
    S_base = 100.0
    Z_base = (V_base ** 2) / S_base  # 190.44 Ohms
    I_base = S_base / (np.sqrt(3) * V_base)  # 0.41837 kA
    
    # Handle line parameter defaults if missing length or standard units
    lengths = net.line.length_km.values if 'length_km' in net.line.columns else np.ones(len(net.line))
    R = net.line.r_ohm_per_km.values * lengths
    X = net.line.x_ohm_per_km.values * lengths
    Z_ohm = np.sqrt(R**2 + X**2)
    Z_ohm = np.maximum(Z_ohm, 1e-4) # Avoid division by zero
    Z_pu = Z_ohm / Z_base
    
    current_sensitivities = I_base / Z_pu
    
    # Map from bus index to list of tuples: (line_idx, sensitivity)
    bus_line_mappings = {b: [] for b in range(num_buses)}
    for idx in range(len(net.line)):
        from_b = from_buses[idx]
        to_b = to_buses[idx]
        sens = current_sensitivities[idx]
        bus_line_mappings[from_b].append((idx, sens))
        bus_line_mappings[to_b].append((idx, sens))
        
    # Base load values
    base_load_p = net.load.p_mw.values.copy()
    base_load_q = net.load.q_mvar.values.copy()
    
    # We will build a matrix of shape (num_snapshots * 118, 34)
    # Columns: Time_Step, Bus_Index, F1..F31, Label
    data_rows = []
    
    print(f"Simulating {num_snapshots} power flow snapshots...")
    
    for t in range(num_snapshots):
        # 1. Randomly vary loads by +/- 5% to simulate load changes (Natural variations)
        load_perturbation = np.random.uniform(0.95, 1.05, len(net.load))
        net.load.p_mw = base_load_p * load_perturbation
        net.load.q_mvar = base_load_q * load_perturbation
        
        # Decide if this snapshot is an attack (30% probability)
        is_attack = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Decide if this snapshot has a natural line fault (10% probability, Class 0)
        has_fault = False
        outage_line = None
        if is_attack == 0 and np.random.rand() < 0.1:
            has_fault = True
            # Disconnect a random line to simulate a physical event
            outage_line = np.random.randint(0, len(net.line))
            net.line.loc[outage_line, 'in_service'] = False
            
        # 2. Run Power Flow (suppress numba warnings and overhead)
        try:
            pp.runpp(net, numba=False)
            success = True
        except Exception:
            # If power flow fails to converge (common on random outages), restore and retry
            success = False
            
        # Restore line if it was out of service
        if outage_line is not None:
            net.line.loc[outage_line, 'in_service'] = True
            
        if not success:
            # Skip this snapshot if it didn't converge
            continue
            
        # 3. Extract Node Measurements
        v_mag = net.res_bus.vm_pu.values
        v_ang = net.res_bus.va_degree.values
        p_inj = net.res_bus.p_mw.values
        q_inj = net.res_bus.q_mvar.values
        
        # Select target bus and shift parameter under attack if is_attack == 1
        target_bus = None
        v_m_shift = 0.0
        v_a_shift = 0.0
        if is_attack == 1:
            target_bus = np.random.randint(0, num_buses)
            v_m_shift = np.random.uniform(0.04, 0.06)
            v_a_shift = -np.random.uniform(1.0, 3.0)
            
        # Gather line currents for each bus (and copy to perturb it coordinately)
        i_line = net.res_line.i_from_ka.values.copy()
        
        # Coordinated physics-conforming attack: perturb line currents connected to target bus
        if is_attack == 1:
            for line_idx, sens in bus_line_mappings[target_bus]:
                i_line[line_idx] += v_m_shift * sens
                
        # Create an list of line connections for each bus
        bus_currents = {b: [] for b in range(num_buses)}
        for idx in range(len(from_buses)):
            from_b = from_buses[idx]
            to_b = to_buses[idx]
            cur = i_line[idx]
            bus_currents[from_b].append(cur)
            bus_currents[to_b].append(cur)
            
        # 4. Generate the 31 Features for each of the 118 buses
        for b in range(num_buses):
            # Base features:
            v_m = v_mag[b]
            v_a = v_ang[b]
            p_i = p_inj[b]
            q_i = q_inj[b]
            
            # Add measurement noise (simulating PMUs)
            v_m += np.random.normal(0, 0.001)
            v_a += np.random.normal(0, 0.05)
            p_i += np.random.normal(0, 0.05)
            q_i += np.random.normal(0, 0.05)
            
            # If under attack and this is the target bus, inject False Data (Stealthy Coordinated Attack)
            if is_attack == 1 and b == target_bus:
                v_m += v_m_shift
                v_a += v_a_shift
                
            # Get line currents
            currents = bus_currents[b]
            c1 = currents[0] if len(currents) > 0 else 0.0
            c2 = currents[1] if len(currents) > 1 else 0.0
            c3 = currents[2] if len(currents) > 2 else 0.0
            
            # Simulate frequency features
            freq = 60.0 + np.random.normal(0, 0.005)
            if has_fault:
                freq -= np.random.uniform(0.05, 0.15)
            df_dt = np.random.normal(0, 0.01)
            
            # Security logs
            relay_log = 0.0
            snort_log = 0.0
            if has_fault:
                relay_log = 1.0 if np.random.rand() < 0.8 else 0.0
            if is_attack == 1 and b == target_bus:
                snort_log = 1.0 if np.random.rand() < 0.1 else 0.0
                
            # Pack features into a list of size 31
            features = [
                v_m, v_a, p_i, q_i, c1, c2, c3, freq, df_dt, relay_log, snort_log
            ]
            # Fill the remaining 20 features with variations and noise to match 31 dimensions
            for f_idx in range(20):
                features.append(features[f_idx % len(features)] * 0.1 + np.random.normal(0, 0.01))
                
            node_label = 1.0 if (is_attack == 1 and b == target_bus) else 0.0
            snapshot_label = 1.0 if is_attack == 1 else 0.0
            
            row = [t, b] + features + [node_label, snapshot_label]
            data_rows.append(row)
            
        if (t + 1) % 100 == 0:
            print(f"  Processed {t+1}/{num_snapshots} snapshots...")
            
    # Save the dataset
    cols = ['Time_Step', 'Bus_Index'] + [f'F{i}' for i in range(1, 32)] + ['Node_Label', 'Snapshot_Label']
    dataset_df = pd.DataFrame(data_rows, columns=cols)
    dataset_df.to_csv('data/ieee118_extracted.csv', index=False)
    print(f"\nDataset generation complete! Saved to 'data/ieee118_extracted.csv'.")
    print(f"Dataset Shape: {dataset_df.shape}")
    print(f"Snapshot Class Value Counts:\n{dataset_df['Snapshot_Label'].value_counts() / 118}")
    print(f"Node Class Value Counts:\n{dataset_df['Node_Label'].value_counts()}")

if __name__ == '__main__':
    generate_ieee118_dataset()

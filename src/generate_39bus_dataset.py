import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def generate_ieee39_dataset():
    print("Initializing IEEE 39-bus network...")
    net = pn.case39()
    
    print("Constructing combined topology (lines + transformers)...")
    from_buses = np.concatenate([net.line.from_bus.values, net.trafo.hv_bus.values]).astype(int)
    to_buses = np.concatenate([net.line.to_bus.values, net.trafo.lv_bus.values]).astype(int)
    
    # Save the static topology
    topo_df = pd.DataFrame({
        'source': from_buses,
        'target': to_buses
    })
    os.makedirs('data', exist_ok=True)
    topo_df.to_csv('data/grid_topology_39bus.csv', index=False)
    print(f"Topology saved with {len(topo_df)} branches (35 lines, 11 transformers).")
    
    num_buses = len(net.bus)
    num_snapshots = 1000
    np.random.seed(39)
    
    V_base = 345.0
    S_base = 100.0
    Z_base = (V_base ** 2) / S_base  # 1190.25 Ohms
    I_base = S_base / (np.sqrt(3) * V_base)  # 0.16735 kA
    
    Z_ohm = []
    lengths = net.line.length_km.values if 'length_km' in net.line.columns else np.ones(len(net.line))
    R_line = net.line.r_ohm_per_km.values * lengths
    X_line = net.line.x_ohm_per_km.values * lengths
    Z_line = np.sqrt(R_line**2 + X_line**2)
    Z_line = np.maximum(Z_line, 1e-4)
    Z_ohm.extend(Z_line)
    
    for idx, row in net.trafo.iterrows():
        sn_mva = row['sn_mva'] if row['sn_mva'] > 0 else 100.0
        vn_hv = row['vn_hv_kv'] if row['vn_hv_kv'] > 0 else 345.0
        Z_nom = (vn_hv ** 2) / sn_mva
        Z_tr = (row['vk_percent'] / 100.0) * Z_nom
        Z_tr = np.maximum(Z_tr, 1e-4)
        Z_ohm.append(Z_tr)
        
    Z_ohm = np.array(Z_ohm)
    Z_pu = Z_ohm / Z_base
    current_sensitivities = I_base / Z_pu
    
    bus_branch_mappings = {b: [] for b in range(num_buses)}
    for idx in range(len(from_buses)):
        from_b = from_buses[idx]
        to_b = to_buses[idx]
        sens = current_sensitivities[idx]
        bus_branch_mappings[from_b].append((idx, sens))
        bus_branch_mappings[to_b].append((idx, sens))
        
    base_load_p = net.load.p_mw.values.copy()
    base_load_q = net.load.q_mvar.values.copy()
    
    data_rows = []
    dynamic_edges_rows = []
    
    print(f"Simulating {num_snapshots} power flow snapshots...")
    
    clean_t = 0
    for t in range(num_snapshots):
        load_perturbation = np.random.uniform(0.95, 1.05, len(net.load))
        net.load.p_mw = base_load_p * load_perturbation
        net.load.q_mvar = base_load_q * load_perturbation
        
        is_attack = np.random.choice([0, 1], p=[0.7, 0.3])
        has_outage = np.random.rand() < 0.15
        
        outage_type = None
        outage_idx = None
        
        if has_outage:
            outage_type = np.random.choice(['line', 'trafo'], p=[0.75, 0.25])
            if outage_type == 'line':
                outage_idx = np.random.randint(0, len(net.line))
                net.line.loc[outage_idx, 'in_service'] = False
            else:
                outage_idx = np.random.randint(0, len(net.trafo))
                net.trafo.loc[outage_idx, 'in_service'] = False
                
        try:
            pp.runpp(net, numba=False)
            success = True
        except Exception:
            if outage_idx is not None:
                if outage_type == 'line':
                    net.line.loc[outage_idx, 'in_service'] = True
                else:
                    net.trafo.loc[outage_idx, 'in_service'] = True
                outage_idx = None
            try:
                pp.runpp(net, numba=False)
                success = True
            except Exception:
                success = False
                
        if not success:
            continue
            
        v_mag = net.res_bus.vm_pu.values
        v_ang = net.res_bus.va_degree.values
        p_inj = net.res_bus.p_mw.values
        q_inj = net.res_bus.q_mvar.values
        
        # Verify that there are no NaNs in the pandapower load flow results
        if np.isnan(v_mag).any() or np.isnan(v_ang).any() or np.isnan(p_inj).any() or np.isnan(q_inj).any():
            if outage_idx is not None:
                if outage_type == 'line':
                    net.line.loc[outage_idx, 'in_service'] = True
                else:
                    net.trafo.loc[outage_idx, 'in_service'] = True
            continue
            
        target_bus = None
        v_m_shift = 0.0
        v_a_shift = 0.0
        if is_attack == 1:
            target_bus = np.random.randint(0, num_buses)
            v_m_shift = np.random.uniform(0.04, 0.06)
            v_a_shift = -np.random.uniform(1.0, 3.0)
            
        i_line = net.res_line.i_from_ka.values.copy()
        i_trafo = net.res_trafo.i_hv_ka.values.copy()
        i_branch = np.concatenate([i_line, i_trafo])
        i_branch = np.nan_to_num(i_branch, nan=0.0)
        
        if is_attack == 1:
            for branch_idx, sens in bus_branch_mappings[target_bus]:
                i_branch[branch_idx] += v_m_shift * sens
                
        outage_branch_idx = None
        if outage_idx is not None:
            if outage_type == 'line':
                outage_branch_idx = outage_idx
            else:
                outage_branch_idx = len(net.line) + outage_idx
                
        bus_currents = {b: [] for b in range(num_buses)}
        for idx in range(len(from_buses)):
            if outage_branch_idx is not None and idx == outage_branch_idx:
                cur = 0.0
            else:
                cur = i_branch[idx]
                from_b = from_buses[idx]
                to_b = to_buses[idx]
                dynamic_edges_rows.append([clean_t, from_b, to_b])
                dynamic_edges_rows.append([clean_t, to_b, from_b])
            from_b = from_buses[idx]
            to_b = to_buses[idx]
            bus_currents[from_b].append(cur)
            bus_currents[to_b].append(cur)
            
        for b in range(num_buses):
            v_m = v_mag[b]
            v_a = v_ang[b]
            p_i = p_inj[b]
            q_i = q_inj[b]
            
            v_m += np.random.normal(0, 0.001)
            v_a += np.random.normal(0, 0.05)
            p_i += np.random.normal(0, 0.05)
            q_i += np.random.normal(0, 0.05)
            
            if is_attack == 1 and b == target_bus:
                v_m += v_m_shift
                v_a += v_a_shift
                
            currents = bus_currents[b]
            c1 = currents[0] if len(currents) > 0 else 0.0
            c2 = currents[1] if len(currents) > 1 else 0.0
            c3 = currents[2] if len(currents) > 2 else 0.0
            
            freq = 60.0 + np.random.normal(0, 0.005)
            if outage_idx is not None:
                freq -= np.random.uniform(0.05, 0.15)
            df_dt = np.random.normal(0, 0.01)
            
            relay_log = 0.0
            snort_log = 0.0
            if outage_idx is not None:
                outage_from = from_buses[outage_branch_idx]
                outage_to = to_buses[outage_branch_idx]
                if b == outage_from or b == outage_to:
                    relay_log = 1.0 if np.random.rand() < 0.9 else 0.0
            if is_attack == 1 and b == target_bus:
                snort_log = 1.0 if np.random.rand() < 0.1 else 0.0
                
            features = [
                v_m, v_a, p_i, q_i, c1, c2, c3, freq, df_dt, relay_log, snort_log
            ]
            for f_idx in range(20):
                features.append(features[f_idx % len(features)] * 0.1 + np.random.normal(0, 0.01))
                
            node_label = 1.0 if (is_attack == 1 and b == target_bus) else 0.0
            snapshot_label = 1.0 if is_attack == 1 else 0.0
            
            row = [clean_t, b] + features + [node_label, snapshot_label]
            data_rows.append(row)
            
        if outage_idx is not None:
            if outage_type == 'line':
                net.line.loc[outage_idx, 'in_service'] = True
            else:
                net.trafo.loc[outage_idx, 'in_service'] = True
                
        clean_t += 1
        if (t + 1) % 100 == 0:
            print(f"  Processed {t+1}/{num_snapshots} snapshots (Clean snapshots: {clean_t})...")
            
    cols = ['Time_Step', 'Bus_Index'] + [f'F{i}' for i in range(1, 32)] + ['Node_Label', 'Snapshot_Label']
    dataset_df = pd.DataFrame(data_rows, columns=cols)
    dataset_df.to_csv('data/ieee39_extracted.csv', index=False)
    
    edges_df = pd.DataFrame(dynamic_edges_rows, columns=['Time_Step', 'source', 'target'])
    edges_df.to_csv('data/grid_topology_39bus_dynamic.csv', index=False)
    
    print(f"\nIEEE 39-bus Dataset generation complete!")
    print(f"Dataset Shape: {dataset_df.shape}")
    print(f"Attack Snapshot Ratio: {dataset_df['Snapshot_Label'].value_counts() / 39}")
    print(f"Node Attack Ratio: {dataset_df['Node_Label'].value_counts()}")

if __name__ == '__main__':
    generate_ieee39_dataset()

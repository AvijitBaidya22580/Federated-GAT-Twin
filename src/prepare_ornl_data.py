import pandas as pd
import numpy as np

def process_ornl_fast():
    print("Loading raw ORNL CSV...")
    df = pd.read_csv('datasets_public/MSU_ORNL/data_injection_and_normal_events_dataset.csv')
    
    print("Building topology...")
    # 4 relays act as 4 nodes in a line
    edges = pd.DataFrame({'source': [0, 1, 2], 'target': [1, 2, 3]})
    edges.to_csv('data/grid_topology_ornl.csv', index=False)
    
    print("Vectorizing data extraction...")
    r1_cols = [c for c in df.columns if c.startswith('R1')]
    r2_cols = [c for c in df.columns if c.startswith('R2')]
    r3_cols = [c for c in df.columns if c.startswith('R3')]
    r4_cols = [c for c in df.columns if c.startswith('R4')]
    
    # Extract values
    r1_data = df[r1_cols + ['relay1_log', 'snort_log1']].values
    r2_data = df[r2_cols + ['relay2_log', 'snort_log2']].values
    r3_data = df[r3_cols + ['relay3_log', 'snort_log3']].values
    r4_data = df[r4_cols + ['relay4_log', 'snort_log4']].values
    
    t_vals = np.arange(len(df))
    lbl_vals = df['marker'].values
    
    nodes_data = []
    for i, data in enumerate([r1_data, r2_data, r3_data, r4_data]):
        node_idx = np.full((len(df), 1), i)
        t_col = t_vals.reshape(-1, 1)
        lbl_col = lbl_vals.reshape(-1, 1)
        combined = np.hstack((t_col, node_idx, data, lbl_col))
        nodes_data.append(combined)
        
    all_data = np.vstack(nodes_data)
    
    cols = ['Time_Step', 'Bus_Index'] + [f'F{i}' for i in range(1, 32)] + ['Label']
    out_df = pd.DataFrame(all_data, columns=cols)
    
    # Sort by Time_Step then Bus_Index
    out_df = out_df.sort_values(by=['Time_Step', 'Bus_Index'])
    
    print("Saving data/ornl_extracted.csv...")
    out_df.to_csv('data/ornl_extracted.csv', index=False)
    print("ORNL Data extraction complete. Shape:", out_df.shape)

if __name__ == '__main__':
    process_ornl_fast()

import subprocess
import sys
import time
import shutil
import os

def run_script(script_name, description):
    print(f"\n======================================================================")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"======================================================================")
    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, script_name], check=True, text=True)
        elapsed = time.time() - start_time
        print(f"\nSUCCESS: {description} completed in {elapsed:.2f} seconds.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {description} failed with exit code {e.returncode}.\n")
        return False

def sync_artifacts():
    print("======================================================================")
    print("SYNCING RESULTS TO ARTFACTS DIRECTORY...")
    print("======================================================================")
    artifact_dir = r"C:\Users\Avijit Baidya\.gemini\antigravity\brain\6271d2d8-6e69-4b25-b4b4-5fd8c79f964d"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Files to sync
    plots = [
        "federated_gat_39bus_results.png",
        "federated_gat_118bus_results.png",
        "federated_gat_ornl_results.png",
        "gat_explanation_bus105.png",
        "closed_loop_mitigation_error.png",
        "multibus_scaling_recall.png"
    ]
    
    for plot in plots:
        src = os.path.join("plots", plot)
        dst = os.path.join(artifact_dir, plot)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  Synced plot: {plot} --> Artifacts")
        else:
            print(f"  Warning: Plot not found: {src}")
            
    reports = [
        ("report/comparative_analysis_report.md", "comparative_analysis_report.md"),
        ("report/walkthrough.md", "walkthrough.md")
    ]
    for src_path, dst_name in reports:
        if os.path.exists(src_path):
            dst = os.path.join(artifact_dir, dst_name)
            shutil.copy(src_path, dst)
            print(f"  Synced report: {src_path} --> Artifacts/{dst_name}")
        else:
            print(f"  Warning: Report not found: {src_path}")
            
    print("\nSUCCESS: All outputs synchronized to artifacts folder.")

def main():
    steps = [
        ("src/prepare_ornl_data.py", "MSU/ORNL Data Preprocessing"),
        ("src/generate_39bus_dataset.py", "IEEE 39-Bus Dataset Generation"),
        ("src/generate_118bus_dataset.py", "IEEE 118-Bus Dataset Generation"),
        ("src/train_ornl_federated.py", "MSU/ORNL Federated ResGAT Training"),
        ("src/train_39bus_federated.py", "IEEE 39-Bus Federated ResGAT Training"),
        ("src/train_118bus_federated.py", "IEEE 118-Bus Federated ResGAT Training"),
        ("src/explain_gat_118bus.py", "GNN Anomaly Attribution (XAI)"),
        ("src/closed_loop_mitigation.py", "State Quarantine and Spatial Mitigation"),
        ("src/test_multibus_scaling.py", "Multi-Bus Attack Scaling Bounds"),
        ("src/compare_grid_networks.py", "Unified Cross-Scale Comparative Evaluation")
    ]
    
    total_start = time.time()
    for script, desc in steps:
        success = run_script(script, desc)
        if not success:
            print("\nPipeline aborted due to script execution failure.")
            sys.exit(1)
            
    sync_artifacts()
    
    print("\n======================================================================")
    print(f"FULL PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - total_start:.2f} SECONDS!")
    print("======================================================================\n")

if __name__ == '__main__':
    main()

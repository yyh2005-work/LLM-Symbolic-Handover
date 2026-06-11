"""
Aggregate multiple LLM-RULE experiment results.

Reads data from multiple successful batch runs, computes averages,
and generates aggregated data for plotting.
"""

import os
import shutil
import sys

import numpy as np
from typing import Any, Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LLM_RESULTS_ROOT = os.path.join(RESULTS_DIR, "llm_generated_symbxrl")
AGGREGATED_DIR = os.path.join(RESULTS_DIR, "llm_rule_aggregated")


def collect_results_for_speed(results_list: List[Dict[str, Any]], key: str) -> np.ndarray:
    """Collect per-speed metric across all runs and return the mean values."""
    if not results_list:
        return np.array([])
    
    first_result = results_list[0]
    speeds = first_result["speeds"]
    n_speeds = len(speeds)
    
    all_values = []
    for result in results_list:
        values = result.get(key, [])
        if len(values) == n_speeds:
            all_values.append(values)
    
    if not all_values:
        return np.array([])
    
    all_values = np.array(all_values)
    mean_values = np.mean(all_values, axis=0)
    std_values = np.std(all_values, axis=0)
    
    print(f"[AGG] {key}: samples={len(all_values)}, mean={mean_values}, std={std_values}")
    return mean_values


def collect_sinr_results(sinr_lists: List[np.ndarray]) -> np.ndarray:
    """Merge all SINR arrays into a single array for ECDF plotting."""
    all_sinr = []
    for sinr_array in sinr_lists:
        all_sinr.extend(sinr_array.tolist())
    return np.array(all_sinr)


def main():
    print("="*70)
    print("          Aggregate LLM-RULE Results")
    print("="*70)
    
    # Find all successfully completed runs
    successful_runs = []
    i = 1
    consecutive_missing = 0
    max_consecutive_missing = 10  # Stop after this many consecutive missing directories
    
    while consecutive_missing < max_consecutive_missing:
        run_dir = os.path.join(LLM_RESULTS_ROOT, f"run_{i:03d}")
        result_path = os.path.join(run_dir, "rule_llm_agent_results.npz")
        if os.path.exists(result_path):
            successful_runs.append(run_dir)
            consecutive_missing = 0
        else:
            consecutive_missing += 1
        i += 1
    
    print(f"Found {len(successful_runs)} successful runs")
    
    if not successful_runs:
        print("[ERROR] No successful run results found")
        sys.exit(1)
    
    # Load results from all successful runs
    all_results = []
    all_sinr_before = []
    all_sinr_after = []
    
    for run_dir in successful_runs:
        result_path = os.path.join(run_dir, "rule_llm_agent_results.npz")
        try:
            data = np.load(result_path, allow_pickle=True)
            result_dict = {
                "speeds": data["speeds"],
                "rate_mbps": data["rate_mbps"],
                "r_rel": data["r_rel"],
                "mean_pp_prob": data["mean_pp_prob"],
                "mean_rlf_prob": data["mean_rlf_prob"],
            }
            all_results.append(result_dict)
            all_sinr_before.append(data["sinr_before_ho"])
            all_sinr_after.append(data["sinr_after_ho"])
            print(f"[INFO] Loaded: {run_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to load {result_path}: {e}")
    
    if not all_results:
        print("[ERROR] No results loaded successfully")
        sys.exit(1)
    
    # Compute aggregate statistics
    print("\n[AGG] Computing aggregate statistics")
    aggregated = {
        "speeds": all_results[0]["speeds"],
        "rate_mbps": collect_results_for_speed(all_results, "rate_mbps"),
        "r_rel": collect_results_for_speed(all_results, "r_rel"),
        "mean_pp_prob": collect_results_for_speed(all_results, "mean_pp_prob"),
        "mean_rlf_prob": collect_results_for_speed(all_results, "mean_rlf_prob"),
        "sinr_before_ho": collect_sinr_results(all_sinr_before),
        "sinr_after_ho": collect_sinr_results(all_sinr_after),
    }
    
    # Save aggregated results
    os.makedirs(AGGREGATED_DIR, exist_ok=True)
    aggregated_path = os.path.join(AGGREGATED_DIR, "rule_llm_agent_results_aggregated.npz")
    np.savez(
        aggregated_path,
        speeds=aggregated["speeds"],
        rate_mbps=aggregated["rate_mbps"],
        r_rel=aggregated["r_rel"],
        mean_pp_prob=aggregated["mean_pp_prob"],
        mean_rlf_prob=aggregated["mean_rlf_prob"],
        sinr_before_ho=aggregated["sinr_before_ho"],
        sinr_after_ho=aggregated["sinr_after_ho"],
    )
    print(f"\n[INFO] Aggregated results saved to {aggregated_path}")
    
    # Copy aggregated results to the location expected by plot_results.py
    target_path = os.path.join(RESULTS_DIR, "rule", "rule_llm_agent_results.npz")
    shutil.copy(aggregated_path, target_path)
    print(f"[INFO] Copied aggregated results to {target_path}")
    print(f"[INFO] Overwrote single-run result; now contains means over {len(successful_runs)} runs")
    
    print("\n" + "="*70)
    print("          Aggregation complete!")
    print("="*70)
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Aggregated path: {aggregated_path}")
    print("\nNext, run the plotting script:")
    print(f"  python scripts/plot_results.py")


if __name__ == "__main__":
    main()

"""
Batch LLM-RULE experiment runner.

Repeatedly calls generate_llm_symbol_mapper.py until the target number of successful LLM calls is reached. 
For each success:
- Check stage1_mapper_meta.json mode field to confirm LLM was used.
- Run validate_ppo.py and validate_ppo_llm_rule.py.
- Save results for the run.

Note:
- aggregate_llm_results.py and plot_results.py are NOT called here;
  run them manually after the batch completes.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# Script paths
VALIDATE_PPO_SCRIPT = os.path.join(SCRIPTS_DIR, "validate_ppo.py")
GENERATE_MAPPER_SCRIPT = os.path.join(SCRIPTS_DIR, "generate_llm_symbol_mapper.py")
VALIDATE_PPO_LLM_RULE_SCRIPT = os.path.join(SCRIPTS_DIR, "validate_ppo_llm_rule.py")

# Output directories
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LLM_RESULTS_ROOT = os.path.join(RESULTS_DIR, "llm_generated_symbxrl")


def run_script(script_path: str, args: Optional[List[str]] = None) -> int:
    """Run a Python script with optional arguments and return its exit code."""
    cmd_args = [sys.executable, script_path]
    if args:
        cmd_args.extend(args)
    
    print(f"[RUN] {' '.join(cmd_args)}")
    start_time = time.time()
    result = subprocess.run(cmd_args, cwd=PROJECT_ROOT, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    print(f"[DONE] exit_code={result.returncode}, elapsed={elapsed:.2f}s\n")
    
    return result.returncode


def run_validate_ppo() -> int:
    """Run validate_ppo.py."""
    return run_script(VALIDATE_PPO_SCRIPT, ["--root-path", PROJECT_ROOT])


def check_llm_success(stage1_dir: str) -> bool:
    """Check whether the LLM call succeeded by inspecting stage1_mapper_meta.json.

    mode == "llm"-> success
    other values-> failure (e.g. "builtin_fallback", "llm_failed_fallback")
    """
    meta_path = os.path.join(stage1_dir, "stage1_mapper_meta.json")
    if not os.path.exists(meta_path):
        print(f"[WARN] Missing: {meta_path}")
        return False
    
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mode = meta.get("mode", "")
        success = mode == "llm"
        print(f"[LLM-CHECK] mode={mode}, success={success}")
        return success
    except Exception as e:
        print(f"[ERROR] Failed to read {meta_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch LLM-RULE experiment runner")
    parser.add_argument("--num-runs", type=int, default=30, help="Number of successful LLM runs ")
    parser.add_argument("--max-attempts", type=int, default=60, help="Maximum total attempts")
    args = parser.parse_args()
    
    print("="*70)
    print("Batch LLM-RULE Experiment")
    print("="*70)
    print(f"Target successful runs: {args.num_runs}")
    print(f"Maximum attempts: {args.max_attempts}")
    print("="*70)
    
    # Repeat LLM-mapper generation + validation until target is reached
    success_count = 0
    attempt_count = 0
    successful_runs = []
    
    print(f"\nRunning experiments (target: {args.num_runs} successes)...")
    
    while success_count < args.num_runs and attempt_count < args.max_attempts:
        attempt_count += 1
        print(f"\n--- Attempt {attempt_count} ---")
        
        run_dir = os.path.join(LLM_RESULTS_ROOT, f"run_{attempt_count:03d}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Step 1: Generate LLM mapper
        print("[Step 1] generate_llm_symbol_mapper.py")
        mapper_args = [
            "--root-path", PROJECT_ROOT,
            "--output-dir", os.path.join(run_dir, "stage1"),
            "--force-regenerate"
        ]
        exit_code = run_script(GENERATE_MAPPER_SCRIPT, mapper_args)
        if exit_code != 0:
            print(f"[WARN] Attempt {attempt_count}: generate_llm_symbol_mapper failed")
            continue
        
        stage1_dir = os.path.join(run_dir, "stage1")
        if not check_llm_success(stage1_dir):
            print(f"[WARN] Attempt {attempt_count}: LLM call failed (not counted)")
            continue
        
        success_count += 1
        print(f"[SUCCESS] #{success_count}/{args.num_runs}")
        
        # Copy generated mapper to default location so downstream scripts find it
        generated_mapper_path = os.path.join(run_dir, "stage1", "stage1_generated_mapper.py")
        default_mapper_dir = os.path.join(RESULTS_DIR, "llm_generated_symbxrl", "stage1")
        os.makedirs(default_mapper_dir, exist_ok=True)
        if os.path.exists(generated_mapper_path):
            shutil.copy(generated_mapper_path, os.path.join(default_mapper_dir, "stage1_generated_mapper.py"))
            print(f"[INFO] Mapper copied to default location")
        
        # Step 2: PPO validation
        print("[Step 2] validate_ppo.py")
        exit_code = run_validate_ppo()
        if exit_code != 0:
            print(f"[ERROR] Attempt {attempt_count}: validate_ppo failed")
            success_count -= 1
            continue
        
        # Step 3: LLM-RULE validation
        print("[Step 3] validate_ppo_llm_rule.py")
        llm_rule_args = ["--root-path", PROJECT_ROOT]
        exit_code = run_script(VALIDATE_PPO_LLM_RULE_SCRIPT, llm_rule_args)
        if exit_code != 0:
            print(f"[ERROR] Attempt {attempt_count}: validate_ppo_llm_rule failed")
            success_count -= 1
            continue
        
        # Save this run's result
        rule_result_path = os.path.join(RESULTS_DIR, "rule", "rule_llm_agent_results.npz")
        if os.path.exists(rule_result_path):
            dest_path = os.path.join(run_dir, "rule_llm_agent_results.npz")
            shutil.copy(rule_result_path, dest_path)
            successful_runs.append(run_dir)
            print(f"[INFO] Saved result to {dest_path}")
    
    print(f"\nDone. Attempts: {attempt_count}, Successes: {success_count}")
    
    if success_count < args.num_runs:
        print(f"[ERROR] Could not reach {args.num_runs} successes within {args.max_attempts} attempts")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("          Batch Experiment Complete!")
    print("="*70)
    print(f"Successful runs: {success_count}")
    print(f"Total attempts:  {attempt_count}")


if __name__ == "__main__":
    main()
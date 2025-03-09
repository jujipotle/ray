#!/usr/bin/env python3
"""
Script to run a benchmark sweep for prefix-aware serving.
This script runs a series of benchmarks with different prefix reuse rates or prefix length ratios.
"""

import argparse
import json
import os
import requests
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

DEFAULT_CONFIG = {
    # Server Info
    "host": "127.0.0.1",
    "router_port": 8000,
    "worker_ports": "8001,8002,8003,8004",
    # "worker_ports": "8001",

    # Model Info
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "gpu_type": "L4",
    "num_servers": 4,

    # Benchmark Info
    "sweep_types": "prefix_length_ratio",
    "num_points": 10, # 11 default
    "total_requests": 512, # 1024 default; total requests per server
    "total_prompt_len": 1088, # 2176 default; total prompt length (system + question)
    "output_len": 16, # 256 default; output length
    "max_concurrency": 32, # 32 default; max concurrency per server
    "prefix_cached": True,
    "default_system_prompt_ratio": 2048/2176,
    "default_num_groups_ratio": 64/1024,
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a benchmark sweep for prefix-aware serving")

    # Server info
    parser.add_argument("--host", type=str, default=DEFAULT_CONFIG["host"], help="Host")
    parser.add_argument("--router-port", type=int, default=DEFAULT_CONFIG["router_port"], help="Router port")
    parser.add_argument("--worker-ports", type=str, default=DEFAULT_CONFIG["worker_ports"], help="Comma-separated list of worker ports")

    # Model info
    parser.add_argument("--model-name", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name")
    parser.add_argument("--gpu-type", type=str, default=DEFAULT_CONFIG["gpu_type"], help="GPU type (e.g., A100, H100)")
    parser.add_argument("--num-servers", type=int, default=DEFAULT_CONFIG["num_servers"], help="Number of servers")
    # Benchmark info
    parser.add_argument("--sweep-type", type=str, default=DEFAULT_CONFIG["sweep_types"], 
                        choices=["prefix_reuse_rate", "prefix_length_ratio"], 
                        help="Type of sweep to run")
    parser.add_argument("--num-points", type=int, default=DEFAULT_CONFIG["num_points"], help="Number of points in the sweep")
    parser.add_argument("--total-requests", type=int, default=DEFAULT_CONFIG["total_requests"], help="Total number of requests per server")
    parser.add_argument("--total-prompt-len", type=int, default=DEFAULT_CONFIG["total_prompt_len"], help="Total prompt length (system + question) per server")
    parser.add_argument("--output-len", type=int, default=DEFAULT_CONFIG["output_len"], help="Output length")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_CONFIG["max_concurrency"], help="Maximum concurrency per server")
    parser.add_argument("--is-prefix-cached", type=bool, default=DEFAULT_CONFIG["prefix_cached"], help="Whether prefix caching is enabled")
    parser.add_argument("--default-system-prompt-ratio", type=float, default=DEFAULT_CONFIG["default_system_prompt_ratio"], help="Default system prompt ratio")
    parser.add_argument("--default-num-groups-ratio", type=float, default=DEFAULT_CONFIG["default_num_groups_ratio"], help="Default number of groups ratio")
    return parser.parse_args()

def generate_sweep_values(sweep_type, num_points, total_requests):
    """Generate values for the sweep based on the type."""
    if sweep_type == "prefix_reuse_rate":
        return np.geomspace(1 / total_requests, 1.0, num=num_points).tolist()
    elif sweep_type == "prefix_length_ratio":
        return np.linspace(0.0, 1.0, num=num_points).tolist()
    else:
        raise ValueError("Invalid sweep type")

def calculate_sweep_parameters(sweep_type, r, total_requests, total_prompt_len, default_system_prompt_ratio, default_num_groups_ratio):
    """Calculate parameters for a single benchmark run based on the sweep type and value."""
    epsilon = 1e-6
    if sweep_type == "prefix_reuse_rate":
        num_groups = max(1, int(1 / r + epsilon))
        prompts_per_group = total_requests // num_groups
        system_prompt_len = int(total_prompt_len * default_system_prompt_ratio)
        question_len = total_prompt_len - system_prompt_len
    elif sweep_type == "prefix_length_ratio":
        system_prompt_len = int(total_prompt_len * r)
        question_len = total_prompt_len - system_prompt_len
        num_groups = max(1, int(total_requests * default_num_groups_ratio))
        prompts_per_group = total_requests // num_groups
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")
    
    return num_groups, prompts_per_group, system_prompt_len, question_len

def reset_prefix_caches(host, worker_ports):
    """Reset prefix caches for all workers"""
    print("Resetting prefix caches for all workers...")

    # Reset worker caches
    for port in worker_ports.split(","):
        try:
            response = requests.post(f"http://{host}:{port.strip()}/reset_prefix_cache")
            print(f"Worker cache reset on port {port}: {response.status_code}")
        except Exception as e:
            print(f"Failed to reset worker cache on port {port}: {e}")

def run_single_benchmark(sweep_type, r, args, num_groups, prompts_per_group, system_prompt_len, question_len):
    """Run a single benchmark with the given parameters and return the result."""
    print(f"\nRunning benchmark with {sweep_type} = {r:.4f} ...")

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"sweep_{sweep_type}_{r:.4f}_{now}.jsonl"
    
    # Reset prefix caches before the benchmark
    print(f"Resetting prefix caches before benchmark ...")
    reset_prefix_caches(args.host, args.worker_ports)
    time.sleep(5)

    """Run the bench_serving.py script with the given parameters."""
    cmd = [
        "python", "-m", "bench_serving",
        "--backend", "vllm",
        "--model", args.model_name,
        "--host", str(args.host),
        "--port", str(args.router_port),
        "--dataset-name", "generated-shared-prefix",
        "--gen-num-groups", str(num_groups),
        "--gen-prompts-per-group", str(prompts_per_group * args.num_servers),
        "--gen-system-prompt-len", str(system_prompt_len),
        "--gen-question-len", str(question_len),
        "--gen-output-len", str(args.output_len),
        "--output-file", str(output_file),
        "--max-concurrency", str(args.max_concurrency * args.num_servers)
    ]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    with open(output_file, "r") as f:
        line = f.readline().strip()
        result = json.loads(line)
        os.remove(output_file)

    # Add additional metadata to the result.
    result.update({
        "model_name": args.model_name,
        "gpu_type": args.gpu_type,
        "num_servers": args.num_servers,
        "is_prefix_cached": args.is_prefix_cached,
        "sweep_type": sweep_type,
        "sweep_value": r,
        "num_groups": num_groups,
        "prompts_per_group": prompts_per_group,
        "system_prompt_len": system_prompt_len,
        "question_len": question_len,
        "output_len": args.output_len,
        "total_prompt_len": system_prompt_len + question_len,
        "total_requests": num_groups * prompts_per_group,
        "max_concurrency": args.max_concurrency,
    })
    
    return result

def save_results_to_csv(sweep_results):
    """Save the benchmark results to a CSV file."""
    # Define CSV column order
    server_params = ["gpu_type", "model_name", "num_servers", "is_prefix_cached"]
    sweep_params = ["sweep_type", "sweep_value", "num_groups", "prompts_per_group", "system_prompt_len", "question_len", "output_len", "total_prompt_len", "total_requests", "max_concurrency"]
    result_keys = [
        "duration",
        "completed",
        # "total_input_tokens",
        # "total_output_tokens",
        # "total_output_tokens_retokenized",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
        # "input_lens",
        # "output_lens",
        # "ttfts",
        # "itls",
        # "generated_texts",
        # "errors",
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
    ]

    ordered_columns = server_params + sweep_params + result_keys
    df = pd.DataFrame(sweep_results)
    df = df[[col for col in ordered_columns if col in df.columns]]
    
    # Round numeric values to 4 decimal places
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    df[numeric_columns] = df[numeric_columns].round(4)

    csv_file = "sweep_results.csv"
    df.to_csv(csv_file, mode='a', index=False)
    print(f"\nAppended sweep results to {csv_file}")

def main():
    """Main function to run the benchmark sweep."""
    args = parse_arguments()
    
    # Generate sweep values
    sweep_values = generate_sweep_values(args.sweep_type, args.num_points, args.total_requests)
    
    # Store results for all sweep points
    sweep_results = []
    
    for r in sweep_values:
        # Calculate parameters for this sweep point
        num_groups, prompts_per_group, system_prompt_len, question_len = calculate_sweep_parameters(
            args.sweep_type, r, args.total_requests, args.total_prompt_len,
            default_system_prompt_ratio=DEFAULT_CONFIG["default_system_prompt_ratio"], 
            default_num_groups_ratio=DEFAULT_CONFIG["default_num_groups_ratio"]
        )
        
        result = run_single_benchmark(
            args.sweep_type, r, args, num_groups, prompts_per_group, 
            system_prompt_len, question_len
        )
        
        sweep_results.append(result)

    save_results_to_csv(sweep_results)
    
if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        sys.exit(1)
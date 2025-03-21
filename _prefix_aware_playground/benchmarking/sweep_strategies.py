"""
python sweep_strategies.py
For each routing strategy, will run 
"""

#!/usr/bin/env python3
"""
Script to run a benchmark sweep for sharegpt with different routing strategies.
This script runs a series of benchmarks with different routing strategies.
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
    "worker_ports": "8001",
    "router_strategies": ["prefix_aware", "pow_of_2"],

    # Model Info
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "gpu_type": "L4",
    "num_servers": 1,
    "is_prefix_cached": True,

    # Benchmark Info
    "benchmark_label": "1-gpu_1-concurrency_with-warmup",
    "dataset_name": "sharegpt",
    "max_concurrency": 1,  # Max concurrency (total)
    "output_len": 32,
    "with_warmup": "False",

    # Generate Shared Prefix Info
    "gen-num-groups": 1,
    "gen-prompts-per-group": 8,
    "gen-system-prompt-len": 1024,
    "gen-question-len": 512,

    # ShareGPT Info
    "num_prompts": 100,  # Number of prompts to sample from ShareGPT
    "max_conversations": 100,  # Max conversations to include from ShareGPT; num_unique_prefixes is approximately max_conversations / 10. To aim for num_unique_prefixes = num_prompts / 10, set max_conversations = num_prompts.
    "dataset_path": "/home/ray/default/work/ray/_prefix_aware_playground/benchmarking/sharegpt.json",  # Path to ShareGPT dataset
}



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a benchmark sweep for ShareGPT with different routing strategies")

    # Server info
    parser.add_argument("--host", type=str, default=DEFAULT_CONFIG["host"], help="Host")
    parser.add_argument("--router-port", type=int, default=DEFAULT_CONFIG["router_port"], help="Router port")
    parser.add_argument("--worker-ports", type=str, default=DEFAULT_CONFIG["worker_ports"], help="Comma-separated list of worker ports")
    parser.add_argument("--router-strategies", type=str, nargs="+", default=DEFAULT_CONFIG["router_strategies"], 
                        help="List of router strategies to benchmark")

    # Model info
    parser.add_argument("--model-name", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name")
    parser.add_argument("--gpu-type", type=str, default=DEFAULT_CONFIG["gpu_type"], help="GPU type (e.g., A100, H100)")
    parser.add_argument("--num-servers", type=int, default=DEFAULT_CONFIG["num_servers"], help="Number of servers")
    parser.add_argument("--is-prefix-cached", type=bool, default=DEFAULT_CONFIG["is_prefix_cached"], 
                        help="Whether prefix caching is enabled")

    # Benchmark info
    parser.add_argument("--benchmark-label", type=str, default=DEFAULT_CONFIG["benchmark_label"], help="Benchmark label")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_CONFIG["dataset_name"], help="Dataset name")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_CONFIG["max_concurrency"], 
                        help="Maximum concurrency (total)")
    parser.add_argument("--output-len", type=int, default=DEFAULT_CONFIG["output_len"], 
                        help="Output length")
    parser.add_argument("--with-warmup", type=str, default=DEFAULT_CONFIG["with_warmup"], 
                        help="Whether to run warmup")

    # Generate Shared Prefix info
    parser.add_argument("--gen-num-groups", type=int, default=DEFAULT_CONFIG["gen-num-groups"],
                        help="Number of groups for generate-shared-prefix dataset")
    parser.add_argument("--gen-prompts-per-group", type=int, default=DEFAULT_CONFIG["gen-prompts-per-group"],
                        help="Prompts per group for generate-shared-prefix dataset")
    parser.add_argument("--gen-system-prompt-len", type=int, default=DEFAULT_CONFIG["gen-system-prompt-len"],
                        help="System prompt length for generate-shared-prefix dataset")
    parser.add_argument("--gen-question-len", type=int, default=DEFAULT_CONFIG["gen-question-len"],
                        help="Question length for generate-shared-prefix dataset")

    # ShareGPT info
    parser.add_argument("--max-conversations", type=int, default=DEFAULT_CONFIG["max_conversations"], 
                        help="Maximum number of conversations to include from ShareGPT")
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_CONFIG["num_prompts"], 
                        help="Number of prompts to sample from ShareGPT")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_CONFIG["dataset_path"],
                        help="Path to ShareGPT dataset")
    
    return parser.parse_args()

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

def restart_server_with_strategy(host, router_port, worker_ports, strategy):
    """Restart the server with a specific routing strategy."""
    print(f"\nRestarting server with routing strategy: {strategy}")
    
    # Kill existing server process if running
    try:
        subprocess.run(["pkill", "-f", f"uvicorn.*{router_port}"], check=False)
        time.sleep(2)  # Give it time to shut down
    except Exception as e:
        print(f"Error stopping server: {e}")

    original_dir = os.getcwd()

    # Change to new_backend directory
    os.chdir("old_backend")
    # Start server with the specified strategy
    cmd = [
        "python", "-m", "server",
        "--host", host,
        "--port", str(router_port),
        "--worker-ports", worker_ports,
        "--policy", strategy
    ]
    print("Starting server with command:", " ".join(cmd))
    # Start server in background
    server_process = subprocess.Popen(cmd)
    
    # Change back to original directory
    os.chdir(original_dir)
    
    # Wait for server to start - give it more time and retry health checks
    print("Waiting for server to start...")
    max_retries = 10
    retry_interval = 3  # seconds
    
    for i in range(max_retries):
        time.sleep(retry_interval)
        try:
            response = requests.get(f"http://{host}:{router_port}/health")
            if response.status_code == 200:
                print(f"Server started successfully with strategy {strategy}")
                return server_process
            else:
                print(f"Health check attempt {i+1}/{max_retries}: Status code {response.status_code}")
        except Exception as e:
            print(f"Health check attempt {i+1}/{max_retries}: {e}")
    
    print(f"Failed to start server with strategy {strategy} after {max_retries} attempts")
    return None

def run_single_benchmark(strategy, args):
    """Run a single benchmark with the given routing strategy and return the result."""
    print(f"\nRunning benchmark with routing strategy = {strategy} ...")

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"sharegpt_{strategy}_{now}.jsonl"
    
    # Reset prefix caches before the benchmark
    print(f"Resetting prefix caches before benchmark ...")
    reset_prefix_caches(args.host, args.worker_ports)
    time.sleep(5)

    if args.dataset_name == "generate-shared-prefix":
        cmd = [
            "python", "-m", "benchmark",
            "--backend", "vllm",
            "--model", args.model_name,
            "--host", str(args.host),
            "--port", str(args.router_port),
            "--dataset-name", "generated-shared-prefix",
            "--output-file", str(output_file),
            "--output-len", str(args.output_len),
            "--max-concurrency", str(args.max_concurrency),
            "--with-warmup", str(args.with_warmup),

            # Parameters specific to dataset
            "--gen-num-groups", str(args.gen_num_groups),
            "--gen-prompts-per-group", str(args.gen_prompts_per_group),
            "--gen-system-prompt-len", str(args.gen_system_prompt_len),
            "--gen-question-len", str(args.gen_question_len),
        ]
    elif args.dataset_name == "sharegpt":
        cmd = [
            "python", "-m", "benchmark",
            "--backend", "vllm",
            "--model", args.model_name,
            "--host", str(args.host),
            "--port", str(args.router_port),
            "--dataset-name", "sharegpt",
            "--dataset-path", args.dataset_path,
            "--output-file", str(output_file),
            "--output-len", str(args.output_len),
            "--max-concurrency", str(args.max_concurrency),
            "--with-warmup", str(args.with_warmup),

            # Parameters specific to dataset
            "--max-conversations", str(args.max_conversations),
            "--num-prompts", str(args.num_prompts),
        ]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    with open(output_file, "r") as f:
        line = f.readline().strip()
        result = json.loads(line)
        os.remove(output_file)

    # Add additional metadata to the result.
    result.update({
        "gpu_type": args.gpu_type,
        "model_name": args.model_name,
        "num_servers": args.num_servers,
        "is_prefix_cached": args.is_prefix_cached,
        "benchmark_label": args.benchmark_label,
        "router_strategy": strategy,
        "output_len": args.output_len,
        "max_concurrency": args.max_concurrency,
        "with_warmup": args.with_warmup,
    })
    if args.dataset_name == "generate-shared-prefix":
        result.update({
            "num_groups": args.gen_num_groups,
            "prompts_per_group": args.gen_prompts_per_group,
            "system_prompt_len": args.gen_system_prompt_len,
            "question_len": args.gen_question_len,
        })
    elif args.dataset_name == "sharegpt":
        result.update({
            "max_conversations": args.max_conversations,
            "num_prompts": args.num_prompts,
        })
        
    return result

def save_results_to_csv(sweep_results, args):
    """Save the benchmark results to a CSV file."""
    # Define CSV column order
    shared_params = ["gpu_type", "model_name", "num_servers", "is_prefix_cached", "benchmark_label", "router_strategy", "output_len", "max_concurrency", "with_warmup"]
    if args.dataset_name == "generate-shared-prefix":
        dataset_params = ["num_groups", "prompts_per_group", "system_prompt_len", "question_len"]
    elif args.dataset_name == "sharegpt":
        dataset_params = ["max_conversations", "num_prompts"]
    result_keys = [
        "duration",
        "completed",
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
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
    ]

    ordered_columns = shared_params + dataset_params + result_keys
    df = pd.DataFrame(sweep_results)
    df = df[[col for col in ordered_columns if col in df.columns]]
    
    # Round numeric values to 4 decimal places
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    df[numeric_columns] = df[numeric_columns].round(4)

    csv_file = f"results/{args.dataset_name}_sweep_results.csv"
    df.to_csv(csv_file, mode='a', index=False)
    print(f"\nAppended sweep results to {csv_file}")

def main():
    """Main function to run the benchmark sweep."""
    args = parse_arguments()
    
    # Define sweep configurations
    sweeps_configs = [
        {"worker_ports": "8001,8002,8003,8004", "num_servers": 4, "benchmark_label": "custom-prefix-router_long-conversations_4-gpu_40-concurrency", "max_concurrency": 40, "with_warmup": "False", "num_prompts": 1000, "max_conversations": 10000},
        # {"worker_ports": "8001,8002,8003,8004", "num_servers": 4, "benchmark_label": "custom-prefix-router_long-conversations_4-gpu_40-concurrency_with-warmup", "max_concurrency": 40, "with_warmup": "True", "num_prompts": 1000, "max_conversations": 10000},
    ]
    
    # Loop through each sweep configuration
    for config_idx, sweep_config in enumerate(sweeps_configs):
        print(f"\n{'='*80}")
        print(f"Running sweep configuration {config_idx+1}/{len(sweeps_configs)}")
        print(f"Configuration: {sweep_config}")
        print(f"{'='*80}\n")
        
        # Update args with the current sweep configuration
        for key, value in sweep_config.items():
            setattr(args, key, value)
        
        # Store results for all strategies in this configuration
        sweep_results = []
        server_process = None
        
        try:
            for strategy in args.router_strategies:
                # Stop previous server if running
                if server_process:
                    print(f"Stopping previous server process...")
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print("Server didn't terminate gracefully, killing it...")
                        server_process.kill()
                    time.sleep(2)
                
                # Start server with the current strategy
                server_process = restart_server_with_strategy(
                    args.host, args.router_port, args.worker_ports, strategy
                )
                
                if not server_process:
                    print(f"Failed to start server with strategy {strategy}, skipping...")
                    continue
                
                # Run benchmark with this strategy
                try:
                    result = run_single_benchmark(strategy, args)
                    sweep_results.append(result)
                except Exception as e:
                    print(f"Error running benchmark with strategy {strategy}: {e}")
        
        finally:
            # Clean up: stop server if still running
            if server_process:
                print("Stopping server...")
                try:
                    server_process.terminate()
                    server_process.wait(timeout=5)
                except:
                    print("Forcing server shutdown...")
                    server_process.kill()
        
        # Save results for this configuration to CSV
        if sweep_results:
            save_results_to_csv(sweep_results, args)
        else:
            print(f"No benchmark results were collected for configuration {config_idx+1}. Check for errors above.")
    
if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        sys.exit(1)
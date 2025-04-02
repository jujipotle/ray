"""
python sweep_strategies.py
For each routing strategy, will call "serve run config.yaml"
"""

#!/usr/bin/env python3
"""
Script to run a benchmark sweep for sharegpt with different routing strategies.
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
import tempfile
import yaml

DEFAULT_CONFIG = {
    # Server Info
    "host": "127.0.0.1",
    "router_port": 8000,
    # "worker_ports": "8001,8002,8003,8004",
    "scheduler_strategies_dict": {
        # "fake": "ray.serve._private.replica_scheduler.fake_replica_scheduler.FakeReplicaScheduler",
        # "random": "ray.serve._private.replica_scheduler.random_scheduler.RandomReplicaScheduler",
        # "round_robin": "ray.serve._private.replica_scheduler.round_robin_scheduler.RoundRobinReplicaScheduler",
        # "pow_of_2": "ray.serve._private.replica_scheduler.llm_pow_2_scheduler.LLMPowerOfTwoChoicesReplicaScheduler",
        "prefix_aware": "ray.serve._private.replica_scheduler.prefix_aware_scheduler.PrefixAwareReplicaScheduler",
    },

    # Model Info
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "gpu_type": "L4",
    "num_servers": 4,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    # Benchmark Info
    "benchmark_label": "no-overhead-no-tracking",
    "dataset_name": "sharegpt",
    "max_concurrency": 40,  # Max concurrency (total)
    "min_output_len": 10,
    "max_output_len": 200,
    "with_warmup": False,
    "disable_ignore_eos": False, # If false, will ignore EOS token and generate output_len tokens. If True, will stop at EOS token. Use false for more control.
    "request_rate": 100,

    # Generate Shared Prefix Info
    "gen-num-groups": 1,
    "gen-prompts-per-group": 8,
    "gen-system-prompt-len": 1024,
    "gen-question-len": 512,

    # ShareGPT Info
    "num_prompts": 1000,  # Number of prompts to sample from ShareGPT
    "max_conversations": 10000,  # Max conversations to include from ShareGPT; num_unique_prefixes is approximately 3/100 * max_conversations.
    # To aim for average_prompts_per_prefix = 3, set max_conversations = 10 * num_prompts.
    "dataset_path": "/home/ray/default/work/ray/_prefix_aware_playground/shared/sharegpt.json",  # Path to ShareGPT dataset
    # "dataset_path": ""
}



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a benchmark sweep for ShareGPT with different routing strategies")

    # Server info
    parser.add_argument("--host", type=str, default=DEFAULT_CONFIG["host"], help="Host")
    parser.add_argument("--router-port", type=int, default=DEFAULT_CONFIG["router_port"], help="Router port")
    # parser.add_argument("--worker-ports", type=str, default=DEFAULT_CONFIG["worker_ports"], help="Comma-separated list of worker ports")
    parser.add_argument("--scheduler-strategies-dict", type=str, nargs="+", default=DEFAULT_CONFIG["scheduler_strategies_dict"], 
                        help="List of scheduler strategies paths to benchmark")

    # Model info
    parser.add_argument("--model-name", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name")
    parser.add_argument("--gpu-type", type=str, default=DEFAULT_CONFIG["gpu_type"], help="GPU type (e.g., A100, H100)")
    parser.add_argument("--num-servers", type=int, default=DEFAULT_CONFIG["num_servers"], help="Number of servers")
    parser.add_argument("--enable-prefix-caching", type=bool, default=DEFAULT_CONFIG["enable_prefix_caching"], 
                        help="Whether prefix caching is enabled")
    parser.add_argument("--enable-chunked-prefill", type=bool, default=DEFAULT_CONFIG["enable_chunked_prefill"], 
                        help="Whether chunked prefill is enabled")

    # Benchmark info
    parser.add_argument("--benchmark-label", type=str, default=DEFAULT_CONFIG["benchmark_label"], help="Benchmark label")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_CONFIG["dataset_name"], help="Dataset name")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_CONFIG["max_concurrency"], 
                        help="Maximum concurrency (total)")
    parser.add_argument("--request-rate", type=Any, default=DEFAULT_CONFIG["request_rate"], 
                        help="Request rate")
    parser.add_argument("--min-output-len", type=int, default=DEFAULT_CONFIG["min_output_len"], 
                        help="Minimum output length")
    parser.add_argument("--max-output-len", type=int, default=DEFAULT_CONFIG["max_output_len"], 
                        help="Maximum output length")
    parser.add_argument("--with-warmup", type=bool, default=DEFAULT_CONFIG["with_warmup"], 
                        help="Whether to run warmup")
    parser.add_argument("--disable-ignore-eos", type=bool, default=DEFAULT_CONFIG["disable_ignore_eos"], 
                        help="Whether to disable ignoring EOS")

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

# def reset_prefix_caches(host, worker_ports):
#     """Reset prefix caches for all workers"""
#     print("Resetting prefix caches for all workers...")

#     # Reset worker caches
#     for port in worker_ports.split(","):
#         try:
#             response = requests.post(f"http://{host}:{port.strip()}/reset_prefix_cache")
#             print(f"Worker cache reset on port {port}: {response.status_code}")
#         except Exception as e:
#             print(f"Failed to reset worker cache on port {port}: {e}")

def restart_server_with_strategy(strategy, args):
    """Restart the server with a specific routing strategy."""
    print(f"\nRestarting server with routing strategy: {strategy}")
    
    # # Kill existing server process if running
    # print(f"Shutting down existing server...")
    # try:
    #     # Shut down the Ray Serve instance
    #     subprocess.run(["serve", "shutdown"], check=False)
    #     time.sleep(2)  # Give it time to shut down
    # except Exception as e:
    #     print(f"Error stopping server: {e}")

    # Create a temporary JSON config for passing arguments
    config = {
        "applications": [
            {
                "args": {
                    "llm_configs": [
                        {
                            "model_loading_config": {
                                "model_id": args.model_name,
                                "model_source": args.model_name
                            },
                            "accelerator_type": args.gpu_type,
                            "engine_kwargs": {
                                "disable_log_requests": True,
                                "enable_prefix_caching": args.enable_prefix_caching,
                                "enable_chunked_prefill": args.enable_chunked_prefill
                            },
                            "deployment_config": {
                                "autoscaling_config": {
                                    "min_replicas": args.num_servers,
                                    "max_replicas": args.num_servers,
                                    "initial_replicas": args.num_servers
                                },
                            },
                            "replica_scheduler_cls_path": args.scheduler_strategies_dict[strategy]
                        }
                    ]
                },
                "import_path": "ray.serve.llm:build_openai_app",
                "name": "llm_app",
                "route_prefix": "/"
            }
        ]
    }
    # Write to a temporary JSON file
    temp_path = "temp_config.yaml"
    with open(temp_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Starting server with strategy '{strategy}': {args.scheduler_strategies_dict[strategy]}")
    cmd = ["serve", "run", temp_path]
    print(f"Executing: {' '.join(cmd)}")
    
    # Open log files for writing (will overwrite if they exist)
    stdout_log = open(f"logs/{strategy}_stdout.log", "w")
    stderr_log = open(f"logs/{strategy}_stderr.log", "w")
    
    server_process = subprocess.Popen(
        cmd,
        # stdout=stdout_log,
        # stderr=stderr_log
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for server to start - give it more time and retry health checks
    print("Waiting for server to start...")
    max_retries = 20
    retry_interval = 5  # seconds
    for i in range(max_retries):
        time.sleep(retry_interval)
        try:
            response = requests.get(f"http://{args.host}:{args.router_port}/v1/models")
            if response.status_code == 200:
                print(f"Health check attempt {i+1}/{max_retries}: Server started successfully with strategy {strategy}")
                return server_process, stdout_log, stderr_log
            else:
                print(f"Health check attempt {i+1}/{max_retries}: Status code {response.status_code}")
        except Exception as e:
            print(f"Health check attempt {i+1}/{max_retries}: {e}")
    
    print(f"Failed to start server with strategy {strategy} after {max_retries} attempts")
    return None, None, None

def run_single_benchmark(strategy, args):
    """Run a single benchmark with the given routing strategy and return the result."""
    print(f"\nRunning benchmark with routing strategy = {strategy} ...")

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"sharegpt_{strategy}_{now}.jsonl"
    
    # Reset prefix caches before the benchmark
    # print(f"Resetting prefix caches before benchmark ...")
    # reset_prefix_caches(args.host, args.worker_ports)
    # time.sleep(5)

    # if args.dataset_name == "generate-shared-prefix":
    #     cmd = [
    #         "python", "-m", "benchmark",
    #         "--backend", "vllm",
    #         "--model", args.model_name,
    #         "--host", str(args.host),
    #         "--port", str(args.router_port),
    #         "--dataset-name", "generated-shared-prefix",
    #         "--output-file", str(output_file),
    #         "--output-len", str(args.output_len),
    #         "--max-concurrency", str(args.max_concurrency),
    #         "--with-warmup", str(args.with_warmup),
    #         "--disable-ignore-eos", str(args.disable_ignore_eos),

    #         # Parameters specific to dataset
    #         "--gen-num-groups", str(args.gen_num_groups),
    #         "--gen-prompts-per-group", str(args.gen_prompts_per_group),
    #         "--gen-system-prompt-len", str(args.gen_system_prompt_len),
    #         "--gen-question-len", str(args.gen_question_len),
    #     ]
    if args.dataset_name == "sharegpt":
        cmd = [
            "python", "-m", "benchmark",
            "--backend", "vllm",
            "--model", args.model_name,
            "--host", str(args.host),
            "--port", str(args.router_port),
            "--dataset-name", "sharegpt",
            "--dataset-path", args.dataset_path,
            "--output-file", str(output_file),
            "--min-output-len", str(args.min_output_len),
            "--max-output-len", str(args.max_output_len),
            "--max-concurrency", str(args.max_concurrency),
            "--request-rate", str(args.request_rate),
            "--with-warmup", str(args.with_warmup),
            "--disable-ignore-eos", str(args.disable_ignore_eos),

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
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "benchmark_label": args.benchmark_label,
        "scheduler_strategy": strategy,
        "min_output_len": args.min_output_len,
        "max_output_len": args.max_output_len,
        "max_concurrency": args.max_concurrency,
        "request_rate": args.request_rate,
        "with_warmup": args.with_warmup,
        "disable_ignore_eos": args.disable_ignore_eos,
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

def save_results_to_csv(results, args):
    """Save the benchmark results to a CSV file."""
    # Define CSV column order
    shared_params = ["gpu_type", "model_name", "num_servers", "enable_prefix_caching", "enable_chunked_prefill", "benchmark_label", "scheduler_strategy", "min_output_len", "max_output_len", "max_concurrency", "request_rate", "with_warmup", "disable_ignore_eos"]
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
    df = pd.DataFrame(results)
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
        {},
        # {},
        # {},
        # {},
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
        
        # try:
        for strategy in args.scheduler_strategies_dict.keys():
            # Store the returned log file handles
            server_process, stdout_log, stderr_log = restart_server_with_strategy(strategy, args)
            
            # Run benchmark with this strategy
            try:
                result = run_single_benchmark(strategy, args)
                # Save result immediately after each strategy run
                save_results_to_csv([result], args)
            except Exception as e:
                print(f"Error running benchmark with strategy {strategy}: {e}")        
            finally:
                print("Sleeping for 10 seconds (to allow load distribution to be written to file)...")
                time.sleep(10)
                # Close log files
                stdout_log.close()
                stderr_log.close()
                
                # Clean up: stop server if still running
                subprocess.run(["serve", "shutdown", "--yes"], check=False)
                print("Sleeping for 5 seconds (to allow server to shut down)...")
                time.sleep(5)  # Give it time to shut down
        

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        sys.exit(1)
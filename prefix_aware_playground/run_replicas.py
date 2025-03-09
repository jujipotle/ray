#!/usr/bin/env python3
"""
Script to run VLLM worker replicas.
This script starts multiple VLLM servers on different ports.

Usage:
    python run_replicas.py --host 127.0.0.1 --worker-ports 8001,8002,8003,8004 --gpu-indices 0,1,2,3 --model-name "Qwen/Qwen2.5-1.5B-Instruct" --enable-prefix-caching
"""

import argparse
import os
import subprocess
import sys
import time
from typing import List
import requests
def parse_args():
    parser = argparse.ArgumentParser(description="Run VLLM worker replicas")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="Host to bind the servers to")
    parser.add_argument("--worker-ports", type=str, default="8001,8002,8003,8004", 
                        help="Comma-separated list of worker ports")
    parser.add_argument("--gpu-indices", type=str, default="0,1,2,3",
                        help="Comma-separated list of GPU indices to use")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name to load")
    parser.add_argument("--enable-prefix-caching", action="store_true", 
                        dest="enable_prefix_caching", default=True,
                        help="Enable prefix caching in VLLM")
    parser.add_argument("--no-enable-prefix-caching", action="store_false", 
                        dest="enable_prefix_caching",
                        help="Disable prefix caching in VLLM")
    return parser.parse_args()

def start_replicas(
    host: str,
    worker_ports: List[int],
    gpu_indices: List[int],
    model_name: str,
    enable_prefix_caching: bool,
    disable_log_requests: bool
) -> List[subprocess.Popen]:
    """Start VLLM worker replicas on the specified ports and GPUs."""
    processes = []
    
    for i, (port, gpu_idx) in enumerate(zip(worker_ports, gpu_indices)):
        cmd = [
            "vllm", "serve", model_name,
            "--host", host,
            "--port", str(port),
            "--enable-prefix-caching" if enable_prefix_caching else "--no-enable-prefix-caching",
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        env["VLLM_SERVER_DEV_MODE"] = "1"
        
        print(f"Starting VLLM worker on port {port} using GPU {gpu_idx}")
        process = subprocess.Popen(
            cmd, env=env
        )
        processes.append(process)
    
    return processes

def main():
    args = parse_args()
    
    worker_ports = [int(port) for port in args.worker_ports.split(",")]
    gpu_indices = [int(idx) for idx in args.gpu_indices.split(",")]
    
    processes = start_replicas(
        host=args.host,
        worker_ports=worker_ports,
        gpu_indices=gpu_indices,
        model_name=args.model_name,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_log_requests=args.disable_log_requests
    )
    
    try:
        # Keep the script running until interrupted
        print("Press Ctrl+C to stop the workers")
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("Stopping workers...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        
        # Wait for processes to terminate
        for process in processes:
            process.wait()
        
        print("All workers stopped")

if __name__ == "__main__":
    main() 
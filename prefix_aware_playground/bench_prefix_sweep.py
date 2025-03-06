import argparse
import subprocess
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sweep prefix caching metrics for bench_serving.")
    
    # Server and model parameters
    parser.add_argument("--gpu-type", type=str, required=True, help="GPU type.")
    parser.add_argument("--model-name", type=str, required=True, help="Model name.")
    parser.add_argument("--base-url", type=str, required=True, help="Base URL of the server.")
    parser.add_argument("--is-prefix-cached", type=bool, required=True, help="Indicates if prefix caching is enabled.")
    
    # Sweep parameters
    parser.add_argument("--sweep-type", choices=["prefix_reuse_rate", "prefix_length_ratio"], required=True, help="Metric to sweep.")
    parser.add_argument("--num-points", type=int, default=11, help="Number of points to sweep.")
    parser.add_argument("--total-requests", type=int, default=1024, help="Total number of requests.")
    parser.add_argument("--total-prompt-len", type=int, default=2176, help="Total prompt length.")
    parser.add_argument("--output-len", type=int, default=256, help="Output length per request.")
    parser.add_argument("--max-concurrency", type=int, default=32, help="Maximum number of concurrent requests.")
    
    # Default scaling factors for maintaining proportionality
    parser.add_argument("--default-system-prompt-ratio", type=float, default=2048 / 2176, help="Default ratio of system prompt length to total prompt length.")
    parser.add_argument("--default-num-groups-ratio", type=float, default=64 / 1024, help="Default ratio of num_groups to total_requests.")
    
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
    """Calculate parameters for a specific sweep value."""
    if sweep_type == "prefix_reuse_rate":
        prompts_per_group = max(1, int(round(r * total_requests)))
        num_groups = max(1, total_requests // prompts_per_group)
        system_prompt_len = int(round(total_prompt_len * default_system_prompt_ratio))
        question_len = total_prompt_len - system_prompt_len
    elif sweep_type == "prefix_length_ratio":
        system_prompt_len = int(round(r * total_prompt_len))
        question_len = total_prompt_len - system_prompt_len
        num_groups = max(1, int(round(total_requests * default_num_groups_ratio)))
        prompts_per_group = max(1, total_requests // num_groups)
    else:
        raise ValueError(f"Invalid sweep type: {sweep_type}")
        
    return num_groups, prompts_per_group, system_prompt_len, question_len

def run_bench_serving(num_groups, prompts_per_group, system_prompt_len, question_len, output_len, base_url, output_file, max_concurrency):
    """Runs the bench_serving benchmark with the specified parameters."""
    cmd = [
        "python3", "-m", "bench_serving",
        "--backend", "vllm",
        "--base-url", base_url,
        "--dataset-name", "generated-shared-prefix",
        "--gen-num-groups", str(num_groups),
        "--gen-prompts-per-group", str(prompts_per_group),
        "--gen-system-prompt-len", str(system_prompt_len),
        "--gen-question-len", str(question_len),
        "--gen-output-len", str(output_len),
        "--num-prompts", str(num_groups * prompts_per_group),
        "--output-file", output_file,
        "--max-concurrency", str(max_concurrency)
    ]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_single_benchmark(sweep_type, r, args, num_groups, prompts_per_group, system_prompt_len, question_len):
    """Run a single benchmark with the given parameters and return the result."""
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"sweep_{sweep_type}_{r:.4f}_{now}.jsonl"

    print(f"\nRunning benchmark with {sweep_type} = {r:.4f} ...")
    run_bench_serving(
        num_groups=num_groups,
        prompts_per_group=prompts_per_group,
        system_prompt_len=system_prompt_len,
        question_len=question_len,
        output_len=args.output_len,
        base_url=args.base_url,
        output_file=output_file,
        max_concurrency=args.max_concurrency
    )

    with open(output_file, "r") as f:
        line = f.readline().strip()
        result = json.loads(line)
        print(result)

    # Add additional metadata to the result.
    result.update({
        "gpu_type": args.gpu_type,
        "model_name": args.model_name,
        "is_prefix_cached": args.is_prefix_cached,
        "sweep_type": sweep_type,
        "sweep_value": r,
        "num_groups": num_groups,
        "prompts_per_group": prompts_per_group,
        "system_prompt_len": system_prompt_len,
        "question_len": question_len,
        "output_len": args.output_len
    })

    os.remove(output_file)
    print(f"  {sweep_type} = {r:.4f} => median TTFT = {result.get('median_ttft_ms')} ms")
    
    return result

def save_results_to_csv(sweep_results):
    """Save the benchmark results to a CSV file."""
    # Define CSV column order
    server_params = ["gpu_type", "model_name", "is_prefix_cached"]
    sweep_params = ["sweep_type", "sweep_value", "num_groups", "prompts_per_group", "system_prompt_len", "question_len", "output_len", "max_concurrency"]
    result_keys = [
        "duration", "completed", "total_input_tokens", "total_output_tokens",
        "total_output_tokens_retokenized", "output_throughput", "median_ttft_ms",
        "median_itl_ms", "mean_e2e_latency_ms", "median_e2e_latency_ms"
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
    args = parse_arguments()
    sweep_values = generate_sweep_values(args.sweep_type, args.num_points, args.total_requests)
    sweep_results = []
    
    for r in sweep_values:
        num_groups, prompts_per_group, system_prompt_len, question_len = calculate_sweep_parameters(
            args.sweep_type, r, args.total_requests, args.total_prompt_len, 
            args.default_system_prompt_ratio, args.default_num_groups_ratio
        )
        
        result = run_single_benchmark(
            args.sweep_type, r, args, num_groups, prompts_per_group, 
            system_prompt_len, question_len
        )
        
        sweep_results.append(result)

    save_results_to_csv(sweep_results)

if __name__ == "__main__":
    main()
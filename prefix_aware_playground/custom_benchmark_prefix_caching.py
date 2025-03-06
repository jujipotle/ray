import requests
import re
import time
import json
import argparse
import pandas as pd
import random
from typing import List, Tuple
from transformers import PreTrainedTokenizerFast

# Constants
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def query_vllm_api(server_url: str, prompt: str, temperature: float, max_tokens: int):
    """ Sends a request to a vLLM server and returns the response and elapsed time. """
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {"Content-Type": "application/json"}

    try:
        start_time = time.time()
        response = requests.post(f"{server_url}/v1/completions", headers=headers, data=json.dumps(payload))
        end_time = time.time()
        response.raise_for_status()
        return response.json()["choices"][0]["text"], end_time - start_time
    except requests.exceptions.RequestException as e:
        print(f"Error querying vLLM API at {server_url}: {e}")
        return None, None


def parse_prometheus_metrics(prometheus_text: str):
    """ Parses Prometheus-style text metrics into a dictionary. """
    metrics = {}

    for line in prometheus_text.splitlines():
        if line.startswith("#"):
            continue  # Skip comments and HELP/TYPE metadata

        match = re.match(r'([\w:]+)\{([^}]*)\}\s+([\d.e+-]+)', line)
        if match:
            metric_name, labels, value = match.groups()
            metrics[metric_name] = float(value)
        else:
            parts = line.split()
            if len(parts) == 2:
                metric_name, value = parts
                metrics[metric_name] = float(value)

    return metrics


def get_vllm_metrics(server_url: str):
    """ Fetches and parses vLLM metrics from the /metrics endpoint. """
    try:
        response = requests.get(f"{server_url}/metrics")
        response.raise_for_status()

        if not response.text.strip():
            print(f"Error: Empty response from {server_url}/metrics.")
            return None

        metrics = parse_prometheus_metrics(response.text)

        # Get token counts
        prompt_tokens = metrics.get("vllm:prompt_tokens_total", 0.0)
        generation_tokens = metrics.get("vllm:generation_tokens_total", 0.0)

        # Get total time for processing requests
        time_to_first_token = metrics.get("vllm:time_to_first_token_seconds_sum", 0.0)
        total_generation_time = metrics.get("vllm:time_per_output_token_seconds_sum", 0.0)

        # Compute throughput
        avg_prompt_throughput = prompt_tokens / time_to_first_token if time_to_first_token > 0 else 0.0
        avg_generation_throughput = generation_tokens / total_generation_time if total_generation_time > 0 else 0.0

        return {
            "num_requests_running": metrics.get("vllm:num_requests_running", 0.0),
            "num_requests_swapped": metrics.get("vllm:num_requests_swapped", 0.0),
            "num_requests_waiting": metrics.get("vllm:num_requests_waiting", 0.0),
            "gpu_cache_usage": metrics.get("vllm:gpu_cache_usage_perc", 0.0),
            "cpu_cache_usage": metrics.get("vllm:cpu_cache_usage_perc", 0.0),
            "prefix_cache_hit_rate_gpu": metrics.get("vllm:gpu_prefix_cache_hit_rate", 0.0),
            "prefix_cache_hit_rate_cpu": metrics.get("vllm:cpu_prefix_cache_hit_rate", 0.0),
            "prompt_tokens_total": prompt_tokens,
            "generation_tokens_total": generation_tokens,
            "avg_prompt_throughput": avg_prompt_throughput,
            "avg_generation_throughput": avg_generation_throughput,
            "time_to_first_token": time_to_first_token,
            "time_per_output_token": total_generation_time,
            "e2e_request_latency": metrics.get("vllm:e2e_request_latency_seconds_sum", 0.0),
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching vLLM metrics from {server_url}: {e}")
        return None


def gen_prompt(tokenizer, token_num):
    """ Generate a random prompt of specified token length using tokenizer vocabulary. """
    all_available_tokens = list(tokenizer.get_vocab().values())
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerFast,
) -> List[Tuple[str, int, int]]:
    """ Generate benchmark requests with shared system prompts using random tokens. """
    system_prompts = [gen_prompt(tokenizer, system_prompt_len) for _ in range(num_groups)]
    questions = [gen_prompt(tokenizer, question_len) for _ in range(num_groups * prompts_per_group)]

    input_requests = []
    for group_idx in range(num_groups):
        system_prompt = system_prompts[group_idx]
        for prompt_idx in range(prompts_per_group):
            question = questions[group_idx * prompts_per_group + prompt_idx]
            full_prompt = f"{system_prompt}\n\n{question}"
            prompt_len = len(tokenizer.encode(full_prompt))
            input_requests.append((full_prompt, prompt_len, output_len))

    return input_requests


def benchmark_vllm(server_url: str, dataset: str, tokenizer, **kwargs):
    """ Runs the benchmark and collects performance metrics. """
    print(f"Running benchmark on vLLM server: {server_url}")

    if dataset == "generated-shared-prefix":
        requests = sample_generated_shared_prefix_requests(
            num_groups=kwargs["num_groups"],
            prompts_per_group=kwargs["prompts_per_group"],
            system_prompt_len=kwargs["system_prompt_len"],
            question_len=kwargs["question_len"],
            output_len=kwargs["output_len"],
            tokenizer=tokenizer,
        )
    else:
        raise ValueError("Unsupported dataset!")

    results = []
    total_time = 0

    for i, (prompt, _, max_tokens) in enumerate(requests):
        response, elapsed_time = query_vllm_api(server_url, prompt, kwargs["temperature"], max_tokens)
        total_time += elapsed_time if response else 0

        metrics = get_vllm_metrics(server_url)
        if metrics:
            results.append({
                "Iteration": i + 1,
                "Prompt Thruput (tok/s)": round(metrics['avg_prompt_throughput'], 2),
                "Gen Thruput (tok/s)": round(metrics['avg_generation_throughput'], 2),
                "GPU Cache (%)": round(metrics['gpu_cache_usage'] * 100, 2),
                "CPU Cache (%)": round(metrics['cpu_cache_usage'] * 100, 2),
                "Prefix Hit Rate (GPU) (%)": round(metrics['prefix_cache_hit_rate_gpu'] * 100, 2),
                "Prefix Hit Rate (CPU) (%)": round(metrics['prefix_cache_hit_rate_cpu'] * 100, 2),
            })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    avg_time = total_time / len(requests)
    print(f"\nAverage time per request: {avg_time:.3f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Benchmark a vLLM server instance.")
    parser.add_argument("--server-url", type=str, required=True, help="URL of the vLLM server (e.g., http://localhost:8000)")
    parser.add_argument("--dataset-name", type=str, choices=["standard", "generated-shared-prefix"], default="standard",
                        help="Choose the dataset: 'standard' for single requests, 'generated-shared-prefix' for prefix caching tests.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum number of tokens to generate")

    # Arguments for shared-prefix caching benchmark
    parser.add_argument("--gen-num-groups", type=int, default=64, help="Number of system prompt groups for shared-prefix dataset")
    parser.add_argument("--gen-prompts-per-group", type=int, default=16, help="Number of prompts per system prompt group")
    parser.add_argument("--gen-system-prompt-len", type=int, default=2048, help="Target length in tokens for system prompts")
    parser.add_argument("--gen-question-len", type=int, default=128, help="Target length in tokens for questions")
    parser.add_argument("--gen-output-len", type=int, default=256, help="Target length in tokens for outputs")

    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")

    if args.dataset_name == "generated-shared-prefix":
        benchmark_vllm(
            args.server_url, 
            "generated-shared-prefix", 
            tokenizer,
            num_groups=args.gen_num_groups,
            prompts_per_group=args.gen_prompts_per_group,
            system_prompt_len=args.gen_system_prompt_len,
            question_len=args.gen_question_len,
            output_len=args.gen_output_len,
            temperature=args.temperature
        )
    else:
        prompt = LONG_PROMPT + "Question: what is the age of John Doe? Your answer: The age of John Doe is "
        benchmark_vllm(args.server_url, "standard", tokenizer, prompt=prompt, temperature=args.temperature, max_tokens=args.max_tokens, repeat_count=args.repeat_count)

if __name__ == "__main__":
    main()
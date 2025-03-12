import argparse
import json
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests

from router import PolicyType, Router


def generate_random_text(length: int = 100) -> str:
    """Generate random text of the given length."""
    return ''.join(random.choice(string.ascii_letters + ' ') for _ in range(length))


def generate_prompt_with_prefix(prefix: str, length: int = 100) -> str:
    """Generate a prompt with the given prefix."""
    suffix_length = max(0, length - len(prefix))
    suffix = ''.join(random.choice(string.ascii_letters + ' ') for _ in range(suffix_length))
    return prefix + suffix


def create_mock_completion_request(prompt: str, model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Dict:
    """Create a mock OpenAI completion request."""
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
    }


def create_mock_chatcompletion_request(prompt: str, model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Dict:
    """Create a mock OpenAI chat completion request."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }


def send_request(router: Router, prompt: str, endpoint: str = "/v1/completions", model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Tuple[int, float]:
    """
    Send a request to the router and return the status code and response time.
    
    Args:
        router: Router instance
        prompt: Prompt text
        endpoint: API endpoint to use
        model_name: Model name to use in the request
        
    Returns:
        Tuple of (status_code, response_time_in_seconds)
    """
    if endpoint == "/v1/completions":
        request_body = create_mock_completion_request(prompt, model_name)
    elif endpoint == "/v1/chat/completions":
        request_body = create_mock_chatcompletion_request(prompt, model_name)
    else:
        request_body = {"prompt": prompt, "model": model_name}

    # Add headers to the request
    headers = {
        "Content-Type": "application/json",
    }
    start_time = time.time()
    status_code, _, _ = router.route_request(endpoint, request_body, headers=headers)
    end_time = time.time()
    
    return status_code, end_time - start_time


def benchmark_single_router(
    router: Router,
    num_requests: int = 100,
    endpoint: str = "/v1/completions",
    max_workers: int = 10,
    random_length: int = 100,
    prefix_ratio: float = 0.5,
    prefixes: List[str] = None,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> Dict:
    """
    Benchmark a single router with random requests.
    
    Args:
        router: Router instance
        num_requests: Number of requests to send
        endpoint: API endpoint to use
        max_workers: Maximum number of concurrent workers
        random_length: Length of random prompts
        prefix_ratio: Ratio of requests that will have a shared prefix
        prefixes: List of prefixes to use (if None, generates random prefixes)
        model_name: Model name to use in the request
        
    Returns:
        Dictionary with benchmark results
    """
    # Generate or use provided prefixes
    if prefixes is None:
        num_prefixes = 5
        prefixes = [generate_random_text(55) for _ in range(num_prefixes)]
    
    # Generate prompts (mix of completely random and prefix-based)
    prompts = []
    for i in range(num_requests):
        if random.random() < prefix_ratio:
            # Use a prefix from the list
            prefix = random.choice(prefixes)
            prompt = generate_prompt_with_prefix(prefix, random_length)
        else:
            # Completely random prompt
            prompt = generate_random_text(random_length)
        prompts.append(prompt)
    
    # Send requests in parallel
    response_times = []
    success_count = 0
    
    def process_request(prompt):
        status, time_taken = send_request(router, prompt, endpoint, model_name)
        return status, time_taken
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_request, prompts))
    
    for status, time_taken in results:
        if 200 <= status < 300:
            success_count += 1
            response_times.append(time_taken)
    
    # Calculate statistics
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        p50_time = sorted(response_times)[len(response_times) // 2]
        p95_time = sorted(response_times)[int(len(response_times) * 0.95)]
        min_time = min(response_times)
        max_time = max(response_times)
    else:
        avg_time = p50_time = p95_time = min_time = max_time = 0
    
    return {
        "success_rate": success_count / num_requests if num_requests > 0 else 0,
        "avg_time": avg_time,
        "p50_time": p50_time,
        "p95_time": p95_time,
        "min_time": min_time,
        "max_time": max_time,
        "raw_times": response_times,
    }


def run_policy_comparison(
    worker_urls: List[str],
    num_requests: int = 1000,
    endpoint: str = "/v1/completions",
    max_workers: int = 10,
    random_length: int = 100,
    prefix_ratio: float = 0.7,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
):
    """
    Compare different routing policies.
    
    Args:
        worker_urls: List of worker URLs
        num_requests: Number of requests per policy
        endpoint: API endpoint to use
        max_workers: Maximum number of concurrent workers
        random_length: Length of random prompts
        prefix_ratio: Ratio of requests that will have a shared prefix
        model_name: Model name to use in the request
    """
    # Generate common prefixes for fair comparison
    num_prefixes = 5
    prefixes = [generate_random_text(55) for _ in range(num_prefixes)]
    
    # Define policies to test
    policies = [
        (PolicyType.RANDOM, "Random"),
        (PolicyType.ROUND_ROBIN, "Round Robin"),
        (PolicyType.CACHE_AWARE, "Cache Aware"),
    ]
    
    # Run benchmarks for each policy
    results = {}
    for policy_type, policy_name in policies:
        print(f"Benchmarking {policy_name} policy...")
        
        router = Router(
            worker_urls=worker_urls,
            policy=policy_type,
            timeout_secs=10,
            interval_secs=2,
            cache_threshold=0.5,
            balance_abs_threshold=32,
            balance_rel_threshold=1.0001,
            eviction_interval_secs=60,
            max_tree_size=2**24,
        )
        
        policy_results = benchmark_single_router(
            router=router,
            num_requests=num_requests,
            endpoint=endpoint,
            max_workers=max_workers,
            random_length=random_length,
            prefix_ratio=prefix_ratio,
            prefixes=prefixes,
            model_name=model_name,
        )
        
        results[policy_name] = policy_results
        print(f"  Success rate: {policy_results['success_rate']:.2%}")
        print(f"  Avg response time: {policy_results['avg_time']:.4f} sec")
        print(f"  P50 response time: {policy_results['p50_time']:.4f} sec")
        print(f"  P95 response time: {policy_results['p95_time']:.4f} sec")
        print()
    
    # Plot results
    plot_comparison(results)
    
    return results


def plot_comparison(results: Dict):
    """Plot comparison of different policies."""
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Response Time Comparison (Avg, P50, P95)
    plt.subplot(2, 2, 1)
    policies = list(results.keys())
    avg_times = [results[p]["avg_time"] for p in policies]
    p50_times = [results[p]["p50_time"] for p in policies]
    p95_times = [results[p]["p95_time"] for p in policies]
    
    x = np.arange(len(policies))
    width = 0.25
    
    plt.bar(x - width, avg_times, width, label='Avg Time')
    plt.bar(x, p50_times, width, label='P50 Time')
    plt.bar(x + width, p95_times, width, label='P95 Time')
    
    plt.xlabel('Policy')
    plt.ylabel('Response Time (s)')
    plt.title('Response Time Comparison')
    plt.xticks(x, policies)
    plt.legend()
    
    # Plot 2: Success Rate
    plt.subplot(2, 2, 2)
    success_rates = [results[p]["success_rate"] * 100 for p in policies]
    
    plt.bar(policies, success_rates)
    plt.xlabel('Policy')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison')
    plt.ylim(0, 100)
    
    # Plot 3: Response Time Distribution
    plt.subplot(2, 1, 2)
    for policy in policies:
        plt.hist(results[policy]["raw_times"], alpha=0.5, bins=50, label=policy)
    
    plt.xlabel('Response Time (s)')
    plt.ylabel('Frequency')
    plt.title('Response Time Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('policy_comparison.png')
    plt.close()
    
    print(f"Plot saved to policy_comparison.png")


def compare_cache_thresholds(
    worker_urls: List[str],
    thresholds: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
    num_requests: int = 500,
    endpoint: str = "/v1/completions",
    max_workers: int = 10,
    random_length: int = 100,
    prefix_ratio: float = 0.7,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
):
    """
    Compare different cache thresholds for the Cache Aware policy.
    
    Args:
        worker_urls: List of worker URLs
        thresholds: List of cache thresholds to test
        num_requests: Number of requests per threshold
        endpoint: API endpoint to use
        max_workers: Maximum number of concurrent workers
        random_length: Length of random prompts
        prefix_ratio: Ratio of requests that will have a shared prefix
        model_name: Model name to use in the request
    """
    # Generate common prefixes for fair comparison
    num_prefixes = 5
    prefixes = [generate_random_text(30) for _ in range(num_prefixes)]
    
    # Run benchmarks for each threshold
    results = {}
    for threshold in thresholds:
        print(f"Benchmarking Cache Aware policy with threshold {threshold}...")
        
        router = Router(
            worker_urls=worker_urls,
            policy=PolicyType.CACHE_AWARE,
            timeout_secs=10,
            interval_secs=2,
            cache_threshold=threshold,
            balance_abs_threshold=32,
            balance_rel_threshold=1.0001,
            eviction_interval_secs=60,
            max_tree_size=2**24,
        )
        
        threshold_results = benchmark_single_router(
            router=router,
            num_requests=num_requests,
            endpoint=endpoint,
            max_workers=max_workers,
            random_length=random_length,
            prefix_ratio=prefix_ratio,
            prefixes=prefixes,
            model_name=model_name,
        )
        
        results[f"Threshold {threshold}"] = threshold_results
        print(f"  Success rate: {threshold_results['success_rate']:.2%}")
        print(f"  Avg response time: {threshold_results['avg_time']:.4f} sec")
        print(f"  P50 response time: {threshold_results['p50_time']:.4f} sec")
        print(f"  P95 response time: {threshold_results['p95_time']:.4f} sec")
        print()
    
    # Plot results
    plot_comparison(results)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Router Benchmark")
    parser.add_argument(
        "--worker-urls",
        type=str,
        nargs="+",
        required=True,
        help="List of worker URLs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name to use in requests",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of requests per policy",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        choices=["/v1/completions", "/v1/chat/completions", "/generate"],
        help="API endpoint to use",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=100,
        help="Maximum number of concurrent workers",
    )
    parser.add_argument(
        "--prefix-ratio",
        type=float,
        default=0.7,
        help="Ratio of requests that will have a shared prefix",
    )
    parser.add_argument(
        "--benchmark-type",
        type=str,
        default="policy",
        choices=["policy", "threshold"],
        help="Type of benchmark to run (policy comparison or threshold comparison)",
    )
    
    args = parser.parse_args()
    
    if args.benchmark_type == "policy":
        run_policy_comparison(
            worker_urls=args.worker_urls,
            num_requests=args.num_requests,
            endpoint=args.endpoint,
            max_workers=args.max_workers,
            prefix_ratio=args.prefix_ratio,
            model_name=args.model_name,
        )
    else:
        compare_cache_thresholds(
            worker_urls=args.worker_urls,
            num_requests=args.num_requests,
            endpoint=args.endpoint,
            max_workers=args.max_workers,
            prefix_ratio=args.prefix_ratio,
            model_name=args.model_name,
        ) 
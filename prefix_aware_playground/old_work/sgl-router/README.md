# SGL Router Python Implementation

This is a Python implementation of the SGL Router, which includes a cache-aware routing strategy for LLMs. The implementation is designed to benchmark and compare different routing strategies.

## Features

- Three routing strategies:
  - **Random**: Randomly selects workers for each request
  - **Round Robin**: Distributes requests sequentially across workers
  - **Cache Aware**: Uses prefix matching to route requests to workers that have processed similar requests, potentially leveraging cached data

- The Cache Aware strategy combines two approaches:
  - Cache-aware routing using a prefix tree for request history
  - Load balancing when the system is imbalanced

## Requirements

```
pip install requests fastapi uvicorn matplotlib numpy
```

## Files

- `tree.py`: Implements the prefix tree for cache-aware routing
- `router.py`: Implements the router with various routing strategies
- `server.py`: Implements a FastAPI server for the router
- `benchmark.py`: Tool for benchmarking and comparing routing strategies

## Usage

### Running the Server

```bash
python server.py --worker-urls http://worker1:8000 http://worker2:8000 http://worker3:8000 --policy cache_aware
```

Options:

- `--worker-urls`: List of worker URLs to route requests to (required)
- `--policy`: Routing policy - `random`, `round_robin`, or `cache_aware` (default: `round_robin`)
- `--host`: Host to bind the server (default: `127.0.0.1`)
- `--port`: Port to bind the server (default: `8000`)
- `--timeout-secs`: Timeout for requests in seconds (default: `60`)
- `--interval-secs`: Interval between health checks in seconds (default: `10`)
- `--cache-threshold`: Threshold for cache hits in cache-aware routing (default: `0.5`)
- `--balance-abs-threshold`: Absolute threshold for load imbalance detection (default: `32`)
- `--balance-rel-threshold`: Relative threshold for load imbalance detection (default: `1.0001`)
- `--eviction-interval-secs`: Interval between cache evictions in seconds (default: `60`)
- `--max-tree-size`: Maximum size of the prefix tree (default: `2^24`)

### Running Benchmarks

```bash
python benchmark.py --worker-urls http://worker1:8000 http://worker2:8000 http://worker3:8000 --benchmark-type policy
```

Options:

- `--worker-urls`: List of worker URLs to route requests to (required)
- `--benchmark-type`: Type of benchmark - `policy` or `threshold` (default: `policy`)
- `--num-requests`: Number of requests per policy/threshold (default: `1000`)
- `--endpoint`: API endpoint to use - `/v1/completions`, `/v1/chat/completions`, or `/generate` (default: `/v1/completions`)
- `--max-workers`: Maximum number of concurrent workers (default: `10`)
- `--prefix-ratio`: Ratio of requests that will have a shared prefix (default: `0.7`)

## How It Works

### Cache-Aware Routing Strategy

The cache-aware routing strategy dynamically switches between two approaches:

1. **Cache-Aware Routing**:
   - Maintains an approximate radix tree for tracking request patterns
   - Routes requests to workers that have processed similar requests before
   - If match rate > `cache_threshold`, routes to worker with highest match
   - If match rate ≤ `cache_threshold`, routes to worker with smallest tree size

2. **Load Balancing**:
   - Used when the system is detected to be imbalanced
   - A system is considered imbalanced if:
     - (max_load - min_load) > `balance_abs_threshold` AND
     - max_load > min_load * `balance_rel_threshold`

### Prefix Tree

The prefix tree (approximate radix tree) is a key data structure that:

- Efficiently stores text patterns from previous requests
- Allows for quick prefix matching to find workers that processed similar requests
- Tracks usage patterns per worker (tenant)
- Uses thread-safe operations for concurrent access

## Benchmarking

The benchmark tool allows you to:

1. Compare different routing policies
2. Test different cache thresholds for the cache-aware policy
3. Generate visualizations of the results

Results are saved as a PNG file with the following charts:
- Response time comparison (Avg, P50, P95)
- Success rate comparison
- Response time distribution

## API Endpoints

The server exposes several endpoints:

- `/health` - Health check endpoint
- `/health/generate` - Health check for generate endpoint
- `/server_info` - Get server information
- `/v1/models` - OpenAI compatible models endpoint
- `/v1/models/{model_id}` - OpenAI compatible model info endpoint
- `/generate` - SGLang generate endpoint
- `/v1/chat/completions` - OpenAI compatible chat completions endpoint
- `/v1/completions` - OpenAI compatible completions endpoint
- `/add_worker` - Add a worker to the router
- `/remove_worker` - Remove a worker from the router 
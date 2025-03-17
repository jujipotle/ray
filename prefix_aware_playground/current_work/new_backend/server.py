import argparse
import uvicorn
from fastapi import FastAPI, Request, Query
from typing import List

from router import Router, RandomConfig, RoundRobinConfig, CacheAwareConfig

app = FastAPI(title="Fast vLLM Router")
router_instance = None


@app.post("/v1/completions")
async def completions(request: Request):
    """Handle completions API calls."""
    try:
        body = await request.json()
        if body.get("stream", False):
            return await router_instance.forward_streaming_request(request, "/v1/completions")
    except Exception:
        pass
    return await router_instance.forward_request(request, "/v1/completions")


@app.get("/v1/models")
async def models(request: Request):
    """List all models from all workers."""
    return await router_instance.get_all_models()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return router_instance.get_health()


@app.post("/reset_prefix_cache")
async def reset_prefix_cache():
    """Reset prefix cache on all workers."""
    return await router_instance.reset_prefix_cache()


@app.get("/router_health")
async def router_health():
    """Router-only health check endpoint that doesn't depend on worker availability."""
    return router_instance.get_router_health()


@app.get("/policy")
async def get_policy():
    """Get the current routing policy."""
    # Return the class name of the policy config in use.
    return {"policy": router_instance.policy_config.__class__.__name__}


def start_server(
    worker_urls: List[str],
    policy: str = "round_robin",
    timeout_secs: int = 60
):
    """Initialize the router with the appropriate policy config."""
    global router_instance

    # Build policy config based on the policy string.
    if policy.lower() == "random":
        policy_config = RandomConfig(timeout_secs=timeout_secs, interval_secs=10)
    elif policy.lower() in ["prefix_aware", "cacheaware"]:
        policy_config = CacheAwareConfig(
            cache_threshold=0.50,
            balance_abs_threshold=32,
            balance_rel_threshold=1.0001,
            eviction_interval_secs=60,
            max_tree_size=(2**24),
            timeout_secs=timeout_secs,
            interval_secs=10
        )
    else:  # Defaults to round_robin.
        policy_config = RoundRobinConfig(timeout_secs=timeout_secs, interval_secs=10)

    router_instance = Router(worker_urls=worker_urls, policy_config=policy_config)


def main():
    """Parse command line arguments and start the server."""
    parser = argparse.ArgumentParser(description="Run the fast router server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument(
        "--worker-ports",
        type=str,
        default="8001,8002,8003,8004",
        help="Comma-separated list of worker ports"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="round_robin",
        choices=["CacheAware", "RoundRobin", "Random"],
        help="Routing policy to use"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for requests to workers"
    )
    
    args = parser.parse_args()
    
    # Convert worker ports to URLs (assuming workers run on localhost).
    worker_ports = [int(port.strip()) for port in args.worker_ports.split(",")]
    worker_urls = [f"http://localhost:{port}" for port in worker_ports]
    
    # Initialize the router with the specified policy config.
    start_server(worker_urls=worker_urls, policy=args.policy, timeout_secs=args.timeout)
    
    print(f"Starting fast router server on {args.host}:{args.port}")
    print(f"Using policy: {args.policy}")
    print(f"Worker URLs: {worker_urls}")
    
    # Run the FastAPI app with uvicorn.
    uvicorn.run(app, host=args.host, port=args.port, log_level="error")


if __name__ == "__main__":
    main()
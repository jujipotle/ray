import argparse
import json
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from router import PolicyType, Router


app = FastAPI(title="SGL Router")
router_instance = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/health/generate")
async def health_generate():
    """Health check for generate endpoint."""
    return {"status": "ok"}


@app.get("/server_info")
async def get_server_info():
    """Get server information."""
    return {
        "status": "ok",
        "worker_urls": router_instance.worker_urls,
        "policy": router_instance.policy.value,
    }


@app.post("/v1/models")
async def v1_models(request: Request):
    """OpenAI compatible models endpoint."""
    body = await request.body()
    headers = dict(request.headers)
    status_code, resp_headers, content = router_instance.route_request(
        "/v1/models", body, headers
    )
    return Response(content=content, status_code=status_code, headers=resp_headers)


@app.get("/v1/models/{model_id}")
async def get_model_info(request: Request, model_id: str):
    """OpenAI compatible model info endpoint."""
    headers = dict(request.headers)
    status_code, resp_headers, content = router_instance.route_request(
        f"/v1/models/{model_id}", {}, headers
    )
    return Response(content=content, status_code=status_code, headers=resp_headers)


@app.post("/generate")
async def generate(request: Request):
    """SGLang generate endpoint."""
    body = await request.body()
    headers = dict(request.headers)
    
    # Check if it's a streaming request
    is_stream = False
    try:
        body_json = json.loads(body)
        is_stream = body_json.get("stream", False)
    except json.JSONDecodeError:
        pass
    
    status_code, resp_headers, content = router_instance.route_request(
        "/generate", body, headers
    )
    
    if is_stream:
        # Handle streaming response
        async def stream_response():
            yield content
        
        return StreamingResponse(
            stream_response(),
            status_code=status_code,
            headers=resp_headers,
        )
    else:
        # Handle normal response
        return Response(content=content, status_code=status_code, headers=resp_headers)


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    """OpenAI compatible chat completions endpoint."""
    body = await request.body()
    headers = dict(request.headers)
    
    # Check if it's a streaming request
    is_stream = False
    try:
        body_json = json.loads(body)
        is_stream = body_json.get("stream", False)
    except json.JSONDecodeError:
        pass
    
    status_code, resp_headers, content = router_instance.route_request(
        "/v1/chat/completions", body, headers
    )
    
    if is_stream:
        # Handle streaming response
        async def stream_response():
            yield content
        
        return StreamingResponse(
            stream_response(),
            status_code=status_code,
            headers=resp_headers,
        )
    else:
        # Handle normal response
        return Response(content=content, status_code=status_code, headers=resp_headers)


@app.post("/v1/completions")
async def v1_completions(request: Request):
    """OpenAI compatible completions endpoint."""
    body = await request.body()
    headers = dict(request.headers)
    
    # Check if it's a streaming request
    is_stream = False
    try:
        body_json = json.loads(body)
        is_stream = body_json.get("stream", False)
    except json.JSONDecodeError:
        pass
    
    status_code, resp_headers, content = router_instance.route_request(
        "/v1/completions", body, headers
    )
    
    if is_stream:
        # Handle streaming response
        async def stream_response():
            yield content
        
        return StreamingResponse(
            stream_response(),
            status_code=status_code,
            headers=resp_headers,
        )
    else:
        # Handle normal response
        return Response(content=content, status_code=status_code, headers=resp_headers)


@app.post("/add_worker")
async def add_worker(worker_url: str):
    """Add a worker to the router."""
    success = router_instance.add_worker(worker_url)
    if success:
        return {"status": "ok", "message": f"Worker {worker_url} added"}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to add worker {worker_url}",
        )


@app.post("/remove_worker")
async def remove_worker(worker_url: str):
    """Remove a worker from the router."""
    success = router_instance.remove_worker(worker_url)
    if success:
        return {"status": "ok", "message": f"Worker {worker_url} removed"}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to remove worker {worker_url}",
        )
@app.post("/reset_prefix_cache")
async def reset_prefix_cache():
    """Reset prefix cache on all workers."""
    results = {}
    
    available_workers = router_instance.worker_urls
    
    async with httpx.AsyncClient() as client:
        for i, worker_url in enumerate(available_workers):
            try:
                response = await client.post(f"{worker_url}/reset_prefix_cache", timeout=10.0)
                results[worker_url] = {
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
            except Exception as e:
                results[worker_url] = {
                    "error": str(e),
                    "success": False
                }
    
    return {
        "status": "completed",
        "results": results
    }

def start_server(
    worker_urls: List[str],
    policy: str = "round_robin",
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout_secs: int = 60,
    interval_secs: int = 10,
    cache_threshold: float = 0.5,
    balance_abs_threshold: int = 32,
    balance_rel_threshold: float = 1.0001,
    eviction_interval_secs: int = 60,
    max_tree_size: int = 2**24,
):
    """Start the router server."""
    global router_instance
    
    # Convert policy string to enum
    policy_map = {
        "random": PolicyType.RANDOM,
        "round_robin": PolicyType.ROUND_ROBIN,
        "cache_aware": PolicyType.CACHE_AWARE,
    }
    policy_enum = policy_map.get(policy.lower(), PolicyType.ROUND_ROBIN)
    
    # Create router instance
    router_instance = Router(
        worker_urls=worker_urls,
        policy=policy_enum,
        timeout_secs=timeout_secs,
        interval_secs=interval_secs,
        cache_threshold=cache_threshold,
        balance_abs_threshold=balance_abs_threshold,
        balance_rel_threshold=balance_rel_threshold,
        eviction_interval_secs=eviction_interval_secs,
        max_tree_size=max_tree_size,
    )
    
    # Start server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGL Router")
    parser.add_argument(
        "--worker-urls",
        type=str,
        nargs="+",
        required=True,
        help="List of worker URLs",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="round_robin",
        choices=["random", "round_robin", "cache_aware"],
        help="Routing policy",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind server",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=60,
        help="Timeout in seconds for requests",
    )
    parser.add_argument(
        "--interval-secs",
        type=int,
        default=10,
        help="Interval in seconds between health checks",
    )
    parser.add_argument(
        "--cache-threshold",
        type=float,
        default=0.5,
        help="Cache threshold for cache-aware routing",
    )
    parser.add_argument(
        "--balance-abs-threshold",
        type=int,
        default=32,
        help="Absolute threshold for load balancing",
    )
    parser.add_argument(
        "--balance-rel-threshold",
        type=float,
        default=1.0001,
        help="Relative threshold for load balancing",
    )
    parser.add_argument(
        "--eviction-interval-secs",
        type=int,
        default=60,
        help="Interval in seconds between cache evictions",
    )
    parser.add_argument(
        "--max-tree-size",
        type=int,
        default=2**24,
        help="Maximum size of the tree",
    )
    
    args = parser.parse_args()
    
    start_server(
        worker_urls=args.worker_urls,
        policy=args.policy,
        host=args.host,
        port=args.port,
        timeout_secs=args.timeout_secs,
        interval_secs=args.interval_secs,
        cache_threshold=args.cache_threshold,
        balance_abs_threshold=args.balance_abs_threshold,
        balance_rel_threshold=args.balance_rel_threshold,
        eviction_interval_secs=args.eviction_interval_secs,
        max_tree_size=args.max_tree_size,
    ) 
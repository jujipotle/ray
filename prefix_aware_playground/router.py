# Quickstart: uvicorn router:app --port 8000
#
# API ENDPOINTS:
# -------------
# 1. Text Completions (Non-streaming):
#    POST /v1/completions
#    Body: {"model": "model_name", "prompt": "Your prompt", "max_tokens": 100}
#
# 2. Text Completions (Streaming):
#    POST /v1/completions
#    Body: {"model": "model_name", "prompt": "Your prompt", "max_tokens": 100, "stream": true}
#
# 3. List Available Models:
#    GET /v1/models
#
# 4. Health Check:
#    GET /health
#
# EXAMPLE CURL COMMANDS:
# --------------------
# 1. Non-streaming completion:
#    curl -X POST http://localhost:8000/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"Hello, world!","max_tokens":50}'
#
# 2. Streaming completion:
#    curl -X POST http://localhost:8000/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"Hello, world!","max_tokens":50,"stream":true}'
#
# 3. List models:
#    curl http://localhost:8000/v1/models
#
# 4. Health check:
#    curl http://localhost:8000/health

import json
import os
from fastapi import FastAPI, Request, Response, HTTPException
from starlette.responses import StreamingResponse
import httpx
import asyncio
import argparse
from typing import List

# Initialize FastAPI application
app = FastAPI()

# Global variable to store worker URLs
worker_urls = []

def initialize_worker_urls(ports: List[int]):
    """Initialize the worker URLs from the provided ports."""
    global worker_urls
    worker_urls = [f"http://localhost:{port}" for port in ports]
    print(f"Initialized worker URLs: {worker_urls}")

# @app.on_event("startup")
# async def startup_event():
#     """Initialize worker URLs from environment variable if not already set."""
#     global worker_urls
#     if not worker_urls:
#         # Try to get worker ports from environment variable
#         worker_ports_str = os.environ.get("WORKER_PORTS", "")
#         if worker_ports_str:
#             ports = [int(port.strip()) for port in worker_ports_str.split(",")]
#             initialize_worker_urls(ports)
#             print(f"Initialized worker URLs from environment: {worker_urls}")
#         else:
#             print("WARNING: No worker URLs initialized. Router will not function properly.")

# Cache for available workers
# available_workers_cache = []
# last_health_check = 0
# HEALTH_CHECK_INTERVAL = 5  # seconds

counter = 0  # Round-robin counter

# async def get_available_workers():
#     """Get a list of available worker URLs."""
#     global available_workers_cache, last_health_check, worker_urls
    
#     # Check if we need to refresh the cache
#     current_time = asyncio.get_event_loop().time()
#     if current_time - last_health_check > HEALTH_CHECK_INTERVAL or not available_workers_cache:
#         available = []
        
#         if not worker_urls:
#             print("ERROR: No worker URLs configured. Please initialize worker URLs.")
#             raise HTTPException(status_code=503, detail="No worker URLs configured")
        
#         print(f"Checking {len(worker_urls)} workers: {worker_urls}")
        
#         async with httpx.AsyncClient() as client:
#             for worker_url in worker_urls:
#                 try:
#                     print(f"Checking worker at {worker_url}...")
#                     response = await client.get(f"{worker_url}/v1/models", timeout=2.0)
#                     print(f"Response from {worker_url}: {response.status_code}")
#                     if response.status_code == 200:
#                         available.append(worker_url)
#                     else:
#                         print(f"Worker {worker_url} returned non-200 status: {response.status_code}")
#                 except Exception as e:
#                     print(f"Error connecting to {worker_url}: {str(e)}")
        
#         print(f"Available workers: {available}")
#         available_workers_cache = available
#         last_health_check = current_time
    
#     if not available_workers_cache:
#         raise HTTPException(status_code=503, detail="No available workers")
    
#     return available_workers_cache

async def forward_request(request: Request, endpoint: str):
    """Forward a regular request to one of the GPU workers."""
    global counter
    
    # available_workers = await get_available_workers()
    available_workers = worker_urls
    worker_url = available_workers[counter % len(available_workers)]
    counter += 1
    
    try:
        # Extract request body
        request_body = await request.json() if await request.body() else None
        
        # Forward the request
        async with httpx.AsyncClient() as client:
            if request_body:
                response = await client.post(f"{worker_url}{endpoint}", json=request_body, timeout=60.0)
            else:
                response = await client.get(f"{worker_url}{endpoint}", timeout=60.0)

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response.headers,
                media_type=response.headers.get("content-type", "application/json")
            )
            
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json"
        )

async def forward_streaming_request(request: Request, endpoint: str):
    """Forward a streaming request to one of the GPU workers."""
    global counter
    
    # available_workers = await get_available_workers()
    available_workers = worker_urls
    worker_url = available_workers[counter % len(available_workers)]
    counter += 1
    
    try:
        # Extract request body
        request_body = await request.json() if await request.body() else None
        
        # Ensure streaming is enabled
        if request_body and "stream" not in request_body:
            request_body["stream"] = True
        
        # Create streaming response
        async def stream_generator():
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{worker_url}{endpoint}", json=request_body, timeout=120.0) as response:
                    if response.status_code != 200:
                        yield json.dumps({"error": f"Worker returned status code: {response.status_code}"}).encode("utf-8")
                        return
                    
                    async for chunk in response.aiter_bytes():
                        yield chunk
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
            
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json"
        )

@app.post("/v1/completions")
async def completions(request: Request):
    """Handle completions API calls."""
    try:
        body = await request.json()
        if body.get("stream", False):
            return await forward_streaming_request(request, "/v1/completions")
    except:
        pass
    return await forward_request(request, "/v1/completions")

@app.get("/v1/models")
async def models(request: Request):
    """List all models from all workers."""
    all_models = []
    
    # available_workers = await get_available_workers()
    available_workers = worker_urls
    
    async with httpx.AsyncClient() as client:
        for i, worker_url in enumerate(available_workers):
            try:
                response = await client.get(f"{worker_url}/v1/models", timeout=10.0)
                if response.status_code == 200:
                    worker_models = response.json()
                    if "data" in worker_models:
                        for model in worker_models["data"]:
                            model["worker_index"] = i
                            model["worker_url"] = worker_url
                            all_models.append(model)
            except:
                pass
    
    return {"object": "list", "data": all_models}

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # available_workers = await get_available_workers()
        available_workers = worker_urls
        worker_indices = [i for i, _ in enumerate(available_workers)]
        
        return {
            "status": "healthy" if worker_indices else "unhealthy",
            "available_workers": worker_indices,
            "worker_count": len(worker_indices),
            "worker_urls": available_workers
        }
    except HTTPException:
        return {
            "status": "unhealthy",
            "available_workers": [],
            "worker_count": 0,
            "worker_urls": []
        }

@app.post("/reset_prefix_cache")
async def reset_prefix_cache():
    """Reset prefix cache on all workers."""
    results = {}
    
    # available_workers = await get_available_workers()
    available_workers = worker_urls
    
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

@app.get("/router_health")
async def router_health():
    """Router-only health check endpoint that doesn't depend on worker availability."""
    return {
        "status": "healthy",
        "worker_urls_configured": len(worker_urls),
        "worker_urls": worker_urls
    }
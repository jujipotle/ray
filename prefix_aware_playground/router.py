import asyncio
import json
from fastapi import FastAPI, Request
import httpx

# Initialize FastAPI application
app = FastAPI()

# List of available GPU workers (adjust if you have more/less than 4)
GPU_WORKERS = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
    "http://localhost:8004"
]

counter = 0  # Round-robin counter


async def forward_request(request: Request):
    """Forward the incoming request to one of the GPU workers."""
    global counter
    worker_url = GPU_WORKERS[counter % len(GPU_WORKERS)]  # Select worker
    counter += 1  # Increment counter for next request

    # Extract request body
    request_body = await request.json()

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{worker_url}/v1/completions", json=request_body)
        return response.json()


@app.post("/v1/completions")
async def completions(request: Request):
    """Handle `/v1/completions` API calls and forward them to GPU workers."""
    return await forward_request(request)
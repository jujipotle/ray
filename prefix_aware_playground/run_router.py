#!/usr/bin/env python3
"""
Script to run the router server.
This script starts a FastAPI server that routes requests to vLLM worker servers.

Usage:
    python run_router.py --host 127.0.0.1 --port 8000 --worker-ports 8001,8002,8003,8004
"""

import argparse
import os
import uvicorn
from router import app, initialize_worker_urls

def parse_args():
    parser = argparse.ArgumentParser(description="Run the router server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the router server on")
    parser.add_argument("--worker-ports", type=str, default="8001,8002,8003,8004", 
                        help="Comma-separated list of worker ports")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse worker ports
    worker_ports = [int(port.strip()) for port in args.worker_ports.split(",")]
    
    # Initialize worker URLs in the router
    initialize_worker_urls(worker_ports)
    
    print(f"Starting router server on {args.host}:{args.port}")
    print(f"Configured to route to worker ports: {args.worker_ports}")
    
    # Start the FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 
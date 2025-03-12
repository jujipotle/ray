#!/usr/bin/env python3
"""
Fast Router Implementation for vLLM Workers

This router implements a simple round-robin policy for distributing requests
across multiple vLLM worker instances.
"""

import json
import random
import threading
from enum import Enum
from typing import Dict, List, Optional, Union
import httpx
from fastapi import Request, Response, HTTPException
from starlette.responses import StreamingResponse

# Import Tree for prefix-aware routing
from tree import Tree

class PolicyType(Enum):
    """Router policy types."""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    PREFIX_AWARE = "prefix_aware"

class Router:
    """
    A simple, high-performance router for distributing requests across worker nodes.
    
    This router supports three policies:
    1. Random - Randomly select workers
    2. RoundRobin - Distribute requests in round-robin fashion
    3. PrefixAware - Tree-based prefix routing for better cache utilization
    """
    
    def __init__(
        self, 
        worker_urls: List[str], 
        policy: PolicyType = PolicyType.ROUND_ROBIN,
        timeout_secs: int = 60
    ):
        """
        Initialize the router with a list of worker URLs.
        
        Args:
            worker_urls: List of worker URLs to route requests to
            policy: Routing policy to use
            timeout_secs: Timeout in seconds for requests to workers
        """
        self.worker_urls = worker_urls
        self.policy = policy
        self.counter = 0  # Round-robin counter
        self.timeout = timeout_secs
        self.round_robin_lock = threading.Lock()
        
        # For prefix-aware routing using Tree
        if policy == PolicyType.PREFIX_AWARE:
            self.tree = Tree()
            self.tree.tenant_char_count = {url: 0 for url in worker_urls}
            self.running_queue = {url: 0 for url in worker_urls}
            self.queue_lock = threading.Lock()
        
        print(f"Initialized Router with policy: {policy.value}, worker URLs: {self.worker_urls}")
    
    def _select_worker(self, text: str = "", prefix_id: str = "-1") -> str:
        """
        Select a worker based on the current policy.
        
        Args:
            text: Text to use for prefix-aware routing
            prefix_id: Prefix ID for explicit prefix grouping
            
        Returns:
            Selected worker URL
        """
        if self.policy == PolicyType.RANDOM:
            return random.choice(self.worker_urls)
        
        elif self.policy == PolicyType.ROUND_ROBIN:
            with self.round_robin_lock:
                selected_url = self.worker_urls[self.counter % len(self.worker_urls)]
                self.counter += 1
                return selected_url
        
        elif self.policy == PolicyType.PREFIX_AWARE:
            with self.queue_lock:
                # Use tree-based prefix matching
                matched_text, matched_worker = self.tree.prefix_match(text)
                
                if matched_worker != "empty" and matched_worker in self.worker_urls:
                    # Calculate match rate
                    match_rate = len(matched_text) / len(text) if text else 0
                    print(f"Match rate: {match_rate}")
                    if match_rate > 0.5:  # Use 0.5 as threshold
                        # Update running queue
                        self.running_queue[matched_worker] += 1
                        
                        # Insert text into tree for future matching
                        if text:
                            self.tree.insert(text, matched_worker)
                        
                        return matched_worker
                
                # No good match, use worker with smallest tree
                selected_url = self.tree.get_smallest_tenant()
                if selected_url == "empty" or selected_url not in self.worker_urls:
                    # Fall back to first worker
                    selected_url = self.worker_urls[0]
                
                # Update running queue
                self.running_queue[selected_url] += 1
                
                # Insert text into tree for future matching
                if text:
                    self.tree.insert(text, selected_url)
                
                return selected_url
            # with self.queue_lock:
            #     # If we have a valid prefix_id, use it for routing
            #     if prefix_id != "-1":
            #         try:
            #             # Convert prefix_id to integer and use modulo to select worker
            #             prefix_id_int = int(prefix_id)
            #             worker_index = prefix_id_int % len(self.worker_urls)  # Use modulo 4 to select worker
            #             selected_url = self.worker_urls[worker_index]
            #             # print(f"Routing prefix_id {prefix_id} to worker {worker_index}: {selected_url}")
            #         except (ValueError, IndexError):
            #             # Fallback if prefix_id is not a valid integer or worker index is out of range
            #             print(f"Invalid prefix_id {prefix_id}, falling back to round-robin")
            #             selected_url = self.worker_urls[self.counter % len(self.worker_urls)]
            #             self.counter += 1
            #     else:
            #         # Fallback to round-robin if no prefix_id is provided
            #         selected_url = self.worker_urls[self.counter % len(self.worker_urls)]
            #         self.counter += 1
                
            #     # Update running queue and processed queue
            #     self.running_queue[selected_url] += 1
            #     if hasattr(self, 'processed_queue'):
            #         self.processed_queue[selected_url] += 1
                
            #     # Insert text into tree for future matching if needed
            #     if text:
            #         self.tree.insert(text, selected_url)
                
            #     return selected_url
        
        # Default to random worker if policy is not recognized
        return random.choice(self.worker_urls)
    
    def _extract_text(self, request_body: dict, endpoint: str = "") -> str:
        """
        Extract text from request body for prefix-aware routing.
        
        Args:
            request_body: Request body
            endpoint: API endpoint path
            
        Returns:
            Extracted text
        """
        if not request_body:
            return ""
            
        # Extract based on endpoint
        if "/v1/chat/completions" in endpoint:
            messages = request_body.get("messages", [])
            if messages:
                return "".join(msg.get("content", "") for msg in messages if "content" in msg)
            
        elif "/v1/completions" in endpoint:
            return request_body.get("prompt", "")
            
        elif "/generate" in endpoint:
            return request_body.get("prompt", "")
            
        # Default extraction
        if "messages" in request_body:
            messages = request_body.get("messages", [])
            return "".join(msg.get("content", "") for msg in messages if "content" in msg)
            
        if "prompt" in request_body:
            return request_body.get("prompt", "")
            
        return ""
    
    async def forward_request(self, request: Request, endpoint: str):
        """Forward a regular request to one of the GPU workers."""
        try:
            # Extract request body
            request_body = await request.json() if await request.body() else None
            
            # Get prefix_id from headers if available
            prefix_id = request.headers.get("X-Prefix-ID", "-1")
            # Select worker based on policy
            text = self._extract_text(request_body, endpoint) if self.policy == PolicyType.PREFIX_AWARE else ""
            worker_url = self._select_worker(text, prefix_id)
            
            # Forward the request
            async with httpx.AsyncClient() as client:
                if request_body:
                    response = await client.post(f"{worker_url}{endpoint}", json=request_body, timeout=self.timeout)
                else:
                    response = await client.get(f"{worker_url}{endpoint}", timeout=self.timeout)

                # Decrease queue count for PREFIX_AWARE
                if self.policy == PolicyType.PREFIX_AWARE:
                    with self.queue_lock:
                        self.running_queue[worker_url] = max(0, self.running_queue[worker_url] - 1)

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response.headers,
                    media_type=response.headers.get("content-type", "application/json")
                )
                
        except Exception as e:
            # Decrease queue count for PREFIX_AWARE if there was an error
            if self.policy == PolicyType.PREFIX_AWARE and 'worker_url' in locals():
                with self.queue_lock:
                    self.running_queue[worker_url] = max(0, self.running_queue[worker_url] - 1)
            
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )

    async def forward_streaming_request(self, request: Request, endpoint: str):
        """Forward a streaming request to one of the GPU workers."""
            # Get prefix_id from headers if available
        prefix_id = request.headers.get("x-prefix-id", "-1")
        try:
            # Extract request body
            request_body = await request.json() if await request.body() else None
            
            # Select worker based on policy
            text = self._extract_text(request_body) if self.policy == PolicyType.PREFIX_AWARE else ""
            worker_url = self._select_worker(text, prefix_id)
            
            # Ensure streaming is enabled
            if request_body and "stream" not in request_body:
                request_body["stream"] = True
            
            # Create streaming response
            async def stream_generator():
                try:
                    async with httpx.AsyncClient() as client:
                        async with client.stream("POST", f"{worker_url}{endpoint}", json=request_body, timeout=self.timeout*2) as response:
                            if response.status_code != 200:
                                yield json.dumps({"error": f"Worker returned status code: {response.status_code}"}).encode("utf-8")
                                return
                            
                            async for chunk in response.aiter_bytes():
                                yield chunk
                finally:
                    # Decrease queue count for PREFIX_AWARE when streaming is done
                    if self.policy == PolicyType.PREFIX_AWARE:
                        with self.queue_lock:
                            self.running_queue[worker_url] = max(0, self.running_queue[worker_url] - 1)
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
                
        except Exception as e:
            # Decrease queue count for PREFIX_AWARE if there was an error
            if self.policy == PolicyType.PREFIX_AWARE and 'worker_url' in locals():
                with self.queue_lock:
                    self.running_queue[worker_url] = max(0, self.running_queue[worker_url] - 1)
            
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )
    
    async def get_all_models(self):
        """List all models from all workers."""
        all_models = []
        
        async with httpx.AsyncClient() as client:
            for i, worker_url in enumerate(self.worker_urls):
                try:
                    response = await client.get(f"{worker_url}/v1/models", timeout=10.0)
                    if response.status_code == 200:
                        worker_models = response.json()
                        if "data" in worker_models:
                            for model in worker_models["data"]:
                                model["worker_index"] = i
                                model["worker_url"] = worker_url
                                all_models.append(model)
                except Exception as e:
                    print(f"Error getting models from {worker_url}: {str(e)}")
        
        return {"object": "list", "data": all_models}
    
    async def reset_prefix_cache(self):
        """Reset prefix cache on all workers."""
        results = {}
        
        async with httpx.AsyncClient() as client:
            for i, worker_url in enumerate(self.worker_urls):
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
        
        # Also reset local tree if using PREFIX_AWARE policy
        if self.policy == PolicyType.PREFIX_AWARE:
            self.tree = Tree()
            self.tree.tenant_char_count = {url: 0 for url in self.worker_urls}
        
        return {
            "status": "completed",
            "results": results
        }
    
    def get_health(self):
        """Get health status of the router."""
        return {
            "status": "healthy" if self.worker_urls else "unhealthy",
            "available_workers": list(range(len(self.worker_urls))),
            "worker_count": len(self.worker_urls),
            "worker_urls": self.worker_urls,
            "policy": self.policy.value
        }
    
    def get_router_health(self):
        """Get router-only health status."""
        health_info = {
            "status": "healthy",
            "worker_urls_configured": len(self.worker_urls),
            "worker_urls": self.worker_urls,
            "policy": self.policy.value
        }
        
        # Add tree info if using PREFIX_AWARE policy
        if self.policy == PolicyType.PREFIX_AWARE:
            health_info["tree_size"] = sum(self.tree.tenant_char_count.values())
            health_info["tenant_sizes"] = self.tree.tenant_char_count
        
        return health_info 
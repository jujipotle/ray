import json
import random
import threading
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import requests

from tree import Tree


class PolicyType(Enum):
    """Router policy types."""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    CACHE_AWARE = "cache_aware"


class Router:
    """
    A high-performance router for distributing requests across worker nodes.
    
    This router supports three policies:
    1. Random - Randomly select workers
    2. RoundRobin - Distribute requests in round-robin fashion
    3. CacheAware - Distribute requests based on cache state and load balance
    """
    
    def __init__(
        self,
        worker_urls: List[str],
        policy: PolicyType = PolicyType.ROUND_ROBIN,
        timeout_secs: int = 60,
        interval_secs: int = 10,
        cache_threshold: float = 0.5,
        balance_abs_threshold: int = 32,
        balance_rel_threshold: float = 1.0001,
        eviction_interval_secs: int = 60,
        max_tree_size: int = 2**24,
    ):
        """Initialize the router with the given parameters."""
        self.worker_urls = worker_urls
        self.policy = policy
        self.timeout_secs = timeout_secs
        self.interval_secs = interval_secs
        
        # Cache-aware specific parameters
        self.cache_threshold = cache_threshold
        self.balance_abs_threshold = balance_abs_threshold
        self.balance_rel_threshold = balance_rel_threshold
        self.eviction_interval_secs = eviction_interval_secs
        self.max_tree_size = max_tree_size
        
        # For RoundRobin policy
        self.current_index = 0
        self.round_robin_lock = threading.Lock()
        
        # For CacheAware policy
        self.tree = Tree()
        self.tree.tenant_char_count = {url: 0 for url in worker_urls}
        self.running_queue = {url: 0 for url in worker_urls}
        self.processed_queue = {url: 0 for url in worker_urls}
        self.queue_lock = threading.Lock()
        
        # Make sure all workers are healthy before starting
        self._wait_for_healthy_workers()
        
        # Start eviction thread for CacheAware
        if policy == PolicyType.CACHE_AWARE:
            self.eviction_thread = threading.Thread(
                target=self._eviction_thread_func,
                daemon=True
            )
            self.eviction_thread.start()
    
    def _wait_for_healthy_workers(self) -> None:
        """
        Wait for all workers to become healthy.
        
        A worker is considered healthy if it responds to a GET /health request
        with a 200 status code.
        """
        print(f"Waiting for all workers to become healthy...")
        start_time = time.time()
        
        while time.time() - start_time < self.timeout_secs:
            unhealthy_workers = []
            
            for worker_url in self.worker_urls:
                try:
                    response = requests.get(f"{worker_url}/health", timeout=5)
                    if response.status_code != 200:
                        unhealthy_workers.append(worker_url)
                except requests.RequestException:
                    unhealthy_workers.append(worker_url)
            
            if not unhealthy_workers:
                print("All workers are healthy!")
                return
            
            print(f"Waiting for {len(unhealthy_workers)} unhealthy workers: {unhealthy_workers}")
            time.sleep(self.interval_secs)
        
        raise TimeoutError(f"Timed out waiting for workers to become healthy after {self.timeout_secs} seconds")
    
    def _eviction_thread_func(self) -> None:
        """Background thread that periodically evicts nodes from the tree."""
        while True:
            time.sleep(self.eviction_interval_secs)
            self.tree.evict_tenant_by_size(self.max_tree_size)
    
    def _get_text_from_request(self, body: dict, route: str) -> str:
        """
        Extract text from the request body based on the route.
        
        This is used to determine what text to match in the tree for cache-aware routing.
        """
        if "/v1/chat/completions" in route:
            # Extract messages from OpenAI chat completion request
            messages = body.get("messages", [])
            if messages:
                # Concatenate all messages, focusing on content
                return "".join(msg.get("content", "") for msg in messages if "content" in msg)
        
        elif "/v1/completions" in route:
            # Extract prompt from OpenAI completion request
            return body.get("prompt", "")
        
        elif "/generate" in route:
            # Extract prompt from sglang generate request
            return body.get("prompt", "")
        
        # Default: return empty string if no matching pattern
        return ""
    
    def _select_worker(self, body: dict, route: str) -> str:
        """
        Select a worker URL based on the current policy.
        
        For cache-aware routing, this involves checking the tree for a matching pattern
        and either using the matched worker or the least loaded worker.
        """
        if self.policy == PolicyType.ROUND_ROBIN:
            with self.round_robin_lock:
                selected_url = self.worker_urls[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.worker_urls)
                return selected_url
        
        elif self.policy == PolicyType.RANDOM:
            return random.choice(self.worker_urls)
        
        elif self.policy == PolicyType.CACHE_AWARE:
            text = self._get_text_from_request(body, route)
            
            with self.queue_lock:
                # Get current load statistics
                max_load = max(self.running_queue.values()) if self.running_queue else 0
                min_load = min(self.running_queue.values()) if self.running_queue else 0
                
                # Check if load is imbalanced
                is_imbalanced = (max_load - min_load > self.balance_abs_threshold and
                                max_load > min_load * self.balance_rel_threshold)
                
                if is_imbalanced:
                    # Use shortest queue routing when load is imbalanced
                    print(f"Load balancing triggered. Max load: {max_load}, Min load: {min_load}")
                    min_queue_items = sorted(self.running_queue.items(), key=lambda x: x[1])
                    selected_url = min_queue_items[0][0] if min_queue_items else self.worker_urls[0]
                else:
                    # Use cache-aware routing when load is balanced
                    matched_text, matched_worker = self.tree.prefix_match(text)
                    
                    if matched_worker != "empty" and matched_worker in self.worker_urls:
                        # Calculate match rate
                        match_rate = len(matched_text) / len(text) if text else 0
                        
                        if match_rate > self.cache_threshold:
                            selected_url = matched_worker
                            print(f"Cache hit! Match rate: {match_rate:.2f}, Worker: {selected_url}")
                        else:
                            # Get worker with smallest tree
                            selected_url = self.tree.get_smallest_tenant()
                            if selected_url == "empty" or selected_url not in self.worker_urls:
                                selected_url = self.worker_urls[0]
                            print(f"Cache miss. Match rate: {match_rate:.2f}, Selected smallest tree: {selected_url}")
                    else:
                        # No match found, use worker with smallest tree
                        selected_url = self.tree.get_smallest_tenant()
                        if selected_url == "empty" or selected_url not in self.worker_urls:
                            selected_url = self.worker_urls[0]
                
                # Update running queue and tree
                self.running_queue[selected_url] += 1
                self.processed_queue[selected_url] += 1
                
                # Insert text into tree for future matching
                if text:
                    self.tree.insert(text, selected_url)
                
                return selected_url
        
        # Default to first worker if policy is not recognized
        return self.worker_urls[0]
    
    def route_request(self, route: str, body: Union[str, bytes, dict], headers: Optional[Dict[str, str]] = None) -> Tuple[int, dict, bytes]:
        """
        Route a request to an appropriate worker based on the current policy.
        
        Args:
            route: The API route being called
            body: The request body (string, bytes, or dict)
            headers: Optional request headers
            
        Returns:
            Tuple of (status_code, response_headers, response_body)
        """
        # Parse body if it's a string or bytes
        parsed_body = None
        if isinstance(body, (str, bytes)):
            try:
                parsed_body = json.loads(body)
            except json.JSONDecodeError:
                parsed_body = {}
        else:
            parsed_body = body
        
        # Select worker
        worker_url = self._select_worker(parsed_body, route)
        
        # Prepare headers
        request_headers = headers or {}
        
        # Make request to worker
        try:
            # Convert body back to JSON if it was parsed
            request_body = json.dumps(parsed_body) if parsed_body else body
            
            response = requests.post(
                f"{worker_url}{route}",
                data=request_body,
                headers=request_headers,
                timeout=self.timeout_secs
            )
            
            # Decrease queue count for CacheAware
            if self.policy == PolicyType.CACHE_AWARE:
                with self.queue_lock:
                    self.running_queue[worker_url] = max(0, self.running_queue[worker_url] - 1)
            
            # Return response
            return response.status_code, dict(response.headers), response.content
        
        except requests.RequestException as e:
            print(f"Error routing request to {worker_url}: {str(e)}")
            
            # Decrease queue count for CacheAware
            if self.policy == PolicyType.CACHE_AWARE:
                with self.queue_lock:
                    self.running_queue[worker_url] = max(0, self.running_queue[worker_url] - 1)
            
            # Return error response
            return 500, {}, json.dumps({"error": str(e)}).encode()
    
    def add_worker(self, worker_url: str) -> bool:
        """
        Add a new worker to the router.
        
        Returns True if worker was added successfully, False otherwise.
        """
        if worker_url in self.worker_urls:
            print(f"Worker {worker_url} already exists")
            return False
        
        # Check if worker is healthy
        try:
            response = requests.get(f"{worker_url}/health", timeout=5)
            if response.status_code != 200:
                print(f"Worker {worker_url} is not healthy")
                return False
        except requests.RequestException:
            print(f"Failed to connect to worker {worker_url}")
            return False
        
        # Add worker
        self.worker_urls.append(worker_url)
        
        # Update CacheAware structures
        if self.policy == PolicyType.CACHE_AWARE:
            with self.queue_lock:
                self.running_queue[worker_url] = 0
                self.processed_queue[worker_url] = 0
        
        print(f"Added worker {worker_url}")
        return True
    
    def remove_worker(self, worker_url: str) -> bool:
        """
        Remove a worker from the router.
        
        Returns True if worker was removed successfully, False otherwise.
        """
        if worker_url not in self.worker_urls:
            print(f"Worker {worker_url} does not exist")
            return False
        
        # Remove worker
        self.worker_urls.remove(worker_url)
        
        # Update CacheAware structures
        if self.policy == PolicyType.CACHE_AWARE:
            with self.queue_lock:
                if worker_url in self.running_queue:
                    del self.running_queue[worker_url]
                if worker_url in self.processed_queue:
                    del self.processed_queue[worker_url]
                self.tree.remove_tenant(worker_url)
        
        print(f"Removed worker {worker_url}")
        return True 
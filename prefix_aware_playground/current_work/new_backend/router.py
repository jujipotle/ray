import asyncio
import aiohttp
import threading
import time
import random
import json
import logging
import requests
from tree import Tree

logger = logging.getLogger(__name__)

# --- Policy configuration classes ---

class PolicyConfig:
    pass

class RandomConfig(PolicyConfig):
    def __init__(self, timeout_secs, interval_secs):
        self.timeout_secs = timeout_secs
        self.interval_secs = interval_secs

class RoundRobinConfig(PolicyConfig):
    def __init__(self, timeout_secs, interval_secs):
        self.timeout_secs = timeout_secs
        self.interval_secs = interval_secs

class CacheAwareConfig(PolicyConfig):
    def __init__(self, cache_threshold, balance_abs_threshold, balance_rel_threshold,
                 eviction_interval_secs, max_tree_size, timeout_secs, interval_secs):
        self.cache_threshold = cache_threshold
        self.balance_abs_threshold = balance_abs_threshold
        self.balance_rel_threshold = balance_rel_threshold
        self.eviction_interval_secs = eviction_interval_secs
        self.max_tree_size = max_tree_size
        self.timeout_secs = timeout_secs
        self.interval_secs = interval_secs

# --- Health-check helper ---

def wait_for_healthy_workers(worker_urls, timeout_secs, interval_secs):
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout_secs:
            logger.error(f"Timeout {timeout_secs}s waiting for workers {worker_urls} to become healthy.")
            raise Exception("Timeout waiting for workers to become healthy.")
        all_healthy = True
        unhealthy_workers = []
        for url in worker_urls:
            try:
                res = requests.get(f"{url}/health")
                if res.status_code != 200:
                    logger.info(f"Worker {url} health check pending with status: {res.status_code}")
                    all_healthy = False
                    unhealthy_workers.append((url, f"Status: {res.status_code}"))
            except Exception as e:
                logger.info(f"Worker {url} health check pending with error: {e}")
                all_healthy = False
                unhealthy_workers.append((url, f"Error: {e}"))
        if all_healthy:
            logger.info("All workers are healthy")
            return
        else:
            logger.info("Unhealthy workers:")
            for url, reason in unhealthy_workers:
                logger.info(f"  {url} - {reason}")
            time.sleep(interval_secs)

# --- Router class ---

class Router:
    def __init__(self, worker_urls, policy_config):
        # wait until all workers are healthy
        wait_for_healthy_workers(worker_urls, policy_config.timeout_secs, policy_config.interval_secs)
        self.worker_urls = worker_urls[:]  # copy of list
        self.timeout_secs = policy_config.timeout_secs
        self.interval_secs = policy_config.interval_secs
        self.lock = threading.Lock()
        if isinstance(policy_config, RandomConfig):
            self.policy = "Random"
        elif isinstance(policy_config, RoundRobinConfig):
            self.policy = "RoundRobin"
            self.current_index = 0
        elif isinstance(policy_config, CacheAwareConfig):
            self.policy = "CacheAware"
            self.tree = Tree()
            self.running_queue = {url: 0 for url in self.worker_urls}
            self.processed_queue = {url: 0 for url in self.worker_urls}
            self.cache_threshold = policy_config.cache_threshold
            self.balance_abs_threshold = policy_config.balance_abs_threshold
            self.balance_rel_threshold = policy_config.balance_rel_threshold
            self.eviction_interval_secs = policy_config.eviction_interval_secs
            self.max_tree_size = policy_config.max_tree_size
            self.eviction_thread = threading.Thread(target=self._eviction_loop, daemon=True)
            self.eviction_thread.start()
            for url in self.worker_urls:
                self.tree.insert("", url)
        else:
            raise Exception("Unknown policy config")

    def _eviction_loop(self):
        while True:
            time.sleep(self.eviction_interval_secs)
            with self.lock:
                self.tree.evict_tenant_by_size(self.max_tree_size)
                logger.info(f"Processed Queue: {self.processed_queue}")
                logger.info(f"Running Queue: {self.running_queue}")

    def select_first_worker(self):
        with self.lock:
            if not self.worker_urls:
                raise Exception("No workers available")
            return self.worker_urls[0]

    async def send_request(self, session, worker_url, route, req_headers):
        url = f"{worker_url}{route}"
        headers = {k: v for k, v in req_headers.items()}
        try:
            async with session.get(url, headers=headers) as response:
                status = response.status
                body = await response.read()
                return status, body
        except Exception as e:
            logger.error(f"Error sending request to {worker_url}: {e}")
            return 500, str(e).encode()

    async def route_to_first(self, session, route, req_headers):
        MAX_REQUEST_RETRIES = 3
        MAX_TOTAL_RETRIES = 6
        total_retries = 0
        while total_retries < MAX_TOTAL_RETRIES:
            try:
                worker_url = self.select_first_worker()
            except Exception as e:
                return 500, str(e).encode()
            request_retries = 0
            while request_retries < MAX_REQUEST_RETRIES:
                if total_retries >= 1:
                    logger.info(f"Retrying request after {total_retries} failed attempts")
                status, body = await self.send_request(session, worker_url, route, req_headers)
                if status == 200:
                    return status, body
                else:
                    h_status, _ = await self.send_request(session, worker_url, "/health", req_headers)
                    if h_status == 200:
                        return status, body
                logger.warning(f"Request to {worker_url} failed (attempt {request_retries+1}/{MAX_REQUEST_RETRIES})")
                request_retries += 1
                total_retries += 1
                if request_retries == MAX_REQUEST_RETRIES:
                    logger.warning(f"Removing failed worker: {worker_url}")
                    self.remove_worker(worker_url)
                    break
        return 500, b"All retry attempts failed"

    def get_text_from_request(self, body: bytes, route: str) -> str:
        try:
            data = json.loads(body.decode())
        except Exception:
            logger.warning("Failed to parse JSON from request body.")
            return ""
        if route == "/generate":
            return data.get("text", "")
        elif route in ["/v1/chat/completions", "/v1/completions"]:
            if "messages" in data:
                return json.dumps(data["messages"])
            elif "prompt" in data:
                return data.get("prompt", "")
            else:
                logger.warning("Failed to find 'messages' or 'prompt' in request body.")
                return ""
        else:
            logger.warning(f"Unknown route: {route} - defaulting to fallback string")
            return ""

    def select_generate_worker(self, body: bytes, route: str):
        text = self.get_text_from_request(body, route)
        selected_url = None
        with self.lock:
            if self.policy == "RoundRobin":
                idx = self.current_index % len(self.worker_urls)
                self.current_index = (self.current_index + 1) % len(self.worker_urls)
                selected_url = self.worker_urls[idx]
            elif self.policy == "Random":
                selected_url = random.choice(self.worker_urls)
            elif self.policy == "CacheAware":
                # Compute current load statistics from the running queue.
                if self.running_queue:
                    max_load = max(self.running_queue.values())
                    min_load = min(self.running_queue.values())
                else:
                    max_load = min_load = 0

                # Determine if the system is imbalanced.
                is_imbalanced = ((max_load - min_load) > self.balance_abs_threshold and
                                (max_load > min_load * self.balance_rel_threshold))
                if is_imbalanced:
                    logging.info(
                        f"Load balancing triggered due to workload imbalance:\n"
                        f"Max load: {max_load}, Min load: {min_load}\n"
                        f"Current running queue: {self.running_queue}"
                    )
                    # Use shortest queue routing when load is imbalanced.
                    selected_url = min(self.running_queue.items(), key=lambda kv: kv[1])[0]
                else:
                    # Use cache-aware routing when load is balanced.
                    matched_text, matched_worker = self.tree.prefix_match(text)
                    match_rate = (len(matched_text) / len(text)) if text else 0
                    if match_rate > self.cache_threshold:
                        selected_url = matched_worker
                    else:
                        selected_url = self.tree.get_smallest_tenant()

                if selected_url in self.running_queue:
                    self.running_queue[selected_url] += 1
                else:
                    self.running_queue[selected_url] = 1
                if selected_url in self.processed_queue:
                    self.processed_queue[selected_url] += 1
                else:
                    self.processed_queue[selected_url] = 1
                self.tree.insert(text, selected_url)
            else:
                selected_url = self.select_first_worker()
        return selected_url

    async def send_generate_request(self, session, req_headers, body, route, worker_url):
        url = f"{worker_url}{route}"
        try:
            async with session.post(url, headers=req_headers, data=body) as response:
                status = response.status
                try:
                    data_json = json.loads(body.decode())
                    is_stream = data_json.get("stream", False)
                except Exception:
                    is_stream = False
                if not is_stream:
                    resp_body = await response.read()
                    if self.policy == "CacheAware":
                        with self.lock:
                            if worker_url in self.running_queue:
                                self.running_queue[worker_url] = max(self.running_queue[worker_url] - 1, 0)
                    return status, resp_body
                else:
                    chunks = []
                    async for chunk in response.content.iter_chunked(1024):
                        chunks.append(chunk)
                        if b"data: [DONE]" in chunk:
                            with self.lock:
                                if worker_url in self.running_queue:
                                    self.running_queue[worker_url] = max(self.running_queue[worker_url] - 1, 0)
                    return status, b"".join(chunks)
        except Exception as e:
            logger.error(f"Error sending generate request to {worker_url}: {e}")
            return 500, str(e).encode()

    async def route_generate_request(self, session, req_headers, body, route):
        MAX_REQUEST_RETRIES = 3
        MAX_TOTAL_RETRIES = 6
        total_retries = 0
        while total_retries < MAX_TOTAL_RETRIES:
            worker_url = self.select_generate_worker(body, route)
            request_retries = 0
            while request_retries < MAX_REQUEST_RETRIES:
                if total_retries >= 1:
                    logger.info(f"Retrying generate request after {total_retries} failed attempts")
                status, resp_body = await self.send_generate_request(session, req_headers, body, route, worker_url)
                if status == 200:
                    return status, resp_body
                else:
                    h_status, _ = await self.send_request(session, worker_url, "/health", req_headers)
                    if h_status == 200:
                        return status, resp_body
                logger.warning(f"Generate request to {worker_url} failed (attempt {request_retries+1}/{MAX_REQUEST_RETRIES})")
                request_retries += 1
                total_retries += 1
                if request_retries == MAX_REQUEST_RETRIES:
                    logger.warning(f"Removing failed worker: {worker_url}")
                    self.remove_worker(worker_url)
                    break
        return 500, b"All retry attempts failed"

    async def add_worker(self, worker_url: str):
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            while True:
                if time.time() - start_time > self.timeout_secs:
                    logger.error(f"Timeout waiting for worker {worker_url} to become healthy.")
                    return f"Timeout waiting for worker {worker_url} to become healthy."
                try:
                    async with session.get(f"{worker_url}/health") as resp:
                        if resp.status == 200:
                            with self.lock:
                                if worker_url in self.worker_urls:
                                    return f"Worker {worker_url} already exists"
                                self.worker_urls.append(worker_url)
                                logger.info(f"Added worker: {worker_url}")
                                if self.policy == "CacheAware":
                                    self.running_queue[worker_url] = 0
                                    self.processed_queue[worker_url] = 0
                                    self.tree.insert("", worker_url)
                            return f"Successfully added worker: {worker_url}"
                        else:
                            logger.info(f"Worker {worker_url} health check pending with status: {resp.status}")
                except Exception as e:
                    logger.info(f"Worker {worker_url} health check pending with error: {e}")
                await asyncio.sleep(self.interval_secs)

    def remove_worker(self, worker_url: str):
        with self.lock:
            if worker_url in self.worker_urls:
                self.worker_urls.remove(worker_url)
                logger.info(f"Removed worker: {worker_url}")
            else:
                logger.warning(f"Worker {worker_url} not found, skipping removal")
                return
            if self.policy == "CacheAware":
                self.tree.remove_tenant(worker_url)
                if worker_url in self.running_queue:
                    del self.running_queue[worker_url]
                if worker_url in self.processed_queue:
                    del self.processed_queue[worker_url]
                logger.info(f"Removed worker from tree and cleaned up queues: {worker_url}")
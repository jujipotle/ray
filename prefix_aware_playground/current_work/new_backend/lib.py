import enum
import uvicorn
from router import Router, RandomConfig, RoundRobinConfig, CacheAwareConfig

class PolicyType(enum.Enum):
    Random = "Random"
    RoundRobin = "RoundRobin"
    CacheAware = "CacheAware"

class RouterWrapper:
    def __init__(self, worker_urls, policy=PolicyType.RoundRobin, host="127.0.0.1", port=3001, 
                 worker_startup_timeout_secs=300, worker_startup_check_interval=10, 
                 cache_threshold=0.50, balance_abs_threshold=32, balance_rel_threshold=1.0001,
                 eviction_interval_secs=60, max_tree_size=(2**24), max_payload_size=4*1024*1024,
                 verbose=False):
        self.host = host
        self.port = port
        self.worker_urls = worker_urls
        self.worker_startup_timeout_secs = worker_startup_timeout_secs
        self.worker_startup_check_interval = worker_startup_check_interval
        self.cache_threshold = cache_threshold
        self.balance_abs_threshold = balance_abs_threshold
        self.balance_rel_threshold = balance_rel_threshold
        self.eviction_interval_secs = eviction_interval_secs
        self.max_tree_size = max_tree_size
        self.max_payload_size = max_payload_size
        self.verbose = verbose
        if policy == PolicyType.Random:
            self.policy_config = RandomConfig(worker_startup_timeout_secs, worker_startup_check_interval)
        elif policy == PolicyType.RoundRobin:
            self.policy_config = RoundRobinConfig(worker_startup_timeout_secs, worker_startup_check_interval)
        elif policy == PolicyType.CacheAware:
            self.policy_config = CacheAwareConfig(cache_threshold, balance_abs_threshold, balance_rel_threshold,
                                                  eviction_interval_secs, max_tree_size, 
                                                  worker_startup_timeout_secs, worker_startup_check_interval)
        else:
            raise Exception("Unknown policy type")
        self.router = Router(worker_urls, self.policy_config)
    
    def start(self):
        uvicorn.run("server:app", host=self.host, port=self.port)
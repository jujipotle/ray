import ray
from ray import serve
import subprocess
print("Starting Ray Serve...")
serve.start(http_options={"host": "0.0.0.0", "port": 8001})
print("Ray Serve started")
# Run serve run config.yaml
print("Running serve run config.yaml")
server_process = subprocess.Popen(["serve", "run", "config.yaml"])
print("Server process started")

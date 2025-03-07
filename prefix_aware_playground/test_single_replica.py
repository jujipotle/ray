import subprocess
import os
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
host = "127.0.0.1"
port = "8002"
enable_prefix_caching = True
gpu_idx = 1
cmd = [
    "vllm", "serve", model_name,
    "--host", host,
    "--port", port,
    "--enable-prefix-caching" if enable_prefix_caching else "--no-enable-prefix-caching",
]
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
env["VLLM_SERVER_DEV_MODE"] = "1"
print(f"Starting VLLM worker on port {port} using GPU {gpu_idx}")
process = subprocess.Popen(
    cmd, env=env
)

try:
    process.wait()
except KeyboardInterrupt:
    process.terminate()
    process.wait()
    print("Worker stopped")
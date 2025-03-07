#!/bin/bash
# Script to run the benchmark sweep with server restarts between runs

# Default values
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
GPU_TYPE="L4"
ROUTER_PORT=8000
WORKER_PORTS="8001,8002,8003,8004"
GPU_INDICES="0,1,2,3"
SWEEP_TYPES="prefix_reuse_rate,prefix_length_ratio"
NUM_POINTS=1
TOTAL_REQUESTS=10
TOTAL_PROMPT_LEN=2176
OUTPUT_LEN=256
MAX_CONCURRENCY=128
PREFIX_CACHED=true
DISABLE_LOG_REQUESTS=true
HOST="127.0.0.1"

# Process IDs for router and replicas
ROUTER_PID=""
REPLICAS_PID=""

# Function to clean up and exit
cleanup_and_exit() {
  echo -e "\nBenchmark interrupted. Cleaning up..."
  
  # Kill the replicas if they're running
  if [ ! -z "$REPLICAS_PID" ]; then
    echo "Stopping replicas..."
    kill $REPLICAS_PID 2>/dev/null
    wait $REPLICAS_PID 2>/dev/null
  fi
  
  # Kill the router if it's running
  if [ ! -z "$ROUTER_PID" ]; then
    echo "Stopping router..."
    kill $ROUTER_PID 2>/dev/null
    wait $ROUTER_PID 2>/dev/null
  fi
  
  echo "Exiting benchmark sweep."
  exit 1
}

# Function to reset prefix caches for all workers
reset_prefix_caches() {
  echo "Resetting prefix caches for all workers..."
  
  # Reset router cache
  echo "Resetting router prefix cache..."
  curl -s -X POST "http://$HOST:$ROUTER_PORT/reset_prefix_cache" > /dev/null
  
  # Reset each worker's cache
  IFS=',' read -ra PORTS <<< "$WORKER_PORTS"
  for PORT in "${PORTS[@]}"; do
    echo "Resetting worker prefix cache on port $PORT..."
    curl -s -X POST "http://$HOST:$PORT/reset_prefix_cache" > /dev/null
  done
  
  echo "All prefix caches reset."
}

# Set up trap to catch Ctrl+C and other termination signals
trap cleanup_and_exit SIGINT SIGTERM

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --router-port)
      ROUTER_PORT="$2"
      shift 2
      ;;
    --worker-ports)
      WORKER_PORTS="$2"
      shift 2
      ;;
    --gpu-indices)
      GPU_INDICES="$2"
      shift 2
      ;;
    --sweep-types)
      SWEEP_TYPES="$2"
      shift 2
      ;;
    --num-points)
      NUM_POINTS="$2"
      shift 2
      ;;
    --total-requests)
      TOTAL_REQUESTS="$2"
      shift 2
      ;;
    --total-prompt-len)
      TOTAL_PROMPT_LEN="$2"
      shift 2
      ;;
    --output-len)
      OUTPUT_LEN="$2"
      shift 2
      ;;
    --max-concurrency)
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    --prefix-cached)
      if [[ "$2" == "false" ]]; then
        PREFIX_CACHED=false
      else
        PREFIX_CACHED=true
      fi
      shift 2
      ;;
    --disable-log-requests)
      if [[ "$2" == "false" ]]; then
        DISABLE_LOG_REQUESTS=false
      else
        DISABLE_LOG_REQUESTS=true
      fi
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 1. Start the router in the background
echo "Starting router on port $ROUTER_PORT..."
python run_router.py --port $ROUTER_PORT --host $HOST --worker-ports $WORKER_PORTS &
ROUTER_PID=$!

# Give the router time to start
sleep 2

# Check if router started successfully
if ! kill -0 $ROUTER_PID 2>/dev/null; then
  echo "Router failed to start. Exiting."
  cleanup_and_exit
fi

# 2. Start the replicas in the background
echo "Starting VLLM replicas..."
python run_replicas.py \
  --worker-ports $WORKER_PORTS \
  --gpu-indices $GPU_INDICES \
  --model-name "$MODEL_NAME" \
  --host $HOST &
REPLICAS_PID=$!

# Give the replicas time to start
echo "Waiting for replicas to start..."
sleep 20

# Check if replicas started successfully
if ! kill -0 $REPLICAS_PID 2>/dev/null; then
  echo "Replicas failed to start. Exiting."
  cleanup_and_exit
fi

# 3. Run the benchmark sweeps
# Convert comma-separated sweep types to array
IFS=',' read -ra SWEEP_TYPE_ARRAY <<< "$SWEEP_TYPES"

# Run each sweep type
for SWEEP_TYPE in "${SWEEP_TYPE_ARRAY[@]}"; do
  echo "Running benchmark sweep for $SWEEP_TYPE..."
  
  # Run the Python script
  python bench_prefix_sweep.py \
    --gpu-type "$GPU_TYPE" \
    --model-name "$MODEL_NAME" \
    --base-url "http://$HOST:$ROUTER_PORT" \
    --is-prefix-cached $PREFIX_CACHED \
    --gpu-indices "$GPU_INDICES" \
    --worker-ports "$WORKER_PORTS" \
    --sweep-type "$SWEEP_TYPE" \
    --num-points $NUM_POINTS \
    --total-requests $TOTAL_REQUESTS \
    --total-prompt-len $TOTAL_PROMPT_LEN \
    --output-len $OUTPUT_LEN \
    --max-concurrency $MAX_CONCURRENCY \
    --disable-log-requests $DISABLE_LOG_REQUESTS
  
  # Check if the Python script exited normally
  if [ $? -ne 0 ]; then
    echo "Benchmark sweep for $SWEEP_TYPE was interrupted or failed."
    cleanup_and_exit
  fi
  
  echo "Completed $SWEEP_TYPE sweep"
  
  # Optional: Add a short pause between sweeps
  sleep 5
done

# 5. Stop the replicas
echo "Stopping replicas..."
kill $REPLICAS_PID 2>/dev/null
wait $REPLICAS_PID 2>/dev/null

# 6. Stop the router
echo "Stopping router..."
kill $ROUTER_PID 2>/dev/null
wait $ROUTER_PID 2>/dev/null

echo "All benchmark sweeps complete!" 
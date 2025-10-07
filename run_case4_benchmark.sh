#!/bin/bash

# Case 4: JIT + Async Rolling CUDA Graph Benchmark
# Runs two processes: JIT + CUDA Graph Generator (server) + Context Creator (client)

set -e

MODEL_NAME="meta-llama/Llama-2-7b-hf"
SOCKET_PATH="/tmp/jit_async_cuda_graph.sock"
MAX_GRAPHS=50

echo "🚀 Starting Case 4: JIT + Async Rolling CUDA Graph Benchmark"
echo "Model: $MODEL_NAME"
echo "Socket: $SOCKET_PATH"
echo "Max Graphs: $MAX_GRAPHS"
echo ""

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "case4_jit_cuda_generator.py" || true
pkill -f "case4_jit_cuda_client.py" || true
sleep 2

# Remove existing socket file
rm -f "$SOCKET_PATH"

# Start the JIT + CUDA graph generator server in background
echo "🔄 Starting JIT + CUDA Graph Generator Server..."
cd /home/azureuser/divakar_projects/cuda_graph_sharing
source venv/bin/activate
python case4_jit_cuda_generator.py \
    --model "$MODEL_NAME" \
    --socket "$SOCKET_PATH" \
    --max-graphs "$MAX_GRAPHS" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start and pre-capture graphs
echo "⏳ Waiting for server to start and pre-capture 50 graphs..."
sleep 15

# Check if server is ready
if [ ! -S "$SOCKET_PATH" ]; then
    echo "❌ Server failed to start. Socket file not found: $SOCKET_PATH"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Server is ready!"

# Run the benchmark client
echo "📊 Running benchmark client..."
python case4_jit_cuda_client.py \
    --socket "$SOCKET_PATH" \
    --ttft-iterations 5 \
    --p99-iterations 100

# Clean up
echo "🧹 Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
sleep 2
rm -f "$SOCKET_PATH"

echo "✅ Case 4 benchmark completed!"
echo "📁 Results saved to: output/case4_jit_cuda_ttft.csv, output/case4_jit_cuda_p99.csv"

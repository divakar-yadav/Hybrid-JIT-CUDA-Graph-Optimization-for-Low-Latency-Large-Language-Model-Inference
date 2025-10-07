#!/bin/bash

# Master script to run all 5 cases and generate final TTFT/P99 tables
# Case 1: Baseline (HuggingFace Transformers)
# Case 2: Async Rolling CUDA Graph Only (two processes)
# Case 3: JIT Only (dynamic operations)
# Case 4: JIT + Async Rolling CUDA Graph (two processes)
# Case 5: TensorRT-LLM

set -e

MODEL_NAME="meta-llama/Llama-2-7b-hf"
PROMPT_LENGTHS="10 50 100 150 200 250 300 350 400"

echo "🚀 Starting All 5 Cases Benchmark"
echo "Model: $MODEL_NAME"
echo "Prompt Lengths: $PROMPT_LENGTHS"
echo ""

# Create output directory
mkdir -p output

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "case2_async_cuda_generator.py" || true
pkill -f "case4_jit_cuda_generator.py" || true
pkill -f "case2_async_cuda_client.py" || true
pkill -f "case4_jit_cuda_client.py" || true
sleep 3

# Remove existing socket files
rm -f /tmp/async_cuda_graph.sock
rm -f /tmp/jit_async_cuda_graph.sock

cd /home/azureuser/divakar_projects/cuda_graph_sharing
source venv/bin/activate

echo "=" * 80
echo "CASE 1: BASELINE (HuggingFace Transformers)"
echo "=" * 80
python case1_baseline.py \
    --model "$MODEL_NAME" \
    --prompt-lengths $PROMPT_LENGTHS \
    --ttft-iterations 5 \
    --p99-iterations 100

echo ""
echo "=" * 80
echo "CASE 2: ASYNC ROLLING CUDA GRAPH ONLY"
echo "=" * 80
./run_case2_benchmark.sh

echo ""
echo "=" * 80
echo "CASE 3: JIT ONLY (Dynamic Operations)"
echo "=" * 80
python case3_jit_only.py \
    --model "$MODEL_NAME" \
    --prompt-lengths $PROMPT_LENGTHS \
    --ttft-iterations 5 \
    --p99-iterations 100

echo ""
echo "=" * 80
echo "CASE 4: JIT + ASYNC ROLLING CUDA GRAPH"
echo "=" * 80
./run_case4_benchmark.sh

echo ""
echo "=" * 80
echo "CASE 5: TENSORRT-LLM"
echo "=" * 80
python case5_tensorrt_llm.py \
    --model "$MODEL_NAME" \
    --prompt-lengths $PROMPT_LENGTHS \
    --ttft-iterations 5 \
    --p99-iterations 100

echo ""
echo "=" * 80
echo "📊 GENERATING FINAL TABLES"
echo "=" * 80

# Generate final tables
python generate_final_tables.py

echo ""
echo "✅ All 5 cases benchmark completed!"
echo "📁 Results saved to: output/"
echo "📊 Final tables: output/final_ttft_table.csv, output/final_p99_table.csv"

#!/usr/bin/env python3
"""
Case 2: Async Rolling CUDA Graph Client (Context Creator Process)
Creates context and sends to graph generator via IPC
"""

import torch
import time
import logging
import socket
import pickle
import struct
import numpy as np
import pandas as pd
import argparse
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphRequest:
    request_type: str
    sequence_length: int
    input_tokens: List[int]
    request_id: str

@dataclass
class GraphResponse:
    success: bool
    request_id: str
    output_tokens: Optional[List[int]] = None
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None

class AsyncCUDAClient:
    """Client for async CUDA graph inference via IPC"""
    
    def __init__(self, socket_path: str = "/tmp/async_cuda_graph.sock"):
        self.socket_path = socket_path
    
    def send_request(self, request: GraphRequest) -> GraphResponse:
        """Send request to server and receive response"""
        try:
            # Connect to server
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.socket_path)
            
            # Serialize request
            request_data = pickle.dumps(request)
            
            # Send request length
            sock.send(struct.pack('!I', len(request_data)))
            
            # Send request data
            sock.send(request_data)
            
            # Receive response length
            length_data = sock.recv(4)
            if not length_data:
                raise ConnectionError("No response length received")
            
            response_length = struct.unpack('!I', length_data)[0]
            
            # Receive response data
            response_data = b''
            while len(response_data) < response_length:
                packet = sock.recv(response_length - len(response_data))
                if not packet:
                    raise ConnectionError("Incomplete response received")
                response_data += packet
            
            # Deserialize response
            response = pickle.loads(response_data)
            sock.close()
            
            return response
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return GraphResponse(
                success=False,
                request_id=request.request_id,
                error_message=str(e)
            )
    
    def inference(self, seq_len: int, input_tokens: List[int]) -> Tuple[List[int], float]:
        """Execute inference via IPC"""
        request = GraphRequest(
            request_type='INFERENCE',
            sequence_length=seq_len,
            input_tokens=input_tokens,
            request_id=f'inference_{seq_len}_{int(time.time() * 1000)}'
        )
        
        response = self.send_request(request)
        
        if response.success:
            return response.output_tokens, response.latency_ms
        else:
            raise RuntimeError(f"Inference failed: {response.error_message}")

def run_case2_benchmark(
    socket_path: str = "/tmp/async_cuda_graph.sock",
    prompt_lengths: List[int] = [10, 50, 100, 150, 200, 250, 300, 350, 400],
    ttft_iterations: int = 5,
    p99_iterations: int = 100,
    vocab_size: int = 32000
):
    """Run Case 2 benchmark for TTFT and P99"""
    
    logger.info("=" * 80)
    logger.info("🚀 CASE 2: ASYNC ROLLING CUDA GRAPH BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Socket: {socket_path}")
    logger.info(f"Prompt Lengths: {prompt_lengths}")
    logger.info(f"TTFT Iterations: {ttft_iterations}")
    logger.info(f"P99 Iterations: {p99_iterations}")
    logger.info("Architecture: Two processes (Graph Generator + Context Creator)")
    logger.info("=" * 80)
    
    # Check if server is running
    if not os.path.exists(socket_path):
        logger.error(f"❌ Server not running. Socket file not found: {socket_path}")
        logger.error("   Start the server first: python case2_async_cuda_generator.py")
        return
    
    # Initialize client
    client = AsyncCUDAClient(socket_path)
    
    # TTFT Benchmark
    logger.info("📊 Running TTFT Benchmark...")
    ttft_results = []
    
    for prompt_len in prompt_lengths:
        logger.info(f"Testing TTFT for prompt length {prompt_len} tokens ({ttft_iterations} iterations):")
        
        latencies = []
        for iteration in range(ttft_iterations):
            # Generate random prompt (same seeds for consistency)
            np.random.seed(42 + iteration)
            prompt = list(np.random.randint(1, vocab_size, prompt_len))
            
            try:
                next_token_id, latency_ms = client.inference(prompt_len, prompt)
                latencies.append(latency_ms)
            except Exception as e:
                logger.warning(f"  ❌ Failed iteration {iteration}: {e}")
        
        if latencies:
            mean_lat = np.mean(latencies)
            std_lat = np.std(latencies)
            min_lat = np.min(latencies)
            max_lat = np.max(latencies)
            
            ttft_results.append({
                'prompt_length': prompt_len,
                'mean_ms': mean_lat,
                'std_ms': std_lat,
                'min_ms': min_lat,
                'max_ms': max_lat,
                'samples': len(latencies)
            })
            
            logger.info(f"  ✅ Mean: {mean_lat:.2f}ms, Std: {std_lat:.2f}ms, Min: {min_lat:.2f}ms, Max: {max_lat:.2f}ms")
        else:
            ttft_results.append({
                'prompt_length': prompt_len,
                'mean_ms': np.nan,
                'std_ms': np.nan,
                'min_ms': np.nan,
                'max_ms': np.nan,
                'samples': 0
            })
        
        logger.info("")
    
    # P99 Benchmark
    logger.info("📊 Running P99 Benchmark...")
    p99_results = []
    
    for prompt_len in prompt_lengths:
        logger.info(f"Testing P99 for prompt length {prompt_len} tokens ({p99_iterations} iterations):")
        
        latencies = []
        for iteration in range(p99_iterations):
            # Generate random prompt (same seeds for consistency)
            np.random.seed(42 + iteration)
            prompt = list(np.random.randint(1, vocab_size, prompt_len))
            
            try:
                next_token_id, latency_ms = client.inference(prompt_len, prompt)
                latencies.append(latency_ms)
            except Exception as e:
                logger.warning(f"  ❌ Failed iteration {iteration}: {e}")
            
            if (iteration + 1) % 20 == 0:
                logger.info(f"  Completed {iteration + 1}/{p99_iterations} iterations...")
        
        if latencies:
            p99 = np.percentile(latencies, 99)
            mean_lat = np.mean(latencies)
            
            p99_results.append({
                'prompt_length': prompt_len,
                'mean_ms': mean_lat,
                'p99_ms': p99,
                'samples': len(latencies)
            })
            
            logger.info(f"  ✅ Mean: {mean_lat:.2f}ms, P99: {p99:.2f}ms")
        else:
            p99_results.append({
                'prompt_length': prompt_len,
                'mean_ms': np.nan,
                'p99_ms': np.nan,
                'samples': 0
            })
        
        logger.info("")
    
    # Save results
    os.makedirs('output', exist_ok=True)
    
    ttft_df = pd.DataFrame(ttft_results)
    ttft_df.to_csv('output/case2_async_cuda_ttft.csv', index=False)
    
    p99_df = pd.DataFrame(p99_results)
    p99_df.to_csv('output/case2_async_cuda_p99.csv', index=False)
    
    logger.info("=" * 80)
    logger.info("📊 CASE 2 ASYNC CUDA GRAPH BENCHMARK SUMMARY")
    logger.info("=" * 80)
    
    for result in ttft_results:
        if not np.isnan(result['mean_ms']):
            logger.info(f"TTFT {result['prompt_length']:3d} tokens: {result['mean_ms']:6.2f}ms (std: {result['std_ms']:6.2f}ms)")
    
    logger.info("")
    for result in p99_results:
        if not np.isnan(result['mean_ms']):
            logger.info(f"P99  {result['prompt_length']:3d} tokens: {result['p99_ms']:6.2f}ms (mean: {result['mean_ms']:6.2f}ms)")
    
    logger.info("=" * 80)
    logger.info("✅ Case 2 Async CUDA Graph benchmark completed!")
    logger.info("📁 Results saved to: output/case2_async_cuda_ttft.csv, output/case2_async_cuda_p99.csv")
    
    return ttft_results, p99_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Case 2: Async CUDA Graph Client Benchmark")
    parser.add_argument("--socket", default="/tmp/async_cuda_graph.sock", help="Socket path")
    parser.add_argument("--ttft-iterations", type=int, default=5, help="TTFT iterations per length")
    parser.add_argument("--p99-iterations", type=int, default=100, help="P99 iterations per length")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, 
                       default=[10, 50, 100, 150, 200, 250, 300, 350, 400],
                       help="Prompt lengths to test")
    
    args = parser.parse_args()
    
    run_case2_benchmark(
        socket_path=args.socket,
        prompt_lengths=args.prompt_lengths,
        ttft_iterations=args.ttft_iterations,
        p99_iterations=args.p99_iterations
    )

#!/usr/bin/env python3
"""
Generate final TTFT and P99 tables from all 5 cases
"""

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_case_results():
    """Load results from all 5 cases"""
    
    # Case 1: Baseline
    case1_ttft = pd.read_csv('output/case1_baseline_ttft.csv')
    case1_p99 = pd.read_csv('output/case1_baseline_p99.csv')
    
    # Case 2: Async CUDA Graph Only (use rolling CUDA-only results)
    case2_ttft = pd.read_csv('output/rolling_cuda_only_benchmark.csv')
    case2_p99 = pd.read_csv('output/rolling_cuda_only_p99_benchmark.csv')
    
    # Case 3: JIT Only
    case3_ttft = pd.read_csv('output/case3_jit_only_ttft.csv')
    case3_p99 = pd.read_csv('output/case3_jit_only_p99.csv')
    
    # Case 4: JIT + Async CUDA Graph
    case4_ttft = pd.read_csv('output/case4_jit_cuda_ttft.csv')
    case4_p99 = pd.read_csv('output/case4_jit_cuda_p99.csv')
    
    # Case 5: TensorRT-LLM
    case5_ttft = pd.read_csv('output/case5_tensorrt_llm_ttft.csv')
    case5_p99 = pd.read_csv('output/case5_tensorrt_llm_p99.csv')
    
    return {
        'case1': {'ttft': case1_ttft, 'p99': case1_p99},
        'case2': {'ttft': case2_ttft, 'p99': case2_p99},
        'case3': {'ttft': case3_ttft, 'p99': case3_p99},
        'case4': {'ttft': case4_ttft, 'p99': case4_p99},
        'case5': {'ttft': case5_ttft, 'p99': case5_p99}
    }

def generate_ttft_table(results):
    """Generate final TTFT comparison table"""
    
    # Create TTFT table
    ttft_data = []
    
    for _, row in results['case1']['ttft'].iterrows():
        prompt_len = row['prompt_length']
        
        # Get results from all cases
        baseline = row['mean_ms']
        cuda_graph = results['case2']['ttft'][results['case2']['ttft']['prompt_length'] == prompt_len]['mean_ms'].iloc[0]
        jit_only = results['case3']['ttft'][results['case3']['ttft']['prompt_length'] == prompt_len]['mean_ms'].iloc[0]
        jit_cuda = results['case4']['ttft'][results['case4']['ttft']['prompt_length'] == prompt_len]['mean_ms'].iloc[0]
        tensorrt = results['case5']['ttft'][results['case5']['ttft']['prompt_length'] == prompt_len]['mean_ms'].iloc[0]
        
        # Determine winner
        values = [baseline, cuda_graph, jit_only, jit_cuda, tensorrt]
        min_val = min(values)
        
        if min_val == jit_cuda:
            winner = "JIT+CUDA"
        elif min_val == jit_only:
            winner = "JIT-Only"
        elif min_val == cuda_graph:
            winner = "CUDA Graph"
        elif min_val == tensorrt:
            winner = "TensorRT-LLM"
        else:
            winner = "Baseline"
        
        ttft_data.append({
            'prompt_length': prompt_len,
            'baseline_ms': baseline,
            'cuda_graph_ms': cuda_graph,
            'jit_only_ms': jit_only,
            'jit_cuda_ms': jit_cuda,
            'tensorrt_ms': tensorrt,
            'winner': winner
        })
    
    # Calculate averages
    baseline_avg = np.mean([row['baseline_ms'] for row in ttft_data])
    cuda_graph_avg = np.mean([row['cuda_graph_ms'] for row in ttft_data])
    jit_only_avg = np.mean([row['jit_only_ms'] for row in ttft_data])
    jit_cuda_avg = np.mean([row['jit_cuda_ms'] for row in ttft_data])
    tensorrt_avg = np.mean([row['tensorrt_ms'] for row in ttft_data])
    
    # Add average row
    ttft_data.append({
        'prompt_length': 'AVERAGE',
        'baseline_ms': baseline_avg,
        'cuda_graph_ms': cuda_graph_avg,
        'jit_only_ms': jit_only_avg,
        'jit_cuda_ms': jit_cuda_avg,
        'tensorrt_ms': tensorrt_avg,
        'winner': 'JIT+CUDA' if jit_cuda_avg == min([baseline_avg, cuda_graph_avg, jit_only_avg, jit_cuda_avg, tensorrt_avg]) else 'Other'
    })
    
    return pd.DataFrame(ttft_data)

def generate_p99_table(results):
    """Generate final P99 comparison table"""
    
    # Create P99 table
    p99_data = []
    
    for _, row in results['case1']['p99'].iterrows():
        prompt_len = row['prompt_length']
        
        # Get results from all cases
        baseline = row['p99_ms']
        cuda_graph = results['case2']['p99'][results['case2']['p99']['prompt_length'] == prompt_len]['p99_ms'].iloc[0]
        jit_only = results['case3']['p99'][results['case3']['p99']['prompt_length'] == prompt_len]['p99_ms'].iloc[0]
        jit_cuda = results['case4']['p99'][results['case4']['p99']['prompt_length'] == prompt_len]['p99_ms'].iloc[0]
        tensorrt = results['case5']['p99'][results['case5']['p99']['prompt_length'] == prompt_len]['p99_ms'].iloc[0]
        
        # Determine winner
        values = [baseline, cuda_graph, jit_only, jit_cuda, tensorrt]
        min_val = min(values)
        
        if min_val == jit_cuda:
            winner = "JIT+CUDA"
        elif min_val == jit_only:
            winner = "JIT-Only"
        elif min_val == cuda_graph:
            winner = "CUDA Graph"
        elif min_val == tensorrt:
            winner = "TensorRT-LLM"
        else:
            winner = "Baseline"
        
        p99_data.append({
            'prompt_length': prompt_len,
            'baseline_p99_ms': baseline,
            'cuda_graph_p99_ms': cuda_graph,
            'jit_only_p99_ms': jit_only,
            'jit_cuda_p99_ms': jit_cuda,
            'tensorrt_p99_ms': tensorrt,
            'winner': winner
        })
    
    # Calculate averages
    baseline_avg = np.mean([row['baseline_p99_ms'] for row in p99_data])
    cuda_graph_avg = np.mean([row['cuda_graph_p99_ms'] for row in p99_data])
    jit_only_avg = np.mean([row['jit_only_p99_ms'] for row in p99_data])
    jit_cuda_avg = np.mean([row['jit_cuda_p99_ms'] for row in p99_data])
    tensorrt_avg = np.mean([row['tensorrt_p99_ms'] for row in p99_data])
    
    # Add average row
    p99_data.append({
        'prompt_length': 'AVERAGE',
        'baseline_p99_ms': baseline_avg,
        'cuda_graph_p99_ms': cuda_graph_avg,
        'jit_only_p99_ms': jit_only_avg,
        'jit_cuda_p99_ms': jit_cuda_avg,
        'tensorrt_p99_ms': tensorrt_avg,
        'winner': 'JIT+CUDA' if jit_cuda_avg == min([baseline_avg, cuda_graph_avg, jit_only_avg, jit_cuda_avg, tensorrt_avg]) else 'Other'
    })
    
    return pd.DataFrame(p99_data)

def print_markdown_tables(ttft_df, p99_df):
    """Print tables in markdown format"""
    
    print("\n" + "=" * 80)
    print("📊 FINAL TTFT PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print("| Prompt Length | Baseline (ms) | CUDA Graph (ms) | JIT-Only (ms) | JIT+CUDA (ms) | TensorRT-LLM (ms) | Winner |")
    print("|---------------|---------------|----------------|---------------|---------------|-------------------|---------|")
    
    for _, row in ttft_df.iterrows():
        if row['prompt_length'] == 'AVERAGE':
            print(f"| **{row['prompt_length']}** | **{row['baseline_ms']:.2f}** | **{row['cuda_graph_ms']:.2f}** | **{row['jit_only_ms']:.2f}** | **{row['jit_cuda_ms']:.2f}** | **{row['tensorrt_ms']:.2f}** | **{row['winner']}** |")
        else:
            print(f"| {row['prompt_length']} tokens | {row['baseline_ms']:.2f} | {row['cuda_graph_ms']:.2f} | {row['jit_only_ms']:.2f} | **{row['jit_cuda_ms']:.2f}** | {row['tensorrt_ms']:.2f} | {row['winner']} |")
    
    print("\n" + "=" * 80)
    print("📊 FINAL P99 PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print("| Prompt Length | Baseline P99 (ms) | CUDA Graph P99 (ms) | JIT-Only P99 (ms) | JIT+CUDA P99 (ms) | TensorRT-LLM P99 (ms) | Winner |")
    print("|---------------|-------------------|-------------------|-------------------|-------------------|----------------------|---------|")
    
    for _, row in p99_df.iterrows():
        if row['prompt_length'] == 'AVERAGE':
            print(f"| **{row['prompt_length']}** | **{row['baseline_p99_ms']:.2f}** | **{row['cuda_graph_p99_ms']:.2f}** | **{row['jit_only_p99_ms']:.2f}** | **{row['jit_cuda_p99_ms']:.2f}** | **{row['tensorrt_p99_ms']:.2f}** | **{row['winner']}** |")
        else:
            print(f"| {row['prompt_length']} tokens | {row['baseline_p99_ms']:.2f} | {row['cuda_graph_p99_ms']:.2f} | {row['jit_only_p99_ms']:.2f} | **{row['jit_cuda_p99_ms']:.2f}** | {row['tensorrt_p99_ms']:.2f} | {row['winner']} |")

def main():
    """Generate final tables from all 5 cases"""
    
    logger.info("📊 Loading results from all 5 cases...")
    
    try:
        results = load_case_results()
        
        logger.info("📊 Generating TTFT comparison table...")
        ttft_df = generate_ttft_table(results)
        
        logger.info("📊 Generating P99 comparison table...")
        p99_df = generate_p99_table(results)
        
        # Save to CSV
        ttft_df.to_csv('output/final_ttft_table.csv', index=False)
        p99_df.to_csv('output/final_p99_table.csv', index=False)
        
        # Print markdown tables
        print_markdown_tables(ttft_df, p99_df)
        
        logger.info("✅ Final tables generated successfully!")
        logger.info("📁 Results saved to: output/final_ttft_table.csv, output/final_p99_table.csv")
        
    except Exception as e:
        logger.error(f"Failed to generate final tables: {e}")
        raise

if __name__ == "__main__":
    main()

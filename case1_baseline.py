#!/usr/bin/env python3
"""
Case 1: Baseline - Pure HuggingFace Transformers
No optimizations, just direct model.generate() calls
"""

import torch
import time
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModel:
    """Pure HuggingFace Transformers baseline - no optimizations"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Loading baseline model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None,
                trust_remote_code=True
            )
            
            self.model.eval()
            logger.info("✅ Baseline model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load baseline model: {e}")
            raise
    
    def baseline_inference(self, seq_len: int, input_tokens: List[int]) -> Tuple[List[int], float]:
        """Pure HuggingFace transformers inference - no optimizations"""
        start_time = time.time()
        
        try:
            # Convert to tensor
            input_tensor = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            
            # Create generation config
            config = GenerationConfig(
                max_new_tokens=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Pure HuggingFace generation
            with torch.no_grad():
                outputs = self.model.generate(
                    input_tensor,
                    generation_config=config,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                next_token_id = outputs.sequences[0, -1].item()
            
            latency_ms = (time.time() - start_time) * 1000
            return [next_token_id], latency_ms
            
        except Exception as e:
            logger.error(f"Baseline inference failed: {e}")
            raise

def run_baseline_benchmark(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    prompt_lengths: List[int] = [10, 50, 100, 150, 200, 250, 300, 350, 400],
    ttft_iterations: int = 5,
    p99_iterations: int = 100,
):
    """Run baseline benchmark for TTFT and P99"""
    
    logger.info("=" * 80)
    logger.info("🚀 CASE 1: BASELINE BENCHMARK (Pure HuggingFace Transformers)")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Prompt Lengths: {prompt_lengths}")
    logger.info(f"TTFT Iterations: {ttft_iterations}")
    logger.info(f"P99 Iterations: {p99_iterations}")
    logger.info("=" * 80)
    
    # Load model
    baseline_model = BaselineModel(model_name)
    
    # TTFT Benchmark
    logger.info("📊 Running TTFT Benchmark...")
    ttft_results = []
    
    for prompt_len in prompt_lengths:
        logger.info(f"Testing TTFT for prompt length {prompt_len} tokens ({ttft_iterations} iterations):")
        
        latencies = []
        for iteration in range(ttft_iterations):
            # Use MMLU prompt for realistic benchmarking
            from mmlu_prompts import get_mmlu_prompt, tokenize_prompt
            prompt_text = get_mmlu_prompt(prompt_len, iteration)
            prompt = tokenize_prompt(baseline_model.tokenizer, prompt_text, prompt_len)
            
            try:
                next_token_id, latency_ms = baseline_model.baseline_inference(prompt_len, prompt)
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
            # Use MMLU prompt for realistic benchmarking
            from mmlu_prompts import get_mmlu_prompt, tokenize_prompt
            prompt_text = get_mmlu_prompt(prompt_len, iteration)
            prompt = tokenize_prompt(baseline_model.tokenizer, prompt_text, prompt_len)
            
            try:
                next_token_id, latency_ms = baseline_model.baseline_inference(prompt_len, prompt)
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
    import os
    os.makedirs('output', exist_ok=True)
    
    ttft_df = pd.DataFrame(ttft_results)
    ttft_df.to_csv('output/case1_baseline_ttft.csv', index=False)
    
    p99_df = pd.DataFrame(p99_results)
    p99_df.to_csv('output/case1_baseline_p99.csv', index=False)
    
    logger.info("=" * 80)
    logger.info("📊 CASE 1 BASELINE BENCHMARK SUMMARY")
    logger.info("=" * 80)
    
    for result in ttft_results:
        if not np.isnan(result['mean_ms']):
            logger.info(f"TTFT {result['prompt_length']:3d} tokens: {result['mean_ms']:6.2f}ms (std: {result['std_ms']:6.2f}ms)")
    
    logger.info("")
    for result in p99_results:
        if not np.isnan(result['mean_ms']):
            logger.info(f"P99  {result['prompt_length']:3d} tokens: {result['p99_ms']:6.2f}ms (mean: {result['mean_ms']:6.2f}ms)")
    
    logger.info("=" * 80)
    logger.info("✅ Case 1 Baseline benchmark completed!")
    logger.info("📁 Results saved to: output/case1_baseline_ttft.csv, output/case1_baseline_p99.csv")
    
    return ttft_results, p99_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Case 1: Baseline Benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name")
    parser.add_argument("--ttft-iterations", type=int, default=5, help="TTFT iterations per length")
    parser.add_argument("--p99-iterations", type=int, default=100, help="P99 iterations per length")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, 
                       default=[10, 50, 100, 150, 200, 250, 300, 350, 400],
                       help="Prompt lengths to test")
    
    args = parser.parse_args()
    
    run_baseline_benchmark(
        model_name=args.model,
        prompt_lengths=args.prompt_lengths,
        ttft_iterations=args.ttft_iterations,
        p99_iterations=args.p99_iterations
    )

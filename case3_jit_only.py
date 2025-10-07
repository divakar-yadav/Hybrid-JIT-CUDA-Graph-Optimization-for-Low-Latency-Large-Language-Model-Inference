#!/usr/bin/env python3
"""
Case 3: JIT Only for Dynamic Operations
JIT compilation for dynamic operations only
Static operations use PyTorch native functionality
Single process, no CUDA Graphs
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

class JITOnlyModel:
    """JIT Only Model - JIT for dynamic operations, PyTorch native for static"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        # JIT-compiled functions for dynamic operations
        self.jit_preprocessing_fn = None
        self.jit_sampling_fn = None
        
        logger.info(f"Loading JIT-Only model: {model_name}")
        self._load_model()
        self._setup_jit_optimization()
    
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
            logger.info("✅ JIT-Only model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load JIT-Only model: {e}")
            raise
    
    def _setup_jit_optimization(self):
        """Setup JIT optimization for dynamic operations only"""
        logger.info("Setting up JIT optimization for dynamic operations...")
        
        # 1. JIT compile dynamic preprocessing
        @torch.jit.script
        def jit_preprocessing(input_ids: torch.Tensor, target_len: int, pad_token_id: int) -> torch.Tensor:
            """JIT-compiled function for handling dynamic shapes"""
            current_len = input_ids.size(1)
            
            if current_len > target_len:
                # Truncate
                return input_ids[:, :target_len]
            elif current_len < target_len:
                # Pad
                batch_size = input_ids.size(0)
                padding = torch.full((batch_size, target_len - current_len), pad_token_id, 
                                   device=input_ids.device, dtype=input_ids.dtype)
                return torch.cat([input_ids, padding], dim=1)
            else:
                return input_ids
        
        # 2. JIT compile sampling function
        @torch.jit.script
        def jit_sampling(logits: torch.Tensor, temperature: float, do_sample: bool) -> torch.Tensor:
            """JIT-compiled function for sampling"""
            # Handle different logits shapes robustly
            if logits.dim() == 3:
                # Shape: [batch, seq_len, vocab_size]
                next_token_logits = logits[0, -1, :]
            elif logits.dim() == 2:
                # Shape: [seq_len, vocab_size] - take last token
                next_token_logits = logits[-1, :]
            else:
                # Fallback: flatten and take last vocab_size elements
                vocab_size = logits.size(-1)
                next_token_logits = logits.view(-1)[-vocab_size:]
            
            if do_sample:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                return torch.multinomial(probs, 1)
            else:
                return torch.argmax(next_token_logits).unsqueeze(0)
        
        # Create JIT functions
        self.jit_preprocessing_fn = torch.jit.script(jit_preprocessing)
        self.jit_sampling_fn = torch.jit.script(jit_sampling)
        
        logger.info("✅ JIT optimization setup complete for dynamic operations")
    
    def jit_only_inference(self, seq_len: int, input_tokens: List[int]) -> Tuple[List[int], float]:
        """Execute JIT-only inference (JIT for dynamic, PyTorch native for static)"""
        start_time = time.time()
        
        try:
            # Step 1: Use JIT for dynamic preprocessing
            input_tensor = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            processed_input = self.jit_preprocessing_fn(
                input_tensor, 
                seq_len, 
                self.tokenizer.eos_token_id
            )
            
            # Step 2: PyTorch native model forward pass (static operation)
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
            
            with torch.no_grad():
                outputs = self.model.generate(
                    processed_input,
                    generation_config=config,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Get logits for JIT sampling
                logits = outputs.scores[0]  # First (and only) generated token logits
                
                # Step 3: Use JIT for dynamic sampling
                next_token = self.jit_sampling_fn(logits, 0.7, True)
                next_token_id = next_token.item()
            
            latency_ms = (time.time() - start_time) * 1000
            return [next_token_id], latency_ms
            
        except Exception as e:
            logger.error(f"JIT-only inference failed: {e}")
            raise

def run_case3_benchmark(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    prompt_lengths: List[int] = [10, 50, 100, 150, 200, 250, 300, 350, 400],
    ttft_iterations: int = 5,
    p99_iterations: int = 100,
):
    """Run Case 3 benchmark for TTFT and P99"""
    
    logger.info("=" * 80)
    logger.info("🚀 CASE 3: JIT-ONLY BENCHMARK (JIT for Dynamic Operations)")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Prompt Lengths: {prompt_lengths}")
    logger.info(f"TTFT Iterations: {ttft_iterations}")
    logger.info(f"P99 Iterations: {p99_iterations}")
    logger.info("Architecture: Single process, JIT for dynamic ops, PyTorch native for static")
    logger.info("=" * 80)
    
    # Load model
    jit_model = JITOnlyModel(model_name)
    
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
            prompt = tokenize_prompt(jit_model.tokenizer, prompt_text, prompt_len)
            
            try:
                next_token_id, latency_ms = jit_model.jit_only_inference(prompt_len, prompt)
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
            prompt = tokenize_prompt(jit_model.tokenizer, prompt_text, prompt_len)
            
            try:
                next_token_id, latency_ms = jit_model.jit_only_inference(prompt_len, prompt)
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
    ttft_df.to_csv('output/case3_jit_only_ttft.csv', index=False)
    
    p99_df = pd.DataFrame(p99_results)
    p99_df.to_csv('output/case3_jit_only_p99.csv', index=False)
    
    logger.info("=" * 80)
    logger.info("📊 CASE 3 JIT-ONLY BENCHMARK SUMMARY")
    logger.info("=" * 80)
    
    for result in ttft_results:
        if not np.isnan(result['mean_ms']):
            logger.info(f"TTFT {result['prompt_length']:3d} tokens: {result['mean_ms']:6.2f}ms (std: {result['std_ms']:6.2f}ms)")
    
    logger.info("")
    for result in p99_results:
        if not np.isnan(result['mean_ms']):
            logger.info(f"P99  {result['prompt_length']:3d} tokens: {result['p99_ms']:6.2f}ms (mean: {result['mean_ms']:6.2f}ms)")
    
    logger.info("=" * 80)
    logger.info("✅ Case 3 JIT-Only benchmark completed!")
    logger.info("📁 Results saved to: output/case3_jit_only_ttft.csv, output/case3_jit_only_p99.csv")
    
    return ttft_results, p99_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Case 3: JIT-Only Benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name")
    parser.add_argument("--ttft-iterations", type=int, default=5, help="TTFT iterations per length")
    parser.add_argument("--p99-iterations", type=int, default=100, help="P99 iterations per length")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, 
                       default=[10, 50, 100, 150, 200, 250, 300, 350, 400],
                       help="Prompt lengths to test")
    
    args = parser.parse_args()
    
    run_case3_benchmark(
        model_name=args.model,
        prompt_lengths=args.prompt_lengths,
        ttft_iterations=args.ttft_iterations,
        p99_iterations=args.p99_iterations
    )

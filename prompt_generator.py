"""
Simple Prompt Generator - Same text for all cases
"""

from typing import List
from transformers import AutoTokenizer

# Single consistent text for all benchmarks
BENCHMARK_TEXT = "The quick brown fox jumps over the lazy dog. This is a test sentence for benchmarking large language model inference performance. We are measuring the time to first token and 99th percentile latency across different optimization approaches."

def get_benchmark_tokens(tokenizer: AutoTokenizer, target_length: int) -> List[int]:
    """
    Get tokenized benchmark text for target length
    
    Args:
        tokenizer: HuggingFace tokenizer
        target_length: Target number of tokens
        
    Returns:
        List of token IDs
    """
    # Tokenize the benchmark text
    tokens = tokenizer.encode(BENCHMARK_TEXT, add_special_tokens=False)
    
    # Adjust to target length
    if len(tokens) > target_length:
        # Truncate
        tokens = tokens[:target_length]
    elif len(tokens) < target_length:
        # Pad with pad token
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        padding = [pad_token_id] * (target_length - len(tokens))
        tokens = tokens + padding
    
    return tokens
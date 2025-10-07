# Approach Explanation

This folder contains detailed explanations for each of the 5 benchmark approaches implemented in this research.

## Overview

Each approach represents a different optimization strategy for Large Language Model (LLM) inference:

1. **[Baseline Approach](baseline_approach.md)** - Pure HuggingFace Transformers
2. **[CUDA Graph Approach](cuda_graph_approach.md)** - Rolling CUDA Graphs with Native Python
3. **[JIT-Only Approach](jit_only_approach.md)** - JIT Compilation for Dynamic Operations
4. **[JIT+CUDA Approach](jit_cuda_approach.md)** - Hybrid JIT + CUDA Graph Optimization
5. **[TensorRT-LLM Approach](tensorrt_approach.md)** - Real TensorRT-LLM Implementation

## Architecture Comparison

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Baseline      │   CUDA Graph    │   JIT-Only      │   JIT+CUDA      │   TensorRT-LLM  │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ • Pure PyTorch  │ • CUDA Graphs   │ • JIT Compile   │ • CUDA Graphs   │ • TensorRT      │
│ • No Optimize   │ • Native Python │ • Native PyTorch│ • JIT Compile   │ • Optimized     │
│ • Single Process│ • Two Processes │ • Single Process│ • Two Processes │ • Single Process│
│ • Direct Calls  │ • IPC Comm      │ • Direct Calls  │ • IPC Comm      │ • Direct Calls  │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

## Performance Comparison

| Approach | TTFT (ms) | P99 (ms) | Memory Usage | Complexity |
|----------|-----------|----------|--------------|------------|
| Baseline | 27.84 | 18.96 | High | Low |
| CUDA Graph | 121.23 | 17.05 | Medium | Medium |
| JIT-Only | 76.04 | 59.25 | Medium | Low |
| **JIT+CUDA** | **15.57** | **14.14** | **Low** | **High** |
| TensorRT-LLM | 22.07 | 17.73 | Low | Medium |

## Key Insights

1. **JIT+CUDA achieves the best overall performance** with 4.48× speedup over baseline
2. **CUDA Graphs alone struggle** due to dynamic operations in LLM inference
3. **JIT compilation is crucial** for handling dynamic operations efficiently
4. **Rolling management prevents memory exhaustion** and enables unlimited sequence lengths
5. **IPC architecture enables scalability** by isolating CUDA contexts

## Navigation

Click on any approach above to see detailed implementation explanations, code walkthroughs, and architectural diagrams.

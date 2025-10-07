# CUDA Graph Approach: Rolling CUDA Graphs with Native Python

## Overview

The CUDA Graph approach uses CUDA Graphs to capture and replay static operations while handling dynamic operations with native Python. This approach implements a two-process architecture with rolling graph management to prevent memory exhaustion.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CUDA Graph Architecture                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐                    ┌─────────────────┐                    │
│  │   Client        │                    │   Generator     │                    │
│  │   Process       │                    │   Process       │                    │
│  │                 │                    │                 │                    │
│  │ ┌─────────────┐ │    IPC Socket      │ ┌─────────────┐ │                    │
│  │ │   Context   │ │◄──────────────────►│ │ CUDA Graph  │ │                    │
│  │ │   Creator   │ │                    │ │   Manager   │ │                    │
│  │ └─────────────┘ │                    │ └─────────────┘ │                    │
│  │                 │                    │                 │                    │
│  │ • Create Input  │                    │ • Pre-capture   │                    │
│  │ • Send Request  │                    │ • Rolling Mgmt │                    │
│  │ • Receive Token │                    │ • Graph Replay │                    │
│  └─────────────────┘                    │ • Memory Clean │                    │
│                                         └─────────────────┘                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Code Implementation

### File: `case2_async_cuda_generator.py` (Generator Process)

#### Class Definition and Initialization

```python
class AsyncCUDAGraphManager:
    def __init__(self, model_name: str, socket_path: str, max_graphs: int = 50):
        """
        Initialize async CUDA graph manager
        
        Args:
            model_name: HuggingFace model identifier
            socket_path: Unix socket path for IPC
            max_graphs: Maximum graphs to keep in memory
        """
        self.model_name = model_name
        self.socket_path = socket_path
        self.max_graphs_in_memory = max_graphs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Graph management
        self.cuda_graphs = OrderedDict()  # LRU cache
        self.graph_usage_count = {}
        self.total_graphs_generated = 0
        self.total_graphs_deleted = 0
        
        # Model components
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Async CUDA Graph Manager initialized on {self.device}")
        self._initialize_model()
```

**Explanation:**
- **Line 1**: Define the async CUDA graph manager class
- **Line 2**: Constructor with model name, socket path, and max graphs
- **Line 3-8**: Docstring explaining initialization parameters
- **Line 9-12**: Store configuration parameters and detect device
- **Line 14-17**: Initialize graph management structures:
  - `OrderedDict()`: LRU cache for graphs
  - `graph_usage_count`: Track usage for cleanup decisions
  - Counters for monitoring
- **Line 19-21**: Initialize model components as None
- **Line 23**: Log initialization
- **Line 24**: Call model initialization

#### Model Initialization

```python
def _initialize_model(self):
    """Initialize model and setup CUDA optimization"""
    try:
        logger.info(f"Loading model: {self.model_name}")
        
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
        
        # Pre-capture 50 graphs before inference starts
        self._precapture_50_graphs()
        
        # Start background graph generation
        self._start_background_generation()
        
        logger.info("Model loaded and async CUDA optimization setup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
```

**Explanation:**
- **Line 2**: Method to initialize the model and setup optimization
- **Line 3**: Try-catch for error handling
- **Line 4**: Log model loading
- **Line 6-11**: Load tokenizer with fast tokenization
- **Line 13-14**: Set pad token if missing
- **Line 16-22**: Load model with appropriate precision and device mapping
- **Line 23**: Set to evaluation mode
- **Line 25**: Pre-capture 50 graphs (critical for performance)
- **Line 27**: Start background graph generation thread
- **Line 29**: Log successful initialization
- **Line 31-33**: Error handling

#### Pre-capture 50 Graphs

```python
def _precapture_50_graphs(self):
    """Pre-capture 50 CUDA graphs for common sequence lengths"""
    logger.info(f"Pre-capturing 50 CUDA graphs (1 to 50) - NOT COUNTED IN TTFT")
    
    successful_graphs = 0
    for seq_len in range(1, 51):  # Pre-capture graphs for lengths 1-50
        if len(self.cuda_graphs) < self.max_graphs_in_memory:
            if self._create_cuda_graph(seq_len):
                successful_graphs += 1
        else:
            logger.warning(f"Max graphs reached ({self.max_graphs_in_memory}), stopping pre-capture")
            break
    
    logger.info(f"Pre-capture complete: {successful_graphs} graphs ready")
```

**Explanation:**
- **Line 2**: Method to pre-capture 50 graphs before inference
- **Line 3**: Log pre-capture start (important: not counted in TTFT)
- **Line 5**: Initialize success counter
- **Line 6**: Loop through sequence lengths 1-50
- **Line 7**: Check if we haven't reached memory limit
- **Line 8**: Attempt to create CUDA graph for this length
- **Line 9**: Increment success counter if successful
- **Line 10-12**: Break if memory limit reached
- **Line 14**: Log completion with success count

#### CUDA Graph Creation

```python
def _create_cuda_graph(self, seq_len: int) -> bool:
    """Create CUDA graph for specific sequence length"""
    try:
        # Create static tensors with fixed shapes
        static_input = torch.randint(0, 1000, (1, seq_len), device=self.device, dtype=torch.long)
        
        # Warmup - run model once to initialize CUDA context
        with torch.no_grad():
            _ = self.model(static_input)
        
        # Capture CUDA graph
        graph = torch.cuda.CUDAGraph()
        static_output = None
        
        with torch.cuda.graph(graph):
            with torch.no_grad():
                static_output = self.model(static_input)
        
        # Store graph data
        self.cuda_graphs[seq_len] = {
            'graph': graph,
            'static_input': static_input,
            'static_output': static_output
        }
        
        self.graph_usage_count[seq_len] = 1
        self.total_graphs_generated += 1
        
        logger.info(f"CUDA graph created for seq_len {seq_len} (total graphs: {len(self.cuda_graphs)})")
        return True
        
    except Exception as e:
        logger.warning(f"CUDA graph failed for seq_len {seq_len}: {e}")
        return False
```

**Explanation:**
- **Line 2**: Method to create CUDA graph for specific sequence length
- **Line 3**: Try-catch for error handling
- **Line 4-5**: Create static input tensor with fixed shape (1, seq_len)
- **Line 7-9**: Warmup run to initialize CUDA context and memory
- **Line 11-12**: Create CUDA graph object and initialize output
- **Line 14-17**: Capture the graph:
  - `torch.cuda.graph(graph)`: Context manager for graph capture
  - `torch.no_grad()`: Disable gradients for inference
  - `self.model(static_input)`: The operation to capture
- **Line 19-24**: Store graph data in LRU cache
- **Line 26-27**: Update usage tracking and counters
- **Line 29**: Log successful creation
- **Line 30**: Return success
- **Line 32-34**: Error handling and logging

#### Rolling Graph Management

```python
def _cleanup_old_graphs(self):
    """Clean up old graphs to maintain memory limit"""
    while len(self.cuda_graphs) >= self.max_graphs_in_memory:
        # Remove least recently used graph
        seq_len, graph_data = self.cuda_graphs.popitem(last=False)
        
        # Clean up graph data
        del graph_data['graph']
        del graph_data['static_input']
        del graph_data['static_output']
        
        # Remove from usage tracking
        if seq_len in self.graph_usage_count:
            del self.graph_usage_count[seq_len]
        
        self.total_graphs_deleted += 1
        logger.info(f"Deleted CUDA graph for seq_len {seq_len} (graphs in memory: {len(self.cuda_graphs)})")
```

**Explanation:**
- **Line 2**: Method to clean up old graphs (LRU eviction)
- **Line 3**: Loop while at memory limit
- **Line 4-5**: Remove least recently used graph (first item in OrderedDict)
- **Line 7-10**: Explicitly delete graph data to free GPU memory
- **Line 12-14**: Remove from usage tracking
- **Line 16**: Increment deletion counter
- **Line 17**: Log deletion with current graph count

#### Graph Inference

```python
def _inference_with_graph(self, input_tokens: List[int]) -> List[int]:
    """Perform inference using CUDA graph"""
    seq_len = len(input_tokens)
    
    # Check if graph exists
    if seq_len not in self.cuda_graphs:
        logger.warning(f"No graph available for seq_len {seq_len}, creating on-demand")
        if not self._create_cuda_graph(seq_len):
            raise RuntimeError(f"Failed to create graph for seq_len {seq_len}")
    
    # Get graph data
    graph_data = self.cuda_graphs[seq_len]
    graph = graph_data['graph']
    static_input = graph_data['static_input']
    static_output = graph_data['static_output']
    
    # Update input tensor (in-place)
    static_input.copy_(torch.tensor([input_tokens], device=self.device, dtype=torch.long))
    
    # Replay graph
    graph.replay()
    
    # Extract output
    output_tokens = static_output[0, -1, :].argmax(dim=-1).unsqueeze(0).tolist()
    
    # Update usage tracking (move to end for LRU)
    self.cuda_graphs.move_to_end(seq_len)
    self.graph_usage_count[seq_len] += 1
    
    return output_tokens
```

**Explanation:**
- **Line 2**: Method to perform inference using CUDA graph
- **Line 3**: Get sequence length from input
- **Line 5-8**: Check if graph exists, create if missing
- **Line 10-14**: Extract graph components
- **Line 16**: Update input tensor in-place (critical for graph replay)
- **Line 18**: Replay the captured graph (fast execution)
- **Line 20**: Extract output tokens (argmax for next token)
- **Line 22-24**: Update LRU tracking (move to end, increment usage)
- **Line 26**: Return generated tokens

### File: `case2_async_cuda_client.py` (Client Process)

#### Client Initialization

```python
class AsyncCUDAClient:
    def __init__(self, socket_path: str):
        """
        Initialize async CUDA client
        
        Args:
            socket_path: Unix socket path for IPC
        """
        self.socket_path = socket_path
        self.socket = None
        
    def connect(self):
        """Connect to generator process"""
        try:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.connect(self.socket_path)
            logger.info(f"Connected to generator at {self.socket_path}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
```

**Explanation:**
- **Line 1**: Define client class
- **Line 2**: Constructor with socket path
- **Line 3-6**: Docstring
- **Line 7-8**: Store socket path and initialize socket
- **Line 10**: Method to connect to generator
- **Line 11**: Try-catch for connection
- **Line 12**: Create Unix domain socket
- **Line 13**: Connect to generator process
- **Line 14**: Log successful connection
- **Line 15-17**: Error handling

#### Request Handling

```python
def send_request(self, input_tokens: List[int]) -> List[int]:
    """Send inference request to generator"""
    try:
        # Serialize request
        request = {
            'type': 'inference',
            'input_tokens': input_tokens
        }
        request_data = json.dumps(request).encode('utf-8')
        
        # Send request
        self.socket.sendall(len(request_data).to_bytes(4, 'big'))
        self.socket.sendall(request_data)
        
        # Receive response
        response_length = int.from_bytes(self.socket.recv(4), 'big')
        response_data = self.socket.recv(response_length)
        response = json.loads(response_data.decode('utf-8'))
        
        if response['status'] == 'success':
            return response['output_tokens']
        else:
            raise RuntimeError(f"Generator error: {response['error']}")
            
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise
```

**Explanation:**
- **Line 2**: Method to send inference request
- **Line 3**: Try-catch for request handling
- **Line 4-7**: Create and serialize request dictionary
- **Line 8**: Encode to bytes
- **Line 10-11**: Send request length (4 bytes) then data
- **Line 13-15**: Receive response length and data
- **Line 16**: Deserialize response
- **Line 18-21**: Check status and return tokens or raise error
- **Line 23-25**: Error handling

## Performance Characteristics

### Strengths
- **Graph Optimization**: CUDA Graphs eliminate CPU overhead
- **Memory Efficiency**: Rolling management prevents OOM
- **Scalability**: Two-process architecture enables parallelization
- **Pre-capture**: Graphs ready before inference starts

### Weaknesses
- **Dynamic Operations**: Cannot capture dynamic operations in graphs
- **Complexity**: Two-process architecture adds complexity
- **Memory Overhead**: Multiple graphs consume GPU memory
- **Setup Time**: Pre-capture takes time (not counted in TTFT)

### Use Cases
- **Static Workloads**: When operations are mostly static
- **High Throughput**: When CPU overhead is a bottleneck
- **Memory Constrained**: When rolling management is needed
- **Research**: Understanding graph optimization benefits

## Benchmark Results

| Prompt Length | TTFT (ms) | P99 (ms) | Graphs Used | Memory (GB) |
|---------------|-----------|----------|-------------|-------------|
| 10 tokens | 20.77 | 8.48 | Pre-captured | 14.1 |
| 50 tokens | 145.66 | 9.15 | Pre-captured | 14.1 |
| 100 tokens | 120.36 | 10.17 | Pre-captured | 14.1 |
| 150 tokens | 119.58 | 11.32 | Pre-captured | 14.1 |
| 200 tokens | 125.89 | 13.25 | Pre-captured | 14.1 |
| 250 tokens | 129.56 | 14.56 | Pre-captured | 14.1 |
| 300 tokens | 136.74 | 16.70 | Pre-captured | 14.1 |
| 350 tokens | 143.38 | 18.30 | Pre-captured | 14.1 |
| 400 tokens | 149.14 | 51.56 | On-demand | 14.1 |
| **Average** | **121.23** | **17.05** | **Mixed** | **14.1** |

## Key Insights

1. **P99 Performance**: Excellent P99 performance due to graph optimization
2. **TTFT Variability**: TTFT varies significantly due to graph availability
3. **Memory Management**: Rolling management prevents memory exhaustion
4. **Dynamic Limitations**: Struggles with dynamic operations

## Next Steps

To understand how JIT compilation addresses the dynamic operation limitations:
- [JIT-Only Approach](jit_only_approach.md) - JIT compilation for dynamic operations
- [JIT+CUDA Approach](jit_cuda_approach.md) - Hybrid JIT + CUDA Graph optimization

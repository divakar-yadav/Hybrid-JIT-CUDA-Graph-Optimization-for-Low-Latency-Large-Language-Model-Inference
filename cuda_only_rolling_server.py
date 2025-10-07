#!/usr/bin/env python3
"""
CUDA-Only Rolling Graph Generator Server
Same IPC architecture as main experiment, but uses CUDA Graphs + Native Python (no JIT)
WITH rolling CUDA Graph management for memory efficiency
"""

import torch
import time
import logging
import socket
import pickle
import struct
import threading
import os
import argparse
import gc
from collections import OrderedDict
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
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

class RollingCUDAOnlyOptimizedModel:
    """CUDA-Only optimized model with rolling CUDA Graph management (no JIT)"""
    
    def __init__(self, model, tokenizer, device, max_seq_len=500, max_graphs_in_memory=50):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.max_graphs_in_memory = max_graphs_in_memory
        
        # Rolling CUDA graphs (LRU cache)
        self.cuda_graphs = OrderedDict()
        self.graph_usage_count = {}
        self.total_graphs_generated = 0
        self.total_graphs_deleted = 0
        
        # Fallback configs
        self.fallback_configs = {}
        
        logger.info(f"✅ Rolling CUDA-Only model initialized (max {max_graphs_in_memory} graphs in memory)")
        
        # Pre-capture 50 graphs and setup fallback configs
        self._precapture_50_graphs()
        self._setup_fallback_configs()
    
    def _precapture_50_graphs(self):
        """Pre-capture 50 CUDA graphs for common sequence lengths"""
        logger.info(f"🚀 Pre-capturing 50 CUDA graphs (1 to 50)")
        
        successful_graphs = 0
        for seq_len in range(1, 51):  # Pre-capture graphs for lengths 1-50
            if len(self.cuda_graphs) < self.max_graphs_in_memory:
                if self._create_cuda_graph(seq_len):
                    successful_graphs += 1
            else:
                logger.warning(f"⚠️ Max graphs reached ({self.max_graphs_in_memory}), stopping pre-capture")
                break
        
        logger.info(f"✅ Pre-capture complete: {successful_graphs} graphs ready (rolling management for additional lengths)")
    
    def _setup_fallback_configs(self):
        """Setup fallback configs for all sequence lengths"""
        logger.info(f"🚀 Setting up fallback configs for all lengths (1 to {self.max_seq_len})")
        
        for seq_len in range(1, self.max_seq_len + 1):
            self.fallback_configs[seq_len] = GenerationConfig(
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
        
        logger.info(f"✅ Fallback configs setup complete: {len(self.fallback_configs)} configs ready")
    
    def native_python_preprocessing(self, input_ids: torch.Tensor, target_len: int, pad_token_id: int) -> torch.Tensor:
        """Native Python preprocessing (no JIT)"""
        current_len = input_ids.size(1)
        
        if current_len > target_len:
            return input_ids[:, :target_len]
        elif current_len < target_len:
            batch_size = input_ids.size(0)
            padding = torch.full((batch_size, target_len - current_len), pad_token_id, 
                               device=input_ids.device, dtype=input_ids.dtype)
            return torch.cat([input_ids, padding], dim=1)
        else:
            return input_ids
    
    def native_python_sampling(self, logits: torch.Tensor, temperature: float, do_sample: bool) -> torch.Tensor:
        """Native Python sampling (no JIT)"""
        next_token_logits = logits[0, -1, :]
        
        if do_sample:
            scaled_logits = next_token_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            return torch.multinomial(probs, 1)
        else:
            return torch.argmax(next_token_logits).unsqueeze(0)
    
    def _get_or_create_graph(self, seq_len: int):
        """Get existing graph or create new one with rolling management"""
        if seq_len in self.cuda_graphs:
            # Move to end (most recently used)
            graph_data = self.cuda_graphs.pop(seq_len)
            self.cuda_graphs[seq_len] = graph_data
            self.graph_usage_count[seq_len] = self.graph_usage_count.get(seq_len, 0) + 1
            return graph_data
        
        # Create new graph if not exists
        if self._create_cuda_graph(seq_len):
            self._cleanup_old_graphs()
            return self.cuda_graphs.get(seq_len)
        return None
    
    def _create_cuda_graph(self, seq_len: int) -> bool:
        """Create CUDA graph for specific sequence length - MODEL FORWARD PASS ONLY"""
        try:
            # Create static tensors with fixed shapes
            static_input = torch.randint(0, 1000, (1, seq_len), device=self.device, dtype=torch.long)
            
            # Warmup - just the model forward pass
            with torch.no_grad():
                _ = self.model(static_input)
            
            # Capture CUDA graph for MODEL FORWARD PASS ONLY (like the working implementation)
            graph = torch.cuda.CUDAGraph()
            static_output = None
            
            with torch.cuda.graph(graph):
                with torch.no_grad():
                    # Only capture the model forward pass - this is what works in the reference
                    static_output = self.model(static_input)
            
            # Store graph data
            self.cuda_graphs[seq_len] = {
                'graph': graph,
                'static_input': static_input,
                'static_output': static_output
            }
            
            self.graph_usage_count[seq_len] = 1
            self.total_graphs_generated += 1
            
            logger.info(f"✅ CUDA graph created for seq_len {seq_len} (MODEL FORWARD PASS ONLY) (total graphs: {len(self.cuda_graphs)})")
            return True
            
        except Exception as e:
            logger.warning(f"❌ CUDA graph failed for seq_len {seq_len}: {e}")
            # Store fallback config
            self.fallback_configs[seq_len] = GenerationConfig(
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
            return False
    
    def _cleanup_old_graphs(self):
        """Remove oldest graphs when memory limit is reached"""
        while len(self.cuda_graphs) >= self.max_graphs_in_memory:
            # Remove least recently used graph
            seq_len, graph_data = self.cuda_graphs.popitem(last=False)
            
            # Explicit cleanup
            del graph_data['graph']
            del graph_data['static_input']
            del graph_data['static_output']
            
            # Clean up usage count
            if seq_len in self.graph_usage_count:
                del self.graph_usage_count[seq_len]
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            self.total_graphs_deleted += 1
            logger.info(f"🗑️ Deleted CUDA graph for seq_len {seq_len} (graphs in memory: {len(self.cuda_graphs)})")
    
    def rolling_cuda_only_inference(self, seq_len: int, input_tokens: List[int]) -> Tuple[List[int], float]:
        """Execute inference with rolling CUDA graphs but no JIT"""
        start_time = time.time()
        
        try:
            # Step 1: Use native Python for dynamic preprocessing
            input_tensor = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            processed_input = self.native_python_preprocessing(
                input_tensor, 
                seq_len, 
                self.tokenizer.eos_token_id
            )
            
            # Step 2: Get or create CUDA graph (rolling management)
            graph_data = self._get_or_create_graph(seq_len)
            
            if graph_data is not None:
                # Copy to static input and replay CUDA graph (like the working implementation)
                graph_data['static_input'].copy_(processed_input)
                graph_data['graph'].replay()
                
                # Get logits from static output
                logits = graph_data['static_output'].logits
                
                # Step 3: Use native Python for sampling (dynamic operation)
                next_token = self.native_python_sampling(logits, 0.7, True)
                next_token_id = next_token.item()
                
            else:
                # Fallback to regular generation
                config = self.fallback_configs.get(seq_len, GenerationConfig(
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                ))
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        processed_input,
                        generation_config=config,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    next_token_id = outputs.sequences[0, -1].item()
                logger.warning("Fallback to regular generation")
            
            latency_ms = (time.time() - start_time) * 1000
            return [next_token_id], latency_ms
            
        except Exception as e:
            logger.error(f"Rolling CUDA-Only inference failed: {e}")
            raise

class RollingCUDAOnlyGraphManager:
    """Manager for rolling CUDA-Only graph optimization (same IPC architecture as main experiment)"""
    
    def __init__(self, model_name: str, max_graphs_in_memory: int = 50):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_graphs_in_memory = max_graphs_in_memory
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # Initialize rolling CUDA-Only optimized model
        self.optimized_model = RollingCUDAOnlyOptimizedModel(
            self.model, 
            self.tokenizer, 
            self.device,
            max_graphs_in_memory=max_graphs_in_memory
        )
        
        logger.info("✅ Rolling CUDA-Only graph manager initialized")
    
    def handle_request(self, request: GraphRequest) -> GraphResponse:
        """Handle inference request (same as main experiment)"""
        try:
            if request.request_type == 'INFERENCE':
                output_tokens, latency_ms = self.optimized_model.rolling_cuda_only_inference(
                    request.sequence_length, 
                    request.input_tokens
                )
                
                return GraphResponse(
                    success=True,
                    request_id=request.request_id,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms
                )
            else:
                return GraphResponse(
                    success=False,
                    request_id=request.request_id,
                    error_message=f"Unknown request type: {request.request_type}"
                )
                
        except Exception as e:
            logger.error(f"Error handling request {request.request_id}: {e}")
            return GraphResponse(
                success=False,
                request_id=request.request_id,
                error_message=str(e)
            )

def handle_client(client_socket, graph_manager):
    """Handle client connection (same as main experiment)"""
    try:
        while True:
            # Read message length
            length_data = client_socket.recv(4)
            if not length_data:
                break
            
            message_length = struct.unpack('!I', length_data)[0]
            
            # Read message data
            message_data = b''
            while len(message_data) < message_length:
                packet = client_socket.recv(message_length - len(message_data))
                if not packet:
                    break
                message_data += packet
            
            if len(message_data) < message_length:
                break
            
            # Deserialize request
            request = pickle.loads(message_data)
            
            # Process request
            response = graph_manager.handle_request(request)
            
            # Serialize response
            response_data = pickle.dumps(response)
            
            # Send response length
            client_socket.send(struct.pack('!I', len(response_data)))
            
            # Send response data
            client_socket.send(response_data)
            
    except Exception as e:
        logger.error(f"Error handling client: {e}")
    finally:
        client_socket.close()

def main():
    parser = argparse.ArgumentParser(description="Rolling CUDA-Only Graph Generator Server")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name")
    parser.add_argument("--socket", default="/tmp/rolling_cuda_only_graph_generator.sock", help="Socket path")
    parser.add_argument("--max-graphs", type=int, default=50, help="Maximum graphs in memory")
    
    args = parser.parse_args()
    
    # Initialize graph manager
    graph_manager = RollingCUDAOnlyGraphManager(
        model_name=args.model,
        max_graphs_in_memory=args.max_graphs
    )
    
    # Create socket
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(args.socket)
    server_socket.listen(5)
    
    logger.info(f"Rolling CUDA-Only graph generator server started on {args.socket}")
    logger.info(f"Max graphs in memory: {args.max_graphs}")
    
    try:
        while True:
            client_socket, _ = server_socket.accept()
            client_thread = threading.Thread(
                target=handle_client, 
                args=(client_socket, graph_manager)
            )
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        server_socket.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)

if __name__ == "__main__":
    main()

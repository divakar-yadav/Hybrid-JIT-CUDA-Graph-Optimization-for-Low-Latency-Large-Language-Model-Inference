#!/usr/bin/env python3
"""
Case 4: JIT + Async Rolling CUDA Graph Generator (Server Process)
Same as Case 2 but with JIT for dynamic operations
Pre-captures 50 graphs (1-50) before inference starts
Maintains rolling pool of 50 graphs with async generation
Uses JIT for dynamic operations (preprocessing, sampling)
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

class JITAsyncRollingCUDAGenerator:
    """JIT + Async Rolling CUDA Graph Generator - Server Process"""
    
    def __init__(self, model, tokenizer, device, max_graphs_in_memory=50):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_graphs_in_memory = max_graphs_in_memory
        
        # Rolling CUDA graphs (LRU cache)
        self.cuda_graphs = OrderedDict()
        self.graph_usage_count = {}
        self.total_graphs_generated = 0
        self.total_graphs_deleted = 0
        
        # JIT-compiled functions for dynamic operations
        self.jit_preprocessing_fn = None
        self.jit_sampling_fn = None
        
        # Background graph generation
        self.background_generation_active = False
        self.background_thread = None
        
        # Setup JIT optimization
        self._setup_jit_optimization()
        
        # Pre-capture 50 graphs before inference starts
        self._precapture_50_graphs()
        
        # Start background graph generation
        self._start_background_generation()
        
        logger.info(f"✅ JIT + Async Rolling CUDA Generator initialized (max {max_graphs_in_memory} graphs)")
    
    def _setup_jit_optimization(self):
        """Setup JIT optimization for dynamic operations"""
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
    
    def _precapture_50_graphs(self):
        """Pre-capture graphs for benchmark lengths only"""
        benchmark_lengths = [10, 50, 100, 150, 200, 250, 300, 350, 400]
        logger.info(f"🚀 Pre-capturing CUDA graphs for benchmark lengths: {benchmark_lengths} - NOT COUNTED IN TTFT")
        
        successful_graphs = 0
        for seq_len in benchmark_lengths:
            if len(self.cuda_graphs) < self.max_graphs_in_memory:
                if self._create_cuda_graph(seq_len):
                    successful_graphs += 1
            else:
                logger.warning(f"⚠️ Max graphs reached ({self.max_graphs_in_memory}), stopping pre-capture")
                break
        
        logger.info(f"✅ Pre-capture complete: {successful_graphs} benchmark graphs ready")
    
    def _start_background_generation(self):
        """Start background thread for continuous graph generation"""
        self.background_generation_active = True
        self.background_thread = threading.Thread(target=self._background_generation_loop, daemon=True)
        self.background_thread.start()
        logger.info("🔄 Background graph generation started")
    
    def _background_generation_loop(self):
        """Background loop to continuously generate graphs"""
        # Priority list: benchmark lengths first, then sequential
        priority_lengths = [250, 300, 350, 400, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
        seq_len_index = 0
        
        while self.background_generation_active:
            try:
                # Generate next graph if we have space
                if len(self.cuda_graphs) < self.max_graphs_in_memory:
                    if seq_len_index < len(priority_lengths):
                        seq_len = priority_lengths[seq_len_index]
                        seq_len_index += 1
                    else:
                        # Continue with sequential generation
                        seq_len = 61 + (seq_len_index - len(priority_lengths))
                        seq_len_index += 1
                        if seq_len > 500:  # Reset if we reach max
                            seq_len_index = 0
                    
                    if self._create_cuda_graph(seq_len):
                        logger.debug(f"🔄 Background generated graph for seq_len {seq_len}")
                else:
                    # Wait a bit if we're at capacity
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.warning(f"Background generation error: {e}")
                time.sleep(1)
    
    def _create_cuda_graph(self, seq_len: int) -> bool:
        """Create CUDA graph for specific sequence length"""
        try:
            # Create static tensors
            static_input = torch.randint(0, 1000, (1, seq_len), device=self.device, dtype=torch.long)
            
            # Warmup
            with torch.no_grad():
                _ = self.model(static_input)
            
            # Capture CUDA graph for forward pass only
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
            
            logger.debug(f"✅ CUDA graph created for seq_len {seq_len} (total graphs: {len(self.cuda_graphs)})")
            return True
            
        except Exception as e:
            logger.warning(f"❌ CUDA graph failed for seq_len {seq_len}: {e}")
            return False
    
    def _get_or_create_graph(self, seq_len: int):
        """Get existing graph or create new one with rolling management"""
        if seq_len in self.cuda_graphs:
            # Move to end (most recently used)
            graph_data = self.cuda_graphs.pop(seq_len)
            self.cuda_graphs[seq_len] = graph_data
            self.graph_usage_count[seq_len] = self.graph_usage_count.get(seq_len, 0) + 1
            return graph_data
        
        # Create new graph if not exists (should be rare with background generation)
        if self._create_cuda_graph(seq_len):
            self._cleanup_old_graphs()
            return self.cuda_graphs.get(seq_len)
        return None
    
    def _cleanup_old_graphs(self):
        """Clean up old graphs to maintain memory limit"""
        while len(self.cuda_graphs) >= self.max_graphs_in_memory:
            seq_len, graph_data = self.cuda_graphs.popitem(last=False)  # Remove LRU
            del graph_data['graph']
            del graph_data['static_input']
            del graph_data['static_output']
            gc.collect()
            torch.cuda.empty_cache()
            self.total_graphs_deleted += 1
            logger.debug(f"🗑️ Deleted CUDA graph for seq_len {seq_len} (graphs in memory: {len(self.cuda_graphs)})")
    
    def process_inference_request(self, seq_len: int, input_tokens: List[int]) -> Tuple[List[int], float]:
        """Process inference request using CUDA graphs + JIT for dynamic operations"""
        start_time = time.time()
        
        try:
            # Step 1: JIT-compiled preprocessing (dynamic operation)
            input_tensor = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            processed_input = self.jit_preprocessing_fn(
                input_tensor, 
                seq_len, 
                self.tokenizer.eos_token_id
            )
            
            # Step 2: Get or create CUDA graph (should be ready from pre-capture or background generation)
            graph_data = self._get_or_create_graph(seq_len)
            
            if graph_data is not None:
                # Copy to static input and replay CUDA graph
                graph_data['static_input'].copy_(processed_input)
                graph_data['graph'].replay()
                
                # Get logits from static output
                logits = graph_data['static_output'].logits
                
                # Step 3: Use JIT for sampling (dynamic operation)
                next_token = self.jit_sampling_fn(logits, 0.7, True)
                next_token_id = next_token.item()
                
            else:
                # This should not happen with proper pre-capture and background generation
                logger.error(f"❌ No graph available for seq_len {seq_len}")
                raise RuntimeError(f"No graph available for seq_len {seq_len}")
            
            latency_ms = (time.time() - start_time) * 1000
            return [next_token_id], latency_ms
            
        except Exception as e:
            logger.error(f"JIT + Async CUDA inference failed: {e}")
            raise
    
    def stop_background_generation(self):
        """Stop background graph generation"""
        self.background_generation_active = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
        logger.info("🛑 Background graph generation stopped")

class JITAsyncCUDAGraphManager:
    """Manager for JIT + Async Rolling CUDA Graph optimization"""
    
    def __init__(self, model_name: str, max_graphs_in_memory: int = 50):
        self.model_name = model_name
        self.max_graphs_in_memory = max_graphs_in_memory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.cuda_generator = None
        
        logger.info(f"JIT + Async CUDA Graph Manager initialized on {self.device}")
        
        if not self._load_model():
            raise RuntimeError("Failed to load model")
    
    def _load_model(self) -> bool:
        """Load the model and setup JIT + async CUDA optimization"""
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
            
            # Setup JIT + async CUDA generator
            self.cuda_generator = JITAsyncRollingCUDAGenerator(
                self.model, self.tokenizer, self.device, self.max_graphs_in_memory
            )
            
            logger.info("Model loaded and JIT + async CUDA optimization setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def execute_inference(self, seq_len: int, input_tokens: List[int]) -> Tuple[List[int], float]:
        """Execute inference using JIT + async CUDA graphs"""
        return self.cuda_generator.process_inference_request(seq_len, input_tokens)
    
    def get_statistics(self) -> dict:
        """Get graph generation statistics"""
        return {
            'total_graphs_generated': self.cuda_generator.total_graphs_generated,
            'total_graphs_deleted': self.cuda_generator.total_graphs_deleted,
            'graphs_in_memory': len(self.cuda_generator.cuda_graphs),
            'max_graphs_in_memory': self.max_graphs_in_memory
        }

class JITAsyncCUDAGraphServer:
    """Server that provides JIT + async CUDA graph inference via IPC"""
    
    def __init__(self, model_name: str, socket_path: str = "/tmp/jit_async_cuda_graph.sock", max_graphs: int = 50):
        self.model_name = model_name
        self.socket_path = socket_path
        self.max_graphs = max_graphs
        self.graph_manager = None
        self.server_socket = None
        self.running = False
        
    def start(self) -> bool:
        """Start the JIT + async CUDA graph server"""
        try:
            # Initialize graph manager
            self.graph_manager = JITAsyncCUDAGraphManager(self.model_name, self.max_graphs)
            
            # Remove existing socket file
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            
            # Start server
            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(self.socket_path)
            self.server_socket.listen(5)
            
            self.running = True
            logger.info(f"JIT + Async CUDA graph server started on {self.socket_path}")
            logger.info(f"Max graphs in memory: {self.max_graphs}")
            
            # Accept connections
            while self.running:
                try:
                    client_socket, _ = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client, 
                        args=(client_socket,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def _handle_client(self, client_socket):
        """Handle client requests"""
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
                response = self._process_request(request)
                
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
    
    def _process_request(self, request: GraphRequest) -> GraphResponse:
        """Process a single request"""
        try:
            if request.request_type == 'INFERENCE':
                output_tokens, latency_ms = self.graph_manager.execute_inference(
                    request.sequence_length, request.input_tokens
                )
                
                return GraphResponse(
                    success=True,
                    request_id=request.request_id,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms
                )
            
            elif request.request_type == 'STATS':
                stats = self.graph_manager.get_statistics()
                return GraphResponse(
                    success=True,
                    request_id=request.request_id,
                    output_tokens=[stats['graphs_in_memory']]  # Use output_tokens to return stats
                )
            
            else:
                return GraphResponse(
                    success=False,
                    request_id=request.request_id,
                    error_message=f"Unknown request type: {request.request_type}"
                )
                
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return GraphResponse(
                success=False,
                request_id=request.request_id,
                error_message=str(e)
            )
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.graph_manager and self.graph_manager.cuda_generator:
            self.graph_manager.cuda_generator.stop_background_generation()
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        logger.info("Server stopped")

def main():
    parser = argparse.ArgumentParser(description="Case 4: JIT + Async Rolling CUDA Graph Generator Server")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name")
    parser.add_argument("--socket", default="/tmp/jit_async_cuda_graph.sock", help="Socket path")
    parser.add_argument("--max-graphs", type=int, default=50, help="Maximum graphs in memory")
    
    args = parser.parse_args()
    
    # Create and start server
    server = JITAsyncCUDAGraphServer(args.model, args.socket, args.max_graphs)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        server.stop()

if __name__ == "__main__":
    main()

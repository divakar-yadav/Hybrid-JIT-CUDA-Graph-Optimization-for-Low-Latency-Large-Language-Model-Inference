#!/usr/bin/env python3
"""
JIT + Rolling CUDA Graph Generator Server
Implements rolling CUDA Graph deletion to manage memory efficiently
"""

import torch
import time
import logging
import os
import socket
import pickle
import threading
import struct
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from collections import OrderedDict
import gc

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

class RollingCUDAOptimizedModel:
    """Model using JIT for dynamic parts and rolling CUDA graphs for static parts"""
    
    def __init__(self, model, tokenizer, device, max_seq_len=500, max_graphs_in_memory=50):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.max_graphs_in_memory = max_graphs_in_memory
        
        # JIT-compiled functions for dynamic operations
        self.jit_dynamic_fn = None
        self.jit_sampling_fn = None
        
        # Rolling CUDA graphs - using OrderedDict for LRU behavior
        self.cuda_graphs = OrderedDict()
        self.graph_usage_count = {}  # Track usage for statistics
        
        # Fallback configs
        self.fallback_configs = {}
        
        # Graph generation statistics
        self.total_graphs_generated = 0
        self.total_graphs_deleted = 0
        
        self._setup_jit_optimization()
        self._precapture_50_graphs()
    
    def _setup_jit_optimization(self):
        """Setup JIT + rolling CUDA graph optimization"""
        logger.info("Setting up JIT + rolling CUDA graph optimization...")
        
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
            next_token_logits = logits[0, -1, :]
            
            if do_sample:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                return torch.multinomial(probs, 1)
            else:
                return torch.argmax(next_token_logits).unsqueeze(0)
        
        # Create JIT functions
        self.jit_dynamic_fn = torch.jit.script(jit_preprocessing)
        self.jit_sampling_fn = torch.jit.script(jit_sampling)
        
        # Setup fallback configs for all sequence lengths
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
        
        logger.info(f"✅ JIT optimization setup complete: Rolling CUDA graphs (max {self.max_graphs_in_memory} in memory) + JIT dynamic functions")
    
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
    
    def _create_cuda_graph(self, seq_len: int) -> bool:
        """Create a CUDA graph for a specific sequence length"""
        try:
            # Create static tensors
            static_input = torch.randint(0, 1000, (1, seq_len), device=self.device, dtype=torch.long)
            
            # Warmup
            with torch.no_grad():
                _ = self.model(static_input)
            
            # Capture CUDA graph for forward pass only (not generation)
            graph = torch.cuda.CUDAGraph()
            static_output = None
            
            with torch.cuda.graph(graph):
                with torch.no_grad():
                    static_output = self.model(static_input)
            
            # Store in rolling cache
            self.cuda_graphs[seq_len] = {
                'graph': graph,
                'static_input': static_input,
                'static_output': static_output,
                'created_at': time.time()
            }
            
            # Initialize usage count
            self.graph_usage_count[seq_len] = 0
            self.total_graphs_generated += 1
            
            logger.info(f"✅ CUDA graph created for seq_len {seq_len} (total graphs: {len(self.cuda_graphs)})")
            return True
            
        except Exception as e:
            logger.warning(f"❌ CUDA graph creation failed for seq_len {seq_len}: {e}")
            return False
    
    def _cleanup_old_graphs(self):
        """Remove old graphs to free memory when limit is reached"""
        while len(self.cuda_graphs) >= self.max_graphs_in_memory:
            # Remove the least recently used graph (first in OrderedDict)
            if self.cuda_graphs:
                seq_len, graph_data = self.cuda_graphs.popitem(last=False)
                
                # Explicitly delete the graph and tensors
                del graph_data['graph']
                del graph_data['static_input']
                del graph_data['static_output']
                
                # Remove from usage tracking
                if seq_len in self.graph_usage_count:
                    del self.graph_usage_count[seq_len]
                
                self.total_graphs_deleted += 1
                logger.info(f"🗑️ Deleted CUDA graph for seq_len {seq_len} (graphs in memory: {len(self.cuda_graphs)})")
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
    
    def _get_or_create_graph(self, seq_len: int) -> Optional[Dict]:
        """Get existing graph or create new one with rolling management"""
        # Check if graph exists
        if seq_len in self.cuda_graphs:
            # Move to end (most recently used)
            graph_data = self.cuda_graphs.pop(seq_len)
            self.cuda_graphs[seq_len] = graph_data
            return graph_data
        
        # Graph doesn't exist, try to create it
        if self._create_cuda_graph(seq_len):
            # Cleanup old graphs if needed
            self._cleanup_old_graphs()
            return self.cuda_graphs.get(seq_len)
        
        return None
    
    def rolling_inference(self, seq_len: int, input_tokens: List[int]) -> Tuple[List[int], float]:
        """Execute JIT + rolling CUDA graph inference"""
        start_time = time.time()
        
        try:
            # Step 1: Use JIT for dynamic preprocessing
            input_tensor = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            processed_input = self.jit_dynamic_fn(
                input_tensor, 
                seq_len, 
                self.tokenizer.eos_token_id
            )
            
            # Step 2: Get or create CUDA graph for forward pass
            graph_data = self._get_or_create_graph(seq_len)
            
            if graph_data:
                # Copy to static input and replay CUDA graph
                graph_data['static_input'].copy_(processed_input)
                graph_data['graph'].replay()
                
                # Get logits from static output
                logits = graph_data['static_output'].logits
                
                # Update usage statistics
                self.graph_usage_count[seq_len] = self.graph_usage_count.get(seq_len, 0) + 1
                
                # Step 3: Use JIT for sampling (dynamic operation)
                next_token = self.jit_sampling_fn(logits, 0.7, True)
                next_token_id = next_token.item()
                
            else:
                # Fallback to regular generation
                config = self.fallback_configs[seq_len]
                with torch.no_grad():
                    outputs = self.model.generate(
                        processed_input,
                        generation_config=config,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    next_token_id = outputs.sequences[0, -1].item()
                logger.debug(f"Fallback to regular generation for seq_len {seq_len}")
            
            latency_ms = (time.time() - start_time) * 1000
            return [next_token_id], latency_ms
            
        except Exception as e:
            logger.error(f"Rolling inference failed: {e}")
            raise
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about graph usage and memory management"""
        return {
            'graphs_in_memory': len(self.cuda_graphs),
            'max_graphs_in_memory': self.max_graphs_in_memory,
            'total_graphs_generated': self.total_graphs_generated,
            'total_graphs_deleted': self.total_graphs_deleted,
            'graph_usage_counts': dict(self.graph_usage_count),
            'available_sequence_lengths': list(self.cuda_graphs.keys())
        }

class RollingJITGraphManager:
    """Manager for JIT + rolling CUDA graph optimization"""
    
    def __init__(self, model_name: str, max_graphs_in_memory: int = 50):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_graphs_in_memory = max_graphs_in_memory
        
        # Load model and tokenizer
        logger.info(f"Loading model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize rolling JIT model
        self.rolling_model = RollingCUDAOptimizedModel(
            self.model, self.tokenizer, self.device, max_graphs_in_memory=max_graphs_in_memory
        )
        
        logger.info("✅ Rolling JIT graph manager initialized")
    
    def has_graph(self, seq_len: int) -> bool:
        """Check if sequence length is supported (either has graph or can create one)"""
        return seq_len <= 500  # All sequence lengths are supported with fallback
    
    def execute_inference(self, seq_len: int, input_tokens: List[int]) -> Tuple[List[int], float]:
        """Execute rolling JIT inference"""
        return self.rolling_model.rolling_inference(seq_len, input_tokens)
    
    def get_available_graphs(self) -> List[int]:
        """Get all available sequence lengths (all are supported with rolling graphs)"""
        return list(range(1, 501))  # All sequence lengths 1-500 are supported
    
    def get_statistics(self) -> Dict:
        """Get detailed statistics about graph management"""
        return self.rolling_model.get_graph_statistics()

class RollingJITGraphGeneratorServer:
    """Server that provides JIT + rolling CUDA graph inference via IPC"""
    
    def __init__(self, model_name: str, socket_path: str = "/tmp/rolling_graph_generator.sock", max_graphs_in_memory: int = 50):
        self.model_name = model_name
        self.socket_path = socket_path
        self.max_graphs_in_memory = max_graphs_in_memory
        self.graph_manager = None
        self.server_socket = None
        self.running = False
        
    def start(self) -> bool:
        """Start the rolling JIT graph generator server"""
        try:
            # Initialize graph manager
            self.graph_manager = RollingJITGraphManager(self.model_name, self.max_graphs_in_memory)
            
            # Remove existing socket file
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            
            # Start server
            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(self.socket_path)
            self.server_socket.listen(5)
            
            self.running = True
            logger.info(f"Rolling JIT + CUDA graph generator server started on {self.socket_path}")
            logger.info(f"Max graphs in memory: {self.max_graphs_in_memory}")
            
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
            # Receive request
            data = self._receive_data(client_socket)
            if not data:
                return
            
            request = pickle.loads(data)
            response = self._process_request(request)
            
            # Send response
            response_data = pickle.dumps(response)
            self._send_data(client_socket, response_data)
            
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def _process_request(self, request: GraphRequest) -> GraphResponse:
        """Process a graph request"""
        try:
            if request.request_type == "INFERENCE":
                if not self.graph_manager.has_graph(request.sequence_length):
                    return GraphResponse(
                        success=False,
                        request_id=request.request_id,
                        error_message=f"Sequence length {request.sequence_length} not supported"
                    )
                
                # Execute rolling JIT inference
                output_tokens, latency_ms = self.graph_manager.execute_inference(
                    request.sequence_length, 
                    request.input_tokens or []
                )
                
                return GraphResponse(
                    success=True,
                    request_id=request.request_id,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms
                )
                
            elif request.request_type == "GET_GRAPH":
                graph_available = self.graph_manager.has_graph(request.sequence_length)
                return GraphResponse(
                    success=True,
                    request_id=request.request_id,
                    error_message=f"Rolling graph support available: {graph_available}"
                )
                
            elif request.request_type == "STATUS":
                stats = self.graph_manager.get_statistics()
                return GraphResponse(
                    success=True,
                    request_id=request.request_id,
                    error_message=f"Rolling JIT optimization: {stats['graphs_in_memory']}/{stats['max_graphs_in_memory']} graphs in memory, "
                                f"generated: {stats['total_graphs_generated']}, deleted: {stats['total_graphs_deleted']}"
                )
                
            elif request.request_type == "STATISTICS":
                stats = self.graph_manager.get_statistics()
                return GraphResponse(
                    success=True,
                    request_id=request.request_id,
                    error_message=f"Detailed stats: {stats}"
                )
                
            else:
                return GraphResponse(
                    success=False,
                    request_id=request.request_id,
                    error_message=f"Unknown request type: {request.request_type}"
                )
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return GraphResponse(
                success=False,
                request_id=request.request_id,
                error_message=str(e)
            )
    
    def _send_data(self, client_socket, data):
        """Send data with length prefix"""
        length = len(data)
        client_socket.send(struct.pack('!I', length))
        client_socket.send(data)
    
    def _receive_data(self, client_socket):
        """Receive data with length prefix"""
        # Read length
        length_data = client_socket.recv(4)
        if not length_data:
            return None
        length = struct.unpack('!I', length_data)[0]
        
        # Read data
        data = b''
        while len(data) < length:
            chunk = client_socket.recv(length - len(data))
            if not chunk:
                return None
            data += chunk
        
        return data
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        logger.info("Rolling JIT graph generator server stopped")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rolling JIT + CUDA Graph Generator Server")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--socket", default="/tmp/rolling_graph_generator.sock", help="Socket path")
    parser.add_argument("--max-graphs", type=int, default=50, help="Maximum graphs to keep in memory")
    
    args = parser.parse_args()
    
    # Create and start server
    server = RollingJITGraphGeneratorServer(args.model, args.socket, args.max_graphs)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        server.stop()

if __name__ == "__main__":
    main()


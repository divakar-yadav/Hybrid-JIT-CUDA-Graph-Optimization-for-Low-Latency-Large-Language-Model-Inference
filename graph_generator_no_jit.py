#!/usr/bin/env python3
"""
CUDA Graph Generator Server (No JIT)
Same as graph_generator_jit_only.py but without JIT compilation
"""

import torch
import time
import logging
import os
import socket
import pickle
import threading
import struct
from typing import List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

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

class NoJITOptimizedModel:
    """Model using CUDA graphs for static parts but NO JIT for dynamic parts"""
    
    def __init__(self, model, tokenizer, device, max_seq_len=500):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        
        # CUDA graphs for static operations
        self.cuda_graphs = {}
        
        # Fallback configs
        self.fallback_configs = {}
        
        self._setup_cuda_graph_optimization()
    
    def _setup_cuda_graph_optimization(self):
        """Setup CUDA graph optimization (no JIT)"""
        logger.info("Setting up CUDA graph optimization (no JIT)...")
        
        successful_cuda_graphs = 0
        
        for seq_len in range(1, self.max_seq_len + 1):
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
                
                self.cuda_graphs[seq_len] = {
                    'graph': graph,
                    'static_input': static_input,
                    'static_output': static_output
                }
                
                successful_cuda_graphs += 1
                logger.info(f"✅ CUDA graph captured for seq_len {seq_len}")
                
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
        
        logger.info(f"✅ CUDA graph optimization setup complete: {successful_cuda_graphs}/{self.max_seq_len} CUDA graphs (no JIT)")
    
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
    
    def no_jit_inference(self, seq_len: int, input_tokens: List[int]) -> Tuple[List[int], float]:
        """Execute inference with CUDA graphs but no JIT"""
        start_time = time.time()
        
        try:
            # Step 1: Use native Python for dynamic preprocessing
            input_tensor = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            processed_input = self.native_python_preprocessing(
                input_tensor, 
                seq_len, 
                self.tokenizer.eos_token_id
            )
            
            # Step 2: Use CUDA graph for forward pass if available
            if seq_len in self.cuda_graphs:
                # Copy to static input and replay CUDA graph
                cuda_graph_data = self.cuda_graphs[seq_len]
                cuda_graph_data['static_input'].copy_(processed_input)
                cuda_graph_data['graph'].replay()
                
                # Get logits from static output
                logits = cuda_graph_data['static_output'].logits
                
                # Step 3: Use native Python for sampling (dynamic operation)
                next_token = self.native_python_sampling(logits, 0.7, True)
                next_token_id = next_token.item()
                
            else:
                config = self.fallback_configs[seq_len]
                with torch.no_grad():
                    outputs = self.model.generate(
                        processed_input,
                        generation_config=config,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    next_token_id = outputs.sequences[0, -1].item()
                print("Fallback to regular generation")
            
            latency_ms = (time.time() - start_time) * 1000
            return [next_token_id], latency_ms
            
        except Exception as e:
            logger.error(f"No-JIT inference failed: {e}")
            raise

class NoJITGraphManager:
    """Manager for CUDA graph optimization (no JIT)"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        logger.info(f"Loading model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize no-JIT model
        self.no_jit_model = NoJITOptimizedModel(
            self.model, self.tokenizer, self.device
        )
        
        logger.info("✅ No-JIT graph manager initialized")
    
    def has_graph(self, seq_len: int) -> bool:
        """Check if sequence length is supported"""
        return seq_len in self.no_jit_model.cuda_graphs or seq_len in self.no_jit_model.fallback_configs
    
    def execute_inference(self, seq_len: int, input_tokens: List[int]) -> Tuple[List[int], float]:
        """Execute no-JIT inference"""
        return self.no_jit_model.no_jit_inference(seq_len, input_tokens)
    
    def get_available_graphs(self) -> List[int]:
        """Get all available sequence lengths"""
        cuda_graphs = set(self.no_jit_model.cuda_graphs.keys())
        fallback_configs = set(self.no_jit_model.fallback_configs.keys())
        
        return sorted(cuda_graphs.union(fallback_configs))

class NoJITGraphGeneratorServer:
    """Server that provides CUDA graph inference (no JIT) via IPC"""
    
    def __init__(self, model_name: str, socket_path: str = "/tmp/graph_generator_no_jit.sock"):
        self.model_name = model_name
        self.socket_path = socket_path
        self.graph_manager = None
        self.server_socket = None
        self.running = False
        
    def start(self) -> bool:
        """Start the no-JIT graph generator server"""
        try:
            # Initialize graph manager
            self.graph_manager = NoJITGraphManager(self.model_name)
            
            # Remove existing socket file
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            
            # Start server
            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(self.socket_path)
            self.server_socket.listen(5)
            
            self.running = True
            logger.info(f"No-JIT CUDA graph generator server started on {self.socket_path}")
            
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
                        error_message=f"No CUDA graph optimization available for seq_len {request.sequence_length}"
                    )
                
                # Execute no-JIT inference
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
                    error_message=f"CUDA graph optimization available: {graph_available}"
                )
                
            elif request.request_type == "STATUS":
                available_graphs = self.graph_manager.get_available_graphs()
                cuda_graphs = len(self.graph_manager.no_jit_model.cuda_graphs)
                fallback_configs = len(self.graph_manager.no_jit_model.fallback_configs)
                return GraphResponse(
                    success=True,
                    request_id=request.request_id,
                    error_message=f"CUDA graph optimization (no JIT): {cuda_graphs} CUDA graphs + {fallback_configs} fallback configs for seq_lens: {available_graphs[:10]}..."
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
        logger.info("No-JIT CUDA graph generator server stopped")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CUDA Graph Generator Server (No JIT)")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--socket", default="/tmp/graph_generator_no_jit.sock", help="Socket path")
    
    args = parser.parse_args()
    
    # Create and start server
    server = NoJITGraphGeneratorServer(args.model, args.socket)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        server.stop()

if __name__ == "__main__":
    main()


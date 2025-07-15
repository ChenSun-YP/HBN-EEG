#!/usr/bin/env python3
"""
GPU Utility Functions for EEG Foundation Challenge
Handles GPU detection, configuration, and multi-GPU setup.
"""

import torch
import torch.distributed as dist
import os
from typing import List, Dict, Optional, Tuple
import subprocess
import json

def detect_gpus() -> Dict[str, any]:
    """
    Detect available GPUs and their properties.
    
    Returns:
    -------
    Dict containing GPU information:
    - num_gpus: Number of available GPUs
    - gpu_names: List of GPU names
    - gpu_memory: List of GPU memory in GB
    - total_memory: Total GPU memory across all devices
    - cuda_version: CUDA version
    - device_ids: List of device IDs
    """
    gpu_info = {
        'num_gpus': 0,
        'gpu_names': [],
        'gpu_memory': [],
        'total_memory': 0,
        'cuda_version': None,
        'device_ids': [],
        'cuda_available': torch.cuda.is_available()
    }
    
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return gpu_info
    
    # Basic GPU information
    gpu_info['num_gpus'] = torch.cuda.device_count()
    gpu_info['cuda_version'] = torch.version.cuda
    gpu_info['device_ids'] = list(range(gpu_info['num_gpus']))
    
    print(f"Detected {gpu_info['num_gpus']} GPU(s)")
    print(f"CUDA Version: {gpu_info['cuda_version']}")
    
    # Detailed GPU information
    for i in range(gpu_info['num_gpus']):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory_bytes = torch.cuda.get_device_properties(i).total_memory
        gpu_memory_gb = gpu_memory_bytes / (1024**3)
        
        gpu_info['gpu_names'].append(gpu_name)
        gpu_info['gpu_memory'].append(gpu_memory_gb)
        gpu_info['total_memory'] += gpu_memory_gb
        
        print(f"GPU {i}: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    
    print(f"Total GPU Memory: {gpu_info['total_memory']:.1f} GB")
    
    return gpu_info

def get_model_memory_footprint(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate the actual memory footprint of a model.
    
    Parameters:
    ----------
    model : torch.nn.Module
        PyTorch model to analyze
        
    Returns:
    -------
    Dict containing memory information:
    - params_mb: Model parameters memory in MB
    - buffers_mb: Model buffers memory in MB  
    - total_mb: Total model memory in MB
    - dtype: Parameter data type
    - bytes_per_param: Bytes per parameter
    """
    param_size = 0
    buffer_size = 0
    dtype = None
    bytes_per_param = 0
    
    # Calculate parameter memory
    for param in model.parameters():
        if dtype is None:
            dtype = param.dtype
            bytes_per_param = param.element_size()
        param_size += param.nelement() * param.element_size()
    
    # Calculate buffer memory (batch norm stats, etc.)
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return {
        'params_mb': param_size / (1024**2),
        'buffers_mb': buffer_size / (1024**2),
        'total_mb': (param_size + buffer_size) / (1024**2),
        'dtype': dtype,
        'bytes_per_param': bytes_per_param,
        'total_params': sum(p.numel() for p in model.parameters())
    }

def get_dtype_size(dtype: torch.dtype) -> int:
    """
    Get the size in bytes for a given PyTorch data type.
    
    Parameters:
    ----------
    dtype : torch.dtype
        PyTorch data type
        
    Returns:
    -------
    int
        Size in bytes
    """
    # Create a dummy tensor to get element size
    dummy = torch.tensor(0, dtype=dtype)
    return dummy.element_size()

def estimate_training_memory(model_memory_mb: float, batch_size: int = 1,
                           sequence_length: int = 1000, num_channels: int = 128,
                           input_dtype: Optional[torch.dtype] = None,
                           model: Optional[torch.nn.Module] = None,
                           gradient_factor: float = 1.0, activation_factor: float = 2.0) -> Dict[str, float]:
    """
    Estimate total memory required for training with dynamic input type detection.
    
    Parameters:
    ----------
    model_memory_mb : float
        Model parameter memory in MB
    batch_size : int
        Batch size for training
    sequence_length : int
        Sequence length (time points)
    num_channels : int
        Number of input channels
    input_dtype : torch.dtype, optional
        Input data type. If None, will try to infer from model or default to float32
    model : torch.nn.Module, optional
        Model to infer input dtype from (uses first parameter's dtype)
    gradient_factor : float
        Memory multiplier for gradients (1.0 = same as params)
    activation_factor : float
        Memory multiplier for activations relative to model size
        
    Returns:
    -------
    Dict with memory breakdown including input dtype information
    """
    # Determine input data type
    if input_dtype is not None:
        # Use explicitly provided dtype
        input_bytes_per_element = get_dtype_size(input_dtype)
        inferred_dtype = input_dtype
        dtype_source = "explicit"
    elif model is not None:
        # Infer from model's first parameter
        try:
            first_param = next(model.parameters())
            inferred_dtype = first_param.dtype
            input_bytes_per_element = get_dtype_size(inferred_dtype)
            dtype_source = "model_parameters"
        except StopIteration:
            # Model has no parameters, use default
            inferred_dtype = torch.float32
            input_bytes_per_element = 4
            dtype_source = "default_no_params"
    else:
        # Default to float32
        inferred_dtype = torch.float32
        input_bytes_per_element = 4
        dtype_source = "default"
    
    # Base model memory
    model_mb = model_memory_mb
    
    # Gradient memory (typically same size as parameters)
    gradient_mb = model_memory_mb * gradient_factor
    
    # Optimizer states (Adam uses 2x param memory for momentum and variance)
    optimizer_mb = model_memory_mb * 2.0
    
    # Input data memory - now dynamic based on actual input dtype
    input_mb = (batch_size * num_channels * sequence_length * input_bytes_per_element) / (1024**2)
    
    # Activation memory (intermediate features, varies by architecture)
    activation_mb = model_memory_mb * activation_factor * batch_size
    
    total_mb = model_mb + gradient_mb + optimizer_mb + input_mb + activation_mb
    
    return {
        'model_mb': model_mb,
        'gradients_mb': gradient_mb,
        'optimizer_mb': optimizer_mb,
        'input_mb': input_mb,
        'activations_mb': activation_mb,
        'total_mb': total_mb,
        'input_dtype': inferred_dtype,
        'input_bytes_per_element': input_bytes_per_element,
        'dtype_source': dtype_source
    }

def get_optimal_batch_size(model: torch.nn.Module, gpu_memory_gb: float,
                          sequence_length: int = 1000, num_channels: int = 128,
                          safety_factor: float = 0.7,
                          input_dtype: Optional[torch.dtype] = None) -> int:
    """
    Calculate optimal batch size based on actual model and GPU memory with dynamic input dtype.
    
    Parameters:
    ----------
    model : torch.nn.Module
        PyTorch model to analyze
    gpu_memory_gb : float
        Available GPU memory in gigabytes
    sequence_length : int
        Input sequence length
    num_channels : int
        Number of input channels
    safety_factor : float
        Safety factor to prevent OOM (0.7 = use 70% of memory)
    input_dtype : torch.dtype, optional
        Input data type. If None, will infer from model parameters
        
    Returns:
    -------
    int
        Recommended batch size
    """
    # Get actual model memory footprint
    model_info = get_model_memory_footprint(model)
    model_memory_mb = model_info['total_mb']
    
    # Available memory in MB
    available_memory_mb = gpu_memory_gb * 1024 * safety_factor
    
    # Binary search for optimal batch size
    low, high = 1, 256
    optimal_batch_size = 1
    
    while low <= high:
        mid = (low + high) // 2
        memory_estimate = estimate_training_memory(
            model_memory_mb, mid, sequence_length, num_channels,
            input_dtype=input_dtype, model=model
        )
        
        if memory_estimate['total_mb'] <= available_memory_mb:
            optimal_batch_size = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return optimal_batch_size

def setup_distributed_training(rank: int, world_size: int, 
                              backend: str = 'nccl') -> None:
    """
    Initialize distributed training process group.
    
    Parameters:
    ----------
    rank : int
        Process rank (GPU ID)
    world_size : int
        Total number of processes (number of GPUs)
    backend : str
        Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    print(f"Initialized distributed training: Rank {rank}/{world_size}")

def get_gpu_utilization() -> List[Dict[str, float]]:
    """
    Get current GPU utilization statistics.
    
    Returns:
    -------
    List of dictionaries containing utilization info for each GPU
    """
    if not torch.cuda.is_available():
        return []
    
    utilization = []
    
    for i in range(torch.cuda.device_count()):
        # Get memory info
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        memory_cached = torch.cuda.memory_reserved(i) / (1024**3)      # GB
        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        utilization.append({
            'device': i,
            'memory_allocated_gb': memory_allocated,
            'memory_cached_gb': memory_cached,
            'memory_total_gb': memory_total,
            'memory_utilization': (memory_allocated / memory_total) * 100
        })
    
    return utilization

def print_gpu_memory_summary():
    """Print a formatted summary of GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("\n" + "="*60)
    print("GPU Memory Summary")
    print("="*60)
    
    for i, util in enumerate(get_gpu_utilization()):
        print(f"GPU {i}: {util['memory_allocated_gb']:.2f}GB / "
              f"{util['memory_total_gb']:.2f}GB "
              f"({util['memory_utilization']:.1f}%)")
    
    print("="*60)

def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")

def set_gpu_memory_fraction(fraction: float = 0.9):
    """
    Set memory fraction for PyTorch to prevent OOM errors.
    
    Parameters:
    ----------
    fraction : float
        Fraction of GPU memory to use (0.0 to 1.0)
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(fraction, device=i)
        print(f"Set GPU memory fraction to {fraction:.1%}")

def get_recommended_config(model: Optional[torch.nn.Module] = None,
                          sequence_length: int = 1000,
                          num_channels: int = 128) -> Dict[str, any]:
    """
    Get recommended training configuration based on available GPUs and actual model.
    
    Parameters:
    ----------
    model : torch.nn.Module, optional
        PyTorch model to analyze for memory requirements
    sequence_length : int
        Input sequence length for memory estimation
    num_channels : int
        Number of input channels for memory estimation
        
    Returns:
    -------
    Dict with recommended configuration
    """
    gpu_info = detect_gpus()
    
    if gpu_info['num_gpus'] == 0:
        return {
            'device': 'cpu',
            'strategy': 'auto',
            'devices': 1,
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': False,
            'gpu_info': gpu_info
        }
    
    # Get model memory information if model is provided
    if model is not None:
        model_info = get_model_memory_footprint(model)
        print(f"Model Analysis:")
        print(f"   Parameters: {model_info['total_params']:,}")
        print(f"   Data type: {model_info['dtype']}")
        print(f"   Bytes per parameter: {model_info['bytes_per_param']}")
        print(f"   Model memory: {model_info['total_mb']:.2f} MB")
        
        # Get optimal batch size for single GPU (use smallest GPU memory)
        single_gpu_batch_size = get_optimal_batch_size(
            model, 
            min(gpu_info['gpu_memory']),
            sequence_length,
            num_channels
        )
    else:
        # Fallback to conservative estimates without model
        single_gpu_batch_size = 16
        print("No model provided - using conservative batch size estimates")
    
    # Scale batch size for multiple GPUs
    total_batch_size = single_gpu_batch_size * gpu_info['num_gpus']
    
    # Choose strategy based on number of GPUs
    if gpu_info['num_gpus'] == 1:
        strategy = 'auto'
    elif gpu_info['num_gpus'] <= 8:
        strategy = 'ddp'  # Distributed Data Parallel
    else:
        strategy = 'ddp_spawn'
    
    config = {
        'device': 'cuda',
        'strategy': strategy,
        'devices': gpu_info['num_gpus'],
        'batch_size': single_gpu_batch_size,
        'total_batch_size': total_batch_size,
        'num_workers': min(gpu_info['num_gpus'] * 4, 16),  # 4 workers per GPU, max 16
        'pin_memory': True,
        'gpu_info': gpu_info,
        'sequence_length': sequence_length,
        'num_channels': num_channels
    }
    
    print(f"\nRecommended Configuration:")
    print(f"   Strategy: {config['strategy']}")
    print(f"   Devices: {config['devices']} GPUs")
    print(f"   Batch size per GPU: {config['batch_size']}")
    print(f"   Total batch size: {config['total_batch_size']}")
    print(f"   Workers: {config['num_workers']}")
    
    return config

if __name__ == "__main__":
    # Test GPU detection
    gpu_info = detect_gpus()
    print(f"\nGPU Detection Results:")
    print(json.dumps(gpu_info, indent=2))
    
    # Test memory monitoring
    if gpu_info['cuda_available']:
        print_gpu_memory_summary()
        
        # Test configuration recommendation without model
        print(f"\n" + "="*50)
        print("Testing configuration recommendation (no model):")
        config = get_recommended_config()
        print(f"\nBasic config:")
        print(json.dumps({k: v for k, v in config.items() if k != 'gpu_info'}, 
                        indent=2, default=str))
        
        # Test with a dummy model to show dynamic analysis
        print(f"\n" + "="*50)
        print("Testing with dummy model (different dtypes):")
        
        # Create test models with different data types
        test_models = []
        
        # Float32 model
        model_fp32 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        ).float()
        
        # Float16 model
        model_fp16 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 2)
        ).half()
        
        test_models = [
            ("Float32 Model", model_fp32),
            ("Float16 Model", model_fp16)
        ]
        
        for name, model in test_models:
            print(f"\n{name}:")
            config = get_recommended_config(model, sequence_length=1000, num_channels=128)
            model_info = get_model_memory_footprint(model)
            print(f"Memory footprint: {model_info['total_mb']:.4f} MB")
            print(f"Recommended batch size: {config['batch_size']}")
            
    else:
        print("CUDA not available - testing CPU fallback")
        config = get_recommended_config()
        print(json.dumps(config, indent=2, default=str)) 
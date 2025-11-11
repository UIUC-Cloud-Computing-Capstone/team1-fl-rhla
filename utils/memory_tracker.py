"""
Memory tracking utilities for PyTorch models.
Supports both GPU and CPU memory profiling including:
- Parameter memory
- Activation memory
- Optimizer state memory
"""

import torch
from typing import Dict, Optional, Union
from torch.profiler import profile, ProfilerActivity, record_function


class MemoryTracker:
    """
    Track memory usage for PyTorch models including parameters, activations, and optimizer states.
    Supports both GPU and CPU.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize memory tracker.
        
        Args:
            device: Device to track memory on. If None, auto-detects from model.
        """
        self.device = device
        self.is_cuda = device is not None and device.type == 'cuda'
    
    def get_parameter_memory(self, model: torch.nn.Module, precision: str = 'fp32') -> Dict[str, Union[int, float]]:
        """
        Calculate parameter memory usage.
        
        Args:
            model: PyTorch model
            precision: 'fp32' or 'fp16'
        
        Returns:
            Dictionary with parameter memory in bytes and MB
        """
        bytes_per_param = 4 if precision == 'fp32' else 2
        
        total_params = 0
        trainable_params = 0
        
        for param in model.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        param_memory_bytes = total_params * bytes_per_param
        trainable_memory_bytes = trainable_params * bytes_per_param
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_memory_bytes': param_memory_bytes,
            'trainable_memory_bytes': trainable_memory_bytes,
            'total_memory_MB': param_memory_bytes / (1024 * 1024),
            'trainable_memory_MB': trainable_memory_bytes / (1024 * 1024),
        }
    
    def get_optimizer_state_memory(self, optimizer: torch.optim.Optimizer, precision: str = 'fp32') -> Dict[str, Union[int, float]]:
        """
        Calculate optimizer state memory usage.
        
        Args:
            optimizer: PyTorch optimizer
            precision: 'fp32' or 'fp16'
        
        Returns:
            Dictionary with optimizer state memory in bytes and MB
        """
        
        # Optimizer states are typically stored in fp32 for numerical stability
        # Even when training in fp16, optimizer states are usually fp32
        optimizer_bytes_per_param = 4
        
        total_optimizer_memory = 0
        param_count = 0
        
        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            # Adam/AdamW: 2 states per parameter (momentum m and variance v)
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        # 2 states (m and v) per parameter
                        total_optimizer_memory += 2 * p.numel() * optimizer_bytes_per_param
                        param_count += p.numel()
        elif isinstance(optimizer, torch.optim.SGD):
            # SGD: 1 state per parameter (momentum) if momentum > 0
            momentum = optimizer.param_groups[0].get('momentum', 0)
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        if momentum > 0:
                            total_optimizer_memory += p.numel() * optimizer_bytes_per_param
                        param_count += p.numel()
        else:
            # For other optimizers, estimate based on parameter count
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        # Conservative estimate: 1 state per parameter
                        total_optimizer_memory += p.numel() * optimizer_bytes_per_param
                        param_count += p.numel()
        
        return {
            'param_count': param_count,
            'optimizer_memory_bytes': total_optimizer_memory,
            'optimizer_memory_MB': total_optimizer_memory / (1024 * 1024),
        }
        """
        Estimate activation memory usage during forward/backward pass.
        
        Args:
            model: PyTorch model
            input_shape: Input shape (without batch dimension), e.g., (3, 224, 224)
            batch_size: Batch size
            precision: 'fp32' or 'fp16'
            include_backward: Whether to include backward pass activations
        
        Returns:
            Dictionary with activation memory estimates
        
        Note:
            For accurate measurements, especially on CPU, use profile_forward_backward() instead.
        """
        bytes_per_element = 4 if precision == 'fp32' else 2
        
        # Set to train mode to store activations (needed for gradient computation)
        model.train()
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        is_cuda = device.type == 'cuda'
        
        try:
            # Reset memory stats before forward pass
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            
            # Create dummy input - handle HuggingFace models
            if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                # HuggingFace model - use dict input
                model_type = model.config.model_type.lower()
                if 'vit' in model_type or 'deit' in model_type or 'image' in model_type:
                    # Vision transformer - use pixel_values
                    dummy_input = {'pixel_values': torch.randn(batch_size, *input_shape, device=device)}
                else:
                    # Text model - use input_ids
                    seq_len = 128  # Default sequence length
                    dummy_input = {'input_ids': torch.randint(0, 1000, (batch_size, seq_len), device=device)}
            else:
                # Regular PyTorch model - use tensor input
                dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            # Forward pass WITHOUT no_grad() to store activations for backprop
            if isinstance(dummy_input, dict):
                outputs = model(**dummy_input)
            else:
                outputs = model(dummy_input)
            
            # Get forward pass memory
            if is_cuda:
                forward_memory = torch.cuda.max_memory_allocated() - initial_memory
                torch.cuda.reset_peak_memory_stats()
            else:
                # For CPU, estimate based on model architecture
                # This is a rough estimate - use profile_forward_backward for accuracy
                total_params = sum(p.numel() for p in model.parameters())
                # Rough estimate: activations ~= 10-20% of parameter memory for transformers
                # For vision models with batch_size, this can be higher
                # Estimate: batch_size * sequence_length * hidden_dim * num_layers * bytes
                if hasattr(model, 'config'):
                    # Try to get model dimensions from config
                    hidden_dim = getattr(model.config, 'hidden_size', 384)
                    num_layers = getattr(model.config, 'num_hidden_layers', 12)
                    if 'vit' in model_type or 'deit' in model_type:
                        # Vision transformer: sequence_length = patches + 1
                        seq_len = (input_shape[1] // 16) * (input_shape[2] // 16) + 1  # Approximate
                    else:
                        seq_len = 128  # Default for text models
                    
                    # Estimate activation memory: batch * seq * hidden * layers * bytes
                    forward_memory = int(batch_size * seq_len * hidden_dim * num_layers * bytes_per_element * 0.5)
                else:
                    # Fallback: estimate based on parameters
                    forward_memory = int(total_params * bytes_per_element * 0.15)
            
            # Estimate backward (roughly 2-3x forward for activations stored during backprop)
            backward_memory = int(forward_memory * 2.5) if include_backward else 0
            
            total_activation_memory = forward_memory + backward_memory
            
        except Exception as e:
            # If model forward fails, return estimates based on parameters
            total_params = sum(p.numel() for p in model.parameters())
            # Conservative estimate
            forward_memory = int(total_params * bytes_per_element * 0.1)
            backward_memory = int(forward_memory * 2.5) if include_backward else 0
            total_activation_memory = forward_memory + backward_memory
        
        return {
            'forward_memory_bytes': forward_memory,
            'backward_memory_bytes': backward_memory,
            'total_activation_memory_bytes': total_activation_memory,
            'forward_memory_MB': forward_memory / (1024 * 1024),
            'backward_memory_MB': backward_memory / (1024 * 1024),
            'total_activation_memory_MB': total_activation_memory / (1024 * 1024),
        }
    
    def profile_forward_backward(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        batch: Dict,
        precision: str = 'fp32'
    ) -> Dict[str, Union[int, float]]:
        """
        Profile actual memory usage during forward and backward pass using PyTorch Profiler.
        Works with both GPU and CPU.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            batch: Input batch (dict for HuggingFace models or tensor)
            precision: 'fp32' or 'fp16'
        
        Returns:
            Dictionary with detailed memory profiling results
        """
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        is_cuda = device.type == 'cuda'
        
        # Reset memory stats if CUDA
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        activities = [ProfilerActivity.CPU]
        if is_cuda:
            activities.append(ProfilerActivity.CUDA)
        
        try:
            with profile(
                activities=activities,
                profile_memory=True,
                record_shapes=True,
                with_stack=False
            ) as prof:
                with record_function("forward"):
                    # Forward pass
                    if isinstance(batch, dict):
                        outputs = model(**batch)
                    else:
                        outputs = model(batch)
                    
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    elif isinstance(outputs, torch.Tensor):
                        loss = loss_fn(outputs, batch.get('labels'))
                    else:
                        loss = outputs
                
                with record_function("backward"):
                    # Backward pass
                    loss.backward()
                
                with record_function("optimizer_step"):
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
        
        except Exception as e:
            return {
                'error': str(e),
                'forward_memory_bytes': 0,
                'backward_memory_bytes': 0,
                'optimizer_memory_bytes': 0,
                'peak_memory_bytes': 0,
            }
        
        # Extract memory information
        events = prof.key_averages()
        
        forward_memory = 0
        backward_memory = 0
        optimizer_memory = 0
        
        for event in events:
            if 'forward' in event.key:
                if is_cuda:
                    forward_memory += event.cuda_memory_usage
                else:
                    forward_memory += event.cpu_memory_usage
            elif 'backward' in event.key:
                if is_cuda:
                    backward_memory += event.cuda_memory_usage
                else:
                    backward_memory += event.cpu_memory_usage
            elif 'optimizer' in event.key:
                if is_cuda:
                    optimizer_memory += event.cuda_memory_usage
                else:
                    optimizer_memory += event.cpu_memory_usage
        
        # Get peak memory if CUDA
        if is_cuda:
            peak_memory = torch.cuda.max_memory_allocated() - initial_memory
        else:
            peak_memory = forward_memory + backward_memory + optimizer_memory
        
        return {
            'forward_memory_bytes': forward_memory,
            'backward_memory_bytes': backward_memory,
            'optimizer_memory_bytes': optimizer_memory,
            'peak_memory_bytes': peak_memory,
            'forward_memory_MB': forward_memory / (1024 * 1024),
            'backward_memory_MB': backward_memory / (1024 * 1024),
            'optimizer_memory_MB': optimizer_memory / (1024 * 1024),
            'peak_memory_MB': peak_memory / (1024 * 1024),
            'profiler_table': prof.key_averages().table(sort_by="cuda_memory_usage" if is_cuda else "cpu_memory_usage", row_limit=20)
        }
    
    def profile_total_memory(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        batch: Union[Dict, torch.Tensor],
        precision: str = 'fp32'
    ) -> Dict[str, Union[int, float, Dict]]:
        """
        Profile total actual memory usage including parameters, optimizer states, and activations.
        This provides the most accurate measurement of total memory during training.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            batch: Input batch (dict for HuggingFace models or tensor)
            precision: 'fp32' or 'fp16'
        
        Returns:
            Dictionary with complete memory breakdown including actual measurements
        """
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        is_cuda = device.type == 'cuda'
        
        # 1. Get parameter memory (calculated)
        param_memory = self.get_parameter_memory(model, precision)
        
        # 2. Get optimizer state memory (calculated)
        optimizer_memory = self.get_optimizer_state_memory(optimizer, precision)
        
        # 3. Measure peak memory during full training step
        model.train()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            # Get baseline memory (parameters + optimizer states already loaded)
            baseline_memory = torch.cuda.memory_allocated()
        else:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            baseline_memory = process.memory_info().rss
        
        # Run a full training step to measure peak
        try:
            if isinstance(batch, dict):
                outputs = model(**batch)
            else:
                outputs = model(batch)
            
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, torch.Tensor):
                labels = batch.get('labels') if isinstance(batch, dict) else None
                if labels is not None:
                    loss = loss_fn(outputs, labels)
                else:
                    loss = outputs
            else:
                loss = outputs
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # TODO Liam: should we memory before optimizer.zero_grad()?
            if is_cuda:
                peak_memory = torch.cuda.max_memory_allocated()
                current_memory = torch.cuda.memory_allocated()
                # Peak includes everything: params + optimizer + activations
                # So peak_memory is the total actual peak
            else:
                # For CPU, measure process memory
                current_memory = process.memory_info().rss
                peak_memory = current_memory  # CPU doesn't track peak easily
        except Exception as e:
            peak_memory = baseline_memory
            current_memory = baseline_memory
        
        # 4. Profile activation memory separately for detailed breakdown
        profile_results = self.profile_forward_backward(
            model, optimizer, loss_fn, batch, precision
        )
        
        # Calculate total memory
        # peak_memory is the total actual peak including everything
        # For breakdown, we use calculated values for params/optimizer and measured for activations
        total_memory = peak_memory
        
        return {
            'parameters': param_memory,
            'optimizer_states': optimizer_memory,
            'activations': {
                'forward_memory_bytes': profile_results.get('forward_memory_bytes', 0),
                'backward_memory_bytes': profile_results.get('backward_memory_bytes', 0),
                'peak_activation_memory_bytes': profile_results.get('peak_memory_bytes', 0),
                'forward_memory_MB': profile_results.get('forward_memory_MB', 0),
                'backward_memory_MB': profile_results.get('backward_memory_MB', 0),
                'peak_activation_memory_MB': profile_results.get('peak_memory_MB', 0),
            },
            'total': {
                'total_memory_bytes': total_memory,
                'total_memory_MB': total_memory / (1024 * 1024),
                'total_memory_GB': total_memory / (1024 * 1024 * 1024),
                'peak_memory_bytes': peak_memory,
                'peak_memory_MB': peak_memory / (1024 * 1024),
                'peak_memory_GB': peak_memory / (1024 * 1024 * 1024),
                'baseline_memory_bytes': baseline_memory,
                'baseline_memory_MB': baseline_memory / (1024 * 1024),
            },
            'breakdown': {
                'parameter_memory_MB': param_memory['total_memory_MB'],
                'optimizer_memory_MB': optimizer_memory['optimizer_memory_MB'],
                'activation_memory_MB': profile_results.get('peak_memory_MB', 0),
                'total_memory_MB': total_memory / (1024 * 1024),
            },
            'profiler_table': profile_results.get('profiler_table', '')
        }
    
    def print_total_memory_profile(self, profile_results: Dict):
        """
        Print a formatted total memory profile.
        
        Args:
            profile_results: Output from profile_total_memory
        """
        print("=" * 80)
        print("TOTAL ACTUAL MEMORY PROFILE")
        print("=" * 80)
        
        if 'parameters' in profile_results:
            params = profile_results['parameters']
            print(f"\nParameters:")
            print(f"  Total params: {params['total_params']:,}")
            print(f"  Trainable params: {params['trainable_params']:,}")
            print(f"  Memory: {params['total_memory_MB']:.2f} MB")
        
        if 'optimizer_states' in profile_results:
            opt = profile_results['optimizer_states']
            print(f"\nOptimizer States:")
            print(f"  Memory: {opt['optimizer_memory_MB']:.2f} MB")
        
        if 'activations' in profile_results:
            acts = profile_results['activations']
            print(f"\nActivations (actual measured):")
            print(f"  Forward: {acts['forward_memory_MB']:.2f} MB")
            print(f"  Backward: {acts['backward_memory_MB']:.2f} MB")
            print(f"  Peak: {acts['peak_activation_memory_MB']:.2f} MB")
        
        if 'total' in profile_results:
            total = profile_results['total']
            print(f"\nTotal Memory Usage:")
            print(f"  Peak Memory: {total['peak_memory_MB']:.2f} MB ({total['peak_memory_GB']:.2f} GB)")
        
        if 'breakdown' in profile_results and 'total' in profile_results:
            breakdown = profile_results['breakdown']
            total = profile_results['total']
            # Use peak memory as the base for percentages (most accurate)
            peak_mb = total['peak_memory_MB']
            other_mb = peak_mb - breakdown['parameter_memory_MB'] - breakdown['optimizer_memory_MB'] - breakdown['activation_memory_MB']
            if peak_mb > 0:
                print(f"\nMemory Breakdown (based on peak memory):")
                print(f"  Parameters: {breakdown['parameter_memory_MB']:.2f} MB ({breakdown['parameter_memory_MB']/peak_mb*100:.1f}%)")
                print(f"  Optimizer: {breakdown['optimizer_memory_MB']:.2f} MB ({breakdown['optimizer_memory_MB']/peak_mb*100:.1f}%)")
                print(f"  Activations: {breakdown['activation_memory_MB']:.2f} MB ({breakdown['activation_memory_MB']/peak_mb*100:.1f}%)")
                print(f"  Other: {other_mb:.2f} MB ({ other_mb/peak_mb*100:.1f}%)")
                print(f"  Note: Peak memory ({peak_mb:.2f} MB) is the actual measured peak during training")
        
        print("=" * 80)
    
    def print_memory_summary(self, memory_breakdown: Dict):
        """
        Print a formatted memory summary.
        
        Args:
            memory_breakdown: Output from get_full_memory_breakdown
        """
        print("=" * 80)
        print("MEMORY BREAKDOWN")
        print("=" * 80)
        
        if 'parameters' in memory_breakdown:
            params = memory_breakdown['parameters']
            print(f"\nParameters:")
            print(f"  Total params: {params['total_params']:,}")
            print(f"  Trainable params: {params['trainable_params']:,}")
            print(f"  Total memory: {params['total_memory_MB']:.2f} MB")
            print(f"  Trainable memory: {params['trainable_memory_MB']:.2f} MB")
        
        if 'optimizer_states' in memory_breakdown:
            opt = memory_breakdown['optimizer_states']
            print(f"\nOptimizer States:")
            print(f"  Parameters: {opt['param_count']:,}")
            print(f"  Memory: {opt['optimizer_memory_MB']:.2f} MB")
        
        if 'activations' in memory_breakdown:
            acts = memory_breakdown['activations']
            print(f"\nActivations (estimated):")
            print(f"  Forward: {acts['forward_memory_MB']:.2f} MB")
            print(f"  Backward: {acts['backward_memory_MB']:.2f} MB")
            print(f"  Total: {acts['total_activation_memory_MB']:.2f} MB")
        
        if 'total' in memory_breakdown:
            total = memory_breakdown['total']
            print(f"\nTotal Memory:")
            print(f"  {total['total_memory_MB']:.2f} MB ({total['total_memory_GB']:.2f} GB)")
        
        print("=" * 80)


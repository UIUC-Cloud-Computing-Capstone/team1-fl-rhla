"""
Memory tracking utilities for PyTorch models.
Supports both GPU and CPU memory profiling including:
- Parameter memory
- Activation memory
- Optimizer state memory
"""

import os
import torch
from typing import Dict, Optional, Union, Tuple
from torch.profiler import profile, ProfilerActivity, record_function

# Constants
MB_TO_BYTES = 1024 * 1024
GB_TO_BYTES = 1024 * 1024 * 1024
FP32_BYTES_PER_PARAM = 4
FP16_BYTES_PER_PARAM = 2
OPTIMIZER_STATE_BYTES_PER_PARAM = 4  # Optimizer states are typically fp32


class MemoryTracker:
    """
    Track memory usage for PyTorch models including parameters, activations, and optimizer states.
    Supports both GPU and CPU.
    """
    
    def __init__(self):
        """Initialize memory tracker. Device is auto-detected from model in each method."""
        pass
    
    @staticmethod
    def _get_device_info(model: torch.nn.Module) -> Tuple[torch.device, bool]:
        """Get device and CUDA status from model."""
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        is_cuda = device.type == 'cuda'
        return device, is_cuda
    
    @staticmethod
    def _get_cpu_process() -> 'psutil.Process':
        """Get current process for CPU memory tracking."""
        import psutil
        return psutil.Process(os.getpid())
    
    @staticmethod
    def _compute_loss(outputs, batch: Union[Dict, torch.Tensor], loss_fn) -> torch.Tensor:
        """Extract or compute loss from model outputs."""
        if hasattr(outputs, 'loss'):
            return outputs.loss
        elif isinstance(outputs, torch.Tensor):
            labels = batch.get('labels') if isinstance(batch, dict) else None
            return loss_fn(outputs, labels) if labels is not None else outputs
        else:
            return outputs
    
    @staticmethod
    def _bytes_to_mb(bytes_value: Union[int, float]) -> float:
        """Convert bytes to megabytes."""
        return bytes_value / MB_TO_BYTES
    
    @staticmethod
    def _bytes_to_gb(bytes_value: Union[int, float]) -> float:
        """Convert bytes to gigabytes."""
        return bytes_value / GB_TO_BYTES
    
    def get_parameter_memory(self, model: torch.nn.Module, precision: str = 'fp32') -> Dict[str, Union[int, float]]:
        """
        Calculate parameter memory usage.
        
        Args:
            model: PyTorch model
            precision: 'fp32' or 'fp16'
        
        Returns:
            Dictionary with parameter counts and memory in MB
        """
        bytes_per_param = FP32_BYTES_PER_PARAM if precision == 'fp32' else FP16_BYTES_PER_PARAM
        
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        
        param_memory_bytes = total_params * bytes_per_param
        trainable_memory_bytes = trainable_params * bytes_per_param
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_memory_MB': self._bytes_to_mb(param_memory_bytes),
            'trainable_memory_MB': self._bytes_to_mb(trainable_memory_bytes),
        }
    
    def get_optimizer_state_memory(self, optimizer: torch.optim.Optimizer, precision: str = 'fp32') -> Dict[str, Union[int, float]]:
        """
        Calculate optimizer state memory usage.
        
        Args:
            optimizer: PyTorch optimizer
            precision: 'fp32' or 'fp16' (optimizer states are typically fp32)
        
        Returns:
            Dictionary with parameter count and optimizer state memory in MB
        """
        total_optimizer_memory = 0
        param_count = 0
        
        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            # Adam/AdamW: 2 states per parameter (momentum m and variance v)
            states_per_param = 2
        elif isinstance(optimizer, torch.optim.SGD):
            # SGD: 1 state per parameter (momentum) if momentum > 0
            momentum = optimizer.param_groups[0].get('momentum', 0)
            states_per_param = 1 if momentum > 0 else 0
        else:
            # For other optimizers, conservative estimate: 1 state per parameter
            states_per_param = 1
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    num_params = p.numel()
                    total_optimizer_memory += states_per_param * num_params * OPTIMIZER_STATE_BYTES_PER_PARAM
                    param_count += num_params
        
        return {
            'param_count': param_count,
            'optimizer_memory_MB': self._bytes_to_mb(total_optimizer_memory),
        }
    
    def profile_forward_backward(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        batch: Union[Dict, torch.Tensor],
        precision: str = 'fp32'
    ) -> Dict[str, Union[int, float, str]]:
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
            Dictionary with detailed memory profiling results in MB
        """
        device, is_cuda = self._get_device_info(model)
        
        # Initialize memory tracking
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        else:
            process = self._get_cpu_process()
            initial_memory = process.memory_info().rss
            peak_memory_during_profiling = initial_memory
        
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
                    outputs = model(**batch) if isinstance(batch, dict) else model(batch)
                    loss = self._compute_loss(outputs, batch, loss_fn)
                    
                    if not is_cuda:
                        peak_memory_during_profiling = max(peak_memory_during_profiling, process.memory_info().rss)
                
                with record_function("backward"):
                    loss.backward()
                    if not is_cuda:
                        peak_memory_during_profiling = max(peak_memory_during_profiling, process.memory_info().rss)
                
                with record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad()
                    if not is_cuda:
                        peak_memory_during_profiling = max(peak_memory_during_profiling, process.memory_info().rss)
        
        except Exception as e:
            return {
                'error': str(e),
                'forward_memory_MB': 0,
                'backward_memory_MB': 0,
                'optimizer_memory_MB': 0,
                'peak_memory_MB': 0,
                'profiler_table': '',
            }
        
        # Extract memory information from profiler events
        events = prof.key_averages()
        forward_memory = 0
        backward_memory = 0
        optimizer_memory = 0
        
        for event in events:
            memory_usage = event.cuda_memory_usage if is_cuda else event.cpu_memory_usage
            if 'forward' in event.key:
                forward_memory += memory_usage
            elif 'backward' in event.key:
                backward_memory += memory_usage
            elif 'optimizer' in event.key:
                optimizer_memory += memory_usage
        
        # Get peak memory
        if is_cuda:
            peak_memory = torch.cuda.max_memory_allocated() - initial_memory
        else:
            peak_memory = peak_memory_during_profiling - initial_memory
            if peak_memory <= 0:
                peak_memory = forward_memory + backward_memory + optimizer_memory
        
        return {
            'forward_memory_MB': self._bytes_to_mb(forward_memory),
            'backward_memory_MB': self._bytes_to_mb(backward_memory),
            'optimizer_memory_MB': self._bytes_to_mb(optimizer_memory),
            'peak_memory_MB': self._bytes_to_mb(peak_memory),
            'profiler_table': prof.key_averages().table(
                sort_by="cuda_memory_usage" if is_cuda else "cpu_memory_usage",
                row_limit=20
            ),
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
        device, is_cuda = self._get_device_info(model)
        
        # Get calculated memory for parameters and optimizer states
        param_memory = self.get_parameter_memory(model, precision)
        optimizer_memory = self.get_optimizer_state_memory(optimizer, precision)
        
        # Measure peak memory during full training step
        model.train()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = torch.cuda.memory_allocated()
        else:
            process = self._get_cpu_process()
            baseline_memory = process.memory_info().rss
        
        # Run a full training step to measure peak
        try:
            outputs = model(**batch) if isinstance(batch, dict) else model(batch)
            loss = self._compute_loss(outputs, batch, loss_fn)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if is_cuda:
                peak_memory = torch.cuda.max_memory_allocated()
            else:
                peak_memory = process.memory_info().rss
        except Exception as e:
            peak_memory = baseline_memory
        
        # Profile activation memory separately for detailed breakdown
        profile_results = self.profile_forward_backward(model, optimizer, loss_fn, batch, precision)
        
        return {
            'parameters': param_memory,
            'optimizer_states': optimizer_memory,
            'activations': {
                'forward_memory_MB': profile_results.get('forward_memory_MB', 0),
                'backward_memory_MB': profile_results.get('backward_memory_MB', 0),
                'peak_activation_memory_MB': profile_results.get('peak_memory_MB', 0),
            },
            'total': {
                'total_memory_MB': self._bytes_to_mb(peak_memory),
                'peak_memory_MB': self._bytes_to_mb(peak_memory),
                'peak_memory_GB': self._bytes_to_gb(peak_memory),
            },
            'breakdown': {
                'parameter_memory_MB': param_memory['total_memory_MB'],
                'optimizer_memory_MB': optimizer_memory['optimizer_memory_MB'],
                'activation_memory_MB': profile_results.get('peak_memory_MB', 0),
                'total_memory_MB': self._bytes_to_mb(peak_memory),
            },
            'profiler_table': profile_results.get('profiler_table', ''),
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
            peak_mb = total['peak_memory_MB']
            other_mb = peak_mb - breakdown['parameter_memory_MB'] - breakdown['optimizer_memory_MB'] - breakdown['activation_memory_MB']
            
            if peak_mb > 0:
                print(f"\nMemory Breakdown (based on peak memory):")
                print(f"  Parameters: {breakdown['parameter_memory_MB']:.2f} MB ({breakdown['parameter_memory_MB']/peak_mb*100:.1f}%)")
                print(f"  Optimizer: {breakdown['optimizer_memory_MB']:.2f} MB ({breakdown['optimizer_memory_MB']/peak_mb*100:.1f}%)")
                print(f"  Activations: {breakdown['activation_memory_MB']:.2f} MB ({breakdown['activation_memory_MB']/peak_mb*100:.1f}%)")
                print(f"  Other: {other_mb:.2f} MB ({other_mb/peak_mb*100:.1f}%)")
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

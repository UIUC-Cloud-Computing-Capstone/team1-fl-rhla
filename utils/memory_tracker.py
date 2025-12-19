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
import argparse
import unittest
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForImageClassification
from torch.profiler import profile, ProfilerActivity
import torch
import pandas as pd
import gc
import time
import statistics
from peft import LoraConfig, get_peft_model

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
    
    def _get_parameter_memory(self, model: torch.nn.Module, precision: str = 'fp32') -> Dict[str, Union[int, float]]:
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
        #print('trainable_params', trainable_params)
        print('trainable_params * 4 bytes', trainable_params * 4)
        print('trainable_params * 2 * 4 bytes', trainable_params * 2 * 4)
        
        param_memory_bytes = total_params * bytes_per_param
        trainable_memory_bytes = trainable_params * bytes_per_param
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_param_memory_MB': self._bytes_to_mb(param_memory_bytes),
            'trainable_memory_MB': self._bytes_to_mb(trainable_memory_bytes),
        }
    
    

    def _get_optimizer_states_memory_bytes(self, optimizer):
        total_mem = 0
        # Iterate through the actual state dictionary stored in the optimizer
        for param_id, state in optimizer.state.items():
            for key, value in state.items():
                if torch.is_tensor(value):
                    total_mem += value.element_size() * value.numel()
        return total_mem

    def _get_grads_memory_bytes(self, optimizer: torch.optim.Optimizer, precision: str = 'fp32') -> Dict[str, Union[int, float]]:
        """
        Calculate grads memory usage.
        
        Args:
            optimizer: PyTorch optimizer
            precision: 'fp32' or 'fp16' (optimizer states are typically fp32)
        
        Returns:
            Dictionary with parameter count and optimizer state memory in MB
        """
        
        total_grads_mem_bytes = 0
        grad_element_count = 0
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and torch.is_tensor(p.grad):
                    total_grads_mem_bytes += p.grad.element_size() * p.grad.numel()
                    grad_element_count += p.grad.numel()
        
        return total_grads_mem_bytes
    
    def _loss_fn(self, outputs, labels):
        return outputs.loss if hasattr(outputs, 'loss') else torch.nn.functional.cross_entropy(outputs, labels)
    
    def profile_and_compare(self, args, config, base_model, output_file_path, rank, memory_summary_dict):
        model, optimizer, batch, device = self._init_profiling(args, config, rank, base_model)
        profiled_info = self.profile(args, config, model, rank, memory_summary_dict, optimizer, batch, device)
        self._create_comparison(args, memory_summary_dict, profiled_info, output_file_path, rank)




    def _create_statistics_of_all_runs(self, args, is_cuda, model, optimizer, batch):
        
        all_profiled_params, all_profiled_optimizer, all_profiled_fwds, all_profiled_grads, all_profiled_total = \
            self._get_profiled_data_for_all_runs(args, is_cuda, model, optimizer, batch)
        profiled_info = {}
        profiled_info['avg_profiled_params'] = sum(all_profiled_params) / len(all_profiled_params)
        profiled_info['avg_profiled_optimizer'] = sum(all_profiled_optimizer) / len(all_profiled_optimizer)
        profiled_info['avg_profiled_fwd'] = sum(all_profiled_fwds) / len(all_profiled_fwds)
        profiled_info['avg_profiled_grads'] = sum(all_profiled_grads) / len(all_profiled_grads)
        profiled_info['avg_profiled_total'] = sum(all_profiled_total) / len(all_profiled_total)
        profiled_info['profiled_params_std'] = statistics.stdev(all_profiled_params) if len(all_profiled_params) > 1 else 0.0
        profiled_info['profiled_optimizer_std'] = statistics.stdev(all_profiled_optimizer) if len(all_profiled_optimizer) > 1 else 0.0
        profiled_info['profiled_activations_std'] = statistics.stdev(all_profiled_fwds) if len(all_profiled_fwds) > 1 else 0.0
        profiled_info['profiled_grads_std'] = statistics.stdev(all_profiled_grads) if len(all_profiled_grads) > 1 else 0.0
        profiled_info['profiled_total_std'] = statistics.stdev(all_profiled_total) if len(all_profiled_total) > 1 else 0.0
        return profiled_info

    def _create_comparison(self, args, memory_summary_dict, profiled_info, output_file_path, estimated_rank):
        estimated_total_params = memory_summary_dict.get('total_parameters_in_MB', 0)
        estimated_total_activations = memory_summary_dict.get('total_activations_gradients_and_with_safety_margin_in_MB', 0)
        estimated_total_optimizer = memory_summary_dict.get('total_optimizer_states_in_MB', 0)
        estimated_total_grads = memory_summary_dict.get('total_grads_in_MB', 0) # TODO
        estimated_total = memory_summary_dict.get('total_memory_in_MB', 0)

        profiled_params = profiled_info['avg_profiled_params']
        profiled_optimizer = profiled_info['avg_profiled_optimizer']
        profiled_activations = profiled_info['avg_profiled_fwd']
        profiled_grads = profiled_info['avg_profiled_grads']
        profiled_total = profiled_info['avg_profiled_total']     
        profiled_params_std = profiled_info['profiled_params_std']
        profiled_optimizer_std = profiled_info['profiled_optimizer_std']
        profiled_activations_std = profiled_info['profiled_activations_std']
        profiled_grads_std = profiled_info['profiled_grads_std']
        profiled_total_std = profiled_info['profiled_total_std']

        # Calculate errors
        def calculate_error(estimated, profiled):
            if profiled == 0:
                return float('inf') if estimated > 0 else 0.0
            return abs(estimated - profiled) / profiled * 100
        param_error = calculate_error(estimated_total_params, profiled_params)
        activation_error = calculate_error(estimated_total_activations, profiled_activations)
        optimizer_error = calculate_error(estimated_total_optimizer, profiled_optimizer)
        grad_error = calculate_error(estimated_total_grads, profiled_grads)
        total_error = calculate_error(estimated_total, profiled_total)
        
        # Create comparison table
        comparison_data = {
            'Component': ['Parameters', 'Forward Pass', 'Gradients', 'Optimizer States', 'Total Peak'],
            'Estimated (MB)': [
                f'{estimated_total_params:.2f}',
                f'{estimated_total_activations:.2f}',
                f'{estimated_total_grads:.2f}',
                f'{estimated_total_optimizer:.2f}',
                f'{estimated_total:.2f}'
            ],
            'Profiled (MB)': [
                f'{profiled_params:.2f}',
                f'{profiled_activations:.2f}',
                f'{profiled_grads:.2f}',
                f'{profiled_optimizer:.2f}',
                f'{profiled_total:.2f}'
            ],
            'Error (%)': [
                f'{param_error:.2f}',
                f'{activation_error:.2f}',
                f'{grad_error:.2f}',
                f'{optimizer_error:.2f}',
                f'{total_error:.2f}'
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        
        # Print table
        print("\n" + "="*80)
        print("MEMORY BREAKDOWN COMPARISON TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        
        self._create_latex(args, comparison_data, output_file_path)
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Mean Absolute Percentage Error (MAPE): {(param_error + activation_error + optimizer_error + total_error) / 4:.2f}%")
        print(f"  Rank used: {estimated_rank}")
        print(f"  Available memory: {args.gpu_memory_size_for_each_group_in_GB[0]} GB ({args.gpu_memory_size_for_each_group_in_GB[0] * 1024:.2f} MB)")
        print(f"  Memory utilization: {profiled_total / (args.gpu_memory_size_for_each_group_in_GB[0] * 1024) * 100:.2f}%")
        
        return df

    def _create_latex(self, args, comparison_data, output_file_path):
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'diagrams')
        
        # Generate LaTeX table with custom formatting
        latex_table = "% batch size: " + str(args.batch_size) + "\n"
        latex_table += "\\begin{table}[htbp]  % 'htbp' allows flexible placement: here, top, bottom, or page\n"
        latex_table += "\n"
        latex_table += "\\centering\n"
        latex_table += "\n"
        latex_table += "\\caption{Memory Breakdown Comparison Table}\n"
        latex_table += "\n"
        latex_table += "\\begin{tabular}{lrrr}\n"
        latex_table += "\n"
        latex_table += "\\toprule\n"
        latex_table += "Component & Estimated (MB) & Profiled (MB) & Error (\\%) \\\\\n"
        latex_table += "\n"
        latex_table += "\\midrule\n"
        
        # Format each row to match the example format
        component_names = ['Parameters', 'Activations', 'Optimizer States', 'Total Peak']
        for i, component in enumerate(component_names):
            estimated = comparison_data['Estimated (MB)'][i]
            # Replace ± with $\pm$ and format the profiled value
            profiled_str = comparison_data['Profiled (MB)'][i]
            profiled_formatted = profiled_str.replace(' ± ', ' $\\pm$ ')
            error = comparison_data['Error (%)'][i]
            
            # Format with proper spacing (matching the example)
            latex_table += f"{component:<15} & {estimated:>8}  & {profiled_formatted:<20} & {error:>6} \\\\\n"
        
        latex_table += "\n"
        latex_table += "\\bottomrule\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        
        latex_path = os.path.join(output_dir, output_file_path)
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_path}")

    def _init_profiling(self, args, config, r, base_model):
        print(f"\nCreating model with rank {r} and profiling actual memory...")
        config = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=args.lora_target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(base_model, config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._get_opt_batch_and_update_args(args, config, model, device), 
    
    def profile(self, args, config, model, rank, memory_summary_dict, optimizer, batch, device):
        is_cuda = device.type == 'cuda'
        if not is_cuda:
            raise ValueError('CPU memory profiling is not supported yet.')
        
        return self._create_statistics_of_all_runs(args, is_cuda, model, optimizer, batch)

    def _get_profiled_data_for_all_runs(self, args, is_cuda, model, optimizer, batch):
        print(f"\nProfiling actual memory {args.num_profiling_actual_runs} times...")
        all_profiled_params, all_profiled_optimizer, all_profiled_fwds, all_profiled_grads, all_profiled_total = [], [], [], [], []
        for run in range(args.num_profiling_warmup_runs + args.num_profiling_actual_runs):
            if run < args.num_profiling_warmup_runs:
                print('warm up')
            else:
                print(f"  Run {run - args.num_profiling_warmup_runs + 1}/{args.num_profiling_actual_runs}...", end=' ', flush=True)
            
            # Clear memory before each run to avoid interference
            if is_cuda:
                torch.cuda.empty_cache()  # Clear GPU cache
                torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats
            gc.collect()  # Force Python garbage collection
            
            time.sleep(3)
            
            model.train()
            activities = [ProfilerActivity.CPU]
            if is_cuda:
                activities.append(ProfilerActivity.CUDA)
            
            param_memory_dict = self._get_parameter_memory(model, args.precision)
            param_memory_MB = param_memory_dict['total_param_memory_MB']
            print('param', param_memory_MB * MB_TO_BYTES)

            grad_memory_bytes = self._get_grads_memory_bytes(optimizer, args.precision)
            print('grad1', grad_memory_bytes)

            # Profile forward and backward pass to get activation memory
            with profile(
                activities=activities,
                profile_memory=True,
                record_shapes=True
            ) as prof:
                torch.cuda.reset_peak_memory_stats()
                print('baseline_memory (param + overhead)', torch.cuda.max_memory_allocated())
                print('param', (param_memory_MB) * MB_TO_BYTES)
                
                # Forward pass
                
                outputs = model(**batch)
                loss = self._loss_fn(outputs, batch['labels'])
                peak_forward_mem = torch.cuda.max_memory_allocated()
                print('peak forward memory: ', peak_forward_mem)
                
                
                torch.cuda.reset_peak_memory_stats()
                # Backward pass
                loss.backward()
                peak_backward_mem = torch.cuda.max_memory_allocated()
                print('peak backward  memory: ', peak_backward_mem)
                grad_memory_MB2 = peak_backward_mem - peak_forward_mem
                print('grad2', grad_memory_MB2)
                torch.cuda.reset_peak_memory_stats()
                
                
                torch.cuda.reset_peak_memory_stats()
                # Optimizer step (to include optimizer state allocations)
                optimizer.step()
                peak_opt_step_mem = torch.cuda.max_memory_allocated()
                print('peak optimer step memory: ', peak_opt_step_mem)
                torch.cuda.reset_peak_memory_stats()
                
                opt_state_mem = bytes = self._get_optimizer_states_memory_bytes(optimizer)
                optimizer_memory_MB = self._bytes_to_mb(opt_state_mem)
                grad_memory_bytes = self._get_grads_memory_bytes(optimizer, args.precision)
                grad_memory_MB = self._bytes_to_mb(grad_memory_bytes)
                print('grad1 before zero grad', grad_memory_bytes)
                optimizer.zero_grad()
                
                peak_zero_grad_mem = torch.cuda.max_memory_allocated()
                print('peak zero grad memory: ', peak_zero_grad_mem)
            
            #print(prof.key_averages().table(row_limit=10))
            if is_cuda:
                peak_memory_bytes = torch.cuda.max_memory_allocated()
                if run >=  args.num_profiling_warmup_runs:
                    print('peak_memory_bytes', peak_memory_bytes)
                peak_memory_MB = peak_memory_bytes / (MB_TO_BYTES)
            else:
                raise ValueError('CPU memory profiling is not supported yet.')
            
            fwd_memory_MB = max(0, peak_memory_MB - param_memory_MB - optimizer_memory_MB - grad_memory_MB)
            
            # skip first run, to warm up
            if run < args.num_profiling_warmup_runs:
                print(f"Warm-up {run + 1} Done (Total: {peak_memory_MB:.2f} MB)")
            else:
                all_profiled_params.append(param_memory_MB)
                all_profiled_optimizer.append(optimizer_memory_MB)
                all_profiled_fwds.append(fwd_memory_MB)
                all_profiled_grads.append(grad_memory_MB)
                all_profiled_total.append(peak_memory_MB)
                print(f"Done (Total: {peak_memory_MB:.2f} MB)")
                
            
            # Clear memory after each run
            if is_cuda:
                torch.cuda.empty_cache()
            gc.collect()
        
        return all_profiled_params, all_profiled_optimizer, all_profiled_fwds, all_profiled_grads, all_profiled_total

    def _get_opt_batch_and_update_args(self, args, config, model, device):
        args.num_profiling_warmup_runs = 0
        args.num_profiling_actual_runs = 1

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        
        model = model.to(device)
        batch = {
            'pixel_values': torch.randn(args.batch_size, 3, config.image_size, config.image_size).to(device),
            'labels': torch.randint(0, 1000, (args.batch_size,)).to(device)
        }

        return model, optimizer, batch

    def get_base_model_fwd_in_MB_for_estimator(self, args, config, base_model):

        H = config.hidden_size
        r = H
        
        config_0qk = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=["0.attention.attention.query", "0.attention.attention.key"],
            lora_dropout=0.1,
            bias="none",
        )
        
        config_0k = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=["0.attention.attention.key"],
            lora_dropout=0.1,
            bias="none",
        )

        config_0q = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=["0.attention.attention.query"],
            lora_dropout=0.1,
            bias="none",
        )



        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        memory_summary_dict_q = {}
        
        info_q = self._get_base_model_fwd_in_MB_for_estimator_helper(args, config, base_model, r, ["attention.attention.query"], device)
        #info_k = self._get_base_model_fwd_in_MB_for_estimator_helper(args, config, base_model, r, ["attention.attention.key"], device)
        #info_qk = self._get_base_model_fwd_in_MB_for_estimator_helper(args, config, base_model, r, ["attention.attention.query", "attention.attention.key"], device)

        #return info_q['avg_profiled_fwd'] - (info_qk['avg_profiled_fwd'] - info_k['avg_profiled_fwd'])
        
    def _get_base_model_fwd_in_MB_for_estimator_helper(self, args, config, base_model, r, target_modules, device):
        def clear_mem(device):
            is_cuda = device.type == 'cuda'
            if is_cuda:
                #print('before reset', torch.cuda.max_memory_allocated())
                torch.cuda.empty_cache()  # Clear GPU cache
                torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats
            gc.collect()  # Force Python garbage collection
            
            time.sleep(3) # seconds
            #print('after reset', torch.cuda.max_memory_allocated())
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
        model, optimizer, batch = self._get_opt_batch_and_update_args(args, config, model, device)
        clear_mem(device)
        result = self.profile(args, config, model, r, {}, optimizer, batch, device)

        del lora_config
        del model
        del optimizer
        del batch
        #clear_mem(device)
        return result
        



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
from torch.profiler import profile, ProfilerActivity
import statistics
import os
from torch.profiler import profile, ProfilerActivity
import torch
import pandas as pd
import gc
import time
import statistics
from peft import LoraConfig, get_peft_model
import copy
import numpy as np
import random
import matplotlib.pyplot as plt

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
        trainable_memory_bytes = trainable_params * bytes_per_param
        trainable_memory_MB = self._bytes_to_mb(trainable_memory_bytes)
        #print('trainable_memory_MB', trainable_memory_MB)
        
        param_memory_bytes = total_params * bytes_per_param
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_param_memory_bytes': param_memory_bytes,
            'trainable_memory_MB': trainable_memory_MB,
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, optimizer, batch = self._init_profiling(args, config, rank, base_model, device)
        profiled_info = self.profile(args, config, model, rank, memory_summary_dict, optimizer, batch, device)
        self._create_comparison(args, memory_summary_dict, profiled_info, output_file_path, rank)

    def _create_statistics_of_all_runs(self, args, is_cuda, model, optimizer, batch):
        
        all_profiled_params, all_profiled_optimizer, all_profiled_fwds, all_profiled_grads, all_profiled_overhead, all_profiled_total = \
            self._get_profiled_data_for_all_runs(args, is_cuda, model, optimizer, batch)
        profiled_info = {}
        profiled_info['avg_profiled_params'] = sum(all_profiled_params) / len(all_profiled_params)
        profiled_info['avg_profiled_optimizer'] = sum(all_profiled_optimizer) / len(all_profiled_optimizer)
        profiled_info['avg_profiled_fwd'] = sum(all_profiled_fwds) / len(all_profiled_fwds)
        profiled_info['avg_profiled_grads'] = sum(all_profiled_grads) / len(all_profiled_grads)
        profiled_info['avg_profiled_overhead'] = sum(all_profiled_overhead) / len(all_profiled_overhead)
        profiled_info['avg_profiled_total'] = sum(all_profiled_total) / len(all_profiled_total)
        profiled_info['profiled_params_std'] = statistics.stdev(all_profiled_params) if len(all_profiled_params) > 1 else 0.0
        profiled_info['profiled_optimizer_std'] = statistics.stdev(all_profiled_optimizer) if len(all_profiled_optimizer) > 1 else 0.0
        profiled_info['profiled_activations_std'] = statistics.stdev(all_profiled_fwds) if len(all_profiled_fwds) > 1 else 0.0
        profiled_info['profiled_grads_std'] = statistics.stdev(all_profiled_grads) if len(all_profiled_grads) > 1 else 0.0
        profiled_info['profiled_overhead_std'] = statistics.stdev(all_profiled_overhead) if len(all_profiled_overhead) > 1 else 0.0
        profiled_info['profiled_total_std'] = statistics.stdev(all_profiled_total) if len(all_profiled_total) > 1 else 0.0
        return profiled_info

    def _create_comparison(self, args, memory_summary_dict, profiled_info, output_file_path, estimated_rank):
        estimated_total_params = memory_summary_dict.get('total_para_bytes', 0)
        estimated_total_fwd = memory_summary_dict.get('total_fwd_bytes', 0)
        estimated_total_optimizer = memory_summary_dict.get('total_optimizer_states_bytes', 0)
        estimated_total_grads = memory_summary_dict.get('total_grads_bytes', 0)
        estimated_overhead = memory_summary_dict.get('overhead_bytes', 0)
        estimated_total = memory_summary_dict.get('total_memory_bytes', 0)

        profiled_params = profiled_info['avg_profiled_params']
        profiled_optimizer = profiled_info['avg_profiled_optimizer']
        profiled_fwd = profiled_info['avg_profiled_fwd']
        profiled_grads = profiled_info['avg_profiled_grads']
        profiled_overhead = profiled_info['avg_profiled_overhead']
        profiled_total = profiled_info['avg_profiled_total']     
        profiled_params_std = profiled_info['profiled_params_std']
        profiled_optimizer_std = profiled_info['profiled_optimizer_std']
        profiled_activations_std = profiled_info['profiled_activations_std']
        profiled_grads_std = profiled_info['profiled_grads_std']
        profiled_overhead_str = profiled_info['profiled_overhead_std']
        profiled_total_std = profiled_info['profiled_total_std']

        # Calculate errors
        def calculate_error(estimated, profiled):
            if profiled == 0:
                return float('inf') if estimated > 0 else 0.0
            return abs(estimated - profiled) / profiled * 100
        param_error = calculate_error(estimated_total_params, profiled_params)
        activation_error = calculate_error(estimated_total_fwd, profiled_fwd)
        optimizer_error = calculate_error(estimated_total_optimizer, profiled_optimizer)
        grad_error = calculate_error(estimated_total_grads, profiled_grads)
        overhead_error = calculate_error(estimated_overhead, profiled_overhead)
        total_error = calculate_error(estimated_total, profiled_total)

        # percentage
        param_perc = profiled_params / profiled_total * 100
        fwd_perc = profiled_fwd / profiled_total * 100
        grads_perc = profiled_grads / profiled_total * 100
        opt_perc = profiled_optimizer / profiled_total * 100
        overhead_perc = profiled_overhead / profiled_total * 100
        
        total_perc = param_perc + fwd_perc + grads_perc + opt_perc + overhead_perc 

        # convert bytes to MB
        estimated_total_params = self._bytes_to_mb(estimated_total_params)
        estimated_total_fwd = self._bytes_to_mb(estimated_total_fwd)
        estimated_total_grads = self._bytes_to_mb(estimated_total_grads)
        estimated_total_optimizer = self._bytes_to_mb(estimated_total_optimizer)
        estimated_overhead = self._bytes_to_mb(estimated_overhead)
        estimated_total = self._bytes_to_mb(estimated_total)

        profiled_params = self._bytes_to_mb(profiled_params)
        profiled_fwd = self._bytes_to_mb(profiled_fwd)
        profiled_grads = self._bytes_to_mb(profiled_grads)
        profiled_optimizer = self._bytes_to_mb(profiled_optimizer)
        profiled_overhead = self._bytes_to_mb(profiled_overhead)
        profiled_total = self._bytes_to_mb(profiled_total)

        # Create comparison table
        comparison_data = {
            'Component': ['Parameters', 'Forward Pass', 'Gradients', 'Optimizer States', 'Overhead', 'Total Peak'],
            'Estimated (MB)': [
                f'{estimated_total_params:.2f}',
                f'{estimated_total_fwd:.2f}',
                f'{estimated_total_grads:.2f}',
                f'{estimated_total_optimizer:.2f}',
                f'{estimated_overhead:.2f}',
                f'{estimated_total:.2f}'
            ],
            'Profiled (MB)': [
                f'{profiled_params:.2f} ({(param_perc):.2f}%)',
                f'{profiled_fwd:.2f} ({(fwd_perc):.2f}%)',
                f'{profiled_grads:.2f} ({(grads_perc):.2f}%)',
                f'{profiled_optimizer:.2f} ({(opt_perc):.2f}%)',
                f'{profiled_overhead:.2f} ({(overhead_perc):.2f}%)',
                f'{profiled_total:.2f} ({(total_perc):.2f}%)'
            ],
            'Error (%)': [
                f'{param_error:.2f}',
                f'{activation_error:.2f}',
                f'{grad_error:.2f}',
                f'{optimizer_error:.2f}',
                f'{overhead_error:.2f}',
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
        # if overhead_perc > 0:
        #     print('Overhead: ', f'{profiled_overhead:.2f} ({overhead_perc:.2f}%)')
        
        # Calculate summary statistics
        mape = (param_error + activation_error + optimizer_error + total_error) / 4
        available_memory_gb = args.gpu_memory_size_for_each_group_in_GB[0]
        available_memory_mb = available_memory_gb * 1024
        memory_utilization = profiled_total / available_memory_mb * 100
        
        self._create_latex(args, comparison_data, output_file_path, mape, estimated_rank, available_memory_gb, available_memory_mb, memory_utilization)
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"  Rank used: {estimated_rank}")
        print(f"  Available memory: {available_memory_gb} GB ({available_memory_mb:.2f} MB)")
        print(f"  Memory utilization: {memory_utilization:.2f}%")
        
        return df

    def _create_latex(self, args, comparison_data, output_file_path, mape, rank, available_memory_gb, available_memory_mb, memory_utilization):
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'diagrams')
        
        # Generate LaTeX table with custom formatting
        latex_table = f"% Profiling runs to solve betas: {args.beta_profiling_run:d}\n"
        latex_table += f"% batch size: {args.batch_size:.2f}\n"
        latex_table += "% Summary Statistics:\n"
        latex_table += f"%   Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n"
        latex_table += f"%   Rank used: {rank}\n"
        latex_table += f"%   Available memory: {available_memory_gb} GB ({available_memory_mb:.2f} MB)\n"
        latex_table += f"%   Memory utilization: {memory_utilization:.2f}%\n"
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
        component_names = ['Parameters', 'Forward Pass', 'Gradients', 'Optimizer States', 'Overhead', 'Total Peak']
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

    def _init_profiling(self, args, config, r, base_model, device):
        print(f"\nCreating model with rank {r} and profiling actual memory...")
        lora_config = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=args.lora_target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
        return self._get_opt_batch_and_update_args(args, config, model, device)
    
    def profile(self, args, config, model, rank, memory_summary_dict, optimizer, batch, device):
        is_cuda = device.type == 'cuda'
        if not is_cuda:
            raise ValueError('CPU memory profiling is not supported yet.')
        
        return self._create_statistics_of_all_runs(args, is_cuda, model, optimizer, batch)

    def _get_profiled_data_for_all_runs(self, args, is_cuda, model, optimizer, batch):
        print(f"\nProfiling actual memory {args.num_profiling_actual_runs} times...")
        all_profiled_params, all_profiled_optimizer, all_profiled_fwds, all_profiled_grads, all_profiled_overhead, all_profiled_total = [], [], [], [], [], []
    
        activities = [ProfilerActivity.CPU]
        if is_cuda:
            activities.append(ProfilerActivity.CUDA)
        
        if is_cuda:
            torch.cuda.empty_cache()  # Clear GPU cache
            torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats
        gc.collect()  # Force Python garbage collection

        model.train()
        with profile(
                activities=activities,
                profile_memory=True,
                record_shapes=True
            ) as prof:
            for run in range(args.num_profiling_warmup_runs + args.num_profiling_actual_runs):
                torch.cuda.reset_peak_memory_stats()
                if run < args.num_profiling_warmup_runs:
                    print('Model warming up')
                else:
                    print(f"Run {run - args.num_profiling_warmup_runs + 1}/{args.num_profiling_actual_runs}...", end=' ', flush=True)

                param_memory_dict = self._get_parameter_memory(model, args.precision)
                param_memory_bytes = param_memory_dict['total_param_memory_bytes']    
                
                grad_memory_bytes = self._get_grads_memory_bytes(optimizer, args.precision)

                # Profile forward and backward pass to get activation memory
                
                torch.cuda.reset_peak_memory_stats()
                baseline_mem = torch.cuda.max_memory_allocated()
                    
                # Forward pass
                outputs = model(**batch)
                loss = self._loss_fn(outputs, batch['labels'])
                peak_forward_mem = torch.cuda.max_memory_allocated()
                peak_fwd_mem_excl_baseline = peak_forward_mem - baseline_mem
                fwd_memory_bytes = peak_fwd_mem_excl_baseline
                
                    
                torch.cuda.reset_peak_memory_stats()
                # Backward pass
                loss.backward()
                peak_backward_mem = torch.cuda.max_memory_allocated()
                    
                # Optimizer step (to include optimizer state allocations)
                torch.cuda.reset_peak_memory_stats()
                before_opt_step_mem = torch.cuda.max_memory_allocated()
                optimizer.step()
                peak_opt_step_mem = torch.cuda.max_memory_allocated()
                
                opt_state_mem_bytes = self._get_optimizer_states_memory_bytes(optimizer)
                
                    
                # grad
                grad_memory_bytes = self._get_grads_memory_bytes(optimizer, args.precision)
                #grad_memory_MB = self._bytes_to_mb(grad_memory_bytes)                
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad()
                peak_zero_grad_mem = torch.cuda.max_memory_allocated()
                #print('peak memory during zero grad: ', peak_zero_grad_mem)
                torch.cuda.reset_peak_memory_stats()
                peak_after_zero_grad = torch.cuda.max_memory_allocated()
                
                peak_memory_bytes = max(baseline_mem, peak_forward_mem, peak_backward_mem, peak_opt_step_mem, peak_zero_grad_mem, peak_after_zero_grad)
                peak_memory_MB = self._bytes_to_mb(peak_memory_bytes)
                
                # skip first run, to warm up
                if run < args.num_profiling_warmup_runs:
                    print(f"Model warm-up run {run + 1} done (total: {peak_memory_MB:.2f} MB)")
                else:
                    all_excl_overhead_bytes = param_memory_bytes + opt_state_mem_bytes + fwd_memory_bytes + grad_memory_bytes
                    overhead_in_bytes = peak_memory_bytes - all_excl_overhead_bytes
                    all_profiled_params.append(param_memory_bytes)
                    all_profiled_optimizer.append(opt_state_mem_bytes)
                    all_profiled_fwds.append(fwd_memory_bytes)
                    all_profiled_grads.append(grad_memory_bytes)
                    all_profiled_overhead.append(overhead_in_bytes)
                    all_profiled_total.append(peak_memory_bytes)
                    print(f"Done (Total: {peak_memory_MB:.2f} MB)")
                
                
        return all_profiled_params, all_profiled_optimizer, all_profiled_fwds, all_profiled_grads, all_profiled_overhead, all_profiled_total

    def _get_opt_batch_and_update_args(self, args, config, model, device):
        args.num_profiling_warmup_runs = 1
        args.num_profiling_actual_runs = 1

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        
        model = model.to(device)
        batch = {
            'pixel_values': torch.randn(args.batch_size, 3, config.image_size, config.image_size).to(device),
            'labels': torch.randint(0, 1000, (args.batch_size,)).to(device)
        }

        return model, optimizer, batch

    def get_base_model_fwd_in_bytes_for_estimator(self, args, config, base_model):

        H = config.hidden_size
        r = int(H / 2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        info_q = self._get_base_model_fwd_in_bytes_for_estimator_helper(args, config, copy.deepcopy(base_model), r, ["0.attention.attention.query"], device)
        info_k = self._get_base_model_fwd_in_bytes_for_estimator_helper(args, config, copy.deepcopy(base_model), r, ["0.attention.attention.key"], device)
        info_qk = self._get_base_model_fwd_in_bytes_for_estimator_helper(args, config, copy.deepcopy(base_model), r, ["0.attention.attention.query", "0.attention.attention.key"], device)

        fwd_key, overhead_key = 'avg_profiled_fwd', 'avg_profiled_overhead'
        print('q', info_q[fwd_key], 'qk', info_qk[fwd_key], 'k', info_k[fwd_key])
        base_fwd_bytes = info_q[fwd_key] - (info_qk[fwd_key] - info_k[fwd_key])
        overhead_bytes = statistics.mean([info_q[overhead_key], info_qk[overhead_key], info_k[overhead_key]])
        print('base_fw_bytes', base_fwd_bytes, 'overhead_bytes', overhead_bytes)
        return base_fwd_bytes, overhead_bytes
    
    def get_lora_betas_v2(self, args, config, base_model, module_names, B, S, H, bytes_per_parameter, memory_summary_dict):
        '''
        betas for for one module (two matrices) on one layer 
        '''
        H = config.hidden_size
        r1 = int(H / 2)
        r2 = int(H / 3)

        if not hasattr(args, 'beta_profiling_run') or not args.beta_profiling_run:
            args.beta_profiling_run = 2
        beta_profiling_run = args.beta_profiling_run
        if beta_profiling_run < 2:
            raise ValueError('beta_profiling_run should be at least 2')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fwd_key= 'avg_profiled_fwd'
        # generate random r values
        rs = [random.randint(1, H) for _ in range(beta_profiling_run)]
        ys = []
        bsrs = [B * S * r for r in rs]
    
        for i in range(beta_profiling_run):
            y = self._get_base_model_fwd_in_bytes_for_estimator_helper(args, config, copy.deepcopy(base_model), rs[i], module_names, device)[fwd_key]
            y -= memory_summary_dict['base_model_fwd_bytes']
            ys.append(y / 4)
            
        

        bs = B * S
        bsh = bs * H

        # y = beta1 * bsh + beta2 * bsrs
        # regression analysis to get beta1 and beta2
        # y = beta1 * bsh + beta2 * bsrs
        # regression analysis to get beta1 and beta2
        bsh_column = np.full(len(ys), bsh) 

        # Create the feature matrix X by stacking the columns
        # X will be shape (10, 2)
        X = np.column_stack((bsh_column, bsrs))

        y = np.array(ys)

        beta_vector, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta1, beta2 = beta_vector

        # plot the data
        # plt.scatter(bsrs, ys)
        # plt.xlabel('bsrs')
        # plt.ylabel('ys')
        # plt.title('beta regression EDA')
        # plt.show()
        # plt.savefig('beta_regression_eda.png')

        print('betas: ', beta1, beta2)
        return beta1, beta2
        
    def _get_base_model_fwd_in_bytes_for_estimator_helper(self, args, config, base_model, r, target_modules, device):
        def clear_mem(device):
            is_cuda = device.type == 'cuda'
            if is_cuda:
                #print('before reset', torch.cuda.max_memory_allocated())
                torch.cuda.empty_cache()  # Clear GPU cache
                torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats
            gc.collect()  # Force Python garbage collection
            #time.sleep(3) # seconds
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
        return result
        



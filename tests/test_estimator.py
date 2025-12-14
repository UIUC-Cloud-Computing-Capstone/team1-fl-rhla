"""
Unit tests for RankEstimator class in estimator.py
"""
import argparse
import unittest
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForImageClassification
from torch.profiler import profile, ProfilerActivity

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from estimator import RankEstimator
from peft import LoraConfig, get_peft_model
from utils.memory_tracker import MemoryTracker


class TestRankEstimator(unittest.TestCase):
    """Test cases for RankEstimator class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.estimator = RankEstimator()

    def test_get_sequence_length_returns_197(self):
        """Test that _get_sequence_length returns 197 for facebook/deit-small-patch16-224"""
        args = argparse.Namespace()
        args.image_height = 224
        args.image_width = 224
        args.patch_size = 16
        result = self.estimator._get_sequence_length(args)
        
        # Verify the result is 197
        # Calculation: (224 / 16) * (224 / 16) + 1 = 14 * 14 + 1 = 196 + 1 = 197
        self.assertEqual(result, 197)
        
        # Verify it's a number (int or float)
        self.assertIsInstance(result, (int, float))
        
        # Verify the calculation is correct
        H = 224
        P = 16
        W = 224
        CLS_TOKEN = 1
        expected = H / P * W / P + CLS_TOKEN
        self.assertEqual(result, expected)

    def test_get_sequence_length_calculation(self):
        """Test the sequence length calculation formula"""
        args = argparse.Namespace()
        args.image_height = 224
        args.image_width = 224
        args.patch_size = 16
        result = self.estimator._get_sequence_length(args)
        
        # Verify the calculation: (H / P) × (W / P) + CLS_TOKEN
        # For facebook/deit-small-patch16-224:
        # H = 224, P = 16, W = 224, CLS_TOKEN = 1
        # (224 / 16) * (224 / 16) + 1 = 14 * 14 + 1 = 196 + 1 = 197
        patches_per_side = 224 / 16  # 14
        total_patches = patches_per_side * patches_per_side  # 14 * 14 = 196
        expected = total_patches + 1  # 196 + 1 = 197
        
        self.assertEqual(result, expected)
        self.assertEqual(result, 197)



    def test_get_base_model_activations_and_safety_margin_memory_size_in_bytes(self):
        args = argparse.Namespace()
        args.batch_size = 32
        args.precision = "fp16"
        args.image_height = 224
        args.image_width = 224
        args.patch_size = 16
        args.percentage_of_layers_in_memory = 12 / 12
        args.overhead_and_safety_margin_factor = 0.1
        args.train_classifier = False # do not train classifier in the base model. Only train LoRA matrices.
        result = self.estimator._get_base_model_activations_and_safety_margin_memory_size_in_bytes(args)
        result_in_MB = result / 1024 / 1024
        print(result_in_MB)


    def test_get_rank_for_all_client_groups_ours(self):
        args = argparse.Namespace()
        args.rank_estimator_method = 'Ours'

        # resource heterogeneity
        # all clients belong to 3 heterogeneous groups. Each group has different resource limitation.
        args.gpu_memory_size_for_each_group_in_GB = [2, 4, 8]
        args.avg_upload_network_speed_for_each_group_in_Mbps = [1, 1.5, 2]
        args.avg_download_network_speed_for_each_group_in_Mbps = [10, 50, 50]
        args.desired_uploading_time_for_each_group_in_seconds = [15, 15, 15]
        args.desired_downloading_time_for_each_group_in_seconds = [30, 30, 30]
        args.heterogeneous_group = [1/3, 1/3, 1/3] # 1/3 of clients belong to each group.

        # model
        args.model = 'facebook/deit-small-patch16-224'

        # training hyperparameters
        args.precision = 'fp32'
        args.optimizer = 'adam'
        args.num_of_layers_to_allocate_LoRA = 12
        args.lora_target_modules = ["query", "value"]
        args.train_classifier = False # do not train classifier in the base model. Only train LoRA matrices.

        # input data sizes
        args.image_height = 224
        args.image_width = 224
        args.patch_size = 16 # each image is split into 16 × 16 pixel patches.
        args.batch_size = 32
        
        # estimation parameters
        args.percentage_of_layers_in_memory = 12 / 12 # not all layers are in memory at the same time during forward pass and backward pass.
        args.overhead_and_safety_margin_factor = 0.1 # assume 10% of activations and gradients

        model = AutoModelForImageClassification.from_pretrained(args.model)
        
        rank_budgets_for_all_heterogeneous_groups = self.estimator.get_rank_for_all_client_groups(args, model)
        print(rank_budgets_for_all_heterogeneous_groups)

    def test_memory_breakdown_comparison_table(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""
        import torch
        import pandas as pd
        
        # Configuration
        args = argparse.Namespace()
        args.rank_estimator_method = 'Ours'
        args.model = 'facebook/deit-small-patch16-224'
        args.precision = 'fp32'
        args.optimizer = 'adam'
        args.num_of_layers_to_allocate_LoRA = 12
        args.lora_target_modules = ["query", "value"]
        args.train_classifier = False # do not train classifier in the base model. Only train LoRA matrices.
        args.image_height = 224
        args.image_width = 224
        args.patch_size = 16
        args.batch_size = 32
        args.percentage_of_layers_in_memory = 12 / 12
        args.overhead_and_safety_margin_factor = 0.1
        args.desired_uploading_time_for_each_group_in_seconds = [15]
        args.desired_downloading_time_for_each_group_in_seconds = [15]
        args.heterogeneous_group = [1.0]
        args.gpu_memory_size_for_each_group_in_GB = [8.0]
        args.avg_upload_network_speed_for_each_group_in_Mbps = [7.0]
        args.avg_download_network_speed_for_each_group_in_Mbps = [50.0]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)
        
        # Get estimated rank and memory breakdown
        print("Getting estimated rank and memory breakdown...")
        memory_summary_dict = {}
        estimated_rank = self.estimator._get_rank_for_one_client_group(
            args, base_model, 
            args.gpu_memory_size_for_each_group_in_GB[0],
            args.avg_upload_network_speed_for_each_group_in_Mbps[0],
            args.avg_download_network_speed_for_each_group_in_Mbps[0],
            args.desired_uploading_time_for_each_group_in_seconds[0],
            args.desired_downloading_time_for_each_group_in_seconds[0],
            memory_summary_dict
        )
        
        # Calculate total estimated memory components (same as in get_rank_for_all_client_groups)
        memory_summary_dict['total_parameters_in_MB'] = memory_summary_dict.get('base_model_parameter_memory_size_in_MB', 0) + \
                                                       memory_summary_dict.get('lora_portion_parameter_size_in_MB', 0)
        memory_summary_dict['total_activations_gradients_and_with_safety_margin_in_MB'] = memory_summary_dict.get('base_model_activations_gradients_and_safety_margin_memory_size_in_MB', 0) + \
                                                                                          memory_summary_dict.get('lora_portion_activations_gradients_and_workspace_margin_in_MB', 0)
        memory_summary_dict['total_optimizer_states_in_MB'] = memory_summary_dict.get('base_model_optimizer_states_memory_size_in_MB', 0) + \
                                                              memory_summary_dict.get('lora_portion_optimizer_states_size_in_MB', 0)
        memory_summary_dict['total_memory_in_MB'] = round(memory_summary_dict['total_parameters_in_MB'] + 
                                                         memory_summary_dict['total_activations_gradients_and_with_safety_margin_in_MB'] + 
                                                         memory_summary_dict['total_optimizer_states_in_MB'], 2)
        
        # Extract values for comparison
        estimated_total_params = memory_summary_dict['total_parameters_in_MB']
        estimated_total_activations = memory_summary_dict['total_activations_gradients_and_with_safety_margin_in_MB']
        estimated_total_optimizer = memory_summary_dict['total_optimizer_states_in_MB']
        estimated_total = memory_summary_dict['total_memory_in_MB']
        
        print(f"Estimated rank: {estimated_rank}")
        print(f"Estimated memory breakdown:")
        print(f"  Parameters: {estimated_total_params:.2f} MB")
        print(f"  Activations: {estimated_total_activations:.2f} MB")
        print(f"  Optimizer: {estimated_total_optimizer:.2f} MB")
        print(f"  Total: {estimated_total:.2f} MB")
        
        # Create model with estimated rank and profile actual memory
        print(f"\nCreating model with rank {estimated_rank} and profiling actual memory...")
        config = LoraConfig(
            r=estimated_rank,
            lora_alpha=estimated_rank,
            target_modules=args.lora_target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(AutoModelForImageClassification.from_pretrained(args.model), config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create batch
        batch = {
            'pixel_values': torch.randn(args.batch_size, 3, args.image_height, args.image_width).to(device),
            'labels': torch.randint(0, 1000, (args.batch_size,)).to(device)
        }
        
        def loss_fn(outputs, labels):
            return outputs.loss if hasattr(outputs, 'loss') else torch.nn.functional.cross_entropy(outputs, labels)
        
        # Profile actual memory (run 10 times and take average for accuracy)
        num_profiling_runs = 10
        print(f"\nProfiling actual memory {num_profiling_runs} times to get average...")
        
        all_profiled_params = []
        all_profiled_optimizer = []
        all_profiled_activations = []
        all_profiled_total = []
        
        # Check if using GPU
        is_cuda = device.type == 'cuda'
        if not is_cuda:
            raise ValueError('CPU memory profiling is not supported yet.')
        import gc
        import time
        
        for run in range(num_profiling_runs):
            print(f"  Run {run + 1}/{num_profiling_runs}...", end=' ', flush=True)
            
            # Clear memory before each run to avoid interference
            if is_cuda:
                torch.cuda.empty_cache()  # Clear GPU cache
                torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats
            gc.collect()  # Force Python garbage collection
            
            time.sleep(3)
            
            # Use PyTorch profiler directly (like ResNet example)
            model.train()
            activities = [ProfilerActivity.CPU]
            if is_cuda:
                activities.append(ProfilerActivity.CUDA)
            
            # Get parameter and optimizer memory (static values)
            tracker = MemoryTracker()
            
            # Get parameter memory using MemoryTracker method
            param_memory_dict = tracker.get_parameter_memory(model, args.precision)
            param_memory_MB = param_memory_dict['total_memory_MB']
            
            # Get optimizer memory using MemoryTracker method
            optimizer_memory_dict = tracker.get_optimizer_state_memory(optimizer, args.precision)
            optimizer_memory_MB = optimizer_memory_dict['optimizer_memory_MB']
            
            # Profile forward and backward pass to get activation memory
            with profile(
                activities=activities,
                profile_memory=True,
                record_shapes=True
            ) as prof:
                # Forward pass
                outputs = model(**batch)
                loss = loss_fn(outputs, batch['labels'])
                # Backward pass
                loss.backward()
                # Optimizer step (to include optimizer state allocations)
                optimizer.step()
                optimizer.zero_grad()
            
            # Extract memory information from profiler
            # Get peak memory from profiler
            if is_cuda:
                peak_memory_bytes = torch.cuda.max_memory_allocated()
                peak_memory_MB = peak_memory_bytes / (1024 * 1024)
            else:
                raise ValueError('CPU memory profiling is not supported yet.')
            
            # Activation memory is peak memory minus parameters and optimizer states
            # (approximation, as activations are temporary)
            activation_memory_MB = max(0, peak_memory_MB - param_memory_MB - optimizer_memory_MB)
            
            # Structure results in the same format as before
            profiled_results = {
                'parameters': {
                    'total_memory_MB': param_memory_MB
                },
                'optimizer_states': {
                    'optimizer_memory_MB': optimizer_memory_MB
                },
                'breakdown': {
                    'activation_memory_MB': activation_memory_MB
                },
                'total': {
                    'peak_memory_MB': peak_memory_MB
                }
            }
            
            # Collect values from this run
            all_profiled_params.append(profiled_results['parameters']['total_memory_MB'])
            all_profiled_optimizer.append(profiled_results['optimizer_states']['optimizer_memory_MB'])
            all_profiled_activations.append(profiled_results['breakdown']['activation_memory_MB'])
            all_profiled_total.append(profiled_results['total']['peak_memory_MB'])
            print(f"Done (Total: {profiled_results['total']['peak_memory_MB']:.2f} MB)")
            
            # Clear memory after each run
            if is_cuda:
                torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate averages
        profiled_params = sum(all_profiled_params) / len(all_profiled_params)
        profiled_optimizer = sum(all_profiled_optimizer) / len(all_profiled_optimizer)
        profiled_activations = sum(all_profiled_activations) / len(all_profiled_activations)
        profiled_total = sum(all_profiled_total) / len(all_profiled_total)
        
        # Calculate standard deviations for reporting
        import statistics
        profiled_params_std = statistics.stdev(all_profiled_params) if len(all_profiled_params) > 1 else 0.0
        profiled_optimizer_std = statistics.stdev(all_profiled_optimizer) if len(all_profiled_optimizer) > 1 else 0.0
        profiled_activations_std = statistics.stdev(all_profiled_activations) if len(all_profiled_activations) > 1 else 0.0
        profiled_total_std = statistics.stdev(all_profiled_total) if len(all_profiled_total) > 1 else 0.0
        
        print(f"\nProfiled memory breakdown (average over {num_profiling_runs} runs):")
        print(f"  Parameters: {profiled_params:.2f} MB (std: {profiled_params_std:.2f} MB)")
        print(f"  Activations: {profiled_activations:.2f} MB (std: {profiled_activations_std:.2f} MB)")
        print(f"  Optimizer: {profiled_optimizer:.2f} MB (std: {profiled_optimizer_std:.2f} MB)")
        print(f"  Total: {profiled_total:.2f} MB (std: {profiled_total_std:.2f} MB)")
        
        # Calculate errors
        def calculate_error(estimated, profiled):
            if profiled == 0:
                return float('inf') if estimated > 0 else 0.0
            return abs(estimated - profiled) / profiled * 100
        
        param_error = calculate_error(estimated_total_params, profiled_params)
        activation_error = calculate_error(estimated_total_activations, profiled_activations)
        optimizer_error = calculate_error(estimated_total_optimizer, profiled_optimizer)
        total_error = calculate_error(estimated_total, profiled_total)
        
        # Create comparison table
        comparison_data = {
            'Component': ['Parameters', 'Activations', 'Optimizer States', 'Total Peak'],
            'Estimated (MB)': [
                f'{estimated_total_params:.2f}',
                f'{estimated_total_activations:.2f}',
                f'{estimated_total_optimizer:.2f}',
                f'{estimated_total:.2f}'
            ],
            'Profiled (MB)': [
                f'{profiled_params:.2f} ± {profiled_params_std:.2f}',
                f'{profiled_activations:.2f} ± {profiled_activations_std:.2f}',
                f'{profiled_optimizer:.2f} ± {profiled_optimizer_std:.2f}',
                f'{profiled_total:.2f} ± {profiled_total_std:.2f}'
            ],
            'Error (%)': [
                f'{param_error:.2f}',
                f'{activation_error:.2f}',
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
        
        
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'diagrams')
        
        # Generate LaTeX table with custom formatting
        latex_table = "\\begin{table}[htbp]  % 'htbp' allows flexible placement: here, top, bottom, or page\n"
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
        
        latex_path = os.path.join(output_dir, 'memory_breakdown_comparison.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_path}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Mean Absolute Percentage Error (MAPE): {(param_error + activation_error + optimizer_error + total_error) / 4:.2f}%")
        print(f"  Rank used: {estimated_rank}")
        print(f"  Available memory: {args.gpu_memory_size_for_each_group_in_GB[0]} GB ({args.gpu_memory_size_for_each_group_in_GB[0] * 1024:.2f} MB)")
        print(f"  Memory utilization: {profiled_total / (args.gpu_memory_size_for_each_group_in_GB[0] * 1024) * 100:.2f}%")
        
        return df

if __name__ == '__main__':
    unittest.main()


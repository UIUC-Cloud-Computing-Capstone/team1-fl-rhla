"""
Unit tests for RankEstimator class in estimator.py
"""
import argparse
import unittest
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
from transformers import AutoModelForImageClassification
import torch
from torch.profiler import profile, ProfilerActivity

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from estimator import RankEstimator
from peft import LoraConfig, get_peft_model
from torchinfo import summary  # Better alternative to torchsummary for HuggingFace models
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
        result = self.estimator._get_base_model_activations_and_safety_margin_memory_size_in_bytes(args)
        result_in_MB = result / 1024 / 1024
        print(result_in_MB)


    def test_get_rank_for_all_client_groups_ours(self):
        args = argparse.Namespace()
        args.rank_estimator_method = 'Ours'

        # resource heterogeneity
        # all clients belong to 3 heterogeneous groups. Each group has different resource limitation.
        #144, 216, 288
        args.gpu_memory_size_for_each_group_in_GB = [8, 8, 8]
        args.avg_upload_network_speed_for_each_group_in_Mbps = [1.5, 2.5, 3]
        args.avg_download_network_speed_for_each_group_in_Mbps = [10, 50, 50]
        args.desired_uploading_time_for_each_group_in_seconds = [60, 60, 60]
        args.desired_downloading_time_for_each_group_in_seconds = [60, 60, 60]
        args.heterogeneous_group = [1/3, 1/3, 1/3] # 1/3 of clients belong to each group.

        # model
        args.model = 'facebook/deit-small-patch16-224'

        # training hyperparameters
        args.precision = 'fp32'
        args.optimizer = 'adam'
        args.num_of_layers_to_allocate_LoRA = 12
        args.lora_target_modules = ["query", "value"]

        # input data sizes
        args.image_height = 224
        args.image_width = 224
        args.patch_size = 16 # each image is split into 16 × 16 pixel patches.
        args.batch_size = 32
        
        # estimation parameters
        args.percentage_of_layers_in_memory = 12 / 12 # not all layers are in memory at the same time during forward pass and backward pass.
        args.overhead_and_safety_margin_factor = 0.1 # assume 20% of activations and gradients

        model = AutoModelForImageClassification.from_pretrained(args.model)
        # torchinfo works with HuggingFace models - shows model summary
        
        rank_budgets_for_all_heterogeneous_groups = self.estimator.get_rank_for_all_client_groups(args, model)
        print(rank_budgets_for_all_heterogeneous_groups)

        

    def test_get_rank_for_all_client_groups_fedhello(self):
        args = argparse.Namespace()
        args.gpu_memory_size_for_each_group_in_GB = [8, 8, 8]
        args.avg_upload_network_speed_for_each_group_in_Mbps = [1, 7, 7]
        args.avg_download_network_speed_for_each_group_in_Mbps = [10, 50, 50]
        args.rank_estimator_method = 'FedHello'
        args.precision = 'fp32'
        args.batch_size = 32
        args.desired_uploading_time_for_each_group_in_seconds = [60, 60, 60]
        args.desired_downloading_time_for_each_group_in_seconds = [60, 60, 60]
        args.optimizer = 'adam'
        args.heterogeneous_group = [1/3, 1/3, 1/3]
        args.model = 'facebook/deit-small-patch16-224'
        args.num_of_layers_to_allocate_LoRA = 12
        args.lora_target_modules = ["query", "value"]
        args.image_height = 224
        args.image_width = 224
        args.patch_size = 16
        args.percentage_of_layers_in_memory = 12 / 12
        args.overhead_and_safety_margin_factor = 0.1
        
        model = AutoModelForImageClassification.from_pretrained('facebook/deit-small-patch16-224')
        
        result = self.estimator.get_rank_for_all_client_groups(args, model)
        print(result)
    
    def test_profile_total_memory(self):
        """Test total actual memory profiling"""
        import torch
        
        # Create model
        model = AutoModelForImageClassification.from_pretrained('facebook/deit-small-patch16-224')


        r=384
        config = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            #modules_to_save=["classifier"],
        )
        model = get_peft_model(model, config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create batch
        batch_size = 32
        batch = {
            'pixel_values': torch.randn(batch_size, 3, 224, 224).to(device),
            'labels': torch.randint(0, 1000, (batch_size,)).to(device)
        }
        
        def loss_fn(outputs, labels):
            return outputs.loss if hasattr(outputs, 'loss') else torch.nn.functional.cross_entropy(outputs, labels)
        
        # Profile total memory
        tracker = MemoryTracker()
        results = tracker.profile_total_memory(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batch=batch,
            precision='fp32'
        )
        
        # Verify structure
        self.assertIn('parameters', results)
        self.assertIn('optimizer_states', results)
        self.assertIn('activations', results)
        self.assertIn('total', results)
        self.assertIn('breakdown', results)
        
        # Verify peak memory is measured
        peak_memory = results['total']['peak_memory_MB']
        self.assertGreater(peak_memory, 0)
        
        # Print full profile
        tracker.print_total_memory_profile(results)

    def test_memory_breakdown_comparison_table2(self):
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
        args.image_height = 224
        args.image_width = 224
        args.patch_size = 16
        args.batch_size = 32
        args.percentage_of_layers_in_memory = 12 / 12
        args.overhead_and_safety_margin_factor = 0.1
        args.desired_uploading_time_for_each_group_in_seconds = [60]
        args.desired_downloading_time_for_each_group_in_seconds = [60]
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
        import gc
        import time
        
        for run in range(num_profiling_runs):
            print(f"  Run {run + 1}/{num_profiling_runs}...", end=' ', flush=True)
            
            # Clear memory before each run to avoid interference
            if is_cuda:
                torch.cuda.empty_cache()  # Clear GPU cache
                torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats
            gc.collect()  # Force Python garbage collection
            
            # Small sleep to ensure cleanup completes (only for first few runs or if needed)
            #if run < 5 or run % 100 == 0:
            time.sleep(3)  # 10ms sleep for cleanup
            
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
                # For CPU, sum up self CPU memory usage from profiler
                key_averages = prof.key_averages()
                peak_memory_bytes = sum(event.self_cpu_memory_usage() for event in key_averages)
                peak_memory_MB = peak_memory_bytes / (1024 * 1024)
            
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
            
            # Print profiler table for debugging (optional)
            # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
            
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
        
        # Save to CSV
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'diagrams')
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'memory_breakdown_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nTable saved to: {csv_path}")
        
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

class TestRankEstimatorVisualization(unittest.TestCase):
    """Test class for visualizing rank size vs memory with fixed network speeds"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.estimator = RankEstimator()

    def _init_args(self):
        args = argparse.Namespace()
        args.rank_estimator_method = 'Ours'
        args.model = 'facebook/deit-small-patch16-224'
        args.precision = 'fp32'
        args.optimizer = 'adam'
        args.num_of_layers_to_allocate_LoRA = 12
        args.lora_target_modules = ["query", "value"]
        args.image_height = 224
        args.image_width = 224
        args.patch_size = 16
        args.batch_size = 32
        args.percentage_of_layers_in_memory = 12 / 12
        args.overhead_and_safety_margin_factor = 0.1
        args.desired_uploading_time_for_each_group_in_seconds = [60]
        args.desired_downloading_time_for_each_group_in_seconds = [60]
        args.heterogeneous_group = [1.0]  # Single group for simplicity
        return args
    
    def test_rank_vs_memory_diagram(self):
        """Generate a diagram showing rank size vs memory with fixed network speeds"""
        # Fixed network speeds
        fixed_upload_speed_Mbps = 7
        fixed_download_speed_Mbps = 50.0
        
        # Vary memory sizes (realistic range: 4GB to 16GB)
        memory_sizes_GB = [1.5, 1.8, 1.9, 2, 4, 8]
        
        # Model and training configuration
        args = self._init_args()
        
        # Load model once
        model = AutoModelForImageClassification.from_pretrained(args.model)
        
        # Collect rank values for each memory size
        rank_values = []
        
        for memory_size_GB in memory_sizes_GB:
            # Set memory and network speeds for this iteration
            args.gpu_memory_size_for_each_group_in_GB = [memory_size_GB]
            args.avg_upload_network_speed_for_each_group_in_Mbps = [fixed_upload_speed_Mbps]
            args.avg_download_network_speed_for_each_group_in_Mbps = [fixed_download_speed_Mbps]
            
            # Get rank for this memory size
            rank_budgets = self.estimator.get_rank_for_all_client_groups(args, model)
            rank_values.append(rank_budgets[0])  # Get rank for the single group
        
        # Create the diagram with improved x-axis handling for uneven spacing
        self._create_rank_vs_memory_diagram(fixed_upload_speed_Mbps, fixed_download_speed_Mbps, memory_sizes_GB, rank_values)
        
        # Save the diagram
        self._save_diagram('rank_vs_memory_diagram.png')

    def _save_diagram(self, diagram_name):
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'diagrams')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, diagram_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nDiagram saved to: {output_path}")
        
        plt.close()

    def _create_rank_vs_memory_diagram(self, fixed_upload_speed_Mbps, fixed_download_speed_Mbps, memory_sizes_GB, rank_values):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create custom x-axis positions for better visualization
        # Map memory values to evenly spaced positions
        x_positions = list(range(len(memory_sizes_GB)))
        
        # Plot with custom x positions
        ax.plot(x_positions, rank_values, marker='o', linewidth=2, markersize=8, color='blue')
        ax.set_xlabel('GPU Memory Size (GB)', fontsize=16)
        ax.set_ylabel('Rank Size', fontsize=16)
        ax.set_title(f'Rank Size vs GPU Memory\n(Fixed Upload: {fixed_upload_speed_Mbps} Mbps, Download: {fixed_download_speed_Mbps} Mbps)', 
                     fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set custom x-axis labels with actual memory values
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'{mem:.2f}' if mem < 1 else f'{mem:.1f}' if mem < 2 else f'{int(mem)}' 
                            for mem in memory_sizes_GB], rotation=45, ha='right', fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        # Add value labels on points
        for i, (pos, rank) in enumerate(zip(x_positions, rank_values)):
            ax.annotate(f'{int(rank)}', (pos, rank), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=12)
        
        # Add a visual break indicator if there's a large gap
        if max(memory_sizes_GB) > 2 * max([m for m in memory_sizes_GB if m <= 2]):
            # Find where the gap occurs
            gap_start_idx = len([m for m in memory_sizes_GB if m <= 2])
            if gap_start_idx < len(memory_sizes_GB):
                # Add a subtle vertical line to indicate the gap
                ax.axvline(x=gap_start_idx - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                # Add text annotation
                ax.text(gap_start_idx - 0.5, ax.get_ylim()[1] * 0.95, '...', 
                        ha='center', fontsize=16, color='gray', alpha=0.7)
        
        plt.tight_layout()

    def test_rank_vs_network_speed_diagram(self):
        """Generate a diagram showing rank size vs network speeds with fixed memory"""
        # Fixed memory size
        fixed_memory_GB = 8.0
        
        # Vary network speeds (realistic range: 0.5 Mbps to 10 Mbps)
        network_speeds_Mbps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
        
        # Model and training configuration
        args = self._init_args()
        args.gpu_memory_size_for_each_group_in_GB = [fixed_memory_GB]
        
        # Load model once
        model = AutoModelForImageClassification.from_pretrained(args.model)
        
        # Collect rank values for each network speed
        rank_values_upload = []
        
        for network_speed_Mbps in network_speeds_Mbps:
            # Test with varying upload speed (fixed download)
            args.avg_upload_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
            args.avg_download_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]  # Fixed download
            rank_budgets = self.estimator.get_rank_for_all_client_groups(args, model)
            rank_values_upload.append(rank_budgets[0])
        
        # Create the diagram
        plt.figure(figsize=(12, 7))
        plt.plot(network_speeds_Mbps, rank_values_upload, marker='o', linewidth=2, markersize=8, 
                label='Varying Network Speed', color='blue')
        plt.xlabel('Network Speed (Mbps)', fontsize=16)
        plt.ylabel('Rank Size', fontsize=16)
        plt.title(f'Rank Size vs Network Speed\n(Fixed Memory: {fixed_memory_GB} GB)', 
                  fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=14)
        plt.xticks(network_speeds_Mbps)
        plt.tick_params(axis='both', labelsize=14)
        
        # Add value labels on points (only for upload line to avoid clutter)
        for i, (speed, rank) in enumerate(zip(network_speeds_Mbps, rank_values_upload)):
            if i % 2 == 0:  # Label every other point to avoid clutter
                plt.annotate(f'{int(rank)}', (speed, rank), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=12, color='blue')
        
        plt.tight_layout()
        
        # Save the diagram
        self._save_diagram('rank_vs_network_speed_diagram.png')

    def test_rank_vs_memory_and_network_speed_combined(self):
        """Generate a combined diagram with both lines in the same figure, sharing the Y-axis"""
        # Fixed network speeds for memory diagram
        fixed_upload_speed_Mbps = 7
        fixed_download_speed_Mbps = 50.0
        
        # Fixed memory size for network speed diagram
        fixed_memory_GB = 8.0
        
        # Vary memory sizes (realistic range: 4GB to 16GB)
        memory_sizes_GB = [1.5, 1.8, 1.9, 2, 2.1, 2.2, 2.5, 4, 8]
        
        # Vary network speeds (realistic range: 0.5 Mbps to 10 Mbps)
        network_speeds_Mbps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
        
        # Model and training configuration
        args = self._init_args()
        
        # Load model once
        model = AutoModelForImageClassification.from_pretrained(args.model)
        
        # Collect rank values for memory variation
        rank_values_memory = []
        for memory_size_GB in memory_sizes_GB:
            args.gpu_memory_size_for_each_group_in_GB = [memory_size_GB]
            args.avg_upload_network_speed_for_each_group_in_Mbps = [fixed_upload_speed_Mbps]
            args.avg_download_network_speed_for_each_group_in_Mbps = [fixed_download_speed_Mbps]
            rank_budgets = self.estimator.get_rank_for_all_client_groups(args, model)
            rank_values_memory.append(rank_budgets[0])
        
        # Collect rank values for network speed variation
        rank_values_network = []
        args.gpu_memory_size_for_each_group_in_GB = [fixed_memory_GB]
        for network_speed_Mbps in network_speeds_Mbps:
            args.avg_upload_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
            args.avg_download_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
            rank_budgets = self.estimator.get_rank_for_all_client_groups(args, model)
            rank_values_network.append(rank_budgets[0])
        
        # Create a single figure with dual X-axes - even larger size for better readability
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Use a piecewise transformation to spread out small values significantly
        # This makes small differences much more readable while preserving the uneven nature
        def transform_memory(mem_val):
            """Transform memory value to spread small values more for readability"""
            # Use a piecewise approach: more aggressive for small values, less for large
            if mem_val <= 2:
                # Very aggressive transformation for small values (1.5-2 range)
                return 0.5 * mem_val  # Linear scaling for small values gives more space
            else:
                # Less aggressive for larger values
                base = 1.0  # Value at mem_val=2
                return base + 0.3 * np.power(mem_val - 2, 0.5)
        
        # Transform memory values for x-positions
        x_positions_memory = [transform_memory(mem) for mem in memory_sizes_GB]
        
        # Plot memory line on bottom X-axis using transformed positions
        line1 = ax.plot(x_positions_memory, rank_values_memory, marker='o', linewidth=2, markersize=8, 
                       color='blue', label=f'Rank vs Memory (Fixed Network: {fixed_upload_speed_Mbps}/{fixed_download_speed_Mbps} Mbps)')
        
        # Set bottom X-axis for memory sizes - use less rotation for better readability
        ax.set_xlabel('GPU Memory Size (GB)', fontsize=26, color='blue', labelpad=15)
        ax.set_xticks(x_positions_memory)
        # Format labels: show 2 decimals for < 1, 1 decimal for < 3, integer for >= 3
        # Use center alignment so labels are positioned directly below their data points
        ax.set_xticklabels([f'{mem:.2f}' if mem < 1 else f'{mem:.1f}' if mem < 3 else f'{int(mem)}' 
                            for mem in memory_sizes_GB], rotation=30, ha='center', fontsize=24, color='blue')
        ax.tick_params(axis='x', labelsize=24, colors='blue', pad=12)
        
        # Add more padding to x-axis limits for better label readability
        x_min_transformed = min(x_positions_memory)
        x_max_transformed = max(x_positions_memory)
        x_range = x_max_transformed - x_min_transformed
        ax.set_xlim(x_min_transformed - 0.2 * x_range, x_max_transformed + 0.2 * x_range)
        
        # Create top X-axis for network speeds
        ax2 = ax.twiny()  # Create a second axes that shares the same y-axis
        
        # Map network speeds to the same physical X range as transformed memory values
        memory_x_min_transformed = min(x_positions_memory)
        memory_x_max_transformed = max(x_positions_memory)
        network_min = min(network_speeds_Mbps)
        network_max = max(network_speeds_Mbps)
        
        # Transform network speeds to transformed memory value space for plotting
        # Shift mapping left so 0.5 is closer to the left edge
        def network_to_x(network_val):
            """Transform network speed value to X position matching transformed memory range"""
            if network_max == network_min:
                return memory_x_min_transformed
            # Use a compressed mapping: map from network_min to network_max but shift left
            # by using a smaller effective range to compress the left side
            range_size = memory_x_max_transformed - memory_x_min_transformed
            # Compress the mapping by using only part of the range, shifting everything left
            # This brings 0.5 closer to the left edge
            compressed_range = range_size * 0.85  # Use 85% of the range
            left_offset = range_size * 0.15  # Shift left by 15% of the range
            
            normalized = (network_val - network_min) / (network_max - network_min)
            return memory_x_min_transformed - left_offset + normalized * compressed_range
        
        network_x_positions = [network_to_x(speed) for speed in network_speeds_Mbps]
        
        # Plot network speed line on top X-axis
        line2 = ax2.plot(network_x_positions, rank_values_network, marker='s', linewidth=2, markersize=8, 
                        color='green', label=f'Rank vs Network Speed (Fixed Memory: {fixed_memory_GB} GB)')
        
        # Set top X-axis for network speeds - show all labels
        ax2.set_xlabel('Network Speed (Mbps)', fontsize=26, color='green', labelpad=15)
        ax2.set_xlim(ax.get_xlim())  # Match the X limits of the bottom axis
        ax2.set_xticks(network_x_positions)
        ax2.set_xticklabels([f'{speed:.1f}' for speed in network_speeds_Mbps], 
                           rotation=30, ha='left', fontsize=24, color='green')
        ax2.tick_params(axis='x', labelsize=24, colors='green', pad=12)
        
        # Set Y-axis label (shared)
        ax.set_ylabel('Rank Size', fontsize=26, labelpad=15)
        ax.set_ylim(0, 600)
        ax.grid(True, alpha=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=22, framealpha=0.95)
        ax.tick_params(axis='y', labelsize=24)
        
        # Set title
        #ax.set_title('Rank Size vs GPU Memory and Network Speed', fontsize=22, fontweight='bold', pad=20)
        
        # Add value labels on points for memory plot
        for i, (pos, rank) in enumerate(zip(x_positions_memory, rank_values_memory)):
            ax.annotate(f'{int(rank)}', (pos, rank), textcoords="offset points", 
                       xytext=(0,13), ha='center', fontsize=24, color='blue')
        
        # Add value labels on points for network speed plot (every other point to avoid clutter)
        for i, (pos, rank) in enumerate(zip(network_x_positions, rank_values_network)):
            if i % 2 == 0:
                ax2.annotate(f'{int(rank)}', (pos, rank), textcoords="offset points", 
                           xytext=(0,30), ha='center', fontsize=24, color='green')
        
        # Use tight_layout with more padding to accommodate rotated labels
        plt.tight_layout(pad=4.0)
        # Additional margin adjustment for better label spacing - more room for rotated labels
        fig.subplots_adjust(bottom=0.20, top=0.82)
        
        # Save the diagram
        self._save_diagram('rank_vs_memory_and_network_speed_combined.png')

    def test_rank_3d_diagram(self):
        """Generate a 3D diagram showing rank size vs memory and network speed"""
        # Define ranges for memory and network speeds
        memory_sizes_GB = [1.9, 1.95, 2, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4]
        network_speeds_Mbps = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
        
        # Model and training configuration
        args = self._init_args()
        
        # Load model once
        model = AutoModelForImageClassification.from_pretrained(args.model)
        
        # Create meshgrid for surface plot
        X, Y = np.meshgrid(memory_sizes_GB, network_speeds_Mbps)
        Z = np.zeros_like(X)
        
        # Calculate rank for each combination of memory and network speed
        print("Calculating rank values for 3D plot...")
        for i, memory_size_GB in enumerate(memory_sizes_GB):
            for j, network_speed_Mbps in enumerate(network_speeds_Mbps):
                args.gpu_memory_size_for_each_group_in_GB = [memory_size_GB]
                args.avg_upload_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
                args.avg_download_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
                
                rank_budgets = self.estimator.get_rank_for_all_client_groups(args, model)
                Z[j, i] = rank_budgets[0]  # Note: j is row (network), i is col (memory)
                print(f"Memory: {memory_size_GB} GB, Network: {network_speed_Mbps} Mbps, Rank: {rank_budgets[0]}")
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none', linewidth=0.1)
        
        # Add contour lines on the surface
        ax.contour(X, Y, Z, zdir='z', offset=ax.get_zlim()[0], cmap='viridis', alpha=0.3)
        
        # Set labels
        ax.set_xlabel('GPU Memory Size (GB)', fontsize=12, labelpad=10)
        ax.set_ylabel('Network Speed (Mbps)', fontsize=12, labelpad=10)
        ax.set_zlabel('Rank Size', fontsize=12, labelpad=10)
        #ax.set_title('Rank Size vs GPU Memory and Network Speed', fontsize=14, fontweight='bold', pad=20)
        
        # Reverse x-axis
        ax.invert_xaxis()
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Rank Size')
        
        # Set viewing angle for better visualization - adjust to make slope face reader
        ax.view_init(elev=25, azim=280)
        
        plt.tight_layout()
        
        # Save the diagram
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'diagrams')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'rank_3d_diagram.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n3D Diagram saved to: {output_path}")
        
        plt.close()


if __name__ == '__main__':
    unittest.main()


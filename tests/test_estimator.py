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
import torch
import pandas as pd
import gc
import time
import statistics
#import timm

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
        args.avg_download_network_speed_for_each_group_in_Mbps = [10, 10, 10]
        args.desired_uploading_time_for_each_group_in_seconds = [15, 15, 15]
        args.desired_downloading_time_for_each_group_in_seconds = [15, 15, 15]
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

    def estimate(self, args, base_model, estimator, memory_summary_dict):
        print("Getting estimated rank and memory breakdown...")
        estimated_rank = estimator._get_rank_for_one_client_group(
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

        return estimated_rank


    def init_model(self, args, r):
        print(f"\nCreating model with rank {r} and profiling actual memory...")
        config = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=args.lora_target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(AutoModelForImageClassification.from_pretrained(args.model), config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        batch = {
            'pixel_values': torch.randn(args.batch_size, 3, args.image_height, args.image_width).to(device),
            'labels': torch.randint(0, 1000, (args.batch_size,)).to(device)
        }
        return model, optimizer, batch, device
    
    def profile(self, args, base_model, output_file_path, estimated_rank, memory_summary_dict):
        
        tracker = MemoryTracker()


        print("Getting memory breakdown...")
        model, optimizer, batch, device = self.init_model(args, estimated_rank)
        # Check if using GPU
        is_cuda = device.type == 'cuda'
        if not is_cuda:
            raise ValueError('CPU memory profiling is not supported yet.')
        
        def loss_fn(outputs, labels):
            return outputs.loss if hasattr(outputs, 'loss') else torch.nn.functional.cross_entropy(outputs, labels)
        
        # Profile actual memory (run 10 times and take average for accuracy)
        num_warmup_runs = 1
        num_profiling_runs = 1
        print(f"\nProfiling actual memory {num_profiling_runs} times to get average...")
        
        all_profiled_params, all_profiled_optimizer, all_profiled_activations, all_profiled_total = [], [], [], []
        for run in range(num_warmup_runs + num_profiling_runs):
            if run < num_warmup_runs:
                print('warm up')
            else:
                print(f"  Run {run}/{num_profiling_runs - 1}...", end=' ', flush=True)
            
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
            # profiled_results = {
            #     'param_memory_MB': param_memory_MB,
            #     'optimizer_memory_MB': optimizer_memory_MB,
            #     'fwd_memory_MB': activation_memory_MB,
            #     'peak_memory_MB': peak_memory_MB
            # }
            
            # Collect values from this run
            # skip first run, to warm up
            if run < num_warmup_runs:
                print(f"Warm-up {run + 1} Done (Total: {peak_memory_MB:.2f} MB)")
            else:
                all_profiled_params.append(param_memory_MB)
                all_profiled_optimizer.append(optimizer_memory_MB)
                all_profiled_activations.append(activation_memory_MB)
                all_profiled_total.append(peak_memory_MB)
                print(f"Done (Total: {peak_memory_MB:.2f} MB)")
                
            
            # Clear memory after each run
            if is_cuda:
                torch.cuda.empty_cache()
            gc.collect()
        

        # statistics
        profiled_info = {}
        profiled_info['avg_profiled_params'] = sum(all_profiled_params) / len(all_profiled_params)
        profiled_info['avg_profiled_optimizer'] = sum(all_profiled_optimizer) / len(all_profiled_optimizer)
        profiled_info['avg_profiled_activations'] = sum(all_profiled_activations) / len(all_profiled_activations)
        profiled_info['avg_profiled_total'] = sum(all_profiled_total) / len(all_profiled_total)
        profiled_info['profiled_params_std'] = statistics.stdev(all_profiled_params) if len(all_profiled_params) > 1 else 0.0
        profiled_info['profiled_optimizer_std'] = statistics.stdev(all_profiled_optimizer) if len(all_profiled_optimizer) > 1 else 0.0
        profiled_info['profiled_activations_std'] = statistics.stdev(all_profiled_activations) if len(all_profiled_activations) > 1 else 0.0
        profiled_info['profiled_total_std'] = statistics.stdev(all_profiled_total) if len(all_profiled_total) > 1 else 0.0
        
        # comparison
        self.create_comparison(args, memory_summary_dict, profiled_info, output_file_path, estimated_rank)
        

    def create_comparison(self, args, memory_summary_dict, profiled_info, output_file_path, estimated_rank):
        estimated_total_params = memory_summary_dict.get('total_parameters_in_MB', 0)
        estimated_total_activations = memory_summary_dict.get('total_activations_gradients_and_with_safety_margin_in_MB', 0)
        estimated_total_optimizer = memory_summary_dict.get('total_optimizer_states_in_MB', 0)
        estimated_total = memory_summary_dict.get('total_memory_in_MB', 0)

        profiled_params = profiled_info['avg_profiled_params']
        profiled_optimizer = profiled_info['avg_profiled_optimizer']
        profiled_activations = profiled_info['avg_profiled_activations']
        profiled_total = profiled_info['avg_profiled_total']     
        profiled_params_std = profiled_info['profiled_params_std']
        profiled_optimizer_std = profiled_info['profiled_optimizer_std']
        profiled_activations_std = profiled_info['profiled_activations_std']
        profiled_total_std = profiled_info['profiled_total_std']

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
        
        
        self.create_latex(args, comparison_data, output_file_path)
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Mean Absolute Percentage Error (MAPE): {(param_error + activation_error + optimizer_error + total_error) / 4:.2f}%")
        print(f"  Rank used: {estimated_rank}")
        print(f"  Available memory: {args.gpu_memory_size_for_each_group_in_GB[0]} GB ({args.gpu_memory_size_for_each_group_in_GB[0] * 1024:.2f} MB)")
        print(f"  Memory utilization: {profiled_total / (args.gpu_memory_size_for_each_group_in_GB[0] * 1024) * 100:.2f}%")
        
        return df

    def create_latex(self, args, comparison_data, output_file_path):
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


    def _init_args(self):
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
        args.batch_size = 32 * 17
        args.percentage_of_layers_in_memory = 12 / 12
        args.overhead_and_safety_margin_factor = 0.1
        args.desired_uploading_time_for_each_group_in_seconds = [15]
        args.desired_downloading_time_for_each_group_in_seconds = [15]
        args.heterogeneous_group = [1.0]
        args.gpu_memory_size_for_each_group_in_GB = [8.0]
        args.avg_upload_network_speed_for_each_group_in_Mbps = [7.0]
        args.avg_download_network_speed_for_each_group_in_Mbps = [50.0]
        return args

    def test_get_all_named_modules(self):
        args = self._init_args()
        model = AutoModelForImageClassification.from_pretrained(args.model)
        print("All module names in the model:")
        for name, module in model.named_modules():
            print(name)

    def test_lora_shapes(self):
        args = self._init_args()
        model = AutoModelForImageClassification.from_pretrained(args.model)
        r = self._get_rank()
        config = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=['query', 'key', 'value', 'attention.output.dense', 'intermediate.dense', 'output.dense'],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, config)
        print("All module names in the model:")
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                print(f"{name:<50} | {list(module.weight.shape)}")

    def test_memory_breakdown_comparison_table_lora_qv(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["query", "value"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_qv.tex', self.estimator)

    def test_memory_breakdown_comparison_table_lora_q(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        memory_summary_dict = {}
        estimated_rank = 178
        self.profile(args, base_model, 'memory_breakdown_comparison_lora_q.tex', estimated_rank, memory_summary_dict)

    def test_memory_breakdown_comparison_table_lora_q_1(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        memory_summary_dict = {}
        #estimated_rank = self.estimate(args, base_model, estimator, memory_summary_dict)
        estimated_rank = self._get_rank()
        self.profile(args, base_model, 'memory_breakdown_comparison_lora_q_1.tex', estimated_rank, memory_summary_dict)
    
    def test_memory_breakdown_comparison_table_lora_q_1_rank2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)
        
        self.profile(args, base_model, 'memory_breakdown_comparison_lora_q_1_rank2.tex', self._get_rank2(), {})

    def test_memory_breakdown_comparison_table_lora_q_2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.query", "1.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_q_2.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_q_2_rank2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.query", "1.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_q_2_rank2.tex', self._get_rank2(), {})

    def test_memory_breakdown_comparison_table_lora_q_1_head_12(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.model = 'facebook/deit-base-patch16-224'
        args.lora_target_modules = ["0.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        memory_summary_dict = {}
        #estimated_rank = self.estimate(args, base_model, estimator, memory_summary_dict)
        estimated_rank = self._get_rank()
        self.profile(args, base_model, 'memory_breakdown_comparison_lora_q_1_head_12.tex', estimated_rank, memory_summary_dict)

    def test_memory_breakdown_comparison_table_lora_q_2_head_12(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.model = 'facebook/deit-base-patch16-224'
        args.lora_target_modules = ["0.attention.attention.query", "1.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        memory_summary_dict = {}
        #estimated_rank = self.estimate(args, base_model, estimator, memory_summary_dict)
        estimated_rank = self._get_rank()
        self.profile(args, base_model, 'memory_breakdown_comparison_lora_q_2_head_12.tex', estimated_rank, memory_summary_dict)


    


    def test_memory_breakdown_comparison_table_lora_q_6(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = [
            "0.attention.attention.query", "1.attention.attention.query", "4.attention.attention.query",
            "8.attention.attention.query", "10.attention.attention.query", "11.attention.attention.query",
        ]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        memory_summary_dict = {}
        estimated_rank = self._get_rank()
        self.profile(args, base_model, 'memory_breakdown_comparison_lora_q_6.tex', estimated_rank, memory_summary_dict)

    def test_memory_breakdown_comparison_table_lora_q_7(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = [
            "0.attention.attention.query", "1.attention.attention.query", "4.attention.attention.query", "5.attention.attention.query",
            "8.attention.attention.query", "10.attention.attention.query", "11.attention.attention.query",
        ]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        memory_summary_dict = {}
        estimated_rank = self._get_rank()
        self.profile(args, base_model, 'memory_breakdown_comparison_lora_q_7.tex', estimated_rank, memory_summary_dict)

    def test_memory_breakdown_comparison_table_lora_v_1(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.target_modules = []
        args.lora_target_modules = ["0.attention.attention.value"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        memory_summary_dict = {}
        #estimated_rank = self.estimate(args, base_model, estimator, memory_summary_dict)
        estimated_rank = self._get_rank()
        self.profile(args, base_model, 'memory_breakdown_comparison_lora_v_1.tex', estimated_rank, memory_summary_dict)
    
    def test_memory_breakdown_comparison_table_lora_v_2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.value", "1.attention.attention.value"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        memory_summary_dict = {}
        estimated_rank = self._get_rank()
        self.profile(args, base_model, 'memory_breakdown_comparison_lora_v_2.tex', estimated_rank, memory_summary_dict)
    
    def test_memory_breakdown_comparison_table_lora_q_0_and_11(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.query", "11.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_q_0_and_11.tex', self._get_rank(), {})
    
    def test_memory_breakdown_comparison_table_lora_attn_output_dense_1(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_attn_output_dense_1.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_attn_output_dense_2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.output.dense", "1.attention.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_attn_output_dense_2.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_int_dense_1(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.intermediate.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense_1.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_int_dense_2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.intermediate.dense", "1.intermediate.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense_2.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = r".*layer\.\d+\.output\.dense$"
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense_1(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense_2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.output.dense", "1.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense_1_rank2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense_1_rank2.tex', self._get_rank2(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense_2_rank2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.output.dense", "1.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        self.profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense_2_rank2.tex', self._get_rank2(), {})


    def test_formula1(self):
        

        # activation for q/k/v: BSH + 1.25 * BSr
        # mlp output dense: 5BSH + BSr
        r = self._get_rank()
        args = self._init_args()
        B = args.batch_size
        precision_bytes = 4
        S = 197
        H = 384
        C_head = 6
        M = 4

        val1 = self._bytes_to_mb(B * S * H * precision_bytes)
        m = 1.5
        val2 = self._bytes_to_mb(B * S * r * precision_bytes * m)
        print('batch size: ', B)
        print(val1)
        print(val2)
        print(val1 + val2)

        val3 = self._bytes_to_mb(B * S * H * precision_bytes * M)

    def test_formula2(self):
        r = self._get_rank()
        args = self._init_args()
        B = args.batch_size
        precision_bytes = 4
        S = 197
        H = 384
        C_head = 6
        M = 4
        L = 12

        val1 = self._bytes_to_mb(B * S * H * precision_bytes)     
        val2 = self._bytes_to_mb(B * C_head * S * S * precision_bytes)
        val3 = self._bytes_to_mb(B * S * H * M * precision_bytes)
        m = 1.5
        val4 = self._bytes_to_mb(B * S * r * precision_bytes)


        lora_layer = 2
        #print(val1 + val4)
        print('BSH', val1)
        print('BSSCh', val2)
        #print(val3)
        print('BSr', val4)
        base_total = self._get_base_total(val1, val2)
        print(base_total)
        print('total: ', (base_total + lora_layer * (val1 + val4 * m)) * 1)
        #print((val1 * 6 + val2 + val3 + val4) * L)

    def test_formula3(self):
        

        
        r = self._get_rank()
        args = self._init_args()
        B = args.batch_size
        precision_bytes = 4
        S = 197
        H = 384
        C_head = 12
        M = 4
        L = 12

        val1 = self._bytes_to_mb(B * S * H * precision_bytes)     
        val2 = self._bytes_to_mb(B * C_head * S * S * precision_bytes)
        val3 = self._bytes_to_mb(B * S * H * M * precision_bytes)
        m = 1.5
        val4 = self._bytes_to_mb(B * S * r * precision_bytes * m)


        lora_layer = 2
        print(val1 + val4)
        print(val1)
        print(val2)
        print(val3)
        base_total = self._get_base_total(val1, val2)
        print(base_total)
        print('total: ', (base_total + lora_layer * (val1 + val4)) * 1)
        #print((val1 * 6 + val2 + val3 + val4) * L)


    def test_formula4(self):
        r = self._get_rank()
        args = self._init_args()
        B = args.batch_size
        precision_bytes = 4
        S = 197
        H = 384
        C_head = 12
        M = 4
        L = 12

        val1 = self._bytes_to_mb(B * S * H * precision_bytes)
        m = 1.5
        val2 = self._bytes_to_mb(B * S * r * precision_bytes * m)
        print('batch size: ', B)
        print(val1)
        print(val2)
        print(val1 + val2)

        val3 = self._bytes_to_mb(B * S * H * precision_bytes * M)

    def test_formula5(self):
        r = self._get_rank2()
        args = self._init_args()
        B = args.batch_size
        precision_bytes = 4
        S = 197
        H = 384
        C_head = 6
        M = 4
        L = 12

        val1 = self._bytes_to_mb(B * S * H * precision_bytes)     
        val2 = self._bytes_to_mb(B * C_head * S * S * precision_bytes)
        val3 = self._bytes_to_mb(B * S * H * M * precision_bytes)
        m = 1.5
        val4 = self._bytes_to_mb(B * S * r * precision_bytes)


        lora_layer = 2
        #print(val1 + val4)
        print('BSH', val1)
        print('BSSCh', val2)
        #print(val3)
        print('BSr', val4)
        base_total = self._get_base_total(val1, val2)
        print(base_total)
        print('total: ', (base_total + lora_layer * (val1 + val4 * m)) * 1)
        #print((val1 * 6 + val2 + val3 + val4) * L)

    def _get_base_total(self, bsh, bchss):
        return 8 * bsh + 49 * bchss

    def _get_rank(self):
        return 178

    def _get_rank2(self):
        return 64    

    def _bytes_to_mb(self, bytes_value):
        return round(bytes_value / 1024 / 1024, 2)
if __name__ == '__main__':
    unittest.main()


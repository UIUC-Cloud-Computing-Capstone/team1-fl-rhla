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

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_qv.tex', self.estimator)

    def test_memory_breakdown_comparison_table_lora_q(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        memory_summary_dict = {}
        estimated_rank = 178
        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_q.tex', estimated_rank, memory_summary_dict)

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
        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_q_1.tex', estimated_rank, memory_summary_dict)
    
    def test_memory_breakdown_comparison_table_lora_q_1_rank2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)
        
        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_q_1_rank2.tex', self._get_rank2(), {})

    def test_memory_breakdown_comparison_table_lora_q_2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.query", "1.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_q_2.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_q_2_rank2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.query", "1.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_q_2_rank2.tex', self._get_rank2(), {})

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
        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_q_1_head_12.tex', estimated_rank, memory_summary_dict)

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
        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_q_2_head_12.tex', estimated_rank, memory_summary_dict)


    


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
        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_q_6.tex', estimated_rank, memory_summary_dict)

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
        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_q_7.tex', estimated_rank, memory_summary_dict)

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
        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_v_1.tex', estimated_rank, memory_summary_dict)
    
    def test_memory_breakdown_comparison_table_lora_v_2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.value", "1.attention.attention.value"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        memory_summary_dict = {}
        estimated_rank = self._get_rank()
        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_v_2.tex', estimated_rank, memory_summary_dict)
    
    def test_memory_breakdown_comparison_table_lora_q_0_and_11(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.attention.query", "11.attention.attention.query"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_q_0_and_11.tex', self._get_rank(), {})
    
    def test_memory_breakdown_comparison_table_lora_attn_output_dense_1(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_attn_output_dense_1.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_attn_output_dense_2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.attention.output.dense", "1.attention.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_attn_output_dense_2.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_int_dense_1(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.intermediate.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense_1.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_int_dense_2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.intermediate.dense", "1.intermediate.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense_2.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = r".*layer\.\d+\.output\.dense$"
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense_1(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense_2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.output.dense", "1.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense.tex', self._get_rank(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense_1_rank2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense_1_rank2.tex', self._get_rank2(), {})

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense_2_rank2(self):
        """Generate a comparison table using PyTorch profiler dire  ctly (like ResNet example)"""

        
        # Configuration
        args = self._init_args()
        args.lora_target_modules = ["0.output.dense", "1.output.dense"]
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)

        MemoryTracker().profile(args, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense_2_rank2.tex', self._get_rank2(), {})


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


"""
Unit tests for RankEstimator class in estimator.py
"""
import argparse
import unittest
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForImageClassification, AutoConfig
from torch.profiler import profile, ProfilerActivity
import torch
import pandas as pd
import gc
import time
import statistics
import copy
#import timm

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from estimator import RankEstimator
from peft import LoraConfig, get_peft_model
from utils.memory_tracker import MemoryTracker

MEM_ONLY = 'mem_only'


class TestRankEstimator(unittest.TestCase):
    """Test cases for RankEstimator class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.estimator = RankEstimator()
        self.tracker = MemoryTracker()

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



    def test_get_rank_for_all_client_groups_ours(self):
        args = argparse.Namespace()
        args.rank_estimator_method = 'Ours'

        # resource heterogeneity
        # all clients belong to 3 heterogeneous groups. Each group has different resource limitation.
        args.gpu_memory_size_for_each_group_in_GB = [8, 8, 8]
        args.avg_upload_network_speed_for_each_group_in_Mbps = [1, 1.5, 2]
        args.avg_download_network_speed_for_each_group_in_Mbps = [10, 10, 10]
        args.desired_uploading_time_for_each_group_in_seconds = [15, 15, 15]
        args.desired_downloading_time_for_each_group_in_seconds = [15, 15, 15]
        args.heterogeneous_group = [1/3, 1/3, 1/3] # 1/3 of clients belong to each group.

        # model
        args.model = 'facebook/deit-small-patch16-224'
        args.CLS_TOKEN = 1

        # training hyperparameters
        args.precision = 'fp32'
        args.optimizer = 'adam'
        args.num_of_layers_to_allocate_LoRA = 12
        args.lora_target_modules = ["query", "value"]
        args.train_classifier = False # do not train classifier in the base model. Only train LoRA matrices.

        # input data sizes
        args.batch_size = 32
        
        # estimation parameters
        args.overhead_and_safety_margin_factor = 0.1 # assume 10% of activations and gradients

        base_model = AutoModelForImageClassification.from_pretrained(args.model)
        
        config = AutoConfig.from_pretrained(args.model)
        rank_budgets_for_all_heterogeneous_groups = self.estimator.get_rank_for_all_client_groups(args, config, base_model, {})
        print(rank_budgets_for_all_heterogeneous_groups)

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
        args.batch_size = 32
        args.overhead_and_safety_margin_factor = 0.1
        args.desired_uploading_time_for_each_group_in_seconds = [15]
        args.desired_downloading_time_for_each_group_in_seconds = [15]
        args.heterogeneous_group = [1.0]
        args.gpu_memory_size_for_each_group_in_GB = [8.0]
        args.avg_upload_network_speed_for_each_group_in_Mbps = [7.0]
        args.avg_download_network_speed_for_each_group_in_Mbps = [50.0]
        args.CLS_TOKEN = 1
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

    def test_memory_breakdown_comparison_table_qv(self):
        args = self._init_args()
        args.gpu_memory_size_for_each_group_in_GB = [2.35]
        args.lora_target_modules = ['query', 'value']
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)
        config = AutoConfig.from_pretrained(args.model)
        memory_summary_dict = {}
        args.rank_estimator_method = MEM_ONLY
        # 
        base_model1 = copy.copy(base_model)
        rank = self.estimator.get_rank_for_one_client_group(args, config, base_model1, memory_summary_dict)[0]
        #rank = 384
        del base_model1
        print('est rank', rank)
        self.tracker.profile_and_compare(args, config, copy.copy(base_model), 'memory_breakdown_comparison_lora_qv.tex', rank, memory_summary_dict)

    def test_memory_breakdown_comparison_table_lora_mlp_output_dense(self):
        args = self._init_args()
        args.lora_target_modules = r".*layer\.\d+\.output\.dense$"
        
        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(args.model)
        config = AutoConfig.from_pretrained(args.model)
        self.tracker.profile_and_compare(args, config, base_model, 'memory_breakdown_comparison_lora_mlp_int_dense.tex', self._get_rank(), {})

    def test_get_base_model_fwd_in_bytes_for_estimator(self):
        args = self._init_args()
        base_model = AutoModelForImageClassification.from_pretrained(args.model)
        config = AutoConfig.from_pretrained(args.model)
        result = self.tracker.get_base_model_fwd_in_bytes_for_estimator(args, config, base_model)
        print(result)

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


"""
Unit tests for RankEstimator class in estimator.py
"""
import argparse
import unittest
import sys
import os
from transformers import AutoModelForImageClassification

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from estimator import RankEstimator


class TestRankEstimator(unittest.TestCase):
    """Test cases for RankEstimator class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.estimator = RankEstimator()

    def test_get_sequence_length_returns_197(self):
        """Test that _get_sequence_length returns 197 for facebook/deit-small-patch16-224"""
        result = self.estimator._get_sequence_length()
        
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
        result = self.estimator._get_sequence_length()
        
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
        result = self.estimator._get_base_model_activations_and_safety_margin_memory_size_in_bytes(args)
        result_in_MB = result / 1024 / 1024
        print(result_in_MB)


    def test_get_rank_for_all_client_groups_ours(self):
        args = argparse.Namespace()
        args.num_users = 3
        args.gpu_memory_size_for_each_group_in_GB = [24, 24, 8]
        args.avg_upload_network_speed_for_each_group_in_Mbps = [1, 7, 7]
        args.avg_download_network_speed_for_each_group_in_Mbps = [10, 50, 50]
        args.rank_estimator_method = 'Ours'
        args.precision = 'fp32'
        args.batch_size = 32
        args.desired_uploading_time_for_each_group_in_seconds = [60, 60, 60]
        args.desired_downloading_time_for_each_group_in_seconds = [60, 60, 60]
        args.optimizer = 'adamw'
        model = AutoModelForImageClassification.from_pretrained('facebook/deit-small-patch16-224')
        result = self.estimator.get_rank_for_all_client_groups(args, model)
        print(result)

    def test_get_rank_for_all_client_groups_fedhello(self):
        args = argparse.Namespace()
        args.num_users = 3
        args.gpu_memory_size_for_each_group_in_GB = [24, 24, 8]
        args.avg_upload_network_speed_for_each_group_in_Mbps = [1, 7, 7]
        args.avg_download_network_speed_for_each_group_in_Mbps = [10, 50, 50]
        args.rank_estimator_method = 'FedHello'
        args.precision = 'fp32'
        args.batch_size = 32
        args.desired_uploading_time_for_each_group_in_seconds = [60, 60, 60]
        args.desired_downloading_time_for_each_group_in_seconds = [60, 60, 60]
        args.optimizer = 'adamw'
        model = AutoModelForImageClassification.from_pretrained('facebook/deit-small-patch16-224')
        result = self.estimator.get_rank_for_all_client_groups(args, model)
        print(result)


if __name__ == '__main__':
    unittest.main()


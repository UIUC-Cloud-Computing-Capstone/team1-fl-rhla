import unittest
import torch
import numpy as np
from unittest.mock import Mock
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.solver.global_aggregator import (
    weighted_average_lora_depthfl, 
    weighted_average_lora_depthfl_heterogenous_ranks_aggregation,
    get_avg_update
)


class TestWeightedAverageLoraDepthFL(unittest.TestCase):
    """Test cases for weighted_average_lora_depthfl with heterogeneous ranks"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock args object
        self.args = Mock()
        self.args.only_train_b = False
        self.args.block_ids_list = [[0], [1], [0, 1]]
        self.args.rank_list = [[4, 6, 8], [2, 4, 6], [8, 10, 12]]
        self.args.lora_layer = 2
        self.args.lora_max_rank = 12
        self.args.aggregation = 'weighted_average_heterogenous_ranks'
        
        # Create a global model with LoRA parameters
        # Simulating a model with 2 layers, each with lora_A and lora_B
        # Full rank is 12 (max rank)
        self.global_model = {
            'base_model.layer.0.attention.lora_A': torch.zeros(12, 384),  # rank x hidden_dim
            'base_model.layer.0.attention.lora_B': torch.zeros(384, 12),  # hidden_dim x rank
            'base_model.layer.1.attention.lora_A': torch.zeros(12, 384),
            'base_model.layer.1.attention.lora_B': torch.zeros(384, 12),
            'classifier.weight': torch.zeros(100, 384),  # num_classes x hidden_dim
            'base_model.embedding.weight': torch.zeros(1000, 384),  # Should be ignored
        }


class TestGetAvgUpdate(unittest.TestCase):
    """Test cases for get_avg_update function"""
    
    def test_basic_weighted_average(self):
        """Test basic weighted average with same-sized tensors"""
        updates = [
            torch.ones(3, 4) * 1.0,
            torch.ones(3, 4) * 2.0,
            torch.ones(3, 4) * 3.0,
        ]
        weights = [
            torch.ones(3, 4) * 10,  # weight for update 1
            torch.ones(3, 4) * 20,  # weight for update 2
            torch.ones(3, 4) * 30,  # weight for update 3
        ]
        
        result = get_avg_update(updates, weights)
        
        # Expected: (1*10 + 2*20 + 3*30) / (10+20+30) = 140/60 ≈ 2.333
        expected = (1.0 * 10 + 2.0 * 20 + 3.0 * 30) / 60
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
        self.assertEqual(result.shape, (3, 4))
    
    def test_heterogeneous_ranks_with_masked_weights(self):
        """Test weighted average with masked weights (simulating different ranks)"""
        # Client 0: rank 2 (first 2 rows are non-zero)
        # Client 1: rank 3 (first 3 rows are non-zero)
        updates = [
            torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]),  # rank 2
            torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]),  # rank 3
        ]
        weights = [
            torch.tensor([[10.0, 10.0], [10.0, 10.0], [0.0, 0.0]]),  # masked: rank 2
            torch.tensor([[20.0, 20.0], [20.0, 20.0], [20.0, 20.0]]),  # masked: rank 3
        ]
        
        result = get_avg_update(updates, weights)
        
        # For rows 0-1: both clients contribute
        # Expected row 0: (1*10 + 3*20) / (10+20) = 70/30 ≈ 2.333
        # Expected row 1: (2*10 + 4*20) / (10+20) = 100/30 ≈ 3.333
        # For row 2: only client 1 contributes
        # Expected row 2: (5*20) / 20 = 5.0
        
        expected_row0 = (1.0 * 10 + 3.0 * 20) / 30
        expected_row1 = (2.0 * 10 + 4.0 * 20) / 30
        expected_row2 = 5.0
        
        np.testing.assert_allclose(result[0, :].numpy(), expected_row0, rtol=1e-5)
        np.testing.assert_allclose(result[1, :].numpy(), expected_row1, rtol=1e-5)
        np.testing.assert_allclose(result[2, :].numpy(), expected_row2, rtol=1e-5)
    
    def test_single_update(self):
        """Test with a single update"""
        updates = [torch.ones(2, 3) * 5.0]
        weights = [torch.ones(2, 3) * 10.0]
        
        result = get_avg_update(updates, weights)
        
        # Should return the update itself (normalized by its own weight)
        expected = torch.ones(2, 3) * 5.0
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)
    
    def test_lora_B_heterogeneous_ranks(self):
        """Test with lora_B style tensors (columns masked)"""
        # Client 0: rank 2 (first 2 columns)
        # Client 1: rank 3 (first 3 columns)
        updates = [
            torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),  # rank 2
            torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),  # rank 3
        ]
        weights = [
            torch.tensor([[10.0, 10.0, 0.0], [10.0, 10.0, 0.0]]),  # masked: rank 2
            torch.tensor([[20.0, 20.0, 20.0], [20.0, 20.0, 20.0]]),  # masked: rank 3
        ]
        
        result = get_avg_update(updates, weights)
        
        # For column 0: both contribute
        # Expected col 0: (1*10 + 5*20) / 30 = 110/30 ≈ 3.667 for row 0
        # Expected col 0: (3*10 + 8*20) / 30 = 190/30 ≈ 6.333 for row 1
        # For column 1: both contribute
        # Expected col 1: (2*10 + 6*20) / 30 = 140/30 ≈ 4.667 for row 0
        # Expected col 1: (4*10 + 9*20) / 30 = 220/30 ≈ 7.333 for row 1
        # For column 2: only client 1 contributes
        # Expected col 2: (7*20) / 20 = 7.0 for row 0
        # Expected col 2: (10*20) / 20 = 10.0 for row 1
        
        expected_col0_row0 = (1.0 * 10 + 5.0 * 20) / 30
        expected_col0_row1 = (3.0 * 10 + 8.0 * 20) / 30
        expected_col1_row0 = (2.0 * 10 + 6.0 * 20) / 30
        expected_col1_row1 = (4.0 * 10 + 9.0 * 20) / 30
        expected_col2_row0 = 7.0
        expected_col2_row1 = 10.0
        
        np.testing.assert_allclose(result[0, 0].numpy(), expected_col0_row0, rtol=1e-5)
        np.testing.assert_allclose(result[1, 0].numpy(), expected_col0_row1, rtol=1e-5)
        np.testing.assert_allclose(result[0, 1].numpy(), expected_col1_row0, rtol=1e-5)
        np.testing.assert_allclose(result[1, 1].numpy(), expected_col1_row1, rtol=1e-5)
        np.testing.assert_allclose(result[0, 2].numpy(), expected_col2_row0, rtol=1e-5)
        np.testing.assert_allclose(result[1, 2].numpy(), expected_col2_row1, rtol=1e-5)
    
    def test_classifier_weighted_average(self):
        """Test with classifier weights (no masking)"""
        updates = [
            torch.ones(100, 384) * 1.0,
            torch.ones(100, 384) * 2.0,
        ]
        weights = [
            torch.ones(100, 384) * 20.0,
            torch.ones(100, 384) * 30.0,
        ]
        
        result = get_avg_update(updates, weights)
        
        # Expected: (1*20 + 2*30) / 50 = 80/50 = 1.6
        expected = (1.0 * 20 + 2.0 * 30) / 50
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
        self.assertEqual(result.shape, (100, 384))
    
    def test_unequal_weights(self):
        """Test with different weights for different elements"""
        updates = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]
        weights = [
            torch.tensor([[10.0, 20.0], [30.0, 40.0]]),  # Different weights per element
            torch.tensor([[50.0, 60.0], [70.0, 80.0]]),
        ]
        
        result = get_avg_update(updates, weights)
        
        # Element (0,0): (1*10 + 5*50) / (10+50) = 260/60 ≈ 4.333
        # Element (0,1): (2*20 + 6*60) / (20+60) = 400/80 = 5.0
        # Element (1,0): (3*30 + 7*70) / (30+70) = 580/100 = 5.8
        # Element (1,1): (4*40 + 8*80) / (40+80) = 800/120 ≈ 6.667
        
        expected_00 = (1.0 * 10 + 5.0 * 50) / 60
        expected_01 = (2.0 * 20 + 6.0 * 60) / 80
        expected_10 = (3.0 * 30 + 7.0 * 70) / 100
        expected_11 = (4.0 * 40 + 8.0 * 80) / 120
        
        np.testing.assert_allclose(result[0, 0].numpy(), expected_00, rtol=1e-5)
        np.testing.assert_allclose(result[0, 1].numpy(), expected_01, rtol=1e-5)
        np.testing.assert_allclose(result[1, 0].numpy(), expected_10, rtol=1e-5)
        np.testing.assert_allclose(result[1, 1].numpy(), expected_11, rtol=1e-5)
    
    def test_zero_weights_handling(self):
        """Test that zero weights are handled correctly"""
        updates = [
            torch.ones(2, 2) * 1.0,
            torch.ones(2, 2) * 2.0,
        ]
        weights = [
            torch.zeros(2, 2),  # Zero weights
            torch.ones(2, 2) * 10.0,
        ]
        
        result = get_avg_update(updates, weights)
        
        # Only the second update should contribute (weight 10)
        # After normalization: weight becomes 1.0, so result = 2.0
        expected = torch.ones(2, 2) * 2.0
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()


"""
Unit tests for MemoryTracker class in utils/memory_tracker.py
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.memory_tracker import MemoryTracker


class TestMemoryTracker(unittest.TestCase):
    """Test cases for MemoryTracker class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tracker = MemoryTracker()

    def test_get_lora_betas_basic_calculation(self):
        """Test get_lora_betas with known values to verify linear equation solving"""
        # Setup
        args = Mock()
        config = Mock()
        config.hidden_size = 384  # H = 384
        base_model = Mock()
        module_name = "attention.attention.query"
        B = 2  # batch size
        S = 197  # sequence length
        H = 384  # hidden size
        bytes_per_parameter = 4  # fp32
        
        # Calculate expected values
        r1 = int(H / 2)  # 192
        r2 = int(H / 3)  # 128
        bs = B * S  # 2 * 197 = 394
        bsh = bs * H  # 394 * 384 = 151296
        bsr1 = bs * r1  # 394 * 192 = 75648
        bsr2 = bs * r2  # 394 * 128 = 50432
        
        # Set up known beta values to test against
        # We'll use beta1 = 1.5, beta2 = 2.0
        beta1_expected = 1.5
        beta2_expected = 2.0
        
        # Calculate what info_r1_fwd and info_r2_fwd should be
        info_r1_fwd_expected = beta1_expected * bsh + beta2_expected * bsr1
        info_r2_fwd_expected = beta1_expected * bsh + beta2_expected * bsr2
        
        # Convert to bytes (multiply by bytes_per_parameter)
        info_r1_fwd_bytes = info_r1_fwd_expected * bytes_per_parameter
        info_r2_fwd_bytes = info_r2_fwd_expected * bytes_per_parameter
        
        # Mock the helper method to return our calculated values
        mock_info_r1 = {'avg_profiled_fwd': info_r1_fwd_bytes}
        mock_info_r2 = {'avg_profiled_fwd': info_r2_fwd_bytes}
        
        with patch.object(self.tracker, '_get_base_model_fwd_in_bytes_for_estimator_helper') as mock_helper:
            mock_helper.side_effect = [mock_info_r1, mock_info_r2]
            
            # Execute
            beta1, beta2 = self.tracker.get_lora_betas(args, config, base_model, module_name, B, S, H, bytes_per_parameter)
            
            # Verify
            self.assertAlmostEqual(beta1, beta1_expected, places=5)
            self.assertAlmostEqual(beta2, beta2_expected, places=5)

    def test_get_lora_betas_different_values(self):
        """Test get_lora_betas with different beta values"""
        # Setup
        args = Mock()
        config = Mock()
        config.hidden_size = 768
        base_model = Mock()
        module_name = "attention.attention.key"
        B = 4
        S = 100
        H = 768
        bytes_per_parameter = 2  # fp16
        
        r1 = int(H / 2)  # 384
        r2 = int(H / 3)  # 256
        bs = B * S  # 400
        bsh = bs * H  # 400 * 768 = 307200
        bsr1 = bs * r1  # 400 * 384 = 153600
        bsr2 = bs * r2  # 400 * 256 = 102400
        
        # Use different beta values
        beta1_expected = 0.8
        beta2_expected = 1.2
        
        info_r1_fwd_expected = beta1_expected * bsh + beta2_expected * bsr1
        info_r2_fwd_expected = beta1_expected * bsh + beta2_expected * bsr2
        
        info_r1_fwd_bytes = info_r1_fwd_expected * bytes_per_parameter
        info_r2_fwd_bytes = info_r2_fwd_expected * bytes_per_parameter
        
        mock_info_r1 = {'avg_profiled_fwd': info_r1_fwd_bytes}
        mock_info_r2 = {'avg_profiled_fwd': info_r2_fwd_bytes}
        
        with patch.object(self.tracker, '_get_base_model_fwd_in_bytes_for_estimator_helper') as mock_helper:
            mock_helper.side_effect = [mock_info_r1, mock_info_r2]
            
            # Execute
            beta1, beta2 = self.tracker.get_lora_betas(args, config, base_model, module_name, B, S, H, bytes_per_parameter)
            
            # Verify
            self.assertAlmostEqual(beta1, beta1_expected, places=5)
            self.assertAlmostEqual(beta2, beta2_expected, places=5)

    def test_get_lora_betas_verifies_equations(self):
        """Test that the returned beta values satisfy both equations"""
        # Setup
        args = Mock()
        config = Mock()
        config.hidden_size = 512
        base_model = Mock()
        module_name = "intermediate.dense"
        B = 8
        S = 50
        H = 512
        bytes_per_parameter = 4
        
        r1 = int(H / 2)  # 256
        r2 = int(H / 3)  # 170
        bs = B * S  # 400
        bsh = bs * H  # 400 * 512 = 204800
        bsr1 = bs * r1  # 400 * 256 = 102400
        bsr2 = bs * r2  # 400 * 170 = 68000
        
        beta1_expected = 2.5
        beta2_expected = 3.0
        
        info_r1_fwd_expected = beta1_expected * bsh + beta2_expected * bsr1
        info_r2_fwd_expected = beta1_expected * bsh + beta2_expected * bsr2
        
        info_r1_fwd_bytes = info_r1_fwd_expected * bytes_per_parameter
        info_r2_fwd_bytes = info_r2_fwd_expected * bytes_per_parameter
        
        mock_info_r1 = {'avg_profiled_fwd': info_r1_fwd_bytes}
        mock_info_r2 = {'avg_profiled_fwd': info_r2_fwd_bytes}
        
        with patch.object(self.tracker, '_get_base_model_fwd_in_bytes_for_estimator_helper') as mock_helper:
            mock_helper.side_effect = [mock_info_r1, mock_info_r2]
            
            # Execute
            beta1, beta2 = self.tracker.get_lora_betas(args, config, base_model, module_name, B, S, H, bytes_per_parameter)
            
            # Verify equations are satisfied
            equation1_result = beta1 * bsh + beta2 * bsr1
            equation2_result = beta1 * bsh + beta2 * bsr2
            
            info_r1_fwd = info_r1_fwd_bytes / bytes_per_parameter
            info_r2_fwd = info_r2_fwd_bytes / bytes_per_parameter
            
            self.assertAlmostEqual(equation1_result, info_r1_fwd, places=5)
            self.assertAlmostEqual(equation2_result, info_r2_fwd, places=5)

    def test_get_lora_betas_singular_matrix_error(self):
        """Test that get_lora_betas raises ValueError when bsr1 and bsr2 are too close"""
        # Setup - use a very small H where r1 and r2 might be equal or very close
        args = Mock()
        config = Mock()
        config.hidden_size = 3  # Very small: r1 = int(3/2) = 1, r2 = int(3/3) = 1
        base_model = Mock()
        module_name = "test.module"
        B = 1
        S = 1
        H = 3
        bytes_per_parameter = 4
        
        # With H=3: r1 = int(3/2) = 1, r2 = int(3/3) = 1
        # So bsr1 = bs * 1 = 1, bsr2 = bs * 1 = 1
        # They are equal, which should trigger the error
        
        # Mock the helper to return some values (won't matter since we'll hit the error)
        mock_info_r1 = {'avg_profiled_fwd': 100.0}
        mock_info_r2 = {'avg_profiled_fwd': 100.0}
        
        with patch.object(self.tracker, '_get_base_model_fwd_in_bytes_for_estimator_helper') as mock_helper:
            mock_helper.side_effect = [mock_info_r1, mock_info_r2]
            
            # Execute and verify error
            with self.assertRaises(ValueError) as context:
                self.tracker.get_lora_betas(args, config, base_model, module_name, B, S, H, bytes_per_parameter)
            
            error_message = str(context.exception).lower()
            self.assertIn("too close", error_message)
            self.assertIn("singular", error_message)

    def test_get_lora_betas_uses_config_hidden_size(self):
        """Test that get_lora_betas uses config.hidden_size for H calculation"""
        # Setup
        args = Mock()
        config = Mock()
        config.hidden_size = 256
        base_model = Mock()
        module_name = "test"
        B = 2
        S = 10
        H_param = 512  # Different from config.hidden_size
        bytes_per_parameter = 4
        
        # The method should use config.hidden_size, not the H parameter
        expected_H = config.hidden_size  # 256
        r1 = int(expected_H / 2)  # 128
        r2 = int(expected_H / 3)  # 85
        
        bs = B * S  # 20
        bsh = bs * expected_H  # 20 * 256 = 5120
        bsr1 = bs * r1  # 20 * 128 = 2560
        bsr2 = bs * r2  # 20 * 85 = 1700
        
        beta1_expected = 1.0
        beta2_expected = 1.0
        
        info_r1_fwd_expected = beta1_expected * bsh + beta2_expected * bsr1
        info_r2_fwd_expected = beta1_expected * bsh + beta2_expected * bsr2
        
        info_r1_fwd_bytes = info_r1_fwd_expected * bytes_per_parameter
        info_r2_fwd_bytes = info_r2_fwd_expected * bytes_per_parameter
        
        mock_info_r1 = {'avg_profiled_fwd': info_r1_fwd_bytes}
        mock_info_r2 = {'avg_profiled_fwd': info_r2_fwd_bytes}
        
        with patch.object(self.tracker, '_get_base_model_fwd_in_bytes_for_estimator_helper') as mock_helper:
            mock_helper.side_effect = [mock_info_r1, mock_info_r2]
            
            # Execute - note: H_param is passed but method uses config.hidden_size
            beta1, beta2 = self.tracker.get_lora_betas(args, config, base_model, module_name, B, S, H_param, bytes_per_parameter)
            
            # Verify the calculation used config.hidden_size (256), not H_param (512)
            # If it used H_param, the result would be different
            self.assertAlmostEqual(beta1, beta1_expected, places=5)
            self.assertAlmostEqual(beta2, beta2_expected, places=5)

    def test_get_lora_betas_handles_negative_betas(self):
        """Test that get_lora_betas can handle negative beta values (edge case)"""
        # Setup
        args = Mock()
        config = Mock()
        config.hidden_size = 384
        base_model = Mock()
        module_name = "test.module"
        B = 1
        S = 1
        H = 384
        bytes_per_parameter = 4
        
        r1 = int(H / 2)  # 192
        r2 = int(H / 3)  # 128
        bs = B * S  # 1
        bsh = bs * H  # 384
        bsr1 = bs * r1  # 192
        bsr2 = bs * r2  # 128
        
        # Use negative beta values (unlikely in practice but mathematically valid)
        beta1_expected = -0.5
        beta2_expected = -1.0
        
        info_r1_fwd_expected = beta1_expected * bsh + beta2_expected * bsr1
        info_r2_fwd_expected = beta1_expected * bsh + beta2_expected * bsr2
        
        info_r1_fwd_bytes = info_r1_fwd_expected * bytes_per_parameter
        info_r2_fwd_bytes = info_r2_fwd_expected * bytes_per_parameter
        
        mock_info_r1 = {'avg_profiled_fwd': info_r1_fwd_bytes}
        mock_info_r2 = {'avg_profiled_fwd': info_r2_fwd_bytes}
        
        with patch.object(self.tracker, '_get_base_model_fwd_in_bytes_for_estimator_helper') as mock_helper:
            mock_helper.side_effect = [mock_info_r1, mock_info_r2]
            
            # Execute
            beta1, beta2 = self.tracker.get_lora_betas(args, config, base_model, module_name, B, S, H, bytes_per_parameter)
            
            # Verify
            self.assertAlmostEqual(beta1, beta1_expected, places=5)
            self.assertAlmostEqual(beta2, beta2_expected, places=5)

    def test_get_lora_betas_zero_beta2(self):
        """Test get_lora_betas when beta2 is zero"""
        # Setup
        args = Mock()
        config = Mock()
        config.hidden_size = 256
        base_model = Mock()
        module_name = "test"
        B = 2
        S = 10
        H = 256
        bytes_per_parameter = 4
        
        r1 = int(H / 2)  # 128
        r2 = int(H / 3)  # 85
        bs = B * S  # 20
        bsh = bs * H  # 5120
        bsr1 = bs * r1  # 2560
        bsr2 = bs * r2  # 1700
        
        beta1_expected = 2.0
        beta2_expected = 0.0  # Zero beta2
        
        info_r1_fwd_expected = beta1_expected * bsh + beta2_expected * bsr1
        info_r2_fwd_expected = beta1_expected * bsh + beta2_expected * bsr2
        
        info_r1_fwd_bytes = info_r1_fwd_expected * bytes_per_parameter
        info_r2_fwd_bytes = info_r2_fwd_expected * bytes_per_parameter
        
        mock_info_r1 = {'avg_profiled_fwd': info_r1_fwd_bytes}
        mock_info_r2 = {'avg_profiled_fwd': info_r2_fwd_bytes}
        
        with patch.object(self.tracker, '_get_base_model_fwd_in_bytes_for_estimator_helper') as mock_helper:
            mock_helper.side_effect = [mock_info_r1, mock_info_r2]
            
            # Execute
            beta1, beta2 = self.tracker.get_lora_betas(args, config, base_model, module_name, B, S, H, bytes_per_parameter)
            
            # Verify
            self.assertAlmostEqual(beta1, beta1_expected, places=5)
            self.assertAlmostEqual(beta2, beta2_expected, places=5)


if __name__ == '__main__':
    unittest.main()

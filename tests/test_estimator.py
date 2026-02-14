"""
Unit tests for RankEstimator class in estimator.py
"""
import argparse
import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from transformers import AutoModelForImageClassification, AutoConfig
import torch

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from estimator import RankEstimator, FEDHELLO, OURS, MEM_ONLY, UPLOAD_ONLY
from peft import LoraConfig, get_peft_model
from utils.memory_tracker import MemoryTracker


def _make_config(image_size=224, patch_size=16, hidden_size=384, num_hidden_layers=12, intermediate_size=1536):
    """Build a minimal config namespace for testing."""
    config = argparse.Namespace()
    config.image_size = image_size
    config.patch_size = patch_size
    config.hidden_size = hidden_size
    config.num_hidden_layers = num_hidden_layers
    config.intermediate_size = intermediate_size
    return config


class TestRankEstimator(unittest.TestCase):
    """Test cases for RankEstimator class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.estimator = RankEstimator()
        self.tracker = MemoryTracker()

    def test_get_sequence_length_returns_197_for_deit_config(self):
        """_get_sequence_length uses config from args.model; (224/16)^2 + 1 = 197."""
        args = argparse.Namespace()
        args.model = "facebook/deit-small-patch16-224"
        args.CLS_TOKEN = 1
        mock_config = MagicMock()
        mock_config.image_size = 224
        mock_config.patch_size = 16
        with patch("estimator.AutoConfig.from_pretrained", return_value=mock_config):
            result = self.estimator._get_sequence_length(args, None)
        self.assertEqual(result, 197)
        self.assertIsInstance(result, (int, float))

    def test_get_sequence_length_formula(self):
        """Sequence length = (image_size/patch_size)^2 + CLS_TOKEN."""
        args = argparse.Namespace()
        args.model = "dummy"
        args.CLS_TOKEN = 1
        mock_config = MagicMock()
        mock_config.image_size = 224
        mock_config.patch_size = 16
        with patch("estimator.AutoConfig.from_pretrained", return_value=mock_config):
            result = self.estimator._get_sequence_length(args, None)
        expected = (224 / 16) * (224 / 16) + 1
        self.assertEqual(result, expected)
        self.assertEqual(result, 197)

    def test_get_sequence_length_with_custom_cls_token(self):
        """CLS_TOKEN is respected in sequence length."""
        args = argparse.Namespace()
        args.model = "dummy"
        args.CLS_TOKEN = 0
        mock_config = MagicMock()
        mock_config.image_size = 224
        mock_config.patch_size = 16
        with patch("estimator.AutoConfig.from_pretrained", return_value=mock_config):
            result = self.estimator._get_sequence_length(args, None)
        self.assertEqual(result, 196)

    def test_get_byte_per_parameter_fp32(self):
        """fp32 uses 4 bytes per parameter."""
        args = argparse.Namespace(precision="fp32")
        self.assertEqual(self.estimator._get_byte_per_parameter("fp32"), 4)

    def test_get_byte_per_parameter_fp16(self):
        """fp16 uses 2 bytes per parameter."""
        self.assertEqual(self.estimator._get_byte_per_parameter("fp16"), 2)

    def test_get_byte_per_parameter_invalid_raises(self):
        """Invalid precision raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.estimator._get_byte_per_parameter("int8")
        self.assertIn("Invalid precision", str(ctx.exception))

    def test_get_total_gpu_memory_size_in_bytes(self):
        """GPU memory in bytes = GB * 1024^3."""
        args = argparse.Namespace()
        result = self.estimator._get_total_gpu_memory_size_in_bytes(args, 8.0)
        self.assertEqual(result, 8.0 * 1024 ** 3)

    def test_get_num_of_adapted_matrices(self):
        """LoRA uses 2 matrices (A and B)."""
        args = argparse.Namespace()
        self.assertEqual(self.estimator._get_num_of_adapted_matrices(args), 2)

    def test_get_num_of_modules_per_layer(self):
        """Number of modules equals len(lora_target_modules)."""
        args = argparse.Namespace(lora_target_modules=["query", "value"])
        self.assertEqual(self.estimator._get_num_of_modules_per_layer(args), 2)
        args.lora_target_modules = ["query", "key", "value"]
        self.assertEqual(self.estimator._get_num_of_modules_per_layer(args), 3)

    def test_get_final_rank_returns_minimum(self):
        """Final rank is the minimum of memory, upload, and download ranks."""
        args = argparse.Namespace()
        config = _make_config()
        r_mem, r_up, r_down = 64, 128, 96
        result = self.estimator._get_final_rank(
            args, config, r_mem, r_up, r_down
        )
        self.assertEqual(result, 64)
        result = self.estimator._get_final_rank(
            args, config, 200, 50, 100
        )
        self.assertEqual(result, 50)

    def test_get_rank_based_on_network_speed_formula(self):
        """Rank from network = (Mbps * 1e6/8 * time_sec) / (C * num_modules * H * num_layers * bytes_per_param)."""
        args = argparse.Namespace(
            precision="fp32",
            num_of_layers_to_allocate_LoRA=12,
            lora_target_modules=["query", "value"],
        )
        config = _make_config(hidden_size=384)
        # 7 Mbps * 1e6/8 = 875_000 bytes/s; 15 s -> 13_125_000 bytes
        # total_dim = 2 * 2 * 384 * 12 * 4 = 73_728 bytes per rank
        # rank = 13125000 / 73728 ≈ 178
        result = self.estimator._get_rank_based_on_network_speed(
            args, config,
            network_speed_in_Mbps=7.0,
            desired_communication_time_in_seconds=15.0,
        )
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
        self.assertLessEqual(result, config.hidden_size)

    def test_get_rank_based_on_network_speed_capped_by_hidden_size(self):
        """Rank from network is capped at config.hidden_size."""
        args = argparse.Namespace(
            precision="fp16",
            num_of_layers_to_allocate_LoRA=1,
            lora_target_modules=["query"],
        )
        config = _make_config(hidden_size=64)
        result = self.estimator._get_rank_based_on_network_speed(
            args, config,
            network_speed_in_Mbps=1000.0,
            desired_communication_time_in_seconds=1000.0,
        )
        self.assertLessEqual(result, 64)

    def test_invalid_rank_estimator_method_raises(self):
        """Invalid rank_estimator_method raises ValueError."""
        args = self._init_args()
        args.rank_estimator_method = "InvalidMethod"
        config = AutoConfig.from_pretrained(args.model)
        base_model = AutoModelForImageClassification.from_pretrained(args.model)
        memory_summary_dict = {}
        with self.assertRaises(ValueError) as ctx:
            self.estimator._get_rank_for_one_client_group(
                args, config, base_model,
                total_gpu_memory_size_in_GB=8.0,
                upload_network_speed_in_Mbps=7.0,
                download_network_speed_in_Mbps=50.0,
                desired_uploading_time_in_seconds=15.0,
                desired_downloading_time_in_seconds=15.0,
                memory_summary_dict=memory_summary_dict,
            )
        self.assertIn("Invalid rank estimator method", str(ctx.exception))

    def test_bytes_to_mb(self):
        """_bytes_to_mb converts bytes to megabytes rounded to 2 decimals."""
        self.assertEqual(self.estimator._bytes_to_mb(1024 * 1024), 1.0)
        self.assertEqual(self.estimator._bytes_to_mb(1024 * 1024 * 5), 5.0)

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
        args.desired_uploading_time_for_each_group_in_seconds = [15]
        args.desired_downloading_time_for_each_group_in_seconds = [15]
        args.heterogeneous_group = [1.0]
        args.gpu_memory_size_for_each_group_in_GB = [8.0]
        args.avg_upload_network_speed_for_each_group_in_Mbps = [7.0]
        args.avg_download_network_speed_for_each_group_in_Mbps = [50.0]
        args.train_classifier = False
        args.CLS_TOKEN = 1
        return args

    def test_get_all_named_modules(self):
        """Base model has named modules and expected structure (e.g. deit)."""
        args = self._init_args()
        model = AutoModelForImageClassification.from_pretrained(args.model)
        names = [name for name, _ in model.named_modules()]
        self.assertGreater(len(names), 0)
        self.assertTrue(
            any("deit" in n.lower() or "encoder" in n.lower() for n in names),
            msg="Expected ViT/DeiT-like module names",
        )

    def test_lora_shapes(self):
        """LoRA adapter adds low-rank weight matrices with expected shapes."""
        args = self._init_args()
        model = AutoModelForImageClassification.from_pretrained(args.model)
        r = self._get_rank()
        config = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, config)
        lora_shapes = []
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None and "lora" in name.lower():
                lora_shapes.append((name, list(module.weight.shape)))
        self.assertGreater(len(lora_shapes), 0, msg="Expected at least one LoRA weight")
        for name, shape in lora_shapes:
            self.assertIn(r, shape, msg=f"LoRA rank {r} should appear in {name} shape {shape}")

    def test_get_base_model_fwd_in_bytes_for_estimator(self):
        """MemoryTracker returns (fwd_bytes, overhead_bytes) with non-negative values."""
        args = self._init_args()
        base_model = AutoModelForImageClassification.from_pretrained(args.model)
        config = AutoConfig.from_pretrained(args.model)
        result = self.tracker.get_base_model_fwd_in_bytes_for_estimator(args, config, base_model)
        self.assertIsInstance(result, (list, tuple), msg="Expected (fwd_bytes, overhead_bytes)")
        self.assertEqual(len(result), 2)
        fwd_bytes, overhead_bytes = result
        self.assertGreaterEqual(fwd_bytes, 0)
        self.assertGreaterEqual(overhead_bytes, 0)

    def _get_rank(self):
        return 178

    def _get_rank2(self):
        return 64


if __name__ == "__main__":
    unittest.main()


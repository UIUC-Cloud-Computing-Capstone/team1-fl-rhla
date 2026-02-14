"""
Unit tests for algorithms/solver/local_solver.py

Tests cover: get_parameter_names (inclusion/exclusion of layers),
LocalUpdate.__init__ (loss_func by data_type), and lora_tuning early return
when client has no trainable layers.
"""
import argparse
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.solver.local_solver import LocalUpdate, get_parameter_names


# -----------------------------------------------------------------------------
# get_parameter_names
# -----------------------------------------------------------------------------
class TestGetParameterNames(unittest.TestCase):
    def test_flat_linear_returns_weight_and_bias(self):
        model = nn.Linear(2, 3)
        names = get_parameter_names(model, [])
        self.assertIn("weight", names)
        self.assertIn("bias", names)
        self.assertEqual(len(names), 2)

    def test_sequential_linear_prefixed_by_child_name(self):
        model = nn.Sequential(nn.Linear(2, 3))
        names = get_parameter_names(model, [])
        self.assertIn("0.weight", names)
        self.assertIn("0.bias", names)
        self.assertEqual(len(names), 2)

    def test_forbidden_layer_excluded(self):
        model = nn.Sequential(nn.Linear(2, 3))
        names = get_parameter_names(model, [nn.Linear])
        self.assertEqual(names, [])

    def test_nested_forbidden_only_excludes_forbidden_subtree(self):
        # Sequential(Linear, Conv2d): forbid Linear only -> Conv params included
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.Conv2d(1, 2, 3),
        )
        names = get_parameter_names(model, [nn.Linear])
        self.assertTrue(any("1." in n for n in names))
        self.assertFalse(any(n.startswith("0.") for n in names))

    def test_empty_forbidden_returns_all_parameter_names(self):
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.Linear(3, 1),
        )
        names = get_parameter_names(model, [])
        self.assertEqual(len(names), 4)
        self.assertIn("0.weight", names)
        self.assertIn("0.bias", names)
        self.assertIn("1.weight", names)
        self.assertIn("1.bias", names)


# -----------------------------------------------------------------------------
# LocalUpdate.__init__
# -----------------------------------------------------------------------------
class TestLocalUpdateInit(unittest.TestCase):
    def test_image_data_type_uses_cross_entropy(self):
        args = argparse.Namespace(data_type="image")
        lu = LocalUpdate(args)
        self.assertIsInstance(lu.loss_func, nn.CrossEntropyLoss)

    def test_text_data_type_uses_cross_entropy(self):
        args = argparse.Namespace(data_type="text")
        lu = LocalUpdate(args)
        self.assertIsInstance(lu.loss_func, nn.CrossEntropyLoss)

    def test_sentiment_data_type_uses_nll_loss(self):
        args = argparse.Namespace(data_type="sentiment")
        lu = LocalUpdate(args)
        self.assertIsInstance(lu.loss_func, nn.NLLLoss)


# -----------------------------------------------------------------------------
# LocalUpdate.lora_tuning early return
# -----------------------------------------------------------------------------
class TestLoraTuningEarlyReturn(unittest.TestCase):
    """When client has no trainable layers (no_weight_lora == all layers), returns immediately."""

    @patch("builtins.print")
    def test_returns_state_dict_none_loss_and_full_no_weight_lora(self, mock_print):
        model = nn.Linear(2, 3)
        model.state_dict()
        ldr_train = []
        args = argparse.Namespace()
        args.lora_layer = 4
        setattr(args, "heterogeneous_group0_lora", [])  # no layers assigned
        args.block_ids_list = [[]] * 4
        args.rank_list = []
        args.logger = MagicMock()
        lu = LocalUpdate(argparse.Namespace(data_type="image"))

        state_dict, mean_loss, no_weight_lora = lu.lora_tuning(
            model=model,
            ldr_train=ldr_train,
            args=args,
            client_index=0,
            client_real_id=0,
            round=0,
            hete_group_id=0,
        )

        self.assertIsInstance(state_dict, dict)
        self.assertIn("weight", state_dict)
        self.assertIn("bias", state_dict)
        self.assertIsNone(mean_loss)
        self.assertEqual(set(no_weight_lora), {0, 1, 2, 3})
        self.assertEqual(len(no_weight_lora), 4)

    @patch("builtins.print")
    def test_early_return_when_heterogeneous_group_lora_is_empty_list(self, mock_print):
        model = nn.Sequential(nn.Linear(2, 3))
        args = argparse.Namespace()
        args.lora_layer = 2
        setattr(args, "heterogeneous_group0_lora", [])
        args.block_ids_list = [[], []]
        args.rank_list = []
        args.logger = MagicMock()
        lu = LocalUpdate(argparse.Namespace(data_type="text"))

        _, mean_loss, no_weight_lora = lu.lora_tuning(
            model=model,
            ldr_train=[],
            args=args,
            client_index=0,
            client_real_id=0,
            round=0,
            hete_group_id=0,
        )

        self.assertIsNone(mean_loss)
        self.assertEqual(len(no_weight_lora), 2)


if __name__ == "__main__":
    unittest.main()

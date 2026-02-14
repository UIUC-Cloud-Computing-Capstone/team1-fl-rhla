"""
Unit tests for algorithms/solver/global_aggregator.py

Tests cover: average_lora_depthfl, weighted_average_lora_depthfl,
svd_average, and product_average (in-place updates, key filtering, weighting).
"""
import argparse
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.solver.global_aggregator import (
    average_lora_depthfl,
    product_average,
    svd_average,
    weighted_average_lora_depthfl,
)


def _args(lokr=False, logger=None):
    a = argparse.Namespace()
    a.LOKR = lokr
    a.logger = logger or MagicMock()
    return a


# -----------------------------------------------------------------------------
# average_lora_depthfl
# -----------------------------------------------------------------------------
class TestAverageLoraDepthfl(unittest.TestCase):
    @patch("builtins.print")
    def test_single_key_two_clients_in_place(self, mock_print):
        args = _args()
        global_model = {"lora_A.0": torch.tensor([[0.0, 0.0], [0.0, 0.0]])}
        loc_updates = [
            {"lora_A.0": torch.tensor([[1.0, 0.0], [0.0, 0.0]])},
            {"lora_A.0": torch.tensor([[0.0, 2.0], [0.0, 0.0]])},
        ]
        out = average_lora_depthfl(args, global_model, loc_updates)
        self.assertIs(out, global_model)
        expected = torch.tensor([[0.5, 1.0], [0.0, 0.0]])
        torch.testing.assert_close(global_model["lora_A.0"], expected)

    @patch("builtins.print")
    def test_heterogeneous_only_aggregates_present_keys(self, mock_print):
        args = _args()
        global_model = {
            "lora_A.0": torch.tensor([0.0, 0.0]),
            "lora_A.1": torch.tensor([0.0, 0.0]),
        }
        loc_updates = [
            {"lora_A.0": torch.tensor([1.0, 0.0])},
            {"lora_A.1": torch.tensor([0.0, 1.0])},
        ]
        average_lora_depthfl(args, global_model, loc_updates)
        torch.testing.assert_close(global_model["lora_A.0"], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(global_model["lora_A.1"], torch.tensor([0.0, 1.0]))

    @patch("builtins.print")
    def test_ignores_non_lora_keys(self, mock_print):
        args = _args()
        global_model = {
            "base.weight": torch.tensor([1.0]),
            "lora_A.0": torch.tensor([0.0]),
        }
        loc_updates = [{"lora_A.0": torch.tensor([2.0])}]
        average_lora_depthfl(args, global_model, loc_updates)
        self.assertEqual(global_model["base.weight"].item(), 1.0)
        self.assertEqual(global_model["lora_A.0"].item(), 2.0)

    @patch("builtins.print")
    def test_lokr_uses_lokr_keys(self, mock_print):
        args = _args(lokr=True)
        global_model = {"lokr_A.0": torch.tensor([0.0])}
        loc_updates = [{"lokr_A.0": torch.tensor([3.0])}]
        average_lora_depthfl(args, global_model, loc_updates)
        self.assertEqual(global_model["lokr_A.0"].item(), 3.0)

    @patch("builtins.print")
    def test_classifier_key_included(self, mock_print):
        args = _args()
        global_model = {"classifier.weight": torch.tensor([0.0, 0.0])}
        loc_updates = [
            {"classifier.weight": torch.tensor([1.0, 0.0])},
            {"classifier.weight": torch.tensor([0.0, 2.0])},
        ]
        average_lora_depthfl(args, global_model, loc_updates)
        torch.testing.assert_close(
            global_model["classifier.weight"], torch.tensor([0.5, 1.0])
        )


# -----------------------------------------------------------------------------
# weighted_average_lora_depthfl
# -----------------------------------------------------------------------------
class TestWeightedAverageLoraDepthfl(unittest.TestCase):
    @patch("builtins.print", MagicMock())
    def test_weighted_by_num_samples(self):
        args = _args()
        global_model = {"lora_A.0": torch.tensor([0.0])}
        # client0: 1 sample, update +1; client1: 3 samples, update +0 → weight 1/4 and 3/4
        loc_updates = [
            {"lora_A.0": torch.tensor([1.0])},
            {"lora_A.0": torch.tensor([0.0])},
        ]
        num_samples = [1, 3]
        weighted_average_lora_depthfl(args, global_model, loc_updates, num_samples)
        # 0 + (1/4)*1 + (3/4)*0 = 0.25
        self.assertAlmostEqual(global_model["lora_A.0"].item(), 0.25)

    def test_single_client_same_as_average(self):
        args = _args()
        global_model = {"lora_A.0": torch.tensor([0.0, 0.0])}
        loc_updates = [{"lora_A.0": torch.tensor([2.0, 4.0])}]
        num_samples = [10]
        weighted_average_lora_depthfl(args, global_model, loc_updates, num_samples)
        torch.testing.assert_close(global_model["lora_A.0"], torch.tensor([2.0, 4.0]))

    def test_returns_same_object(self):
        args = _args()
        global_model = {"lora_A.0": torch.tensor([0.0])}
        loc_updates = [{"lora_A.0": torch.tensor([1.0])}]
        out = weighted_average_lora_depthfl(
            args, global_model, loc_updates, [1]
        )
        self.assertIs(out, global_model)


# -----------------------------------------------------------------------------
# svd_average
# -----------------------------------------------------------------------------
class TestSvdAverage(unittest.TestCase):
    @patch("builtins.print")
    def test_weights_normalized_and_applied(self, mock_print):
        args = _args()
        global_model = {
            "lora_A.0": torch.tensor([[0.0, 0.0]]),
            "lora_B.0": torch.tensor([[0.0], [0.0]]),
        }
        # Two clients: B@A gives same Frobenius norm (1.0 each), so weights [0.5, 0.5]
        loc_updates = [
            {"lora_A.0": torch.tensor([[1.0, 0.0]]), "lora_B.0": torch.tensor([[1.0], [0.0]])},
            {"lora_A.0": torch.tensor([[0.0, 1.0]]), "lora_B.0": torch.tensor([[0.0], [1.0]])},
        ]
        num_samples = [1, 1]
        svd_average(args, global_model, loc_updates, num_samples)
        # global stays 0; weighted sum = 0.5*u1 + 0.5*u2 → lora_A.0 = [[0.5, 0.5]]
        self.assertEqual(global_model["lora_A.0"].shape, (1, 2))
        self.assertEqual(global_model["lora_B.0"].shape, (2, 1))
        torch.testing.assert_close(
            global_model["lora_A.0"], torch.tensor([[0.5, 0.5]])
        )

    @patch("builtins.print")
    def test_modifies_loc_updates_in_place(self, mock_print):
        args = _args()
        global_model = {
            "lora_A.0": torch.zeros(1, 2),
            "lora_B.0": torch.zeros(2, 1),
        }
        u1 = {"lora_A.0": torch.ones(1, 2), "lora_B.0": torch.ones(2, 1)}
        u2 = {"lora_A.0": torch.ones(1, 2) * 2, "lora_B.0": torch.ones(2, 1) * 2}
        loc_updates = [u1, u2]
        svd_average(args, global_model, loc_updates, [1, 1])
        # Weights applied in place
        self.assertIs(loc_updates[0], u1)
        self.assertIs(loc_updates[1], u2)

    @patch("builtins.print")
    def test_single_client_weight_one(self, mock_print):
        args = _args()
        global_model = {
            "lora_A.0": torch.tensor([[0.0, 0.0]]),
            "lora_B.0": torch.tensor([[0.0], [0.0]]),
        }
        loc_updates = [
            {"lora_A.0": torch.tensor([[1.0, 2.0]]), "lora_B.0": torch.tensor([[1.0], [0.0]])},
        ]
        svd_average(args, global_model, loc_updates, [1])
        torch.testing.assert_close(global_model["lora_A.0"], torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(global_model["lora_B.0"], torch.tensor([[1.0], [0.0]]))


# -----------------------------------------------------------------------------
# product_average
# -----------------------------------------------------------------------------
class TestProductAverage(unittest.TestCase):
    @patch("builtins.print")
    def test_mean_of_global_plus_updates(self, mock_print):
        args = _args()
        global_model = {"lora_A.0": torch.tensor([0.0, 0.0])}
        loc_updates = [
            {"lora_A.0": torch.tensor([1.0, 0.0])},
            {"lora_A.0": torch.tensor([0.0, 2.0])},
        ]
        product_average(args, global_model, loc_updates, [1, 1])
        # (0+1, 0+0) and (0+0, 0+2) → mean = (0.5, 1.0)
        torch.testing.assert_close(
            global_model["lora_A.0"], torch.tensor([0.5, 1.0])
        )

    @patch("builtins.print")
    def test_heterogeneous_keys(self, mock_print):
        args = _args()
        global_model = {
            "lora_A.0": torch.tensor([0.0]),
            "lora_A.1": torch.tensor([0.0]),
        }
        loc_updates = [
            {"lora_A.0": torch.tensor([2.0])},
            {"lora_A.1": torch.tensor([4.0])},
        ]
        product_average(args, global_model, loc_updates, [1, 1])
        self.assertEqual(global_model["lora_A.0"].item(), 2.0)
        self.assertEqual(global_model["lora_A.1"].item(), 4.0)

    @patch("builtins.print")
    def test_returns_same_object(self, mock_print):
        args = _args()
        global_model = {"lora_A.0": torch.tensor([0.0])}
        loc_updates = [{"lora_A.0": torch.tensor([1.0])}]
        out = product_average(args, global_model, loc_updates, [1])
        self.assertIs(out, global_model)

    @patch("builtins.print")
    def test_lokr_keys(self, mock_print):
        args = _args(lokr=True)
        global_model = {"lokr_A.0": torch.tensor([0.0])}
        loc_updates = [{"lokr_A.0": torch.tensor([3.0])}]
        product_average(args, global_model, loc_updates, [1])
        self.assertEqual(global_model["lokr_A.0"].item(), 3.0)


if __name__ == "__main__":
    unittest.main()

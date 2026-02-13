#!/usr/bin/env python3
"""
Run rank estimation for all client groups using the "Ours" method.
Config is loaded from a YAML file (e.g. config/rank_estimator_ours.yaml).

Usage:
    python run_rank_estimation.py [config_path]
    python run_rank_estimation.py config/rank_estimator_ours.yaml
"""
import argparse
import sys
import time
from pathlib import Path

import yaml
from transformers import AutoModelForImageClassification, AutoConfig

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from estimator import RankEstimator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_to_args(cfg: dict) -> argparse.Namespace:
    """Build argparse.Namespace from YAML config for RankEstimator."""
    args = argparse.Namespace()
    args.rank_estimator_method = cfg["rank_estimator_method"]

    rh = cfg.get("resource_heterogeneity", {})
    args.gpu_memory_size_for_each_group_in_GB = rh["gpu_memory_size_for_each_group_in_GB"]
    args.avg_upload_network_speed_for_each_group_in_Mbps = rh["avg_upload_network_speed_for_each_group_in_Mbps"]
    args.avg_download_network_speed_for_each_group_in_Mbps = rh["avg_download_network_speed_for_each_group_in_Mbps"]
    args.desired_uploading_time_for_each_group_in_seconds = rh["desired_uploading_time_for_each_group_in_seconds"]
    args.desired_downloading_time_for_each_group_in_seconds = rh["desired_downloading_time_for_each_group_in_seconds"]
    args.heterogeneous_group = rh["heterogeneous_group"]

    args.model = cfg["model"]
    args.CLS_TOKEN = cfg["CLS_TOKEN"]
    args.precision = cfg["precision"]
    args.optimizer = cfg["optimizer"]
    args.num_of_layers_to_allocate_LoRA = cfg["num_of_layers_to_allocate_LoRA"]
    args.lora_target_modules = cfg["lora_target_modules"]
    args.train_classifier = cfg["train_classifier"]
    args.batch_size = cfg["batch_size"]

    return args


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run rank estimation from YAML config.")
    parser.add_argument(
        "config",
        nargs="?",
        default="config/rank_estimator_ours.yaml",
        help="Path to YAML config (default: config/rank_estimator_ours.yaml)",
    )
    parsed = parser.parse_args()
    config_path = Path(parsed.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    args = config_to_args(cfg)

    base_model = AutoModelForImageClassification.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    estimator = RankEstimator()
    memory_summary_dict = {}

    client_rank_budgets = estimator.get_rank_for_all_client_groups(
        args, config, base_model, memory_summary_dict
    )
    print("Per client (total rank budget):", client_rank_budgets)

    elapsed = time.perf_counter() - start
    print(f"Total time to finish the rank estimation task: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

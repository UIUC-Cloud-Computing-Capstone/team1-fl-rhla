#!/usr/bin/env python3
"""
Run memory breakdown comparison profiling and write LaTeX tables.
Not a test: generates memory_breakdown_comparison_*.tex files.
Config is loaded from a YAML file (default: config/memory_breakdown_comparison.yaml).

Usage:
    python run_memory_breakdown_comparison.py [qv|mlp|all] [--config PATH]
    python run_memory_breakdown_comparison.py qv     # LoRA query/value, 2GB GPU
    python run_memory_breakdown_comparison.py mlp    # LoRA MLP output.dense, fixed rank 178
    python run_memory_breakdown_comparison.py all    # both (default)
"""
import argparse
import copy
import sys
from pathlib import Path

import yaml
from transformers import AutoModelForImageClassification, AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))

from estimator import RankEstimator
from utils.memory_tracker import MemoryTracker

MEM_ONLY = "mem_only"
DEFAULT_RANK = 178
DEFAULT_CONFIG = "config/memory_breakdown_comparison.yaml"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_to_args(cfg: dict) -> argparse.Namespace:
    """Build argparse.Namespace from YAML config."""
    args = argparse.Namespace()
    args.rank_estimator_method = cfg["rank_estimator_method"]
    args.model = cfg["model"]
    args.CLS_TOKEN = cfg["CLS_TOKEN"]
    args.precision = cfg["precision"]
    args.optimizer = cfg["optimizer"]
    args.num_of_layers_to_allocate_LoRA = cfg["num_of_layers_to_allocate_LoRA"]
    args.lora_target_modules = cfg["lora_target_modules"]
    args.train_classifier = cfg["train_classifier"]
    args.batch_size = cfg["batch_size"]

    rh = cfg.get("resource_heterogeneity", {})
    args.desired_uploading_time_for_each_group_in_seconds = rh["desired_uploading_time_for_each_group_in_seconds"]
    args.desired_downloading_time_for_each_group_in_seconds = rh["desired_downloading_time_for_each_group_in_seconds"]
    args.heterogeneous_group = rh["heterogeneous_group"]
    args.gpu_memory_size_for_each_group_in_GB = rh["gpu_memory_size_for_each_group_in_GB"]
    args.avg_upload_network_speed_for_each_group_in_Mbps = rh["avg_upload_network_speed_for_each_group_in_Mbps"]
    args.avg_download_network_speed_for_each_group_in_Mbps = rh["avg_download_network_speed_for_each_group_in_Mbps"]
    return args


def run_qv(tracker: MemoryTracker, estimator: RankEstimator, base_args: argparse.Namespace):
    """LoRA on query/value, 2GB GPU, rank from memory-only estimator."""
    args = copy.deepcopy(base_args)
    args.gpu_memory_size_for_each_group_in_GB = [2]
    args.lora_target_modules = ["query", "value"]

    base_model = AutoModelForImageClassification.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    memory_summary_dict = {}
    args.rank_estimator_method = MEM_ONLY
    base_model_copy = copy.copy(base_model)
    rank = estimator.get_rank_for_one_client_group(
        args, config, base_model_copy, memory_summary_dict
    )[0]
    del base_model_copy
    tracker.profile_and_compare(
        args,
        config,
        copy.copy(base_model),
        "memory_breakdown_comparison_lora_qv.tex",
        rank,
        memory_summary_dict,
    )
    print("Wrote memory_breakdown_comparison_lora_qv.tex")


def run_mlp(tracker: MemoryTracker, base_args: argparse.Namespace):
    """LoRA on MLP output.dense, fixed rank 178."""
    args = copy.deepcopy(base_args)
    args.lora_target_modules = r".*layer\.\d+\.output\.dense$"

    base_model = AutoModelForImageClassification.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    tracker.profile_and_compare(
        args,
        config,
        base_model,
        "memory_breakdown_comparison_lora_mlp_int_dense.tex",
        DEFAULT_RANK,
        {},
    )
    print("Wrote memory_breakdown_comparison_lora_mlp_int_dense.tex")


def main():
    parser = argparse.ArgumentParser(
        description="Run memory breakdown comparison and write .tex tables."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["qv", "mlp", "all"],
        help="qv = LoRA query/value 2GB; mlp = LoRA MLP output.dense; all = both",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    parsed = parser.parse_args()

    config_path = Path(parsed.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    cfg = load_config(config_path)
    base_args = config_to_args(cfg)

    tracker = MemoryTracker()
    estimator = RankEstimator()

    if parsed.mode in ("qv", "all"):
        run_qv(tracker, estimator, base_args)
    if parsed.mode in ("mlp", "all"):
        run_mlp(tracker, base_args)


if __name__ == "__main__":
    main()

"""
Script to visualize rank size vs GPU memory and network speed using RankEstimator.
Generates a combined diagram and saves it to results/diagrams/.
"""
import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForImageClassification, AutoConfig
import copy

MEM_ONLY = 'mem_only'
UPLOAD_ONLY = 'upload_only'

# Ensure project root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from estimator import RankEstimator


def init_args():
    args = argparse.Namespace()
    args.rank_estimator_method = 'Ours'
    args.model = 'facebook/deit-small-patch16-224'
    args.precision = 'fp32'
    args.optimizer = 'adam'
    args.num_of_layers_to_allocate_LoRA = 12
    args.lora_target_modules = ["query", "value"]
    args.image_height = 224
    args.image_width = 224
    args.patch_size = 16
    args.batch_size = 32
    args.desired_uploading_time_for_each_group_in_seconds = [1]
    args.desired_downloading_time_for_each_group_in_seconds = [1]
    args.heterogeneous_group = [1.0]  # Single group for simplicity
    args.train_classifier = False  # do not train classifier in the base model. Only train LoRA matrices.
    args.CLS_TOKEN = 1
    return args


def save_diagram(diagram_name):
    output_dir = os.path.join(os.path.dirname(__file__), 'results', 'diagrams')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, diagram_name)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nDiagram saved to: {output_path}")
    plt.close()


def main():
    # Vary memory sizes
    memory_sizes_GB = [1.5, 1.8, 2, 2.2, 2.5]

    # Vary network speeds
    network_speeds_Mbps = [1.0, 7.0, 15, 30, 50, 80]

    estimator = RankEstimator()
    args = init_args()

    # Load model once
    base_model = AutoModelForImageClassification.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)

    # Collect rank values for memory variation
    rank_values_memory = []
    for memory_size_GB in memory_sizes_GB:
        args.gpu_memory_size_for_each_group_in_GB = [memory_size_GB]
        args.rank_estimator_method = MEM_ONLY
        rank_budgets = estimator.get_rank_for_one_client_group(args, config, copy.deepcopy(base_model), {})
        rank_values_memory.append(rank_budgets[0])

    # Collect rank values for network speed variation
    rank_values_network = []
    for network_speed_Mbps in network_speeds_Mbps:
        args.avg_upload_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
        args.avg_download_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
        args.rank_estimator_method = UPLOAD_ONLY
        rank_budgets = estimator.get_rank_for_one_client_group(args, config, copy.deepcopy(base_model), {})
        rank_values_network.append(rank_budgets[0])

    # Create a single figure with dual X-axes
    fig, ax = plt.subplots(figsize=(16, 9))

    def transform_memory(mem_val):
        if mem_val <= 2:
            return 0.5 * mem_val
        else:
            base = 1.0
            return base + 0.3 * np.power(mem_val - 2, 0.5)

    x_positions_memory = [transform_memory(mem) for mem in memory_sizes_GB]

    # Plot memory line on bottom X-axis
    ax.plot(x_positions_memory, rank_values_memory, marker='o', linewidth=2, markersize=8,
            color='blue', label='Rank vs Memory')

    ax.set_xlabel('GPU Memory Size (GB)', fontsize=26, color='blue', labelpad=15)
    ax.set_xticks(x_positions_memory)
    ax.set_xticklabels([f'{mem:.2f}' if mem < 1 else f'{mem:.1f}' if mem < 3 else f'{int(mem)}'
                        for mem in memory_sizes_GB], rotation=30, ha='center', fontsize=24, color='blue')
    ax.tick_params(axis='x', labelsize=24, colors='blue', pad=12)

    x_min_transformed = min(x_positions_memory)
    x_max_transformed = max(x_positions_memory)
    x_range = x_max_transformed - x_min_transformed
    left_padding = 0.2 * x_range
    right_padding = 0.4 * x_range
    x_lim_min = x_min_transformed - left_padding
    x_lim_max = x_max_transformed + right_padding
    ax.set_xlim(x_lim_min, x_lim_max)

    # Create top X-axis for network speeds
    ax2 = ax.twiny()

    network_min = min(network_speeds_Mbps)
    network_max = max(network_speeds_Mbps)

    def network_to_x(network_val):
        if network_max == network_min:
            return x_lim_min
        available_range = x_lim_max - x_lim_min
        compressed_range = available_range * 0.85
        left_offset = available_range * 0.10
        normalized = (network_val - network_min) / (network_max - network_min)
        return x_lim_min + left_offset + normalized * compressed_range

    network_x_positions = [network_to_x(speed) for speed in network_speeds_Mbps]

    ax2.plot(network_x_positions, rank_values_network, marker='s', linewidth=2, markersize=8,
             color='green', label='Rank vs Network Speed')

    ax2.set_xlabel('Network Speed (Mbps)', fontsize=26, color='green', labelpad=15)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(network_x_positions)
    ax2.set_xticklabels([f'{speed:.1f}' for speed in network_speeds_Mbps],
                        rotation=30, ha='left', fontsize=24, color='green')
    ax2.tick_params(axis='x', labelsize=24, colors='green', pad=12)

    ax.set_ylabel('Rank Size', fontsize=26, labelpad=15)
    ax.set_ylim(0, 11000)
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=22, framealpha=0.95)
    ax.tick_params(axis='y', labelsize=24)

    for pos, rank in zip(x_positions_memory, rank_values_memory):
        ax.annotate(f'{int(rank)}', (pos, rank), textcoords="offset points",
                    xytext=(0, 13), ha='center', fontsize=24, color='blue')

    for pos, rank in zip(network_x_positions, rank_values_network):
        ax2.annotate(f'{int(rank)}', (pos, rank), textcoords="offset points",
                     xytext=(0, 30), ha='center', fontsize=24, color='green')

    plt.tight_layout(pad=4.0)
    fig.subplots_adjust(bottom=0.20, top=0.82)

    save_diagram('rank_vs_memory_and_network_speed_combined.pdf')


if __name__ == '__main__':
    main()

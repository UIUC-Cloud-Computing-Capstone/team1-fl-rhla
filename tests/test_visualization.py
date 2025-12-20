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
import copy
MEM_ONLY = 'mem_only'
UPLOAD_ONLY = 'upload_only'

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from estimator import RankEstimator

class TestRankEstimatorVisualization(unittest.TestCase):
    """Test class for visualizing rank size vs memory with fixed network speeds"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.estimator = RankEstimator()

    def _init_args(self):
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
        args.overhead_and_safety_margin_factor = 0.1
        args.desired_uploading_time_for_each_group_in_seconds = [15]
        args.desired_downloading_time_for_each_group_in_seconds = [15]
        args.heterogeneous_group = [1.0]  # Single group for simplicity
        args.train_classifier = False # do not train classifier in the base model. Only train LoRA matrices.
        args.CLS_TOKEN = 1
        return args
    

    def _save_diagram(self, diagram_name):
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'diagrams')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, diagram_name)
        # Save as PDF for selectable text labels (dpi parameter not needed for vector format)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"\nDiagram saved to: {output_path}")
        
        plt.close()

    def _create_rank_vs_memory_diagram(self, fixed_upload_speed_Mbps, fixed_download_speed_Mbps, memory_sizes_GB, rank_values):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create custom x-axis positions for better visualization
        # Map memory values to evenly spaced positions
        x_positions = list(range(len(memory_sizes_GB)))
        
        # Plot with custom x positions
        ax.plot(x_positions, rank_values, marker='o', linewidth=2, markersize=8, color='blue')
        ax.set_xlabel('GPU Memory Size (GB)', fontsize=16)
        ax.set_ylabel('Rank Size', fontsize=16)
        ax.set_title(f'Rank Size vs GPU Memory\n(Fixed Upload: {fixed_upload_speed_Mbps} Mbps, Download: {fixed_download_speed_Mbps} Mbps)', 
                     fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set custom x-axis labels with actual memory values
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'{mem:.2f}' if mem < 1 else f'{mem:.1f}' if mem < 2 else f'{int(mem)}' 
                            for mem in memory_sizes_GB], rotation=45, ha='right', fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        # Add value labels on points
        for i, (pos, rank) in enumerate(zip(x_positions, rank_values)):
            ax.annotate(f'{int(rank)}', (pos, rank), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=12)
        
        # Add a visual break indicator if there's a large gap
        if max(memory_sizes_GB) > 2 * max([m for m in memory_sizes_GB if m <= 2]):
            # Find where the gap occurs
            gap_start_idx = len([m for m in memory_sizes_GB if m <= 2])
            if gap_start_idx < len(memory_sizes_GB):
                # Add a subtle vertical line to indicate the gap
                ax.axvline(x=gap_start_idx - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                # Add text annotation
                ax.text(gap_start_idx - 0.5, ax.get_ylim()[1] * 0.95, '...', 
                        ha='center', fontsize=16, color='gray', alpha=0.7)
        
        plt.tight_layout()


    def test_rank_vs_memory_and_network_speed_combined(self):
        """Generate a combined diagram with both lines in the same figure, sharing the Y-axis"""
        # Fixed network speeds for memory diagram
        fixed_upload_speed_Mbps = 20
        fixed_download_speed_Mbps = 50.0
        
        # Fixed memory size for network speed diagram
        fixed_memory_GB = 8.0
        
        # Vary memory sizes (realistic range: 4GB to 16GB)
        memory_sizes_GB = [1.5, 1.8, 1.9, 2, 2.1, 2.2, 2.5, 4, 8]
        
        # Vary network speeds (realistic range: 0.5 Mbps to 10 Mbps)
        network_speeds_Mbps = [1.0, 2.0, 4.0, 8.0, 12.0, 15.0, 20.0]
        
        # Model and training configuration
        args = self._init_args()
        
        # Load model once
        base_model = AutoModelForImageClassification.from_pretrained(args.model)
        config = AutoConfig.from_pretrained(args.model)
        
        # Collect rank values for memory variation
        rank_values_memory = []
        for memory_size_GB in memory_sizes_GB:
            args.gpu_memory_size_for_each_group_in_GB = [memory_size_GB]
            args.avg_upload_network_speed_for_each_group_in_Mbps = [fixed_upload_speed_Mbps]
            args.avg_download_network_speed_for_each_group_in_Mbps = [fixed_download_speed_Mbps]
            args.rank_estimator_method = MEM_ONLY
            rank_budgets = self.estimator.get_rank_for_one_client_group(args, config, copy.deepcopy(base_model), {})
            rank_values_memory.append(rank_budgets[0])
        
        # Collect rank values for network speed variation
        rank_values_network = []
        args.gpu_memory_size_for_each_group_in_GB = [fixed_memory_GB]
        for network_speed_Mbps in network_speeds_Mbps:
            args.avg_upload_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
            args.avg_download_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
            args.rank_estimator_method = UPLOAD_ONLY
            rank_budgets = self.estimator.get_rank_for_one_client_group(args, config, copy.deepcopy(base_model), {})
            rank_values_network.append(rank_budgets[0])
        
        # Create a single figure with dual X-axes - even larger size for better readability
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Use a piecewise transformation to spread out small values significantly
        # This makes small differences much more readable while preserving the uneven nature
        def transform_memory(mem_val):
            """Transform memory value to spread small values more for readability"""
            # # Use a piecewise approach: more aggressive for small values, less for large
            # if mem_val <= 2:
            #     # Very aggressive transformation for small values (1.5-2 range)
            #     return 0.5 * mem_val  # Linear scaling for small values gives more space
            # else:
            #     # Less aggressive for larger values
            #     base = 1.0  # Value at mem_val=2
            #     return base + 0.3 * np.power(mem_val - 2, 0.5)
            return mem_val
            
        # Transform memory values for x-positions
        x_positions_memory = [transform_memory(mem) for mem in memory_sizes_GB]
        
        # Plot memory line on bottom X-axis using transformed positions
        line1 = ax.plot(x_positions_memory, rank_values_memory, marker='o', linewidth=2, markersize=8, 
                       color='blue', label=f'Rank vs Memory (Fixed Network: {fixed_upload_speed_Mbps}/{fixed_download_speed_Mbps} Mbps)')
        
        # Set bottom X-axis for memory sizes - use less rotation for better readability
        ax.set_xlabel('GPU Memory Size (GB)', fontsize=26, color='blue', labelpad=15)
        ax.set_xticks(x_positions_memory)
        # Format labels: show 2 decimals for < 1, 1 decimal for < 3, integer for >= 3
        # Use center alignment so labels are positioned directly below their data points
        ax.set_xticklabels([f'{mem:.2f}' if mem < 1 else f'{mem:.1f}' if mem < 3 else f'{int(mem)}' 
                            for mem in memory_sizes_GB], rotation=30, ha='center', fontsize=24, color='blue')
        ax.tick_params(axis='x', labelsize=24, colors='blue', pad=12)
        
        # Add more padding to x-axis limits for better label readability
        x_min_transformed = min(x_positions_memory)
        x_max_transformed = max(x_positions_memory)
        x_range = x_max_transformed - x_min_transformed
        # Increase right padding significantly to accommodate network speed labels
        left_padding = 0.2 * x_range
        right_padding = 0.4 * x_range  # Increased from 0.3 to 0.4
        x_lim_min = x_min_transformed - left_padding
        x_lim_max = x_max_transformed + right_padding
        ax.set_xlim(x_lim_min, x_lim_max)
        
        # Create top X-axis for network speeds
        ax2 = ax.twiny()  # Create a second axes that shares the same y-axis
        
        # Map network speeds to the same physical X range as transformed memory values
        # Use the FULL x-axis range (including padding) to ensure last point has space
        network_min = min(network_speeds_Mbps)
        network_max = max(network_speeds_Mbps)
        
        # Transform network speeds to transformed memory value space for plotting
        def network_to_x(network_val):
            """Transform network speed value to X position matching transformed memory range"""
            if network_max == network_min:
                return x_lim_min
            # Map network speeds to the FULL x-axis range, but leave extra space on the right
            # Use 85% of the available range to ensure last point isn't at the edge
            available_range = x_lim_max - x_lim_min
            compressed_range = available_range * 0.85  # Use 85% of available range
            left_offset = available_range * 0.10  # Start at 10% from left edge
            
            normalized = (network_val - network_min) / (network_max - network_min)
            return x_lim_min + left_offset + normalized * compressed_range
        
        network_x_positions = [network_to_x(speed) for speed in network_speeds_Mbps]
        
        # Plot network speed line on top X-axis
        line2 = ax2.plot(network_x_positions, rank_values_network, marker='s', linewidth=2, markersize=8, 
                        color='green', label=f'Rank vs Network Speed (Fixed Memory: {fixed_memory_GB} GB)')
        
        # Set top X-axis for network speeds - show all labels
        ax2.set_xlabel('Network Speed (Mbps)', fontsize=26, color='green', labelpad=15)
        ax2.set_xlim(ax.get_xlim())  # Match the X limits of the bottom axis
        ax2.set_xticks(network_x_positions)
        ax2.set_xticklabels([f'{speed:.1f}' for speed in network_speeds_Mbps], 
                           rotation=30, ha='left', fontsize=24, color='green')
        ax2.tick_params(axis='x', labelsize=24, colors='green', pad=12)
        
        # Set Y-axis label (shared)
        ax.set_ylabel('Rank Size', fontsize=26, labelpad=15)
        ax.set_ylim(0, 600)
        ax.grid(True, alpha=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=22, framealpha=0.95)
        ax.tick_params(axis='y', labelsize=24)
        
        # Set title
        #ax.set_title('Rank Size vs GPU Memory and Network Speed', fontsize=22, fontweight='bold', pad=20)
        
        # Add value labels on points for memory plot
        for i, (pos, rank) in enumerate(zip(x_positions_memory, rank_values_memory)):
            ax.annotate(f'{int(rank)}', (pos, rank), textcoords="offset points", 
                       xytext=(0,13), ha='center', fontsize=24, color='blue')
        
        # Add value labels on points for network speed plot (every other point to avoid clutter)
        for i, (pos, rank) in enumerate(zip(network_x_positions, rank_values_network)):
            #if i % 2 == 0:
                ax2.annotate(f'{int(rank)}', (pos, rank), textcoords="offset points", 
                           xytext=(0,30), ha='center', fontsize=24, color='green')
        
        # Use tight_layout with more padding to accommodate rotated labels
        plt.tight_layout(pad=4.0)
        # Additional margin adjustment for better label spacing - more room for rotated labels
        fig.subplots_adjust(bottom=0.20, top=0.82)
        
        # Save the diagram
        self._save_diagram('rank_vs_memory_and_network_speed_combined.pdf')

    def test_rank_3d_diagram(self):
        """Generate a 3D diagram showing rank size vs memory and network speed"""
        # Define ranges for memory and network speeds
        memory_sizes_GB = [1.9, 1.95, 2, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4]
        network_speeds_Mbps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        
        # Model and training configuration
        args = self._init_args()
        
        # Load model once
        model = AutoModelForImageClassification.from_pretrained(args.model)
        
        # Create meshgrid for surface plot
        X, Y = np.meshgrid(memory_sizes_GB, network_speeds_Mbps)
        Z = np.zeros_like(X)
        
        points = []
        # Calculate rank for each combination of memory and network speed
        print("Calculating rank values for 3D plot...")
        for i, memory_size_GB in enumerate(memory_sizes_GB):
            for j, network_speed_Mbps in enumerate(network_speeds_Mbps):
                args.gpu_memory_size_for_each_group_in_GB = [memory_size_GB]
                args.avg_upload_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
                args.avg_download_network_speed_for_each_group_in_Mbps = [network_speed_Mbps]
                
                rank_budgets = self.estimator.get_rank_for_all_client_groups(args, model)
                Z[j, i] = rank_budgets[0]  # Note: j is row (network), i is col (memory)
                print(f"Memory: {memory_size_GB} GB, Network: {network_speed_Mbps} Mbps, Rank: {rank_budgets[0]}")
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none', linewidth=0.1)
        
        # Plot scatter points on the surface
        # for every z, only show point at max x and max y
        points = zip(X.flatten(), Y.flatten(), Z.flatten())
        min_x_for_each_z = {}
        min_y_for_each_z = {}
        for x, y, z in points:
            if z not in min_x_for_each_z:
                min_x_for_each_z[z] = x
            if z not in min_y_for_each_z:
                min_y_for_each_z[z] = y
        # Calculate z-range for label offset
        z_range = Z.max() - Z.min()
        # Use a larger offset to ensure labels appear above the surface
        # Use both a percentage of z-range and a percentage of z value for better visibility
        base_offset = z_range * 0.03  # Base offset of 3% of range
        
        for i, z in enumerate(min_x_for_each_z):
            
            ax.scatter(min_x_for_each_z[z], min_y_for_each_z[z], z, c='red', s=100, alpha=1, marker='o')
            if i % 3 == 0:
                # Use a combination of base offset and percentage of z value
                label_z = z + base_offset + z * 0.05  # + 5% of z value
                ax.text(min_x_for_each_z[z], min_y_for_each_z[z], label_z, f'{int(z)}', 
                       fontsize=20, color='black', zorder=100,  # High zorder to ensure it's on top
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='none'))
        
        # Add contour lines on the surface
        ax.contour(X, Y, Z, zdir='z', offset=ax.get_zlim()[0], cmap='viridis', alpha=0.3)
        

        font_size = 26
        label_pad = 32
        z_label_pad = 50  # Increased padding for z-axis to avoid grid overlap
        # Set labels
        ax.set_xlabel('GPU Memory Size (GB)', fontsize=font_size, labelpad=label_pad)
        ax.set_ylabel('Network Speed (Mbps)', fontsize=font_size, labelpad=label_pad)
        ax.set_zlabel('Rank Size', fontsize=font_size, labelpad=z_label_pad)
        #ax.set_title('Rank Size vs GPU Memory and Network Speed', fontsize=14, fontweight='bold', pad=20)
        
        # Make tick labels larger
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ax.tick_params(axis='z', labelsize=font_size, pad=20)  # Add padding to z-axis ticks to avoid overlap
        
        # Reverse x-axis
        ax.invert_xaxis()
        
        # Add colorbar with larger label
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Rank Size')
        cbar.set_label('Rank Size', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)
        
        # Set viewing angle for better visualization - adjust to make slope face reader
        ax.view_init(elev=25, azim=280)
        
        plt.tight_layout()
        
        # Save the diagram
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'diagrams')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'rank_3d_diagram.pdf')
        # Save as PDF for selectable text labels (dpi parameter not needed for vector format)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"\n3D Diagram saved to: {output_path}")
        
        plt.close()


if __name__ == '__main__':
    unittest.main()


"""
Minimal Flower Server - Ultra-simple version that just works
"""
import flwr as fl
import numpy as np
import logging
import argparse
import os
import sys
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.solver.fl_utils import setup_multiprocessing


def start_minimal_flower_server(server_address="0.0.0.0", server_port=8080, num_rounds=10):
    """
    Start a minimal Flower server that just works.
    """
    # Setup multiprocessing for optimal CPU utilization
    num_cores = setup_multiprocessing()
    logging.info(f"Minimal server initialized with {num_cores} CPU cores")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start server with default strategy
    logging.info(f"Starting minimal Flower server on {server_address}:{server_port}")
    
    fl.server.start_server(
        server_address=f"{server_address}:{server_port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,  # Disable evaluation to avoid issues
            min_fit_clients=1,
            min_evaluate_clients=0,
            min_available_clients=1,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Flower Server")
    parser.add_argument("--server_address", type=str, default="0.0.0.0", help="Server address")
    parser.add_argument("--server_port", type=int, default=8080, help="Server port")
    parser.add_argument("--config_name", type=str, 
                       default="experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml",
                       help="Configuration file")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of rounds")
    
    args = parser.parse_args()
    
    # Load configuration to get number of rounds
    try:
        config_path = os.path.join('config/', args.config_name)
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if 'round' in config_dict:
            args.num_rounds = config_dict['round']
            logging.info(f"Using {args.num_rounds} rounds from config")
    except Exception as e:
        logging.warning(f"Could not load config: {e}, using default {args.num_rounds} rounds")
    
    start_minimal_flower_server(args.server_address, args.server_port, args.num_rounds)

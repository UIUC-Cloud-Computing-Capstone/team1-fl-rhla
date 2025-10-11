"""
Utility functions for the Flower client.

This module contains utility functions for configuration loading, logging setup,
device management, and argument parsing.

Author: Team1-FL-RHLA
Version: 1.0.0
"""

import argparse
import logging
import os
import torch
import yaml

from client_config import Config
from client_constants import (
    DEFAULT_CONFIG_PATH, DEFAULT_SERVER_ADDRESS, DEFAULT_SERVER_PORT,
    DEFAULT_CLIENT_ID, DEFAULT_SEED, DEFAULT_GPU_ID, DEFAULT_LOG_LEVEL,
    DEFAULT_CPU_DEVICE, DEFAULT_CUDA_DEVICE
)


def load_configuration(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError("Configuration file is empty or invalid")

    return Config(config_dict)


def setup_logging(client_id: int, log_level: str = "INFO") -> None:
    """
    Setup logging configuration for the client.
    
    Args:
        client_id: Client identifier
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f'%(asctime)s - Client {client_id} - %(levelname)s - %(message)s',
        force=True  # Override any existing logging configuration
    )


def setup_device(gpu_id: int) -> torch.device:
    """
    Setup compute device (CPU or GPU).
    
    Args:
        gpu_id: GPU ID (-1 for CPU)
        
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available() and gpu_id != -1:
        if gpu_id >= torch.cuda.device_count():
            logging.warning(f"GPU {gpu_id} not available, falling back to CPU")
            return torch.device(DEFAULT_CPU_DEVICE)
        return torch.device(f'{DEFAULT_CUDA_DEVICE}:{gpu_id}')
    return torch.device(DEFAULT_CPU_DEVICE)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
        
    Raises:
        SystemExit: If argument parsing fails
    """
    parser = argparse.ArgumentParser(
        description="Flower Client for Federated Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Connection arguments
    parser.add_argument(
        "--server_address", 
        type=str, 
        default=DEFAULT_SERVER_ADDRESS, 
        help="Server address to connect to"
    )
    parser.add_argument(
        "--server_port", 
        type=int, 
        default=DEFAULT_SERVER_PORT, 
        help="Server port to connect to"
    )
    
    # Client configuration
    parser.add_argument(
        "--client_id", 
        type=int, 
        default=DEFAULT_CLIENT_ID, 
        help="Client identifier"
    )
    parser.add_argument(
        "--config_name", 
        type=str, 
        default=DEFAULT_CONFIG_PATH,
        help="Configuration file path (relative to config/ directory)"
    )
    
    # Training configuration
    parser.add_argument(
        "--seed", 
        type=int, 
        default=DEFAULT_SEED, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=DEFAULT_GPU_ID, 
        help="GPU ID (-1 for CPU, 0+ for specific GPU)"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log_level", 
        type=str, 
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        help="Logging level"
    )
    
    # Memory management
    parser.add_argument(
        "--force_cpu", 
        action="store_true", 
        help="Force CPU usage (disable GPU/MPS for memory-constrained environments)"
    )
    
    return parser.parse_args()


def load_and_merge_config(args) -> Config:
    """Load configuration and merge with command line arguments."""
    config_path = os.path.join('config/', args.config_name)
    config = load_configuration(config_path)
    
    # Override config values with command line arguments if provided
    # Use config values as defaults for command line arguments
    if hasattr(config, 'server_address') and not hasattr(args, 'server_address'):
        args.server_address = config.server_address
    if hasattr(config, 'server_port') and not hasattr(args, 'server_port'):
        args.server_port = config.server_port
    if hasattr(config, 'client_id') and not hasattr(args, 'client_id'):
        args.client_id = config.client_id
    if hasattr(config, 'seed') and not hasattr(args, 'seed'):
        args.seed = config.seed
    if hasattr(config, 'gpu_id') and not hasattr(args, 'gpu_id'):
        args.gpu_id = config.gpu_id
    if hasattr(config, 'log_level') and not hasattr(args, 'log_level'):
        args.log_level = config.log_level
    
    # Merge command line arguments into config
    for arg_name in vars(args):
        setattr(config, arg_name, getattr(args, arg_name))
    
    # Handle force_cpu option
    if hasattr(args, 'force_cpu') and args.force_cpu:
        config.force_cpu = True
        logging.info("Force CPU mode enabled")
    
    return config


def setup_environment(config: Config, args) -> None:
    """Setup device."""
    config.device = setup_device(args.gpu)


def create_flower_client(args: Config, client_id: int = 0):
    """
    Create a Flower client instance.
    
    Args:
        args: Configuration object
        client_id: Client identifier
        
    Returns:
        FlowerClient instance
    """
    from flower_client import FlowerClient
    return FlowerClient(args, client_id)


def start_flower_client(args: Config, server_address: str = "localhost",
                              server_port: int = 8080, client_id: int = 0) -> None:
    """
    Start a Flower client.
    
    Args:
        args: Configuration object
        server_address: Server address to connect to
        server_port: Server port to connect to
        client_id: Client identifier
    """
    # Setup logging
    setup_logging(client_id)
    
    # Create client
    client = create_flower_client(args, client_id)
    
    # Start client using Flower API
    logging.info(f"Starting Flower client {client_id} connecting to {server_address}:{server_port}")
    
    # Use the modern Flower API (warnings suppressed)
    import flwr as fl
    fl.client.start_client(
        server_address=f"{server_address}:{server_port}",
        client=client.to_client(),
    )

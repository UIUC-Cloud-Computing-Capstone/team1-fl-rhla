"""
Minimal Flower Server - Ultra-simple version that just works
Refactored for improved code quality, error handling, and maintainability.
"""
import flwr as fl
import numpy as np
import logging
import argparse
import os
import sys
import yaml
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.solver.fl_utils import setup_multiprocessing


class ServerConfig:
    """Configuration class for the Flower server."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize server configuration from dictionary."""
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)


def load_server_configuration(config_path: str) -> ServerConfig:
    """
    Load server configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ServerConfig object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            raise ValueError("Configuration file is empty or invalid")
            
        return ServerConfig(config_dict)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")


def setup_server_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration for the server.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - Server - %(levelname)s - %(message)s',
        force=True  # Override any existing logging configuration
    )


def create_fedavg_strategy(num_rounds: int, 
                          fraction_fit: float = 1.0,
                          fraction_evaluate: float = 0.0,
                          min_fit_clients: int = 1,
                          min_evaluate_clients: int = 0,
                          min_available_clients: int = 1) -> fl.server.strategy.FedAvg:
    """
    Create a FedAvg strategy with specified parameters.
    
    Args:
        num_rounds: Number of training rounds
        fraction_fit: Fraction of clients used for training
        fraction_evaluate: Fraction of clients used for evaluation
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of available clients
        
    Returns:
        Configured FedAvg strategy
    """
    return fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )


def start_minimal_flower_server(server_address: str = "0.0.0.0", 
                               server_port: int = 8080, 
                               num_rounds: int = 10,
                               log_level: str = "INFO",
                               fraction_fit: float = 1.0,
                               fraction_evaluate: float = 0.0,
                               min_fit_clients: int = 1,
                               min_evaluate_clients: int = 0,
                               min_available_clients: int = 1) -> None:
    """
    Start a minimal Flower server that just works.
    
    Args:
        server_address: Server address to bind to
        server_port: Server port to bind to
        num_rounds: Number of training rounds
        log_level: Logging level
        fraction_fit: Fraction of clients used for training
        fraction_evaluate: Fraction of clients used for evaluation
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of available clients
    """
    try:
        # Setup logging first
        setup_server_logging(log_level)
        
        # Setup multiprocessing for optimal CPU utilization
        num_cores = setup_multiprocessing()
        logging.info(f"Minimal server initialized with {num_cores} CPU cores")
        
        # Validate parameters
        if num_rounds <= 0:
            raise ValueError(f"Number of rounds must be positive, got {num_rounds}")
        
        if server_port <= 0 or server_port > 65535:
            raise ValueError(f"Invalid port number: {server_port}")
        
        # Create strategy with custom parameters
        strategy = create_fedavg_strategy(
            num_rounds=num_rounds,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients
        )
        
        # Start server
        logging.info(f"Starting minimal Flower server on {server_address}:{server_port}")
        logging.info(f"Configuration: {num_rounds} rounds, fraction_fit={fraction_fit}, "
                    f"fraction_evaluate={fraction_evaluate}, min_fit_clients={min_fit_clients}")
        
        fl.server.start_server(
            server_address=f"{server_address}:{server_port}",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
        
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        raise


def parse_server_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the server.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Minimal Flower Server")
    parser.add_argument("--server_address", type=str, default="0.0.0.0", help="Server address")
    parser.add_argument("--server_port", type=int, default=8080, help="Server port")
    parser.add_argument("--config_name", type=str, 
                       default="experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml",
                       help="Configuration file")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--fraction_fit", type=float, default=1.0, 
                       help="Fraction of clients used for training")
    parser.add_argument("--fraction_evaluate", type=float, default=0.0, 
                       help="Fraction of clients used for evaluation")
    parser.add_argument("--min_fit_clients", type=int, default=1, 
                       help="Minimum number of clients for training")
    parser.add_argument("--min_evaluate_clients", type=int, default=0, 
                       help="Minimum number of clients for evaluation")
    parser.add_argument("--min_available_clients", type=int, default=1, 
                       help="Minimum number of available clients")
    
    return parser.parse_args()


def load_configuration_with_fallback(config_name: str, default_rounds: int) -> int:
    """
    Load configuration and extract number of rounds with fallback.
    
    Args:
        config_name: Configuration file name
        default_rounds: Default number of rounds if config loading fails
        
    Returns:
        Number of rounds from config or default
    """
    try:
        config_path = os.path.join('config/', config_name)
        config = load_server_configuration(config_path)
        
        rounds = config.get('round', default_rounds)
        if rounds != default_rounds:
            logging.info(f"Using {rounds} rounds from config file")
        else:
            logging.info(f"Using default {default_rounds} rounds (not specified in config)")
            
        return rounds
        
    except Exception as e:
        logging.warning(f"Could not load config: {e}, using default {default_rounds} rounds")
        return default_rounds


def validate_server_arguments(args: argparse.Namespace) -> None:
    """
    Validate server arguments.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    if args.server_port <= 0 or args.server_port > 65535:
        raise ValueError(f"Invalid port number: {args.server_port}")
    
    if args.num_rounds <= 0:
        raise ValueError(f"Number of rounds must be positive: {args.num_rounds}")
    
    if not 0.0 <= args.fraction_fit <= 1.0:
        raise ValueError(f"fraction_fit must be between 0.0 and 1.0: {args.fraction_fit}")
    
    if not 0.0 <= args.fraction_evaluate <= 1.0:
        raise ValueError(f"fraction_evaluate must be between 0.0 and 1.0: {args.fraction_evaluate}")
    
    if args.min_fit_clients < 0:
        raise ValueError(f"min_fit_clients must be non-negative: {args.min_fit_clients}")
    
    if args.min_evaluate_clients < 0:
        raise ValueError(f"min_evaluate_clients must be non-negative: {args.min_evaluate_clients}")
    
    if args.min_available_clients < 0:
        raise ValueError(f"min_available_clients must be non-negative: {args.min_available_clients}")


def main() -> None:
    """Main function to run the Flower server."""
    try:
        # Parse arguments
        args = parse_server_arguments()
        
        # Validate arguments
        validate_server_arguments(args)
        
        # Load configuration
        num_rounds = load_configuration_with_fallback(args.config_name, args.num_rounds)
        
        # Start server
        start_minimal_flower_server(
            server_address=args.server_address,
            server_port=args.server_port,
            num_rounds=num_rounds,
            log_level=args.log_level,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.min_available_clients
        )
        
    except KeyboardInterrupt:
        logging.info("Server interrupted by user")
    except Exception as e:
        logging.error(f"Server failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

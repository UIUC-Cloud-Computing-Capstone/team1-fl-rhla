"""
Flower Server Implementation

A comprehensive Flower server implementation for federated learning.
Provides robust error handling, configuration management, and support for
various federated learning strategies including FedAvg.

Author: Team1-FL-RHLA
Version: 1.0.0
"""
import flwr as fl
import numpy as np
import logging
import argparse
import os
import sys
import yaml
import torch
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.solver.fl_utils import setup_multiprocessing
from algorithms.solver.shared_utils import (
    get_data_loader_list, get_dataset_fim, test_collate_fn, 
    get_model_update, get_norm_updates, update_delta_norms, 
    get_train_loss, get_norm
)


class ServerConfig:
    """Configuration class for the Flower server."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize server configuration from dictionary."""
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)


# =============================================================================
# SERVER-SIDE FEDERATED LEARNING UTILITIES
# =============================================================================

class ServerFederatedLearningUtils:
    """
    Server-side utilities for federated learning operations.
    
    This class provides server-side implementations of functions that were
    previously unused in the client but are needed for server operations
    like aggregation, norm calculations, and data management.
    """
    
    def __init__(self, config: ServerConfig):
        """Initialize with server configuration."""
        self.config = config
        self.training_history = {
            'losses': [],
            'norms': [],
            'rounds': []
        }
    
    def create_data_loader_list(self, dataset_train, dict_users) -> List[DataLoader]:
        """
        Create data loaders for all clients (server-side data management).
        
        Args:
            dataset_train: Training dataset
            dict_users: User data partition dictionary
            
        Returns:
            List of DataLoaders for each client
        """
        return get_data_loader_list(self.config, dataset_train, dict_users)
    
    def create_dataset_fim(self, dataset_fim) -> DataLoader:
        """
        Create DataLoader for FIM dataset (server-side FIM management).
        
        Args:
            dataset_fim: FIM dataset
            
        Returns:
            DataLoader for FIM dataset
        """
        return get_dataset_fim(self.config, dataset_fim)
    
    def get_test_collate_function(self):
        """
        Get test collate function for server-side evaluation.
        
        Returns:
            Test collate function
        """
        return test_collate_fn
    
    def compute_model_updates(self, global_model_state: Dict, local_model_states: List[Dict], 
                            no_weight_lora_lists: List[List]) -> List[Dict]:
        """
        Compute model updates from multiple clients.
        
        Args:
            global_model_state: Global model state dictionary
            local_model_states: List of local model state dictionaries
            no_weight_lora_lists: List of no_weight_lora lists for each client
            
        Returns:
            List of model update dictionaries
        """
        model_updates = []
        for local_state, no_weight_lora in zip(local_model_states, no_weight_lora_lists):
            model_update = get_model_update(self.config, global_model_state, local_state, no_weight_lora)
            model_updates.append(model_update)
        return model_updates
    
    def compute_norm_updates(self, model_updates: List[Dict]) -> List[List[torch.Tensor]]:
        """
        Compute norm updates for model parameters.
        
        Args:
            model_updates: List of model update dictionaries
            
        Returns:
            List of norm updates for each client
        """
        norm_updates_list = []
        for model_update in model_updates:
            norm_updates = get_norm_updates(model_update)
            norm_updates_list.append(norm_updates)
        return norm_updates_list
    
    def update_delta_norms(self, delta_norms: List[torch.Tensor], norm_updates: List[torch.Tensor]) -> None:
        """
        Update delta norms list with new norm updates.
        
        Args:
            delta_norms: List of delta norms to update
            norm_updates: New norm updates to add
        """
        update_delta_norms(delta_norms, norm_updates)
    
    def compute_average_training_loss(self, local_losses: List[float]) -> float:
        """
        Calculate average training loss across clients.
        
        Args:
            local_losses: List of local training losses
            
        Returns:
            Average training loss
        """
        return get_train_loss(local_losses)
    
    def compute_median_norm(self, delta_norms: List[torch.Tensor]) -> float:
        """
        Calculate median norm of model updates.
        
        Args:
            delta_norms: List of delta norms
            
        Returns:
            Median norm
        """
        return get_norm(delta_norms)
    
    def aggregate_parameter_updates(self, parameter_updates: List[List[np.ndarray]], 
                                  num_examples: List[int]) -> List[np.ndarray]:
        """
        Aggregate parameter updates from multiple clients using weighted average.
        
        Args:
            parameter_updates: List of parameter updates from each client
            num_examples: Number of examples used by each client
            
        Returns:
            Aggregated parameter updates
        """
        if not parameter_updates or not num_examples:
            raise ValueError("No parameter updates or examples provided")
        
        # Calculate total examples
        total_examples = sum(num_examples)
        if total_examples == 0:
            raise ValueError("Total examples is zero")
        
        # Initialize aggregated parameters
        num_params = len(parameter_updates[0])
        aggregated_params = []
        
        for param_idx in range(num_params):
            # Calculate weighted sum for this parameter
            weighted_sum = np.zeros_like(parameter_updates[0][param_idx])
            
            for client_updates, num_samples in zip(parameter_updates, num_examples):
                weight = num_samples / total_examples
                weighted_sum += weight * client_updates[param_idx]
            
            aggregated_params.append(weighted_sum)
        
        return aggregated_params
    
    def update_training_history(self, loss: float, norm: float, round_num: int) -> None:
        """
        Update server training history.
        
        Args:
            loss: Training loss
            norm: Model update norm
            round_num: Round number
        """
        self.training_history['losses'].append(loss)
        self.training_history['norms'].append(norm)
        self.training_history['rounds'].append(round_num)
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Get current training metrics.
        
        Returns:
            Dictionary of training metrics
        """
        if not self.training_history['losses']:
            return {'loss': 0.0, 'norm': 0.0, 'round': 0}
        
        return {
            'loss': self.training_history['losses'][-1],
            'norm': self.training_history['norms'][-1],
            'round': self.training_history['rounds'][-1],
            'total_rounds': len(self.training_history['rounds'])
        }


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


class CustomFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy that passes round information to clients and uses server-side utilities.
    """
    
    def __init__(self, *args, server_utils: Optional[ServerFederatedLearningUtils] = None, **kwargs):
        """Initialize with optional server utilities."""
        super().__init__(*args, **kwargs)
        self.server_utils = server_utils
    
    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the next round of training."""
        # Get the default configuration from parent
        config = {}
        if hasattr(self, 'on_fit_config_fn') and self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Add server round to the configuration
        config["server_round"] = server_round
        
        # Get client instructions from parent
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Update each instruction with the round information
        for instruction in client_instructions:
            if hasattr(instruction, 'config'):
                if instruction.config is None:
                    instruction.config = config.copy()
                else:
                    instruction.config.update(config)
        
        return client_instructions
    
    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure the next round of evaluation."""
        # Get the default configuration from parent
        config = {}
        if hasattr(self, 'on_evaluate_config_fn') and self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        # Add server round to the configuration
        config["server_round"] = server_round
        
        # Get client instructions from parent
        client_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        
        # Update each instruction with the round information
        for instruction in client_instructions:
            if hasattr(instruction, 'config'):
                if instruction.config is None:
                    instruction.config = config.copy()
                else:
                    instruction.config.update(config)
        
        return client_instructions
    
    def aggregate_fit(self, server_round: int, results, failures):
        """
        Aggregate training results using server-side utilities.
        
        Args:
            server_round: Current server round
            results: List of (client, FitRes) tuples
            failures: List of BaseException objects
            
        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            logging.warning("No results to aggregate")
            return None, {}
        
        # Extract parameters and metrics from results
        parameters_list = []
        num_examples_list = []
        metrics_list = []
        
        for client, fit_res in results:
            if fit_res.parameters is not None:
                parameters_list.append(fit_res.parameters)
                num_examples_list.append(fit_res.num_examples)
                if fit_res.metrics:
                    metrics_list.append(fit_res.metrics)
        
        if not parameters_list:
            logging.warning("No valid parameters to aggregate")
            return None, {}
        
        # Use server utilities for aggregation if available
        if self.server_utils:
            try:
                # Aggregate parameters using server utilities
                aggregated_parameters = self.server_utils.aggregate_parameter_updates(
                    parameters_list, num_examples_list
                )
                
                # Compute aggregated metrics
                aggregated_metrics = self._compute_aggregated_metrics(metrics_list, server_round)
                
                logging.info(f"Server round {server_round}: Aggregated {len(parameters_list)} client updates")
                return aggregated_parameters, aggregated_metrics
                
            except Exception as e:
                logging.error(f"Error in server-side aggregation: {e}")
                # Fall back to default aggregation
                return super().aggregate_fit(server_round, results, failures)
        else:
            # Use default Flower aggregation
            return super().aggregate_fit(server_round, results, failures)
    
    def _compute_aggregated_metrics(self, metrics_list: List[Dict], server_round: int) -> Dict[str, Any]:
        """
        Compute aggregated metrics from client results.
        
        Args:
            metrics_list: List of client metrics
            server_round: Current server round
            
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics_list:
            return {"server_round": server_round}
        
        # Aggregate common metrics
        aggregated_metrics = {"server_round": server_round}
        
        # Average loss across clients
        losses = [m.get("loss", 0.0) for m in metrics_list if "loss" in m]
        if losses:
            aggregated_metrics["avg_loss"] = sum(losses) / len(losses)
        
        # Count clients
        aggregated_metrics["num_clients"] = len(metrics_list)
        
        # Add other metrics if available
        for key in ["learning_rate", "num_epochs"]:
            values = [m.get(key) for m in metrics_list if key in m]
            if values:
                aggregated_metrics[f"avg_{key}"] = sum(values) / len(values)
        
        return aggregated_metrics


def create_fedavg_strategy(num_rounds: int, 
                          fraction_fit: float = 1.0,
                          fraction_evaluate: float = 0.0,
                          min_fit_clients: int = 1,
                          min_evaluate_clients: int = 0,
                          min_available_clients: int = 1,
                          config: Optional[ServerConfig] = None) -> CustomFedAvgStrategy:
    """
    Create a FedAvg strategy with specified parameters and round information.
    
    Args:
        num_rounds: Number of training rounds
        fraction_fit: Fraction of clients used for training
        fraction_evaluate: Fraction of clients used for evaluation
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of available clients
        config: Server configuration
        
    Returns:
        Configured CustomFedAvgStrategy with round information and server utilities
    """
    
    def fit_config_fn(server_round: int):
        """Configuration function for fit that includes round information."""
        # Use configuration values if available, otherwise use defaults
        local_epochs = config.get('tau', 3) if config else 3
        learning_rate = config.get('local_lr', 0.01) if config else 0.01
        
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
        }
    
    def evaluate_config_fn(server_round: int):
        """Configuration function for evaluate that includes round information."""
        return {
            "server_round": server_round,
        }
    
    # Create server utilities if config is available
    server_utils = None
    if config:
        server_utils = ServerFederatedLearningUtils(config)
        logging.info("Created server-side federated learning utilities")
    
    return CustomFedAvgStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=evaluate_config_fn,
        server_utils=server_utils
    )


def start_flower_server(server_address: str = "0.0.0.0", 
                       server_port: int = 8080, 
                       num_rounds: int = 10,
                       log_level: str = "INFO",
                       fraction_fit: float = 1.0,
                       fraction_evaluate: float = 0.0,
                       min_fit_clients: int = 1,
                       min_evaluate_clients: int = 0,
                       min_available_clients: int = 1,
                       config_name: str = None) -> None:
    """
    Start a Flower server for federated learning.
    
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
        logging.info(f"Flower server initialized with {num_cores} CPU cores")
        
        # Validate parameters
        if num_rounds <= 0:
            raise ValueError(f"Number of rounds must be positive, got {num_rounds}")
        
        if server_port <= 0 or server_port > 65535:
            raise ValueError(f"Invalid port number: {server_port}")
        
        # Load configuration if provided
        server_config = None
        if config_name:
            try:
                config_path = os.path.join('config/', config_name)
                server_config = load_server_configuration(config_path)
                logging.info(f"Loaded server configuration from: {config_path}")
            except Exception as e:
                logging.warning(f"Could not load server configuration: {e}, using defaults")
        
        # Create strategy with custom parameters
        strategy = create_fedavg_strategy(
            num_rounds=num_rounds,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            config=server_config
        )
        
        # Start server
        logging.info(f"Starting Flower server on {server_address}:{server_port}")
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
    parser = argparse.ArgumentParser(description="Flower Server for Federated Learning")
    parser.add_argument("--server_address", type=str, default="0.0.0.0", help="Server address")
    parser.add_argument("--server_port", type=int, default=8080, help="Server port")
    parser.add_argument("--config_name", type=str, 
                       default="experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml",
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
        start_flower_server(
            server_address=args.server_address,
            server_port=args.server_port,
            num_rounds=num_rounds,
            log_level=args.log_level,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.min_available_clients,
            config_name=args.config_name
        )
        
    except KeyboardInterrupt:
        logging.info("Server interrupted by user")
    except Exception as e:
        logging.error(f"Server failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

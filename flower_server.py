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
from typing import Dict, Any, Optional, List, Tuple

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


# =============================================================================
# SERVER-SIDE FEDERATED LEARNING UTILITIES
# =============================================================================

class ServerFederatedLearningUtils:
    """
    Server-side utilities for federated learning operations.
    
    This class provides server-side parameter aggregation functionality.
    """
    
    def __init__(self, config: ServerConfig):
        """Initialize with server configuration."""
        self.config = config
    
    
    def aggregate_parameter_updates(self, parameter_updates: List[List[np.ndarray]], 
                                  num_examples: List[int], 
                                  client_metrics: List[Dict] = None) -> List[np.ndarray]:
        """
        Aggregate parameter updates from multiple clients using weighted average.
        Supports both full model updates and heterogeneous LoRA updates.
        
        Args:
            parameter_updates: List of parameter updates from each client
            num_examples: Number of examples used by each client
            client_metrics: List of client metrics containing LoRA metadata
            
        Returns:
            Aggregated parameter updates
        """
        if not parameter_updates or not num_examples:
            raise ValueError("No parameter updates or examples provided")
        
        # Calculate total examples
        total_examples = sum(num_examples)
        if total_examples == 0:
            raise ValueError("Total examples is zero")
        
        # Check if we have LoRA metadata from clients
        has_lora_metadata = (client_metrics is not None and 
                           any('lora_trained_layers' in metrics for metrics in client_metrics))
        
        if has_lora_metadata:
            logging.info("Detected LoRA updates with metadata, using heterogeneous aggregation")
            return self._aggregate_heterogeneous_lora_parameters(parameter_updates, num_examples, client_metrics)
        else:
            
            first_client_params = len(parameter_updates[0])
            
            logging.info(f"Detected full model updates: {first_client_params} parameters per client")
            return self._aggregate_full_model_parameters(parameter_updates, num_examples)
    
    def _aggregate_full_model_parameters(self, parameter_updates: List[List[np.ndarray]], 
                                       num_examples: List[int]) -> List[np.ndarray]:
        """Aggregate full model parameters using weighted average."""
        # Initialize aggregated parameters
        num_params = len(parameter_updates[0])
        aggregated_params = []
        total_examples = sum(num_examples)
        
        for param_idx in range(num_params):
            # Calculate weighted sum for this parameter
            weighted_sum = np.zeros_like(parameter_updates[0][param_idx])
            
            for client_updates, num_samples in zip(parameter_updates, num_examples):
                weight = num_samples / total_examples
                weighted_sum += weight * client_updates[param_idx]
            
            aggregated_params.append(weighted_sum)
        
        return aggregated_params
    
    def _aggregate_lora_parameters(self, parameter_updates: List[List[np.ndarray]], 
                                 num_examples: List[int]) -> List[np.ndarray]:
        """
        Aggregate LoRA parameters using weighted average.
        For LoRA without metadata, we assume all clients send the same set of parameters.
        """
        # For LoRA aggregation without metadata, we can use the same weighted average approach
        # since all clients should be sending the same LoRA parameters
        return self._aggregate_full_model_parameters(parameter_updates, num_examples)
    
    def _aggregate_heterogeneous_lora_parameters(self, parameter_updates: List[List[np.ndarray]], 
                                               num_examples: List[int], 
                                               client_metrics: List[Dict]) -> List[np.ndarray]:
        """
        Aggregate heterogeneous LoRA parameters from clients that trained different subsets of layers.
        
        Args:
            parameter_updates: List of parameter updates from each client
            num_examples: Number of examples used by each client
            client_metrics: List of client metrics containing LoRA metadata
            
        Returns:
            Aggregated parameter updates
        """
        # Group parameters by layer and parameter name
        layer_param_groups = {}
        client_weights = {}
        total_examples = sum(num_examples)
        
        # Process each client's parameters and metadata
        for client_idx, (params, num_samples, metrics) in enumerate(zip(parameter_updates, num_examples, client_metrics)):
            if 'lora_trained_layers' not in metrics:
                logging.warning(f"Client {client_idx} missing LoRA metadata, skipping")
                continue
                
            # Parse trained layers from string format
            trained_layers_str = metrics.get('lora_trained_layers', "")
            trained_layers = [int(x) for x in trained_layers_str.split(',') if x.strip()] if trained_layers_str else []
            param_count = metrics.get('lora_param_count', len(params))
            
            client_weight = num_samples / total_examples
            client_weights[client_idx] = client_weight
            
            # Group parameters by layer (simplified approach without parameter names)
            # Since we don't have parameter names, we'll group by layer and parameter index
            for param_idx, param in enumerate(params):
                # For now, we'll use a simplified grouping approach
                # Each parameter gets its own group based on its index
                key = f"param_{param_idx}"
                
                if key not in layer_param_groups:
                    layer_param_groups[key] = []
                
                layer_param_groups[key].append({
                    'param': param,
                    'client_idx': client_idx,
                    'weight': client_weight,
                    'num_samples': num_samples,
                    'layer_info': trained_layers  # Store layer info for reference
                })
        
        # Aggregate parameters that were trained by multiple clients
        aggregated_params = []
        aggregation_stats = {
            'total_params': 0,
            'aggregated_params': 0,
            'single_client_params': 0,
            'layers_aggregated': set(),
            'layers_single_client': set()
        }
        
        for param_key, param_group in layer_param_groups.items():
            aggregation_stats['total_params'] += 1
            
            if len(param_group) > 1:
                # Multiple clients trained this parameter - aggregate
                weighted_sum = np.zeros_like(param_group[0]['param'])
                total_weight = 0
                
                for param_info in param_group:
                    weight = param_info['weight']
                    weighted_sum += weight * param_info['param']
                    total_weight += weight
                
                if total_weight > 0:
                    aggregated_param = weighted_sum / total_weight
                    aggregated_params.append(aggregated_param)
                    aggregation_stats['aggregated_params'] += 1
                    # Add layer info from any client (they should be the same for the same parameter)
                    if param_group[0]['layer_info']:
                        aggregation_stats['layers_aggregated'].update(param_group[0]['layer_info'])
                    
                    logging.debug(f"Aggregated parameter {param_key} from {len(param_group)} clients")
            else:
                # Only one client trained this parameter - include it as-is
                aggregated_params.append(param_group[0]['param'])
                aggregation_stats['single_client_params'] += 1
                # Add layer info from the single client
                if param_group[0]['layer_info']:
                    aggregation_stats['layers_single_client'].update(param_group[0]['layer_info'])
                
                logging.debug(f"Single client parameter {param_key} from client {param_group[0]['client_idx']}")
        
        # Log aggregation statistics
        logging.info(f"LoRA aggregation complete:")
        logging.info(f"  - Total parameters: {aggregation_stats['total_params']}")
        logging.info(f"  - Aggregated parameters: {aggregation_stats['aggregated_params']}")
        logging.info(f"  - Single client parameters: {aggregation_stats['single_client_params']}")
        logging.info(f"  - Layers aggregated: {sorted(aggregation_stats['layers_aggregated'])}")
        logging.info(f"  - Layers single client: {sorted(aggregation_stats['layers_single_client'])}")
        
        return aggregated_params
    


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
                # Aggregate parameters using server utilities with client metrics
                aggregated_parameters = self.server_utils.aggregate_parameter_updates(
                    parameters_list, num_examples_list, metrics_list
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

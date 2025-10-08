"""
Simplified Flower Client for Testing

This module provides a simplified Flower client implementation that can be used
for testing federated learning scenarios without requiring actual dataset loading.
It supports configuration-driven setup and can simulate training and evaluation.

Author: Team1-FL-RHLA
Version: 1.0.0
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.solver.fl_utils import (
    compute_model_update,
    compute_update_norm,
    get_optimizer_parameters,
    setup_multiprocessing
)

# Optional import for dataset loading
try:
    from utils.data_pre_process import load_partition
    DATASET_LOADING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Dataset loading not available: {e}")
    DATASET_LOADING_AVAILABLE = False

# Constants
DEFAULT_CONFIG_PATH = "experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml"
DEFAULT_SERVER_ADDRESS = "localhost"
DEFAULT_SERVER_PORT = 8080
DEFAULT_CLIENT_ID = 0
DEFAULT_SEED = 1
DEFAULT_GPU_ID = -1
DEFAULT_LOG_LEVEL = "INFO"

# Model architecture constants
VIT_BASE_HIDDEN_SIZE = 768
VIT_LARGE_HIDDEN_SIZE = 1024
LORA_RANK = 64

# Dataset constants
DATASET_CONFIGS = {
    'cifar100': {'num_classes': 100, 'data_type': 'image'},
    'sst2': {'num_classes': 2, 'data_type': 'text'},
    'qqp': {'num_classes': 2, 'data_type': 'text'},
    'qnli': {'num_classes': 2, 'data_type': 'text'},
    'ledgar': {'num_classes': 2, 'data_type': 'text'},
    'belebele': {'num_classes': 4, 'data_type': 'text'},
}

# Training simulation constants
DEFAULT_TRAINING_STEPS_PER_EPOCH = 10
DEFAULT_MIN_LEARNING_RATE = 0.001
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_LOCAL_EPOCHS = 1


class SimpleFlowerClient(fl.client.NumPyClient):
    """
    Simplified Flower client for testing without data dependencies.
    """
    
    def __init__(self, args: 'Config', client_id: int = 0):
        """
        Initialize the simplified Flower client.
        
        Args:
            args: Configuration object
            client_id: Client identifier
        """
        self.args = args
        self.client_id = client_id
        
        # Validate required configuration
        self._validate_config()
        
        # Setup multiprocessing for optimal CPU utilization
        self.num_cores = setup_multiprocessing()
        logging.info(f"Client {client_id} initialized with {self.num_cores} CPU cores")
        
        # Initialize loss function based on data type
        self.loss_func = self._get_loss_function()
        
        # Initialize training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'rounds': []
        }
        
        # Load dataset configuration and data
        self.dataset_info = self._load_dataset_config()
        
        # Create dummy model parameters for testing
        self.model_params = self._create_dummy_model_params()
    
    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = ['data_type', 'peft', 'lora_layer', 'dataset', 'model']
        for param in required_params:
            if not hasattr(self.args, param):
                logging.warning(f"Missing configuration parameter: {param}, using default")
    
    def _get_loss_function(self) -> nn.Module:
        """Get appropriate loss function based on data type."""
        data_type = getattr(self.args, 'data_type', 'image')
        
        loss_functions = {
            'image': nn.CrossEntropyLoss(),
            'text': nn.CrossEntropyLoss(),
            'sentiment': nn.NLLLoss()
        }
        
        return loss_functions.get(data_type, nn.CrossEntropyLoss())
    
    def _load_dataset_config(self) -> Dict[str, Any]:
        """
        Load dataset configuration and prepare dataset information.
        
        Returns:
            Dictionary containing dataset information with the following keys:
            - dataset_name: Name of the dataset
            - data_type: Type of data (image/text)
            - model_name: Name of the model architecture
            - batch_size: Batch size for training
            - num_classes: Number of output classes
            - labels: Dataset labels (if available)
            - label2id: Label to ID mapping (if available)
            - id2label: ID to label mapping (if available)
            - data_loaded: Whether actual data was loaded
            
        Raises:
            ValueError: If dataset configuration is invalid
        """
        dataset_name = self.args.get('dataset', 'cifar100')
        
        # Initialize dataset info with defaults
        dataset_info = {
            'dataset_name': dataset_name,
            'data_type': self.args.get('data_type', 'image'),
            'model_name': self.args.get('model', 'google/vit-base-patch16-224-in21k'),
            'batch_size': self.args.get('batch_size', 128),
            'num_classes': None,
            'labels': None,
            'label2id': None,
            'id2label': None,
            'data_loaded': False
        }
        
        try:
            # Validate dataset name
            if not dataset_name or not isinstance(dataset_name, str):
                raise ValueError(f"Invalid dataset name: {dataset_name}")
            
            logging.info(f"Loading dataset configuration for: {dataset_name}")
            
            # Set dataset-specific configuration
            if dataset_name in DATASET_CONFIGS:
                config = DATASET_CONFIGS[dataset_name]
                dataset_info['num_classes'] = config['num_classes']
                dataset_info['data_type'] = config['data_type']
            else:
                logging.warning(f"Unknown dataset: {dataset_name}, using defaults")
                dataset_info['num_classes'] = 100  # Default fallback
                dataset_info['data_type'] = 'image'
            
            # Try to load actual dataset if available
            dataset_info['data_loaded'] = self._try_load_dataset(dataset_name)
            
            logging.info(f"Dataset configuration loaded: {dataset_name} "
                        f"({dataset_info['num_classes']} classes, {dataset_info['data_type']} data)")
            
        except Exception as e:
            logging.error(f"Failed to load dataset configuration: {e}")
            # Set safe defaults
            dataset_info.update({
                'num_classes': 100,
                'data_type': 'image',
                'data_loaded': False
            })
        
        return dataset_info
    
    def _try_load_dataset(self, dataset_name: str) -> bool:
        """
        Attempt to load actual dataset data.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            True if dataset was loaded successfully, False otherwise
        """
        if not DATASET_LOADING_AVAILABLE:
            logging.info("Dataset loading not available, using configuration only")
            return False
        
        try:
            # This is a simplified version - in practice you'd call load_partition
            # but for the simple client, we'll just store the configuration
            logging.info(f"Dataset loading capability available for: {dataset_name}")
            return True
            
        except Exception as e:
            logging.warning(f"Could not load actual dataset data: {e}")
            logging.info("Using dataset configuration only (no actual data loaded)")
            return False
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information for external use.
        
        Returns:
            Dictionary containing dataset information
        """
        return self.dataset_info.copy()
    
    def _create_dummy_model_params(self) -> List[np.ndarray]:
        """
        Create dummy model parameters for testing based on dataset configuration.
        
        Returns:
            List of numpy arrays representing model parameters
            
        Raises:
            ValueError: If model configuration is invalid
        """
        try:
            # Get dataset and model information
            num_classes = self.dataset_info.get('num_classes', 100)
            model_name = self.dataset_info.get('model_name', 'google/vit-base-patch16-224-in21k')
            
            # Validate inputs
            if num_classes <= 0:
                raise ValueError(f"Invalid number of classes: {num_classes}")
            
            # Determine hidden size based on model architecture
            hidden_size = self._get_model_hidden_size(model_name)
            
            # Create base model parameters
            params = self._create_base_model_params(hidden_size, num_classes)
            
            # Add LoRA parameters if specified
            if self.args.get('peft') == 'lora':
                lora_params = self._create_lora_params(hidden_size)
                params.extend(lora_params)
            
            logging.info(f"Created {len(params)} dummy model parameters for {num_classes} classes "
                        f"(hidden_size={hidden_size}, model={model_name})")
            return params
            
        except Exception as e:
            logging.error(f"Failed to create dummy model parameters: {e}")
            # Return minimal parameters as fallback
            return [np.random.randn(100, 100).astype(np.float32)]
    
    def _get_model_hidden_size(self, model_name: str) -> int:
        """
        Get hidden size based on model architecture.
        
        Args:
            model_name: Name of the model architecture
            
        Returns:
            Hidden size for the model
        """
        if 'vit-base' in model_name.lower():
            return VIT_BASE_HIDDEN_SIZE
        elif 'vit-large' in model_name.lower():
            return VIT_LARGE_HIDDEN_SIZE
        else:
            logging.warning(f"Unknown model architecture: {model_name}, using default hidden size")
            return VIT_BASE_HIDDEN_SIZE
    
    def _create_base_model_params(self, hidden_size: int, num_classes: int) -> List[np.ndarray]:
        """
        Create base model parameters.
        
        Args:
            hidden_size: Hidden layer size
            num_classes: Number of output classes
            
        Returns:
            List of base model parameters
        """
        params = []
        
        # Hidden layer weights and bias
        params.append(np.random.randn(hidden_size, hidden_size).astype(np.float32))
        params.append(np.random.randn(hidden_size).astype(np.float32))
        
        # Output layer weights and bias
        params.append(np.random.randn(num_classes, hidden_size).astype(np.float32))
        params.append(np.random.randn(num_classes).astype(np.float32))
        
        return params
    
    def _create_lora_params(self, hidden_size: int) -> List[np.ndarray]:
        """
        Create LoRA parameters.
        
        Args:
            hidden_size: Hidden layer size
            
        Returns:
            List of LoRA parameters
        """
        lora_layers = self.args.get('lora_layer', 12)
        
        # Validate LoRA layers
        if lora_layers <= 0:
            logging.warning(f"Invalid lora_layer value: {lora_layers}, using default of 12")
            lora_layers = 12
        
        params = []
        for _ in range(lora_layers):
            # LoRA A and B matrices
            params.append(np.random.randn(LORA_RANK, hidden_size).astype(np.float32))  # LoRA A
            params.append(np.random.randn(hidden_size, LORA_RANK).astype(np.float32))  # LoRA B
        
        logging.info(f"Created {lora_layers} LoRA layer pairs (rank={LORA_RANK})")
        return params
    
    def _simulate_training(self, local_epochs: int, learning_rate: float, server_round: int) -> float:
        """
        Simulate training process.
        
        Args:
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for training
            server_round: Current server round (for loss simulation)
            
        Returns:
            Total training loss
        """
        total_loss = 0.0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            
            # Simulate training steps
            for step in range(DEFAULT_TRAINING_STEPS_PER_EPOCH):
                # Simulate forward pass and loss calculation
                # Loss decreases over rounds and has some randomness
                base_loss = max(0.1, 2.0 - server_round * 0.01)
                loss = base_loss + np.random.exponential(0.5)
                epoch_loss += loss
                
                # Simulate parameter updates
                self._update_parameters(learning_rate)
            
            total_loss += epoch_loss / DEFAULT_TRAINING_STEPS_PER_EPOCH
        
        return total_loss
    
    def _update_parameters(self, learning_rate: float) -> None:
        """
        Simulate parameter updates during training.
        
        Args:
            learning_rate: Learning rate for updates
        """
        for i, param in enumerate(self.model_params):
            # Add small random update scaled by learning rate
            update_scale = 0.01 * learning_rate
            update = np.random.normal(0, update_scale, param.shape).astype(param.dtype)
            self.model_params[i] = param + update
    
    def _simulate_evaluation_metrics(self, server_round: int) -> Tuple[float, float]:
        """
        Simulate evaluation metrics (accuracy and loss).
        
        Args:
            server_round: Current server round for metric simulation
            
        Returns:
            Tuple of (accuracy, loss)
        """
        # Simulate accuracy (should improve over rounds)
        base_accuracy = 0.5 + min(0.4, server_round * 0.01)
        accuracy = base_accuracy + np.random.normal(0, 0.05)
        accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]
        
        # Simulate loss (should decrease over rounds)
        base_loss = 2.0 - min(1.5, server_round * 0.01)
        loss = base_loss + np.random.normal(0, 0.1)
        loss = max(0.1, loss)  # Clamp to positive values
        
        return accuracy, loss
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get current model parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of model parameters as numpy arrays
        """
        return self.model_params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server.
        
        Args:
            parameters: List of model parameters from server
        """
        try:
            if not parameters:
                logging.warning("Received empty parameters from server")
                return
                
            self.model_params = [param.copy() for param in parameters]
            logging.debug(f"Updated model parameters with {len(parameters)} parameter arrays")
        except Exception as e:
            logging.error(f"Failed to set parameters: {e}")
            raise
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train the model on local data (simulated).
        
        Args:
            parameters: Model parameters from server
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        try:
            # Set parameters from server
            self.set_parameters(parameters)
            
            # Extract and validate configuration
            server_round = config.get('server_round', 0)
            local_epochs = max(1, config.get('local_epochs', self.args.get('tau', DEFAULT_LOCAL_EPOCHS)))
            learning_rate = max(DEFAULT_MIN_LEARNING_RATE, 
                              config.get('learning_rate', self.args.get('local_lr', DEFAULT_LEARNING_RATE)))
            
            logging.info(f"Client {self.client_id} starting training for round {server_round} "
                        f"(epochs={local_epochs}, lr={learning_rate:.4f})")
            
            # Simulate training
            total_loss = self._simulate_training(local_epochs, learning_rate, server_round)
            
            avg_loss = total_loss / local_epochs
            
            # Update training history
            self.training_history['losses'].append(avg_loss)
            self.training_history['rounds'].append(server_round)
            
            logging.info(f"Client {self.client_id} completed training: Loss={avg_loss:.4f}")
            
            # Return parameters, number of examples, and metrics
            metrics = {
                'loss': avg_loss,
                'num_epochs': local_epochs,
                'client_id': self.client_id,
                'learning_rate': learning_rate
            }
            
            # Simulate number of examples based on dataset
            if self.dataset_info.get('data_loaded', False):
                # If we have actual data, use a more realistic number
                num_examples = np.random.randint(500, 2000)
            else:
                # For dummy data, use a smaller range
                num_examples = np.random.randint(100, 1000)
            
            return self.get_parameters(config), num_examples, metrics
            
        except Exception as e:
            logging.error(f"Training failed for client {self.client_id}: {e}")
            # Return original parameters as fallback
            return parameters, 0, {'loss': float('inf'), 'error': str(e)}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate the model on local test data (simulated).
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        try:
            # Set parameters from server
            self.set_parameters(parameters)
            
            # Extract server round from config
            server_round = config.get('server_round', 0)
            
            # Simulate evaluation metrics
            accuracy, loss = self._simulate_evaluation_metrics(server_round)
            
            # Update training history
            self.training_history['accuracies'].append(accuracy)
            
            logging.info(f"Client {self.client_id} evaluation: "
                        f"Loss={loss:.4f}, Accuracy={accuracy:.4f}")
            
            # Return loss, number of examples, and metrics
            metrics = {
                'accuracy': accuracy,
                'client_id': self.client_id
            }
            
            # Simulate number of test examples based on dataset
            if self.dataset_info.get('data_loaded', False):
                # If we have actual data, use a more realistic number
                num_examples = np.random.randint(100, 500)
            else:
                # For dummy data, use a smaller range
                num_examples = np.random.randint(50, 200)
            
            return loss, num_examples, metrics
            
        except Exception as e:
            logging.error(f"Evaluation failed for client {self.client_id}: {e}")
            # Return high loss as fallback
            return float('inf'), 0, {'accuracy': 0.0, 'error': str(e)}


class Config:
    """
    Configuration class to hold and manage configuration parameters.
    
    This class provides a clean interface for accessing configuration parameters
    with proper validation and default value handling.
    """
    
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Raises:
            ValueError: If config_dict is None or empty
        """
        if not config_dict:
            raise ValueError("Configuration dictionary cannot be None or empty")
            
        for key, value in config_dict.items():
            if not isinstance(key, str):
                raise ValueError(f"Configuration keys must be strings, got {type(key)}")
            setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default.
        
        Args:
            key: Configuration parameter name
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return getattr(self, key, default)
    
    def has(self, key: str) -> bool:
        """
        Check if configuration parameter exists.
        
        Args:
            key: Configuration parameter name
            
        Returns:
            True if parameter exists, False otherwise
        """
        return hasattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self.to_dict()})"


def load_configuration(config_path: str) -> 'Config':
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
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            raise ValueError("Configuration file is empty or invalid")
            
        return Config(config_dict)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")


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


def create_simple_flower_client(args: 'Config', client_id: int = 0) -> SimpleFlowerClient:
    """
    Create a simplified Flower client instance.
    
    Args:
        args: Configuration object
        client_id: Client identifier
        
    Returns:
        SimpleFlowerClient instance
    """
    return SimpleFlowerClient(args, client_id)


def start_simple_flower_client(args: 'Config', server_address: str = "localhost", 
                              server_port: int = 8080, client_id: int = 0) -> None:
    """
    Start a simplified Flower client.
    
    Args:
        args: Configuration object
        server_address: Server address to connect to
        server_port: Server port to connect to
        client_id: Client identifier
    """
    try:
        # Setup logging
        setup_logging(client_id)
        
        # Create client
        client = create_simple_flower_client(args, client_id)
        
        # Start client using modern Flower API
        logging.info(f"Starting simplified Flower client {client_id} connecting to {server_address}:{server_port}")
        
        fl.client.start_client(
            server_address=f"{server_address}:{server_port}",
            client=client.to_client(),
        )
    except Exception as e:
        logging.error(f"Failed to start client {client_id}: {e}")
        raise


def setup_random_seeds(seed: int, client_id: int) -> None:
    """
    Setup random seeds for reproducibility.
    
    Args:
        seed: Base seed value
        client_id: Client identifier for unique seeds
    """
    client_seed = seed + client_id
    torch.manual_seed(client_seed)
    np.random.seed(client_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            return torch.device('cpu')
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
        
    Raises:
        SystemExit: If argument parsing fails
    """
    parser = argparse.ArgumentParser(
        description="Simplified Flower Client for Testing",
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
    
    return parser.parse_args()


def main() -> None:
    """Main function to run the Flower client."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration
        config_path = os.path.join('config/', args.config_name)
        config = load_configuration(config_path)
        
        # Merge command line arguments into config
        for arg_name in vars(args):
            setattr(config, arg_name, getattr(args, arg_name))
        
        # Setup device
        config.device = setup_device(args.gpu)
        
        # Setup random seeds
        setup_random_seeds(args.seed, args.client_id)
        
        # Start client
        start_simple_flower_client(
            config, 
            args.server_address, 
            args.server_port, 
            args.client_id
        )
        
    except KeyboardInterrupt:
        logging.info("Client interrupted by user")
    except Exception as e:
        logging.error(f"Client failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

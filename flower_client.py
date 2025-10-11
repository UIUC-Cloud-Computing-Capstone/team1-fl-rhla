"""
Flower Client Implementation

This module provides a comprehensive Flower client implementation for federated learning
scenarios. It supports configuration-driven setup, dataset loading, and actual
training and evaluation with real data. The client is designed to work with various
datasets and model architectures including LoRA fine-tuning.

Author: Team1-FL-RHLA
Version: 1.0.0
"""

# Standard library imports
import logging
import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional, Any

# Third-party imports
import flwr as fl
import numpy as np
import torch
import torch.nn as nn

# Suppress Flower deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from algorithms.solver.fl_utils import setup_multiprocessing
from utils.model_utils import model_setup

# Import our modular components
from client_constants import *
from client_config import Config
from client_training import ClientTrainingMixin
from client_evaluation import ClientEvaluationMixin
from client_data_loading import ClientDataLoadingMixin
from client_utils import (start_flower_client, parse_arguments, load_and_merge_config, setup_environment)

# Constants
DATASET_LOADING_AVAILABLE = True


class FlowerClient(fl.client.NumPyClient, ClientTrainingMixin, ClientEvaluationMixin, ClientDataLoadingMixin):
    """
    Comprehensive Flower client for federated learning with dataset support.
    
    This client provides a full-featured implementation that supports:
    - Configuration-driven setup from YAML files
    - Dataset loading and management with non-IID data distribution
    - Model parameter creation based on architecture (ViT, LoRA)
    - Actual training and evaluation with real data
    - LoRA fine-tuning support for efficient parameter updates
    
    The client handles the complete federated learning workflow including:
    local training, parameter aggregation, and evaluation on test data.
    
    Example:
        config = Config({CONFIG_KEY_DATASET: DEFAULT_DATASET, CONFIG_KEY_MODEL: DEFAULT_MODEL})
        client = FlowerClient(config, client_id=DEFAULT_CLIENT_ID)
    """
    
    def __init__(self, args: Config, client_id: int = 0):
        """
        Initialize the Flower client.
        
        Args:
            args: Configuration object containing dataset, model, and training parameters
            client_id: Client identifier for this federated learning client
        """
        self.args = args
        self.client_id = client_id
        
        # Validate required configuration
        self._validate_config()
        
        # Setup multiprocessing for optimal CPU utilization
        num_cores = setup_multiprocessing()
        logging.info(LOG_CLIENT_INITIALIZED.format(client_id=client_id, num_cores=num_cores))
        
        # Initialize loss function based on data type (will be created when needed)
        self._loss_func = None
        
        # Initialize training history
        self.training_history = {
            CONFIG_KEY_LOSSES: [],
            CONFIG_KEY_ACCURACIES: [],
            CONFIG_KEY_ROUNDS: []
        }
        
        # Load dataset configuration and data
        self.dataset_info = self._load_dataset_config()
        
        # Initialize heterogeneous group configuration if needed
        self._initialize_heterogeneous_config()
        
        # Create actual model
        self.model = self._create_actual_model()
    
    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [CONFIG_KEY_DATA_TYPE, CONFIG_KEY_PEFT, CONFIG_KEY_LORA_LAYER, CONFIG_KEY_DATASET, CONFIG_KEY_MODEL]
        for param in required_params:
            if not hasattr(self.args, param): 
                logging.warning(f"Missing configuration parameter: {param}, using default")
    
    def _get_loss_function(self) -> nn.Module:
        """Get appropriate loss function based on data type."""
        # Create loss function on demand to avoid storing as instance variable
        if self._loss_func is None:
            data_type = getattr(self.args, CONFIG_KEY_DATA_TYPE, DEFAULT_IMAGE_DATA_TYPE)
            
            loss_functions = {
                DEFAULT_IMAGE_DATA_TYPE: nn.CrossEntropyLoss(),
                DEFAULT_TEXT_DATA_TYPE: nn.CrossEntropyLoss(),
                DEFAULT_SENTIMENT_DATA_TYPE: nn.NLLLoss()
            }
            
            self._loss_func = loss_functions.get(data_type, nn.CrossEntropyLoss())
        
        return self._loss_func
    
    
    def _create_actual_model(self):
        """
        Create actual model using shared model_setup function.
        
        Returns:
            PyTorch model instance
            
        Raises:
            ValueError: If model configuration is invalid
        """
        # Ensure required attributes are present for model_setup
        self._ensure_model_setup_attributes()
        
        # Use shared model_setup function for consistency with original implementation
        _, model, _, model_dim = model_setup(self.args)
        
        # Update args with model dimension
        self.args.dim = model_dim
        
        logging.info(f"Created model using shared model_setup: {model_dim} dimensions "
                    f"(model={self.args.get(CONFIG_KEY_MODEL, DEFAULT_UNKNOWN_VALUE)}, peft={self.args.get(CONFIG_KEY_PEFT, DEFAULT_NONE_VALUE)})")
        return model
    
    def _ensure_model_setup_attributes(self):
        """Ensure required attributes are present for model_setup function."""
        # Add missing attributes that model_setup expects
        if not hasattr(self.args, CONFIG_KEY_LABEL2ID):
            num_classes = self.dataset_info.get(CONFIG_KEY_NUM_CLASSES, DEFAULT_NUM_CLASSES)
            self.args.label2id = {f"{DEFAULT_CLASS_PREFIX}{i}": i for i in range(num_classes)}
        
        if not hasattr(self.args, CONFIG_KEY_ID2LABEL):
            num_classes = self.dataset_info.get(CONFIG_KEY_NUM_CLASSES, DEFAULT_NUM_CLASSES)
            self.args.id2label = {i: f"{DEFAULT_CLASS_PREFIX}{i}" for i in range(num_classes)}
        
        if not hasattr(self.args, CONFIG_KEY_LOGGER):
            self.args.logger = self._create_simple_logger()
        
        if not hasattr(self.args, CONFIG_KEY_ACCELERATOR):
            # Create a simple accelerator-like object for compatibility
            class SimpleAccelerator:
                def is_local_main_process(self):
                    return True
            self.args.accelerator = SimpleAccelerator()
        
        if not hasattr(self.args, CONFIG_KEY_LOG_PATH):
            self.args.log_path = f"{DEFAULT_LOG_PATH_PREFIX}{self.client_id}"
        
        # Set device with memory management
        device = self._get_optimal_device()
        
        # Ensure other required attributes
        required_attrs = {
            CONFIG_KEY_DEVICE: device,
            CONFIG_KEY_SEED: getattr(self.args, CONFIG_KEY_SEED, DEFAULT_SEED),
            CONFIG_KEY_GPU_ID: getattr(self.args, CONFIG_KEY_GPU_ID, DEFAULT_GPU_ID),
        }
        
        for attr, default_value in required_attrs.items():
            if not hasattr(self.args, attr):
                setattr(self.args, attr, default_value)
    
    def _get_optimal_device(self):
        """Get optimal device with memory management."""
        # Force CPU for memory-constrained environments
        if hasattr(self.args, CONFIG_KEY_FORCE_CPU) and self.args.force_cpu:
            logging.info("Forcing CPU usage due to configuration")
            return torch.device(DEFAULT_CPU_DEVICE)
        
        # Check for MPS (Metal Performance Shaders) on macOS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Set MPS memory management
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = DEFAULT_PYTORCH_MPS_RATIO
            logging.info("Using MPS with memory management enabled")
            return torch.device(DEFAULT_MPS_DEVICE)
        
        # Check for CUDA
        if torch.cuda.is_available():
            # Set CUDA memory management
            torch.cuda.empty_cache()
            logging.info("Using CUDA with memory management enabled")
            return torch.device(DEFAULT_CUDA_DEVICE)
        
        # Fallback to CPU
        logging.info("Using CPU device")
        return torch.device(DEFAULT_CPU_DEVICE)
    
    def _create_simple_logger(self):
        """Create a simple logger for compatibility."""
        class SimpleLogger:
            def info(self, msg, main_process_only=False):
                logging.info(msg)
            def warning(self, msg, main_process_only=False):
                logging.warning(msg)
            def error(self, msg, main_process_only=False):
                logging.error(msg)
        return SimpleLogger()
    
    def _model_to_numpy_params(self) -> List[np.ndarray]:
        """Convert model parameters to numpy arrays."""
        params = []
        for param in self.model.parameters():
            params.append(param.detach().cpu().numpy())
        return params
    
    def _numpy_params_to_model(self, params: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        param_idx = 0
        for param in self.model.parameters():
            if param_idx < len(params):
                param.data = torch.from_numpy(params[param_idx]).to(param.device)
                param_idx += 1
    
    def _initialize_heterogeneous_config(self) -> None:
        """Initialize heterogeneous group configuration if needed."""
        # Only initialize if we have heterogeneous group configuration
        if hasattr(self.args, CONFIG_KEY_HETEROGENEOUS_GROUP):
            from algorithms.solver.shared_utils import get_group_cnt, update_user_groupid_list, update_block_ids_list
            
            # Initialize user group ID list if not present
            if not hasattr(self.args, CONFIG_KEY_USER_GROUPID_LIST):
                # Calculate group counts
                group_cnt = get_group_cnt(self.args)
                
                # Create user group ID list
                update_user_groupid_list(self.args, group_cnt)
                
                logging.info(f"Initialized heterogeneous groups: {group_cnt}")
            
            # Initialize block IDs list if not present
            if not hasattr(self.args, CONFIG_KEY_BLOCK_IDS_LIST):
                update_block_ids_list(self.args)
                logging.info(f"Initialized block IDs list: {len(self.args.block_ids_list)} clients")
            
            logging.info(f"Client {self.client_id} assigned to group {self._get_client_group_id()}")
    
    # Flower client interface methods
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get current model parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of model parameters as numpy arrays
        """
        return self._model_to_numpy_params()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server.
        
        Args:
            parameters: List of model parameters from server
        """
        if not parameters:
            logging.warning("Received empty parameters from server")
            return
            
        self._numpy_params_to_model(parameters)
        logging.debug(f"Updated model parameters with {len(parameters)} parameter arrays")
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Set parameters and validate configuration
        self.set_parameters(parameters)
        server_round, local_epochs, learning_rate = self._extract_training_config(config)
        
        # Log training start
        self._log_training_start(server_round, local_epochs, learning_rate, config)
        
        # Train using actual dataset
        total_loss, num_examples = self._perform_training(local_epochs, learning_rate, server_round)
        
        # Update training history and return results
        avg_loss = total_loss / local_epochs
        self._update_training_history(avg_loss, server_round)
        
        logging.info(LOG_TRAINING_COMPLETED.format(client_id=self.client_id, loss=avg_loss))
        
        metrics = self._create_training_metrics(avg_loss, local_epochs, learning_rate)
        return self.get_parameters(config), int(num_examples), metrics
    
    def _extract_training_config(self, config: Dict[str, Any]) -> Tuple[int, int, float]:
        """Extract and validate training configuration."""
        server_round = config.get('server_round')
        if server_round is None:
            raise ValueError("server_round not provided in config")
        
        local_epochs = config.get('local_epochs', self.args.get(CONFIG_KEY_TAU))
        learning_rate = config.get('learning_rate', self.args.get(CONFIG_KEY_LOCAL_LR))
        
        if local_epochs is None or local_epochs < DEFAULT_ONE_VALUE:
            raise ValueError(f"Invalid local_epochs: {local_epochs}")
        if learning_rate is None or learning_rate <= DEFAULT_ZERO_VALUE:
            raise ValueError(f"Invalid learning_rate: {learning_rate}")
        
        return server_round, local_epochs, learning_rate
    
    def _log_training_start(self, server_round: int, local_epochs: int, learning_rate: float, config: Dict[str, Any]) -> None:
        """Log training start information."""
        logging.info(f"Client {self.client_id} received config: {config}")
        logging.info(f"Client {self.client_id} starting training for round {server_round} "
                     f"(epochs={local_epochs}, lr={learning_rate:.4f})")
    
    def _perform_training(self, local_epochs: int, learning_rate: float, server_round: int) -> Tuple[float, int]:
        """Perform the actual training."""
        data_loaded = self.dataset_info.get(CONFIG_KEY_DATA_LOADED, False)
        dataset_train_available = self.dataset_train is not None
        
        logging.info(f"Client {self.client_id} data status: data_loaded={data_loaded}, dataset_train_available={dataset_train_available}")
        
        if not (data_loaded and dataset_train_available):
            raise ValueError(ERROR_NO_TRAINING_DATASET)
        
        total_loss = self._train_with_actual_data(local_epochs, learning_rate, server_round)
        num_examples = self._get_num_examples()
        
        logging.info(f"Client {self.client_id} trained with actual non-IID dataset: {num_examples} samples")
        return total_loss, num_examples
    
    def _get_num_examples(self) -> int:
        """Get number of training examples for this client."""
        if hasattr(self, 'client_data_indices'):
            return len(self.client_data_indices)
        else:
            return len(self.dataset_info.get(CONFIG_KEY_CLIENT_DATA_INDICES, set()))
    
    def _update_training_history(self, avg_loss: float, server_round: int) -> None:
        """Update training history."""
        self.training_history[CONFIG_KEY_LOSSES].append(avg_loss)
        self.training_history[CONFIG_KEY_ROUNDS].append(server_round)
    
    def _create_training_metrics(self, avg_loss: float, local_epochs: int, learning_rate: float) -> Dict[str, Any]:
        """Create training metrics dictionary with proper types for Flower."""
        metrics = {
            'loss': avg_loss,
            'num_epochs': local_epochs,
            'client_id': self.client_id,
            'learning_rate': learning_rate,
            'data_loaded': self.dataset_info.get(CONFIG_KEY_DATA_LOADED, False),
            'noniid_type': self.dataset_info.get(CONFIG_KEY_NONIID_TYPE, DEFAULT_UNKNOWN_VALUE)
        }
        return self._ensure_flower_compatible_types(metrics)
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate the model on local test data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set parameters and validate configuration
        self.set_parameters(parameters)
        server_round = self._extract_server_round(config)
        
        # Perform evaluation
        accuracy, loss, num_examples = self._perform_evaluation_with_validation(server_round)
        
        # Update training history and return results
        self.training_history[CONFIG_KEY_ACCURACIES].append(accuracy)
        logging.info(LOG_EVALUATION_COMPLETED.format(
            client_id=self.client_id, loss=loss, accuracy=accuracy
        ))
        
        metrics = self._create_evaluation_metrics(accuracy)
        return float(loss), int(num_examples), metrics
    
    def _extract_server_round(self, config: Dict[str, Any]) -> int:
        """Extract server round from config."""
        server_round = config.get('server_round')
        if server_round is None:
            raise ValueError("server_round not provided in config")
        return server_round
    
    def _create_evaluation_metrics(self, accuracy: float) -> Dict[str, Any]:
        """Create evaluation metrics dictionary with proper types for Flower."""
        metrics = {
            'accuracy': accuracy,
            'client_id': self.client_id,
            'data_loaded': self.dataset_info.get(CONFIG_KEY_DATA_LOADED, False),
            'noniid_type': self.dataset_info.get(CONFIG_KEY_NONIID_TYPE, DEFAULT_UNKNOWN_VALUE)
        }
        return self._ensure_flower_compatible_types(metrics)
    
    def _ensure_flower_compatible_types(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all values in metrics dictionary are Flower-compatible types.
        
        Flower expects: int, float, str, bytes, bool, list[int], list[float], list[str], list[bytes], list[bool]
        """
        compatible_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                compatible_metrics[key] = int(value)
            elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
                compatible_metrics[key] = float(value)
            elif isinstance(value, (np.bool_,)):
                compatible_metrics[key] = bool(value)
            elif isinstance(value, (np.str_,)):
                compatible_metrics[key] = str(value)
            elif isinstance(value, (int, float, str, bytes, bool)):
                compatible_metrics[key] = value
            elif isinstance(value, list):
                # Handle lists of numpy types
                if all(isinstance(item, (np.integer, np.int8, np.int16, np.int32, np.int64)) for item in value):
                    compatible_metrics[key] = [int(item) for item in value]
                elif all(isinstance(item, (np.floating, np.float16, np.float32, np.float64)) for item in value):
                    compatible_metrics[key] = [float(item) for item in value]
                elif all(isinstance(item, (np.bool_,)) for item in value):
                    compatible_metrics[key] = [bool(item) for item in value]
                elif all(isinstance(item, (np.str_,)) for item in value):
                    compatible_metrics[key] = [str(item) for item in value]
                else:
                    compatible_metrics[key] = value
            else:
                # Convert to string as fallback
                compatible_metrics[key] = str(value)
        
        return compatible_metrics


def main() -> None:
    """Main function to run the Flower client."""
    try:
        # Parse arguments and load configuration
        args = parse_arguments()
        config = load_and_merge_config(args)
        
        # Setup environment
        setup_environment(config, args)
        
        # Start client
        start_flower_client(
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
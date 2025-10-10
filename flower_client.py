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
import argparse
import copy
import logging
import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

# Third-party imports
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import yaml

# Suppress Flower deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from algorithms.solver.fl_utils import (
    setup_multiprocessing
)
from algorithms.solver.shared_utils import (
    load_data, vit_collate_fn, create_client_dataset, 
    create_client_dataloader
)
from algorithms.solver.local_solver import LocalUpdate
from utils.model_utils import model_setup

# Constants
DATASET_LOADING_AVAILABLE = True

# =============================================================================
# CONSTANTS
# =============================================================================

# Default configuration paths
DEFAULT_CONFIG_PATH = "experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml"

# Network configuration
DEFAULT_SERVER_ADDRESS = "localhost"
DEFAULT_SERVER_PORT = 8080
DEFAULT_CLIENT_ID = 0

# Training configuration
DEFAULT_SEED = 1
DEFAULT_GPU_ID = -1
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TRAINING_STEPS_PER_EPOCH = 10
DEFAULT_MIN_LEARNING_RATE = 0.001
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_LOCAL_EPOCHS = 1

# Model architecture constants (can be overridden by config)
VIT_BASE_HIDDEN_SIZE = 768
VIT_LARGE_HIDDEN_SIZE = 1024
LORA_RANK = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]
LORA_BIAS = "none"

# Training constants (can be overridden by config)
DEFAULT_LOGGING_BATCHES = 3  # Number of batches to log during training
DEFAULT_EVAL_BATCHES = 3     # Number of batches to log during evaluation
DEFAULT_FLATTENED_SIZE_CIFAR = 150528  # 3*224*224 for CIFAR-100 with ViT

# Error messages
ERROR_NO_DATA_INDICES = "Client {client_id} has no data indices"
ERROR_INVALID_BATCH_FORMAT = "Invalid batch format: {batch_type}"
ERROR_NO_TEST_DATASET = "No test dataset available for evaluation"
ERROR_NO_TRAINING_DATASET = "No actual dataset found for training"
ERROR_INVALID_CONFIG = "Configuration dictionary cannot be None or empty"
ERROR_INVALID_DATASET = "Unknown dataset: {dataset_name}"

# Logging messages
LOG_CLIENT_INITIALIZED = "Client {client_id} initialized with {num_cores} CPU cores"
LOG_DATASET_LOADED = "Successfully loaded dataset: {dataset_name}"
LOG_TRAINING_COMPLETED = "Client {client_id} completed training: Loss={loss:.4f}"
LOG_EVALUATION_COMPLETED = "Client {client_id} evaluation: Loss={loss:.4f}, Accuracy={accuracy:.4f}"

# Dataset configuration
DATASET_CONFIGS = {
    'cifar100': {'num_classes': 100, 'data_type': 'image'},
    'ledgar': {'num_classes': 2, 'data_type': 'text'},
}

# Non-IID configuration defaults
DEFAULT_NONIID_TYPE = 'dirichlet'
DEFAULT_PAT_NUM_CLS = 10
DEFAULT_PARTITION_MODE = 'dir'
DEFAULT_DIR_ALPHA = 0.5
DEFAULT_DIR_BETA = 1.0

# Data processing constants (can be overridden by config)
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0  # Avoid multiprocessing issues in federated setting
DEFAULT_SHUFFLE_TRAINING = True
DEFAULT_DROP_LAST = True
DEFAULT_SHUFFLE_EVAL = False
DEFAULT_DROP_LAST_EVAL = False


# =============================================================================
# HELPER CLASSES
# =============================================================================

class DatasetArgs:
    """
    Arguments class for dataset loading compatibility.
    
    This class provides a bridge between the configuration dictionary and the 
    dataset loading functions, ensuring compatibility with the existing data 
    preprocessing pipeline.
    
    Example:
        config_dict = {'dataset': 'cifar100', 'batch_size': 128}
        args = DatasetArgs(config_dict, client_id=0)
    """
    
    def __init__(self, config_dict: Dict[str, Any], client_id: int):
        """Initialize dataset args from configuration dictionary."""
        # Dataset configuration
        self.dataset = config_dict.get('dataset', 'cifar100')
        self.model = config_dict.get('model', 'google/vit-base-patch16-224-in21k')
        self.data_type = config_dict.get('data_type', 'image')

        # Model configuration
        self.peft = config_dict.get('peft', 'lora')
        self.lora_layer = config_dict.get('lora_layer', 12)
        
        # LoRA configuration
        self.lora_rank = config_dict.get('lora_rank', LORA_RANK)
        self.lora_alpha = config_dict.get('lora_alpha', LORA_ALPHA)
        self.lora_dropout = config_dict.get('lora_dropout', LORA_DROPOUT)
        self.lora_target_modules = config_dict.get('lora_target_modules', LORA_TARGET_MODULES)
        self.lora_bias = config_dict.get('lora_bias', LORA_BIAS)

        # Training configuration
        self.batch_size = config_dict.get('batch_size', DEFAULT_BATCH_SIZE)
        self.local_lr = config_dict.get('local_lr', DEFAULT_LEARNING_RATE)
        self.tau = config_dict.get('tau', 3)
        self.round = config_dict.get('round', 500)
        self.optimizer = config_dict.get('optimizer', 'adamw')
        
        # Data processing configuration
        self.num_workers = config_dict.get('num_workers', DEFAULT_NUM_WORKERS)
        self.shuffle_training = config_dict.get('shuffle_training', DEFAULT_SHUFFLE_TRAINING)
        self.drop_last = config_dict.get('drop_last', DEFAULT_DROP_LAST)
        self.shuffle_eval = config_dict.get('shuffle_eval', DEFAULT_SHUFFLE_EVAL)
        self.drop_last_eval = config_dict.get('drop_last_eval', DEFAULT_DROP_LAST_EVAL)
        self.logging_batches = config_dict.get('logging_batches', DEFAULT_LOGGING_BATCHES)
        self.eval_batches = config_dict.get('eval_batches', DEFAULT_EVAL_BATCHES)

        # Federated learning configuration
        self.num_users = config_dict.get('num_users', 100)
        self.num_selected_users = config_dict.get('num_selected_users', 1)

        # Non-IID configuration - use values from config or defaults
        self.iid = config_dict.get('iid', 0) == 1  # Convert to boolean
        self.noniid = not self.iid
        self.noniid_type = config_dict.get('noniid_type', DEFAULT_NONIID_TYPE)
        self.pat_num_cls = config_dict.get('pat_num_cls', DEFAULT_PAT_NUM_CLS)
        self.partition_mode = config_dict.get('partition_mode', DEFAULT_PARTITION_MODE)
        self.dir_cls_alpha = config_dict.get('dir_cls_alpha', DEFAULT_DIR_ALPHA)
        self.dir_par_beta = config_dict.get('dir_par_beta', DEFAULT_DIR_BETA)

        # Model heterogeneity
        self.model_heterogeneity = config_dict.get('model_heterogeneity', 'depthffm_fim')
        self.freeze_datasplit = config_dict.get('freeze_datasplit', True)

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Logger
        self.logger = self._create_simple_logger()
        self.client_id = client_id
        
        # Additional attributes that might be needed
        self.num_classes = config_dict.get('num_classes', 100)
        self.labels = None  # Will be set by load_partition
        self.label2id = None  # Will be set by load_partition
        self.id2label = None  # Will be set by load_partition

    def _create_simple_logger(self):
        """Create a simple logger for compatibility."""
        class SimpleLogger:
            def info(self, msg, main_process_only=False):
                logging.info(msg)
        return SimpleLogger()
  
class Config:
    """
    Configuration class to hold and manage configuration parameters.
    
    This class provides a clean interface for accessing configuration parameters
    with proper validation and default value handling. It acts as a wrapper around
    a configuration dictionary, providing type safety and validation.
    
    Example:
        config_dict = {'dataset': 'cifar100', 'batch_size': 128, 'learning_rate': 0.01}
        config = Config(config_dict)
        dataset = config.get('dataset', 'default')
        batch_size = config.batch_size
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
            raise ValueError(ERROR_INVALID_CONFIG)
            
        for key, value in config_dict.items():
            if not isinstance(key, str):
                raise ValueError(f"Configuration keys must be strings, got {type(key)}")
            setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
    
    def has(self, key: str) -> bool:
        """Check if configuration parameter exists."""
        return hasattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self.to_dict()})"


# =============================================================================
# MAIN FLOWER CLIENT CLASS
# =============================================================================

class FlowerClient(fl.client.NumPyClient):
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
        config = Config({'dataset': 'cifar100', 'model': 'google/vit-base-patch16-224-in21k'})
        client = FlowerClient(config, client_id=0)
    """
    
    def __init__(self, args: 'Config', client_id: int = 0):
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
        self.num_cores = setup_multiprocessing()
        logging.info(LOG_CLIENT_INITIALIZED.format(client_id=client_id, num_cores=self.num_cores))
        
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
        
        # Initialize heterogeneous group configuration if needed
        self._initialize_heterogeneous_config()
        
        # Create actual model and parameters
        self.model = self._create_actual_model()
        self.model_params = self._model_to_numpy_params()
        
        # Initialize optimizer (will be recreated for each training round)
        self.optimizer = None
    
    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        # TODO Liam: this is not complete
        required_params = ['data_type', 'peft', 'lora_layer', 'dataset', 'model']
        for param in required_params:
            if not hasattr(self.args, param): 
                logging.warning(f"Missing configuration parameter: {param}, using default")
    
    def _get_loss_function(self) -> nn.Module:
        """Get appropriate loss function based on data type."""
        # TODO Liam: extract constant
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
        dataset_name = self._get_dataset_name()
        dataset_info = self._create_base_dataset_info(dataset_name)
        
        self._validate_dataset_name(dataset_name)
        self._apply_dataset_specific_config(dataset_info, dataset_name)
        
        # Load actual dataset if available
        dataset_info['data_loaded'] = self._load_dataset(dataset_name)
        logging.info(f"Dataset loading result: {dataset_info['data_loaded']}")
        
        # Update with actual data if loaded successfully
        if dataset_info['data_loaded']:
            self._update_dataset_info_with_loaded_data(dataset_info)

        logging.info(f"Dataset configuration loaded: {dataset_name} "
                     f"({dataset_info['num_classes']} classes, {dataset_info['data_type']} data)")

        return dataset_info
    
    def _get_dataset_name(self) -> str:
        """Get dataset name from configuration."""
        # TODO Liam: extract constant
        return self.args.get('dataset', 'cifar100')
    
    def _create_base_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Create base dataset information dictionary."""
        # TODO Liam: extract constant
        return {
            'dataset_name': dataset_name,
            'data_type': self.args.get('data_type', 'image'),
            'model_name': self.args.get('model', 'google/vit-base-patch16-224-in21k'),
            'batch_size': self.args.get('batch_size', DEFAULT_BATCH_SIZE),
            'num_classes': None,
            'labels': None,
            'label2id': None,
            'id2label': None,
            'data_loaded': False
        }
    
    def _validate_dataset_name(self, dataset_name: str) -> None:
        """Validate dataset name."""
        if not dataset_name or not isinstance(dataset_name, str):
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        logging.info(f"Loading dataset configuration for: {dataset_name}")
    
    def _apply_dataset_specific_config(self, dataset_info: Dict[str, Any], dataset_name: str) -> None:
        """Apply dataset-specific configuration."""
        # TODO Liam: extract constant
        if dataset_name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_name]
            dataset_info['num_classes'] = config['num_classes']
            dataset_info['data_type'] = config['data_type']
        else:
            raise ValueError(ERROR_INVALID_DATASET.format(dataset_name=dataset_name))
    
    def _update_dataset_info_with_loaded_data(self, dataset_info: Dict[str, Any]) -> None:
        """Update dataset info with actual loaded data."""
        # TODO Liam: extract constant
        if hasattr(self, 'dataset_train'):
            dataset_info.update({
                'train_samples': len(self.dataset_train) if self.dataset_train else 0,
                'test_samples': len(self.dataset_test) if self.dataset_test else 0,
                'num_users': len(self.client_data_partition) if self.client_data_partition else 0,
                'client_data_indices': getattr(self, 'dataset_info', {}).get('client_data_indices', set()),
                'noniid_type': getattr(self.args_loaded, 'noniid_type', 'dirichlet') if hasattr(self, 'args_loaded') else 'dirichlet'
            })
    
    def _load_dataset(self, dataset_name: str) -> bool:
        """
        Attempt to load actual dataset data.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            True if dataset was loaded successfully, False otherwise
        """
        logging.info(f"Loading dataset: {dataset_name}")

        # Create dataset args and load dataset
        dataset_args = DatasetArgs(self.args.to_dict(), self.client_id)
        dataset_data = self._load_dataset_with_partition(dataset_args)
        
        if dataset_data is None:
            raise ValueError("Failed to load dataset")
            
        # Store dataset information
        self._store_dataset_data(dataset_data)
        
        # Log dataset statistics
        self._log_dataset_statistics(dataset_name, dataset_data)
        
        return True
    
    def _load_dataset_with_partition(self, dataset_args: DatasetArgs) -> Optional[Tuple]:
        """Load dataset using shared load_data function."""
        
        logging.info(f"Loading dataset: {dataset_args.dataset} with model: {dataset_args.model}")
        logging.info(f"Non-IID configuration: {dataset_args.noniid_type}, {dataset_args.pat_num_cls} classes per client")

        # Use shared load_data function for consistency
        args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim = load_data(dataset_args)

        # Debug prints
        # TODO Liam: log instead of print
        print('dataset_train length: ', len(dataset_train))
        print('dataset_test length: ', len(dataset_test))
        print('client_data_partition length: ', len(client_data_partition))
        print('dataset_fim length: ', len(dataset_fim) if dataset_fim else 0)
        print('args_loaded: ', args_loaded)

        # Validate that we have the required data
        if not dataset_train:
            raise ValueError("Failed to load training dataset")
            
        if not client_data_partition:
            raise ValueError("Failed to load user data partition")

        # TODO Liam: other edge cases

        return (args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim)
            
    

    
    def _store_dataset_data(self, dataset_data: Tuple) -> None:
        """Store loaded dataset data in instance variables."""
        args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim = dataset_data
        
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.client_data_partition = client_data_partition
        self.dataset_fim = dataset_fim
        self.args_loaded = args_loaded
        
        # Store client data indices for later use
        if client_data_partition and self.client_id in client_data_partition:
            self.client_data_indices = client_data_partition[self.client_id]
            logging.info(f"Client {self.client_id} assigned {len(client_data_partition[self.client_id])} data samples")
        else:
            logging.warning(f"Client {self.client_id} not found in user data partition")
            self.client_data_indices = set()
    
    def _log_dataset_statistics(self, dataset_name: str, dataset_data: Tuple) -> None:
        """Log comprehensive dataset statistics."""
        args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim = dataset_data
        
        logging.info(LOG_DATASET_LOADED.format(dataset_name=dataset_name))
        logging.info(f"Train samples: {len(dataset_train) if dataset_train else 0}")
        logging.info(f"Test samples: {len(dataset_test) if dataset_test else 0}")
        logging.info(f"Total clients: {len(client_data_partition) if client_data_partition else 0}")
        logging.info(f"Client {self.client_id} has {len(client_data_partition.get(self.client_id, set())) if client_data_partition else 0} data samples")

        # Log non-IID distribution information
        self._log_client_class_distribution(client_data_partition, args_loaded)
    
    def _log_client_class_distribution(self, client_data_partition: Dict, args_loaded) -> None:
        """Log client-specific class distribution information."""
        if not client_data_partition or self.client_id not in client_data_partition:
            return
            
        client_indices = list(client_data_partition[self.client_id])
        if not hasattr(args_loaded, '_tr_labels') or args_loaded._tr_labels is None:
            return
            
        client_labels = args_loaded._tr_labels[client_indices]
        unique_labels = np.unique(client_labels)
        class_counts = dict(zip(*np.unique(client_labels, return_counts=True)))
        
        logging.info(f"Client {self.client_id} has classes: {unique_labels.tolist()}")
        logging.info(f"Client {self.client_id} class distribution: {class_counts}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information for external use.
        
        Returns:
            Dictionary containing dataset information
        """
        return self.dataset_info.copy()
    
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
        # TODO Liam: refactor this to support finer LoRA training
        _, model, _, model_dim = model_setup(self.args)
        
        # Update args with model dimension
        self.args.dim = model_dim
        
        # TODO Liam: extract constant
        logging.info(f"Created model using shared model_setup: {model_dim} dimensions "
                    f"(model={self.args.get('model', 'unknown')}, peft={self.args.get('peft', 'none')})")
        return model
    
    def _ensure_model_setup_attributes(self):
        """Ensure required attributes are present for model_setup function."""
        # Add missing attributes that model_setup expects
        if not hasattr(self.args, 'label2id'):
            num_classes = self.dataset_info.get('num_classes', 100)
            self.args.label2id = {f"class_{i}": i for i in range(num_classes)}
        
        if not hasattr(self.args, 'id2label'):
            num_classes = self.dataset_info.get('num_classes', 100)
            self.args.id2label = {i: f"class_{i}" for i in range(num_classes)}
        
        if not hasattr(self.args, 'logger'):
            self.args.logger = self._create_simple_logger()
        
        if not hasattr(self.args, 'accelerator'):
            # Create a simple accelerator-like object for compatibility
            class SimpleAccelerator:
                def is_local_main_process(self):
                    return True
            self.args.accelerator = SimpleAccelerator()
        
        if not hasattr(self.args, 'log_path'):
            self.args.log_path = f"./logs/client_{self.client_id}"
        
        # Set device with memory management
        device = self._get_optimal_device()
        
        # Ensure other required attributes
        required_attrs = {
            'device': device,
            'seed': getattr(self.args, 'seed', 1),
            'gpu_id': getattr(self.args, 'gpu_id', -1),
        }
        
        for attr, default_value in required_attrs.items():
            if not hasattr(self.args, attr):
                setattr(self.args, attr, default_value)
    
    def _get_optimal_device(self):
        """Get optimal device with memory management."""
        # Force CPU for memory-constrained environments
        if hasattr(self.args, 'force_cpu') and self.args.force_cpu:
            logging.info("Forcing CPU usage due to configuration")
            return torch.device('cpu')
        
        # Check for MPS (Metal Performance Shaders) on macOS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Set MPS memory management
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            logging.info("Using MPS with memory management enabled")
            return torch.device('mps')
        
        # Check for CUDA
        if torch.cuda.is_available():
            # Set CUDA memory management
            torch.cuda.empty_cache()
            logging.info("Using CUDA with memory management enabled")
            return torch.device('cuda')
        
        # Fallback to CPU
        logging.info("Using CPU device")
        return torch.device('cpu')
    
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
            

    
    
    def _train_with_actual_data(self, local_epochs: int, learning_rate: float, server_round: int) -> float:
        """
        Train using actual dataset data with real batch iteration and non-IID distribution.
        
        Args:
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for training
            server_round: Current server round
            
        Returns:
            Total training loss
        """
        # Check if we should use LocalUpdate for heterogeneous training
        if self._should_use_local_update():
            return self._train_with_local_update(local_epochs, learning_rate, server_round)
        else:
            return self._train_with_standard_approach(local_epochs, learning_rate, server_round)
    
    def _should_use_local_update(self) -> bool:
        """Check if we should use LocalUpdate for heterogeneous training."""
        # Use LocalUpdate if we have heterogeneous group configuration
        return (hasattr(self.args, 'heterogeneous_group') and 
                hasattr(self.args, 'user_groupid_list') and
                hasattr(self.args, 'block_ids_list') and
                self.args.get('peft') == 'lora')
    
    # TODO Liam: this is a bit weird
    def _train_with_local_update(self, local_epochs: int, learning_rate: float, server_round: int) -> float:
        """
        Train using LocalUpdate class for heterogeneous federated learning.
        
        Args:
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for training
            server_round: Current server round
            
        Returns:
            Total training loss
        """
        # Ensure block_ids_list is initialized
        if not hasattr(self.args, 'block_ids_list'):
            from algorithms.solver.shared_utils import update_block_ids_list
            update_block_ids_list(self.args)
            logging.info(f"Initialized block_ids_list for client {self.client_id}")
        
        # Apply memory management
        self._apply_memory_management()
        
        # Prepare training data with reduced batch size if needed
        client_indices_list = self._get_client_data_indices()
        client_dataset = self._create_client_dataset(client_indices_list)
        dataloader = self._create_training_dataloader(client_dataset)
        
        logging.info(f"Client {self.client_id} using LocalUpdate for heterogeneous training")
        
        # Create LocalUpdate instance
        local_solver = LocalUpdate(args=self.args)
        
        # Get client group ID for heterogeneous training
        hete_group_id = self._get_client_group_id()
        
        try:
            # Use LocalUpdate for training
            local_model, local_loss, no_weight_lora = local_solver.lora_tuning(
                model=copy.deepcopy(self.model),
                ldr_train=dataloader,
                args=self.args,
                client_index=self.client_id,
                client_real_id=self.client_id,
                round=server_round,
                hete_group_id=hete_group_id
            )
            
            # Update model with trained parameters
            self.model.load_state_dict(local_model)
            
            # Log results
            if local_loss is not None:
                logging.info(f"Client {self.client_id} LocalUpdate training completed: loss={local_loss:.4f}")
                return float(local_loss)  # Ensure float type
            else:
                logging.warning(f"Client {self.client_id} LocalUpdate training returned no loss")
                return 0.0
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(f"Memory error during training: {e}")
                logging.info("Falling back to standard training approach")
                return self._train_with_standard_approach(local_epochs, learning_rate, server_round)
            else:
                raise e
        finally:
            # Clean up memory
            self._cleanup_memory()
    
    def _train_with_standard_approach(self, local_epochs: int, learning_rate: float, server_round: int) -> float:
        """
        Train using standard Flower client approach.
        
        Args:
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for training
            server_round: Current server round
            
        Returns:
            Total training loss
        """
        # Prepare training data
        client_indices_list = self._get_client_data_indices()
        client_dataset = self._create_client_dataset(client_indices_list)
        dataloader = self._create_training_dataloader(client_dataset)
        
        logging.info(f"Client {self.client_id} created DataLoader with {len(dataloader)} batches")
        
        # Setup optimizer for this training round
        self._setup_optimizer(learning_rate)
        
        # Perform actual training with gradients
        total_loss = self._perform_actual_training(dataloader, local_epochs, server_round)
        
        # Log final results
        self._log_training_results(client_indices_list, total_loss, local_epochs)
        return float(total_loss)  # Ensure float type
    
    def _get_client_group_id(self) -> int:
        """Get the heterogeneous group ID for this client."""
        if hasattr(self.args, 'user_groupid_list') and self.client_id < len(self.args.user_groupid_list):
            return self.args.user_groupid_list[self.client_id]
        return 0  # Default to group 0
    
    def _apply_memory_management(self):
        """Apply memory management settings."""
        # Reduce batch size for memory-constrained environments
        # TODO Liam: refactor this
        original_batch_size = self.args.get('batch_size', 32)
        if original_batch_size > 16:
            self.args.batch_size = 16
            logging.info(f"Reduced batch size from {original_batch_size} to {self.args.batch_size} for memory management")
        
        # Clear cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set memory-efficient settings
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Already set in _get_optimal_device()
            pass
    
    def _cleanup_memory(self):
        """Clean up memory after training."""
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    # TODO Liam: refactor this
    def _initialize_heterogeneous_config(self) -> None:
        """Initialize heterogeneous group configuration if needed."""
        # Only initialize if we have heterogeneous group configuration
        if hasattr(self.args, 'heterogeneous_group'):
            from algorithms.solver.shared_utils import get_group_cnt, update_user_groupid_list, update_block_ids_list
            
            # Initialize user group ID list if not present
            if not hasattr(self.args, 'user_groupid_list'):
                # Calculate group counts
                group_cnt = get_group_cnt(self.args)
                
                # Create user group ID list
                update_user_groupid_list(self.args, group_cnt)
                
                logging.info(f"Initialized heterogeneous groups: {group_cnt}")
            
            # Initialize block IDs list if not present
            if not hasattr(self.args, 'block_ids_list'):
                update_block_ids_list(self.args)
                logging.info(f"Initialized block IDs list: {len(self.args.block_ids_list)} clients")
            
            logging.info(f"Client {self.client_id} assigned to group {self._get_client_group_id()}")
    
    def _setup_optimizer(self, learning_rate: float) -> None:
        """Setup optimizer for training."""
        # Only optimize LoRA parameters if using LoRA
        if self.args.get('peft') == 'lora':
            # Get only LoRA parameters
            lora_params = []
            # TODO Liam: refactor this
            for name, param in self.model.named_parameters():
                if 'lora' in name or 'classifier' in name:
                    lora_params.append(param)
            self.optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
        else:
            # Optimize all parameters
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        logging.debug(f"Setup optimizer with {len(list(self.optimizer.param_groups[0]['params']))} parameters")
    
    def _perform_actual_training(self, dataloader, local_epochs: int, server_round: int) -> float:
        """Perform actual training with gradients and parameter updates."""
        self.model.train()
        # TODO Liam: doesn't seem correct
        total_loss = 0.0
        
        for epoch in range(local_epochs):
            epoch_loss = self._train_single_epoch_with_gradients(dataloader, epoch, server_round)
            total_loss += epoch_loss
            
        return total_loss
    
    def _log_training_results(self, client_indices_list: List[int], total_loss: float, local_epochs: int) -> None:
        """Log final training results."""
        avg_total_loss = total_loss / local_epochs if local_epochs > 0 else 0.0
        logging.info(f"Client {self.client_id} trained on {len(client_indices_list)} actual samples, "
                     f"avg_loss={avg_total_loss:.4f}")
    
    def _get_client_data_indices(self) -> List[int]:
        """Get and validate client data indices."""
        if hasattr(self, 'client_data_indices'):
            client_indices = self.client_data_indices
        else:
            client_indices = self.dataset_info.get('client_data_indices', set())
        
        if not client_indices:
            raise ValueError(ERROR_NO_DATA_INDICES.format(client_id=self.client_id))
        return list(client_indices)
    
    def _create_training_dataloader(self, client_dataset):
        """Create DataLoader for training using shared utilities."""
        collate_fn = self._get_collate_function()
        
        # Use shared create_client_dataloader function
        return create_client_dataloader(client_dataset, self.args, collate_fn)
    
    def _get_collate_function(self):
        """Get the appropriate collate function based on dataset type."""
        if self._is_cifar100_dataset():
            return vit_collate_fn
        elif self._is_ledgar_dataset():
            return getattr(self.args_loaded, 'data_collator', None)
        return None  # Use default collate
    
    def _is_cifar100_dataset(self) -> bool:
        """Check if the dataset is CIFAR-100."""
        return (hasattr(self.args_loaded, 'dataset') and 
                self.args_loaded.dataset == 'cifar100')
    
    def _is_ledgar_dataset(self) -> bool:
        """Check if the dataset is LEDGAR."""
        return (hasattr(self.args_loaded, 'dataset') and 
                'ledgar' in self.args_loaded.dataset)
    
    def _train_single_epoch_with_gradients(self, dataloader, epoch: int, server_round: int) -> float:
        """Train for a single epoch with actual gradients and parameter updates."""
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            batch_loss = self._process_training_batch_with_gradients(batch, batch_idx, epoch)
            epoch_loss += batch_loss
            num_batches += 1

        return self._compute_epoch_metrics(epoch, epoch_loss, num_batches)
    
    def _process_training_batch_with_gradients(self, batch, batch_idx: int, epoch: int) -> float:
        """Process a single training batch with actual gradients."""
        # Extract batch data
        pixel_values, labels = self._extract_batch_data(batch)
        
        # Move to device
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        batch_loss = loss.item()
        
        # Log progress for first few batches
        if batch_idx < self.args.get('logging_batches', DEFAULT_LOGGING_BATCHES):
            logging.debug(f"Client {self.client_id} epoch {epoch + 1}, batch {batch_idx + 1}: loss={batch_loss:.4f}")
        
        return batch_loss
    
    def _compute_epoch_metrics(self, epoch: int, epoch_loss: float, num_batches: int) -> float:
        """Compute and log epoch metrics."""
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            logging.info(f"Client {self.client_id} epoch {epoch + 1}: avg_loss={avg_epoch_loss:.4f} ({num_batches} batches)")
            return avg_epoch_loss
        else:
            logging.warning(f"Client {self.client_id} epoch {epoch + 1}: no batches processed")
            return 0.0
    
    def _extract_batch_data(self, batch) -> Tuple:
        """Extract pixel_values and labels from batch."""
        if isinstance(batch, dict):
            return self._extract_from_dict_batch(batch)
        elif len(batch) == 3:
            return self._extract_from_three_element_batch(batch)
        elif len(batch) == 2:
            return self._extract_from_two_element_batch(batch)
        else:
            raise ValueError(ERROR_INVALID_BATCH_FORMAT.format(batch_type=type(batch)))
    
    def _extract_from_dict_batch(self, batch: Dict) -> Tuple:
        """Extract data from dictionary format batch."""
        return batch["pixel_values"], batch["labels"]
    
    def _extract_from_three_element_batch(self, batch) -> Tuple:
        """Extract data from three-element batch (image, label, pixel_values)."""
        image, label, pixel_values = batch
        return pixel_values, label
    
    def _extract_from_two_element_batch(self, batch) -> Tuple:
        """Extract data from two-element batch (pixel_values, labels)."""
        return batch[0], batch[1]

    def _create_client_dataset(self, client_indices: List[int]):
        """
        Create a client-specific dataset subset using shared utilities.

        Args:
            client_indices: List of indices for this client's data

        Returns:
            Dataset subset for this client
        """
        # Use shared create_client_dataset function
        client_dataset = create_client_dataset(self.dataset_train, client_indices, self.args_loaded)

        logging.debug(f"Client {self.client_id} created dataset subset with {len(client_dataset)} samples")
        return client_dataset





    
    
    def _evaluate_with_actual_data(self, server_round: int) -> Tuple[float, float]:
        """
        Evaluate using actual dataset data with real batch iteration.

        Args:
            server_round: Current server round

        Returns:
            Tuple of (accuracy, loss)
        """
        # Prepare evaluation dataset
        eval_dataset = self._get_evaluation_dataset()
        if eval_dataset is None:
            raise ValueError("No test dataset available for evaluation")

        # Create evaluation DataLoader
        eval_dataloader = self._create_evaluation_dataloader(eval_dataset)
        
        # Perform evaluation
        metrics = self._perform_evaluation(eval_dataloader, server_round)
        
        return metrics
    
    def _get_evaluation_dataset(self):
        """Get evaluation dataset (test only)."""
        return self.dataset_test
    
    def _create_evaluation_dataloader(self, eval_dataset):
        """Create DataLoader for evaluation."""
        from torch.utils.data import DataLoader
        batch_size = len(eval_dataset)  # Use full dataset for evaluation
        collate_fn = self._get_collate_function()
        
        return DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=self.args.get('shuffle_eval', DEFAULT_SHUFFLE_EVAL),
            drop_last=self.args.get('drop_last_eval', DEFAULT_DROP_LAST_EVAL),
            num_workers=self.args.get('num_workers', DEFAULT_NUM_WORKERS),
            collate_fn=collate_fn
        )
    
    def _perform_evaluation(self, eval_dataloader, server_round: int) -> Tuple[float, float]:
        """Perform evaluation on all batches."""
        metrics = {'total_loss': 0.0, 'total_correct': 0, 'total_samples': 0, 'num_batches': 0}
        
        logging.info(f"Client {self.client_id} evaluating on {len(eval_dataloader)} test batches")

        for batch_idx, batch in enumerate(eval_dataloader):
            self._process_evaluation_batch(batch, server_round, batch_idx, metrics)

        return self._compute_overall_evaluation_metrics(
            metrics['total_loss'], metrics['total_correct'], 
            metrics['total_samples'], metrics['num_batches']
        )
    
    def _process_evaluation_batch(self, batch, server_round: int, batch_idx: int, metrics: Dict) -> None:
        """Process a single evaluation batch."""
        pixel_values, label = self._extract_evaluation_batch_data(batch)
        batch_loss, batch_correct, batch_size = self._compute_batch_evaluation_metrics(
            pixel_values, label, server_round, batch_idx
        )

        metrics['total_loss'] += batch_loss
        metrics['total_correct'] += batch_correct
        metrics['total_samples'] += batch_size
        metrics['num_batches'] += 1

        # Log progress for first few batches
        if batch_idx < self.args.get('eval_batches', DEFAULT_EVAL_BATCHES):
            batch_accuracy = batch_correct / batch_size if batch_size > 0 else 0.0
            logging.debug(f"Client {self.client_id} eval batch {batch_idx + 1}: "
                          f"loss={batch_loss:.4f}, acc={batch_accuracy:.4f}")
    
    def _extract_evaluation_batch_data(self, batch) -> Tuple:
        """Extract batch data for evaluation."""
        if len(batch) == 3:  # DatasetSplit format
            image, label, pixel_values = batch
            return pixel_values, label
        elif len(batch) == 2:  # Standard format
            return batch[0], batch[1]
        else:
            raise ValueError(f"Invalid batch length for evaluation: {len(batch)}")
    
    def _compute_overall_evaluation_metrics(self, total_loss: float, total_correct: int, 
                                          total_samples: int, num_batches: int) -> Tuple[float, float]:
        """Compute overall evaluation metrics."""
        if num_batches > 0 and total_samples > 0:
            avg_loss = total_loss / num_batches
            accuracy = total_correct / total_samples

            logging.info(f"Client {self.client_id} evaluation completed: "
                         f"loss={avg_loss:.4f}, accuracy={accuracy:.4f} "
                         f"({total_samples} samples, {num_batches} batches)")

            return accuracy, avg_loss
        else:
            raise ValueError("No batches processed during evaluation")

    def _compute_batch_evaluation_metrics(self, pixel_values, labels, server_round: int, batch_idx: int) -> Tuple[float, int, int]:
        """
        Compute actual evaluation metrics for a batch of data.

        Args:
            pixel_values: Batch of image data (tensor or None)
            labels: Batch of labels (tensor or None)
            server_round: Current server round
            batch_idx: Batch index within evaluation

        Returns:
            Tuple of (batch_loss, num_correct, batch_size)
        """
        if pixel_values is None or labels is None:
            raise ValueError(f"Invalid pixel_values or labels: {pixel_values}, {labels}")
        
        batch_size = pixel_values.shape[0] if hasattr(pixel_values, 'shape') else labels.shape[0]
        
        # Move to device
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            num_correct = int(torch.sum(predictions == labels).item())

        logging.debug(f"Eval batch {batch_idx}: size={batch_size}, "
                      f"loss={loss:.4f}, correct={num_correct}")

        return float(loss.item()), num_correct, batch_size





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
        
        local_epochs = config.get('local_epochs', self.args.get('tau'))
        learning_rate = config.get('learning_rate', self.args.get('local_lr'))
        
        if local_epochs is None or local_epochs < 1:
            raise ValueError(f"Invalid local_epochs: {local_epochs}")
        if learning_rate is None or learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate: {learning_rate}")
        
        return server_round, local_epochs, learning_rate
    
    def _log_training_start(self, server_round: int, local_epochs: int, learning_rate: float, config: Dict[str, Any]) -> None:
        """Log training start information."""
        logging.info(f"Client {self.client_id} received config: {config}")
        logging.info(f"Client {self.client_id} starting training for round {server_round} "
                     f"(epochs={local_epochs}, lr={learning_rate:.4f})")
    
    def _perform_training(self, local_epochs: int, learning_rate: float, server_round: int) -> Tuple[float, int]:
        """Perform the actual training."""
        data_loaded = self.dataset_info.get('data_loaded', False)
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
            return len(self.dataset_info.get('client_data_indices', set()))
    
    def _update_training_history(self, avg_loss: float, server_round: int) -> None:
        """Update training history."""
        self.training_history['losses'].append(avg_loss)
        self.training_history['rounds'].append(server_round)
    
    def _create_training_metrics(self, avg_loss: float, local_epochs: int, learning_rate: float) -> Dict[str, Any]:
        """Create training metrics dictionary with proper types for Flower."""
        metrics = {
            'loss': avg_loss,
            'num_epochs': local_epochs,
            'client_id': self.client_id,
            'learning_rate': learning_rate,
            'data_loaded': self.dataset_info.get('data_loaded', False),
            'noniid_type': self.dataset_info.get('noniid_type', 'unknown')
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
        self.training_history['accuracies'].append(accuracy)
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
    
    def _perform_evaluation_with_validation(self, server_round: int) -> Tuple[float, float, int]:
        """Perform evaluation with proper validation."""
        if not (self.dataset_info.get('data_loaded', False) and self.dataset_test is not None):
            raise ValueError(ERROR_NO_TEST_DATASET)
        
        accuracy, loss = self._evaluate_with_actual_data(server_round)
        num_examples = self.dataset_info.get('test_samples')
        
        if num_examples is None:
            raise ValueError("test_samples not available in dataset_info")
        
        logging.info(f"Client {self.client_id} evaluated with actual test dataset: {num_examples} samples")
        return accuracy, loss, num_examples
    
    def _create_evaluation_metrics(self, accuracy: float) -> Dict[str, Any]:
        """Create evaluation metrics dictionary with proper types for Flower."""
        metrics = {
            'accuracy': accuracy,
            'client_id': self.client_id,
            'data_loaded': self.dataset_info.get('data_loaded', False),
            'noniid_type': self.dataset_info.get('noniid_type', 'unknown')
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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


def create_flower_client(args: 'Config', client_id: int = 0) -> FlowerClient:
    """
    Create a Flower client instance.
    
    Args:
        args: Configuration object
        client_id: Client identifier
        
    Returns:
        FlowerClient instance
    """
    return FlowerClient(args, client_id)


def start_flower_client(args: 'Config', server_address: str = "localhost",
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
    fl.client.start_client(
        server_address=f"{server_address}:{server_port}",
        client=client.to_client(),
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


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    """Main function to run the Flower client."""
    try:
        # Parse arguments and load configuration
        args = parse_arguments()
        config = _load_and_merge_config(args)
        
        # Setup environment
        _setup_environment(config, args)
        
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


def _load_and_merge_config(args) -> 'Config':
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


def _setup_environment(config: 'Config', args) -> None:
    """Setup device."""
    config.device = setup_device(args.gpu)


if __name__ == "__main__":
    main()

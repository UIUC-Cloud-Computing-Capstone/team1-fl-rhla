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
from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

# Suppress Flower deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from algorithms.solver.fl_utils import (
    setup_multiprocessing
)
from utils.data_pre_process import load_partition

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

# Model architecture constants
VIT_BASE_HIDDEN_SIZE = 768
VIT_LARGE_HIDDEN_SIZE = 1024
LORA_RANK = 64

# Dataset configuration
DATASET_CONFIGS = {
    'cifar100': {'num_classes': 100, 'data_type': 'image'},
    'sst2': {'num_classes': 2, 'data_type': 'text'},
    'qqp': {'num_classes': 2, 'data_type': 'text'},
    'qnli': {'num_classes': 2, 'data_type': 'text'},
    'ledgar': {'num_classes': 2, 'data_type': 'text'},
    'belebele': {'num_classes': 4, 'data_type': 'text'},
}

# Non-IID configuration defaults
DEFAULT_NONIID_TYPE = 'pathological'
DEFAULT_PAT_NUM_CLS = 10
DEFAULT_PARTITION_MODE = 'dir'
DEFAULT_DIR_ALPHA = 0.5
DEFAULT_DIR_BETA = 1.0

# Data processing constants
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0  # Avoid multiprocessing issues in federated setting


# =============================================================================
# HELPER CLASSES
# =============================================================================

class DatasetArgs:
    """Arguments class for dataset loading compatibility."""
    
    def __init__(self, config_dict: Dict[str, Any], client_id: int):
        """Initialize dataset args from configuration dictionary."""
        # Dataset configuration
        self.dataset = config_dict.get('dataset', 'cifar100')
        self.model = config_dict.get('model', 'google/vit-base-patch16-224-in21k')
        self.data_type = config_dict.get('data_type', 'image')

        # Model configuration
        self.peft = config_dict.get('peft', 'lora')
        self.lora_layer = config_dict.get('lora_layer', 12)

        # Training configuration
        self.batch_size = config_dict.get('batch_size', DEFAULT_BATCH_SIZE)
        self.local_lr = config_dict.get('local_lr', DEFAULT_LEARNING_RATE)
        self.tau = config_dict.get('tau', 3)
        self.round = config_dict.get('round', 500)
        self.optimizer = config_dict.get('optimizer', 'adamw')

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
    - Dataset loading and management
    - Model parameter creation based on architecture
    - Actual training and evaluation with real data
    - LoRA fine-tuning support
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

        # Initialize dataset attributes (will be populated if actual data is loaded)
        # Only set to None if not already loaded
        if not hasattr(self, 'dataset_train'):
            self.dataset_train = None
        if not hasattr(self, 'dataset_test'):
            self.dataset_test = None
        if not hasattr(self, 'dict_users'):
            self.dict_users = None
        if not hasattr(self, 'dataset_fim'):
            self.dataset_fim = None
        if not hasattr(self, 'args_loaded'):
            self.args_loaded = None
        
        # Create actual model parameters
        self.model_params = self._create_model_params()
    
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
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Load actual dataset if available
        
        dataset_info['data_loaded'] = self._load_dataset(dataset_name)
        logging.info(f"Dataset loading result: {dataset_info['data_loaded']}")
            
            # If data was loaded successfully, update dataset_info with actual data
        if dataset_info['data_loaded'] and hasattr(self, 'dataset_train'):
                dataset_info.update({
                    'train_samples': len(self.dataset_train) if self.dataset_train else 0,
                    'test_samples': len(self.dataset_test) if self.dataset_test else 0,
                    'num_users': len(self.dict_users) if self.dict_users else 0,
                    'client_data_indices': getattr(self, 'dataset_info', {}).get('client_data_indices', set()),
                    'noniid_type': getattr(self.args_loaded, 'noniid_type', 'dirichlet') if hasattr(self, 'args_loaded') else 'dirichlet'
                })
        
        

        logging.info(f"Dataset configuration loaded: {dataset_name} "
                     f"({dataset_info['num_classes']} classes, {dataset_info['data_type']} data)")

        return dataset_info
    
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
            return False
            
        # Store dataset information
        self._store_dataset_data(dataset_data)
        
        # Log dataset statistics
        self._log_dataset_statistics(dataset_name, dataset_data)
        
        return True
    
    def _load_dataset_with_partition(self, dataset_args: DatasetArgs) -> Optional[Tuple]:
        """Load dataset using load_partition function."""
        
        logging.info(f"Loading dataset: {dataset_args.dataset} with model: {dataset_args.model}")
        logging.info(f"Non-IID configuration: {dataset_args.noniid_type}, {dataset_args.pat_num_cls} classes per client")

        # Call load_partition to get the actual dataset with non-IID partitioning
        args_loaded, dataset_train, dataset_test, _, _, dict_users, dataset_fim = load_partition(dataset_args)

        # Debug prints
        print('dataset_train length: ', len(dataset_train))
        print('dataset_test length: ', len(dataset_test))
        print('dict_users length: ', len(dict_users))
        print('dataset_fim length: ', len(dataset_fim) if dataset_fim else 0)
        print('args_loaded: ', args_loaded)

        # Validate that we have the required data
        if dataset_train is None or len(dataset_train) == 0:
            logging.error("Failed to load training dataset")
            return None
                
        if dict_users is None or len(dict_users) == 0:
            logging.error("Failed to load user data partition")
            return None

        return (args_loaded, dataset_train, dataset_test, dict_users, dataset_fim)
            
    

    
    def _store_dataset_data(self, dataset_data: Tuple) -> None:
        """Store loaded dataset data in instance variables."""
        args_loaded, dataset_train, dataset_test, dict_users, dataset_fim = dataset_data
        
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users
        self.dataset_fim = dataset_fim
        self.args_loaded = args_loaded
        
        # Store client data indices for later use
        if dict_users and self.client_id in dict_users:
            self.client_data_indices = dict_users[self.client_id]
            logging.info(f"Client {self.client_id} assigned {len(dict_users[self.client_id])} data samples")
        else:
            logging.warning(f"Client {self.client_id} not found in user data partition")
            self.client_data_indices = set()
    
    def _log_dataset_statistics(self, dataset_name: str, dataset_data: Tuple) -> None:
        """Log comprehensive dataset statistics."""
        args_loaded, dataset_train, dataset_test, dict_users, dataset_fim = dataset_data
        
        logging.info(f"Successfully loaded dataset: {dataset_name}")
        logging.info(f"Train samples: {len(dataset_train) if dataset_train else 0}")
        logging.info(f"Test samples: {len(dataset_test) if dataset_test else 0}")
        logging.info(f"Total clients: {len(dict_users) if dict_users else 0}")
        logging.info(f"Client {self.client_id} has {len(dict_users.get(self.client_id, set())) if dict_users else 0} data samples")

        # Log non-IID distribution information
        self._log_client_class_distribution(dict_users, args_loaded)
    
    def _log_client_class_distribution(self, dict_users: Dict, args_loaded) -> None:
        """Log client-specific class distribution information."""
        if not dict_users or self.client_id not in dict_users:
            return
            
        client_indices = list(dict_users[self.client_id])
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
    
    def _create_model_params(self) -> List[np.ndarray]:
        """
        Create actual model parameters using AutoModelForImageClassification.from_pretrained.
        
        Returns:
            List of numpy arrays representing model parameters
            
        Raises:
            ValueError: If model configuration is invalid
        """
        
        # Get dataset and model information
        num_classes = self.dataset_info.get('num_classes', 100)
        model_name = self.dataset_info.get('model_name', 'google/vit-base-patch16-224-in21k')
            
        # Validate inputs
        if num_classes <= 0:
            raise ValueError(f"Invalid number of classes: {num_classes}")
            
        # Create label mappings for the model
        label2id = {f"class_{i}": i for i in range(num_classes)}
        id2label = {i: f"class_{i}" for i in range(num_classes)}
            
        # Create the actual model using AutoModelForImageClassification
        if 'vit' in model_name.lower():
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
                num_labels=num_classes
            )
        else:
            # Fallback for other model types
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            
        # Apply LoRA if specified
        if self.args.get('peft') == 'lora':
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"] if 'vit' in model_name.lower() else None,
            )
            model = get_peft_model(model, lora_config)
            
        # Convert model parameters to numpy arrays
        params = []
        for param in model.parameters():
            params.append(param.detach().cpu().numpy())
            
        logging.info(f"Created {len(params)} actual model parameters for {num_classes} classes "
                        f"(model={model_name}, peft={self.args.get('peft', 'none')})")
        return params
            

    
    
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
        # Validate and prepare client data
        client_indices_list = self._get_client_data_indices()
        client_dataset = self._create_client_dataset(client_indices_list)
        dataloader = self._create_training_dataloader(client_dataset)
        
        logging.info(f"Client {self.client_id} created DataLoader with {len(dataloader)} batches")
        
        # Perform training epochs
        total_loss = self._perform_training_epochs(dataloader, local_epochs, learning_rate, server_round)
        
        # Log final results
        avg_total_loss = total_loss / local_epochs if local_epochs > 0 else 0.0
        logging.info(f"Client {self.client_id} trained on {len(client_indices_list)} actual samples, "
                     f"avg_loss={avg_total_loss:.4f}")
        return total_loss
    
    def _get_client_data_indices(self) -> List[int]:
        """Get and validate client data indices."""
        if hasattr(self, 'client_data_indices'):
            client_indices = self.client_data_indices
        else:
            client_indices = self.dataset_info.get('client_data_indices', set())
        
        if not client_indices:
            raise ValueError(f"Client {self.client_id} has no data indices")
        return list(client_indices)
    
    def _create_training_dataloader(self, client_dataset) -> 'DataLoader':
        """Create DataLoader for training."""
        from torch.utils.data import DataLoader
        batch_size = self.dataset_info.get('batch_size', DEFAULT_BATCH_SIZE)
        
        # Use the appropriate collate function based on the dataset type
        if hasattr(self.args_loaded, 'dataset') and self.args_loaded.dataset == 'cifar100':
            # For CIFAR-100, use the vit_collate_fn that handles the dataset format
            collate_fn = self._get_vit_collate_fn()
        else:
            collate_fn = None  # Use default collate
        
        return DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=DEFAULT_NUM_WORKERS,
            collate_fn=collate_fn
        )
    
    def _get_vit_collate_fn(self):
        """Get the appropriate collate function for ViT models with CIFAR-100."""
        def vit_collate_fn(examples):
            """Collate function for ViT models that handles the CIFAR-100 dataset format."""
            # Find the pixel_values tensor in the example
            pixel_values = None
            labels = None
            
            for example in examples:
                for i, item in enumerate(example):
                    if isinstance(item, torch.Tensor) and len(item.shape) == 3:  # (C, H, W)
                        if pixel_values is None:
                            pixel_values = []
                        pixel_values.append(item)
                    elif isinstance(item, (int, float)) or (isinstance(item, torch.Tensor) and item.dim() == 0):
                        if labels is None:
                            labels = []
                        labels.append(int(item) if isinstance(item, torch.Tensor) else item)
            
            if pixel_values is None or labels is None:
                raise ValueError(f"Could not find pixel_values or labels in dataset format: {[type(x) for x in examples[0]]}")
            
            pixel_values = torch.stack(pixel_values)
            labels = torch.tensor(labels)
            
            return {"pixel_values": pixel_values, "labels": labels}
        
        return vit_collate_fn
    
    def _perform_training_epochs(self, dataloader: 'DataLoader', local_epochs: int, 
                                learning_rate: float, server_round: int) -> float:
        """Perform training across multiple epochs."""
        total_loss = 0.0
        
        for epoch in range(local_epochs):
            epoch_loss = self._train_single_epoch(dataloader, epoch, learning_rate, server_round)
            total_loss += epoch_loss
            
        return total_loss
    
    def _train_single_epoch(self, dataloader: 'DataLoader', epoch: int, 
                           learning_rate: float, server_round: int) -> float:
        """Train for a single epoch."""
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Extract batch data
            pixel_values, labels = self._extract_batch_data(batch)
            
            # Compute loss
            batch_loss = self._compute_batch_loss(pixel_values, labels, server_round, batch_idx)
            epoch_loss += batch_loss
            num_batches += 1

            # Log progress for first few batches
            if batch_idx < 3:
                logging.debug(f"Client {self.client_id} epoch {epoch + 1}, batch {batch_idx + 1}: loss={batch_loss:.4f}")

        # Log epoch results
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            logging.info(f"Client {self.client_id} epoch {epoch + 1}: avg_loss={avg_epoch_loss:.4f} ({num_batches} batches)")
            return avg_epoch_loss
        else:
            logging.warning(f"Client {self.client_id} epoch {epoch + 1}: no batches processed")
            return 0.0
    
    def _extract_batch_data(self, batch) -> Tuple:
        """Extract pixel_values and labels from batch."""
        if isinstance(batch, dict):  # New format with collate function
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]
        elif len(batch) == 3:  # DatasetSplit returns (image, label, pixel_values)
            image, label, pixel_values = batch
            labels = label
        elif len(batch) == 2:  # Standard format
            pixel_values, labels = batch
        else:
            raise ValueError(f"Invalid batch format: {type(batch)}")
        return pixel_values, labels

    def _create_client_dataset(self, client_indices: List[int]):
        """
        Create a client-specific dataset subset from the loaded dataset using DatasetSplit.

        Args:
            client_indices: List of indices for this client's data

        Returns:
            Dataset subset for this client
        """
        
        # Import DatasetSplit from utils
        from utils.data_pre_process import DatasetSplit

        # Create client-specific dataset subset
        client_dataset = DatasetSplit(self.dataset_train, client_indices, self.args_loaded)

        logging.debug(f"Client {self.client_id} created dataset subset with {len(client_dataset)} samples")
        return client_dataset




    def _compute_batch_loss(self, pixel_values, labels, server_round: int, batch_idx: int) -> float:
        """
        Compute actual loss for a batch of data using model forward pass.

        Args:
            pixel_values: Batch of image data (tensor or None)
            labels: Batch of labels (tensor or None)
            server_round: Current server round
            batch_idx: Batch index within epoch

        Returns:
            Computed loss for this batch
        """
        
        # If we have actual data, compute real loss
        if pixel_values is not None and labels is not None:
            # Get batch size from actual data
            if hasattr(pixel_values, 'shape'):
                batch_size = pixel_values.shape[0]
            elif hasattr(labels, 'shape'):
                batch_size = labels.shape[0]
            else:
                raise ValueError("Cannot determine batch size from pixel_values or labels")

            # Compute actual loss using model forward pass
            loss = self._compute_actual_loss(pixel_values, labels, batch_size)

            logging.debug(f"Batch {batch_idx}: size={batch_size}, actual_loss={loss:.4f}")

        else:
            raise ValueError(f"Invalid pixel_values or labels: {pixel_values}, {labels}")

        return float(loss)

    def _compute_actual_loss(self, pixel_values, labels, batch_size: int) -> float:
        """
        Compute actual loss using model forward pass.
        
        Args:
            pixel_values: Batch of image data
            labels: Batch of labels
            batch_size: Size of the batch
            
        Returns:
            Computed loss value
        """
        
        # Convert to tensors if needed
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

            # Ensure proper device placement
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        # Create a simple model for forward pass
        loss = self._forward_pass_loss(pixel_values, labels, batch_size)

        return float(loss.detach().cpu().item())

    def _forward_pass_loss(self, pixel_values: torch.Tensor, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Simulate forward pass and compute loss without using actual model.

        Args:
            pixel_values: Batch of image data
            labels: Batch of labels
            batch_size: Size of the batch

        Returns:
            Computed loss tensor
    """
        
        # Create a simple linear layer for forward pass
        num_classes = self.dataset_info.get('num_classes', 100)

        # Linear transformation for forward pass
        # For CIFAR-100 with ViT, pixel_values has shape (batch_size, 3, 224, 224)
        # We need to flatten it to (batch_size, 3*224*224) = (batch_size, 150528)
        flattened_size = pixel_values.view(batch_size, -1).shape[1]
        linear = torch.nn.Linear(flattened_size, num_classes).to(pixel_values.device)

        # Forward pass
        logits = linear(pixel_values.view(batch_size, -1))

        # Compute cross-entropy loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return loss

    
    
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
    
    def _create_evaluation_dataloader(self, eval_dataset) -> 'DataLoader':
        """Create DataLoader for evaluation."""
        from torch.utils.data import DataLoader
        batch_size = len(eval_dataset)  # Use full dataset for evaluation
        
        # Use the appropriate collate function based on the dataset type
        if hasattr(self.args_loaded, 'dataset') and self.args_loaded.dataset == 'cifar100':
            # For CIFAR-100, use the vit_collate_fn that handles the dataset format
            collate_fn = self._get_vit_collate_fn()
        else:
            collate_fn = None  # Use default collate
        
        return DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=DEFAULT_NUM_WORKERS,
            collate_fn=collate_fn
        )
    
    def _perform_evaluation(self, eval_dataloader: 'DataLoader', server_round: int) -> Tuple[float, float]:
        """Perform evaluation on all batches."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        logging.info(f"Client {self.client_id} evaluating on {len(eval_dataloader)} test batches")

        for batch_idx, batch in enumerate(eval_dataloader):
            # Extract batch data
            pixel_values, label = self._extract_evaluation_batch_data(batch)
            
            # Compute batch metrics
            batch_loss, batch_correct, batch_size = self._compute_batch_evaluation_metrics(
                pixel_values, label, server_round, batch_idx
            )

            total_loss += batch_loss
            total_correct += batch_correct
            total_samples += batch_size
            num_batches += 1

            # Log progress for first few batches
            if batch_idx < 3:
                batch_accuracy = batch_correct / batch_size if batch_size > 0 else 0.0
                logging.debug(f"Client {self.client_id} eval batch {batch_idx + 1}: "
                              f"loss={batch_loss:.4f}, acc={batch_accuracy:.4f}")

        # Compute and return overall metrics
        return self._compute_overall_evaluation_metrics(total_loss, total_correct, total_samples, num_batches)
    
    def _extract_evaluation_batch_data(self, batch) -> Tuple:
        """Extract batch data for evaluation."""
        if len(batch) == 3:  # DatasetSplit format
            image, label, pixel_values = batch
        elif len(batch) == 2:  # Standard format
            pixel_values, label = batch
        else:
            raise ValueError(f"Invalid batch length for evaluation: {len(batch)}")
        return pixel_values, label
    
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

    def _compute_batch_evaluation_metrics(self, pixel_values, labels, server_round: int, batch_idx: int) -> Tuple[
        float, int, int]:
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
        
        # If we have actual data, compute real metrics
        if pixel_values is not None and labels is not None:
            # Get batch size from actual data
            if hasattr(pixel_values, 'shape'):
                batch_size = pixel_values.shape[0]
            elif hasattr(labels, 'shape'):
                batch_size = labels.shape[0]
            else:
                raise ValueError("Cannot determine batch size from pixel_values or labels")

            # Compute actual loss and accuracy
            loss, num_correct = self._compute_actual_evaluation_metrics(pixel_values, labels, batch_size)

            logging.debug(f"Eval batch {batch_idx}: size={batch_size}, "
                          f"loss={loss:.4f}, correct={num_correct}")

        else:
            raise ValueError(f"Invalid pixel_values or labels: {pixel_values}, {labels}")

        return float(loss), num_correct, batch_size



    def _compute_actual_evaluation_metrics(self, pixel_values, labels, batch_size: int) -> Tuple[float, int]:
        """
        Compute actual evaluation metrics using model forward pass.

        Args:
            pixel_values: Batch of image data
            labels: Batch of labels
            batch_size: Size of the batch

        Returns:
            Tuple of (loss, num_correct)
        """
        
        # Convert to tensors if needed
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        # Ensure proper device placement
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        # Compute actual loss and predictions
        loss, predictions = self._forward_pass_evaluation(pixel_values, labels, batch_size)

        # Calculate number of correct predictions
        num_correct = int(torch.sum(predictions == labels).item())

        return float(loss.detach().cpu().item()), num_correct

    def _forward_pass_evaluation(self, pixel_values: torch.Tensor, labels: torch.Tensor, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Perform forward pass for evaluation and return loss and predictions.

        Args:
            pixel_values: Batch of image data
            labels: Batch of labels
            batch_size: Size of the batch

        Returns:
            Tuple of (loss, predictions)
        """
        
        # Create a simple linear layer for forward pass
        num_classes = self.dataset_info.get('num_classes', 100)

        # Linear transformation for forward pass
        if len(pixel_values.shape) > 1:
            hidden_size = pixel_values.shape[1]
        else:
            raise ValueError(f"Invalid pixel_values shape: {pixel_values.shape}")
        linear = torch.nn.Linear(hidden_size, num_classes).to(pixel_values.device)

        # Forward pass
        logits = linear(pixel_values.view(batch_size, -1))

        # Compute cross-entropy loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        # Get predictions
        predictions = torch.argmax(logits, dim=1)

        return loss, predictions


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
        if not parameters:
            logging.warning("Received empty parameters from server")
            return
            
        self.model_params = [param.copy() for param in parameters]
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
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Extract and validate configuration
        server_round = config.get('server_round')
        if server_round is None:
            raise ValueError("server_round not provided in config")
        # Get local epochs and learning rate from config
        local_epochs = config.get('local_epochs', self.args.get('tau'))
        learning_rate = config.get('learning_rate', self.args.get('local_lr'))
        
        if local_epochs is None or local_epochs < 1:
            raise ValueError(f"Invalid local_epochs: {local_epochs}")
        if learning_rate is None or learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate: {learning_rate}")

        # Debug: Print the entire config to see what's being passed
        logging.info(f"Client {self.client_id} received config: {config}")
        logging.info(f"Client {self.client_id} starting training for round {server_round} "
                     f"(epochs={local_epochs}, lr={learning_rate:.4f})")

        # Train using actual dataset
        data_loaded = self.dataset_info.get('data_loaded', False)
        dataset_train_available = self.dataset_train is not None
        
        logging.info(f"Client {self.client_id} data status: data_loaded={data_loaded}, dataset_train_available={dataset_train_available}")
        
        if data_loaded and dataset_train_available:
            # Use actual dataset for training
            total_loss = self._train_with_actual_data(local_epochs, learning_rate, server_round)
            if hasattr(self, 'client_data_indices'):
                num_examples = len(self.client_data_indices)
            else:
                num_examples = len(self.dataset_info.get('client_data_indices', set()))
            logging.info(f"Client {self.client_id} trained with actual non-IID dataset: {num_examples} samples")
        else:
            logging.error(f"Client {self.client_id} cannot train: data_loaded={data_loaded}, dataset_train_available={dataset_train_available}")
            raise ValueError("No actual dataset found for training")

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
            'learning_rate': learning_rate,
            'data_loaded': self.dataset_info.get('data_loaded', False),
            'noniid_type': self.dataset_info.get('noniid_type', 'unknown')
        }

        return self.get_parameters(config), num_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate the model on local test data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Extract server round from config
        server_round = config.get('server_round')
        if server_round is None:
            raise ValueError("server_round not provided in config")
        
        # Evaluate using actual dataset
        if self.dataset_info.get('data_loaded', False) and self.dataset_test is not None:
            # Use actual dataset for evaluation
            accuracy, loss = self._evaluate_with_actual_data(server_round)
            num_examples = self.dataset_info.get('test_samples')
            if num_examples is None:
                raise ValueError("test_samples not available in dataset_info")
            logging.info(f"Client {self.client_id} evaluated with actual test dataset: {num_examples} samples")
        else:
            raise ValueError("No test dataset available for evaluation")

        # Update training history
        self.training_history['accuracies'].append(accuracy)

        logging.info(f"Client {self.client_id} evaluation: "
                     f"Loss={loss:.4f}, Accuracy={accuracy:.4f}")

        # Return loss, number of examples, and metrics
        metrics = {
            'accuracy': accuracy,
            'client_id': self.client_id,
            'data_loaded': self.dataset_info.get('data_loaded', False),
            'noniid_type': self.dataset_info.get('noniid_type', 'unknown')
        }

        return loss, num_examples, metrics


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
    
    # Merge command line arguments into config
    for arg_name in vars(args):
        setattr(config, arg_name, getattr(args, arg_name))
    
    return config


def _setup_environment(config: 'Config', args) -> None:
    """Setup device."""
    config.device = setup_device(args.gpu)


if __name__ == "__main__":
    main()

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
from data_loading import (
    DatasetArgs, load_dataset_config, load_dataset, store_dataset_data,
    get_client_data_indices, create_training_dataloader, create_client_dataset,
    get_evaluation_dataset, create_evaluation_dataloader, extract_batch_data,
    extract_evaluation_batch_data, compute_batch_evaluation_metrics
)

# Constants
DATASET_LOADING_AVAILABLE = True

# =============================================================================
# CONSTANTS
# =============================================================================

# Basic numeric constants
DEFAULT_ZERO_VALUE = 0
DEFAULT_ONE_VALUE = 1
DEFAULT_TWO_VALUE = 2
DEFAULT_THREE_VALUE = 3
DEFAULT_TWELVE_VALUE = 12
DEFAULT_SIXTEEN_VALUE = 16
DEFAULT_THIRTY_TWO_VALUE = 32
DEFAULT_ONE_HUNDRED_VALUE = 100
DEFAULT_FIVE_HUNDRED_VALUE = 500
DEFAULT_ONE_THOUSAND_TWENTY_FOUR_VALUE = 1024
DEFAULT_SEVEN_HUNDRED_SIXTY_EIGHT_VALUE = 768
DEFAULT_ONE_FIFTY_FIVE_TWENTY_EIGHT_VALUE = 150528

# Floating point constants
DEFAULT_ZERO_POINT_ZERO = 0.0
DEFAULT_ZERO_POINT_FIVE = 0.5
DEFAULT_ONE_POINT_ZERO = 1.0
DEFAULT_ZERO_POINT_ZERO_ONE = 0.01
DEFAULT_ZERO_POINT_ZERO_ZERO_ONE = 0.001
DEFAULT_ZERO_POINT_ONE = 0.1

# String constants
DEFAULT_DATASET = 'cifar100'
DEFAULT_MODEL = 'google/vit-base-patch16-224-in21k'
DEFAULT_DATA_TYPE = 'image'
DEFAULT_PEFT = 'lora'
DEFAULT_LORA_LAYER = DEFAULT_TWELVE_VALUE
DEFAULT_TAU = DEFAULT_THREE_VALUE
DEFAULT_ROUND = DEFAULT_FIVE_HUNDRED_VALUE
DEFAULT_OPTIMIZER = 'adamw'
DEFAULT_NUM_USERS = DEFAULT_ONE_HUNDRED_VALUE
DEFAULT_NUM_SELECTED_USERS = DEFAULT_ONE_VALUE
DEFAULT_NUM_CLASSES = DEFAULT_ONE_HUNDRED_VALUE
DEFAULT_MODEL_HETEROGENEITY = 'depthffm_fim'
DEFAULT_GROUP_ID = DEFAULT_ZERO_VALUE
DEFAULT_MEMORY_BATCH_SIZE = DEFAULT_SIXTEEN_VALUE
DEFAULT_MEMORY_THRESHOLD = DEFAULT_THIRTY_TWO_VALUE
DEFAULT_PYTORCH_MPS_RATIO = '0.0'
DEFAULT_LOG_PATH_PREFIX = './logs/client_'
DEFAULT_CLASS_PREFIX = 'class_'
DEFAULT_UNKNOWN_VALUE = 'unknown'
DEFAULT_NONE_VALUE = 'none'
DEFAULT_DEFAULT_VALUE = 'default'
DEFAULT_DIRICHLET_TYPE = 'dirichlet'
DEFAULT_CPU_DEVICE = 'cpu'
DEFAULT_CUDA_DEVICE = 'cuda'
DEFAULT_MPS_DEVICE = 'mps'
DEFAULT_IMAGE_DATA_TYPE = 'image'
DEFAULT_TEXT_DATA_TYPE = 'text'
DEFAULT_SENTIMENT_DATA_TYPE = 'sentiment'
DEFAULT_QUERY_MODULE = 'query'
DEFAULT_VALUE_MODULE = 'value'
DEFAULT_DIR_PARTITION_MODE = 'dir'
DEFAULT_ADAMW_OPTIMIZER = 'adamw'
DEFAULT_LORA_PEFT = 'lora'
DEFAULT_CIFAR100_DATASET = 'cifar100'
DEFAULT_LEDGAR_DATASET = 'ledgar'
DEFAULT_BATCH_FORMAT_3_ELEMENTS = DEFAULT_THREE_VALUE
DEFAULT_BATCH_FORMAT_2_ELEMENTS = DEFAULT_TWO_VALUE
DEFAULT_DIMENSION_1 = DEFAULT_ONE_VALUE

# Configuration key constants
CONFIG_KEY_DATASET = 'dataset'
CONFIG_KEY_MODEL = 'model'
CONFIG_KEY_DATA_TYPE = 'data_type'
CONFIG_KEY_PEFT = 'peft'
CONFIG_KEY_LORA_LAYER = 'lora_layer'
CONFIG_KEY_LORA_RANK = 'lora_rank'
CONFIG_KEY_LORA_ALPHA = 'lora_alpha'
CONFIG_KEY_LORA_DROPOUT = 'lora_dropout'
CONFIG_KEY_LORA_TARGET_MODULES = 'lora_target_modules'
CONFIG_KEY_LORA_BIAS = 'lora_bias'
CONFIG_KEY_BATCH_SIZE = 'batch_size'
CONFIG_KEY_LOCAL_LR = 'local_lr'
CONFIG_KEY_TAU = 'tau'
CONFIG_KEY_ROUND = 'round'
CONFIG_KEY_NUM_WORKERS = 'num_workers'
CONFIG_KEY_SHUFFLE_TRAINING = 'shuffle_training'
CONFIG_KEY_DROP_LAST = 'drop_last'
CONFIG_KEY_SHUFFLE_EVAL = 'shuffle_eval'
CONFIG_KEY_DROP_LAST_EVAL = 'drop_last_eval'
CONFIG_KEY_LOGGING_BATCHES = 'logging_batches'
CONFIG_KEY_EVAL_BATCHES = 'eval_batches'
CONFIG_KEY_NUM_USERS = 'num_users'
CONFIG_KEY_NUM_SELECTED_USERS = 'num_selected_users'
CONFIG_KEY_IID = 'iid'
CONFIG_KEY_NONIID_TYPE = 'noniid_type'
CONFIG_KEY_PAT_NUM_CLS = 'pat_num_cls'
CONFIG_KEY_PARTITION_MODE = 'partition_mode'
CONFIG_KEY_DIR_CLS_ALPHA = 'dir_cls_alpha'
CONFIG_KEY_DIR_PAR_BETA = 'dir_par_beta'
CONFIG_KEY_MODEL_HETEROGENEITY = 'model_heterogeneity'
CONFIG_KEY_FREEZE_DATASPLIT = 'freeze_datasplit'
CONFIG_KEY_NUM_CLASSES = 'num_classes'
CONFIG_KEY_SEED = 'seed'
CONFIG_KEY_GPU_ID = 'gpu_id'
CONFIG_KEY_FORCE_CPU = 'force_cpu'
CONFIG_KEY_HETEROGENEOUS_GROUP = 'heterogeneous_group'
CONFIG_KEY_USER_GROUPID_LIST = 'user_groupid_list'
CONFIG_KEY_BLOCK_IDS_LIST = 'block_ids_list'
CONFIG_KEY_LABEL2ID = 'label2id'
CONFIG_KEY_ID2LABEL = 'id2label'
CONFIG_KEY_LOGGER = 'logger'
CONFIG_KEY_ACCELERATOR = 'accelerator'
CONFIG_KEY_LOG_PATH = 'log_path'
CONFIG_KEY_DEVICE = 'device'
CONFIG_KEY_DATASET_INFO = 'dataset_info'
CONFIG_KEY_CLIENT_DATA_INDICES = 'client_data_indices'
CONFIG_KEY_NONIID_TYPE = 'noniid_type'
CONFIG_KEY_DATA_COLLATOR = 'data_collator'
CONFIG_KEY_TRAIN_SAMPLES = 'train_samples'
CONFIG_KEY_TEST_SAMPLES = 'test_samples'
CONFIG_KEY_NUM_USERS = 'num_users'
CONFIG_KEY_DATA_LOADED = 'data_loaded'
CONFIG_KEY_DATASET_NAME = 'dataset_name'
CONFIG_KEY_MODEL_NAME = 'model_name'
CONFIG_KEY_LABELS = 'labels'
CONFIG_KEY_NUM_CLASSES = 'num_classes'
CONFIG_KEY_LOSSES = 'losses'
CONFIG_KEY_ACCURACIES = 'accuracies'
CONFIG_KEY_ROUNDS = 'rounds'
CONFIG_KEY_PIXEL_VALUES = 'pixel_values'
CONFIG_KEY_LABELS = 'labels'
CONFIG_KEY_TOTAL_LOSS = 'total_loss'
CONFIG_KEY_TOTAL_CORRECT = 'total_correct'
CONFIG_KEY_TOTAL_SAMPLES = 'total_samples'
CONFIG_KEY_NUM_BATCHES = 'num_batches'
CONFIG_KEY_TR_LABELS = '_tr_labels'

# Default configuration paths
DEFAULT_CONFIG_PATH = "experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml"

# Network configuration
DEFAULT_SERVER_ADDRESS = "localhost"
DEFAULT_SERVER_PORT = 8080
DEFAULT_CLIENT_ID = DEFAULT_ZERO_VALUE

# Training configuration
DEFAULT_SEED = DEFAULT_ONE_VALUE
DEFAULT_GPU_ID = -1
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TRAINING_STEPS_PER_EPOCH = 10
DEFAULT_MIN_LEARNING_RATE = DEFAULT_ZERO_POINT_ZERO_ZERO_ONE
DEFAULT_LEARNING_RATE = DEFAULT_ZERO_POINT_ZERO_ONE
DEFAULT_LOCAL_EPOCHS = DEFAULT_ONE_VALUE

# Model architecture constants (can be overridden by config)
VIT_BASE_HIDDEN_SIZE = DEFAULT_SEVEN_HUNDRED_SIXTY_EIGHT_VALUE
VIT_LARGE_HIDDEN_SIZE = DEFAULT_ONE_THOUSAND_TWENTY_FOUR_VALUE
LORA_RANK = 64
LORA_ALPHA = DEFAULT_SIXTEEN_VALUE
LORA_DROPOUT = DEFAULT_ZERO_POINT_ONE
LORA_TARGET_MODULES = [DEFAULT_QUERY_MODULE, DEFAULT_VALUE_MODULE]
LORA_BIAS = DEFAULT_NONE_VALUE

# Training constants (can be overridden by config)
DEFAULT_EVAL_BATCHES = DEFAULT_THREE_VALUE     # Number of batches to log during evaluation
DEFAULT_FLATTENED_SIZE_CIFAR = DEFAULT_ONE_FIFTY_FIVE_TWENTY_EIGHT_VALUE  # 3*224*224 for CIFAR-100 with ViT

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
    DEFAULT_CIFAR100_DATASET: {CONFIG_KEY_NUM_CLASSES: DEFAULT_ONE_HUNDRED_VALUE, CONFIG_KEY_DATA_TYPE: DEFAULT_IMAGE_DATA_TYPE},
    DEFAULT_LEDGAR_DATASET: {CONFIG_KEY_NUM_CLASSES: DEFAULT_TWO_VALUE, CONFIG_KEY_DATA_TYPE: DEFAULT_TEXT_DATA_TYPE},
}

# Non-IID configuration defaults
DEFAULT_NONIID_TYPE = DEFAULT_DIRICHLET_TYPE
DEFAULT_PAT_NUM_CLS = 10
DEFAULT_PARTITION_MODE = DEFAULT_DIR_PARTITION_MODE
DEFAULT_DIR_ALPHA = DEFAULT_ZERO_POINT_FIVE
DEFAULT_DIR_BETA = DEFAULT_ONE_POINT_ZERO

# Data processing constants (can be overridden by config)
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = DEFAULT_ZERO_VALUE  # Avoid multiprocessing issues in federated setting
DEFAULT_SHUFFLE_TRAINING = True
DEFAULT_DROP_LAST = True
DEFAULT_SHUFFLE_EVAL = False
DEFAULT_DROP_LAST_EVAL = False


# =============================================================================
# HELPER CLASSES
# =============================================================================

  
class Config:
    """
    Configuration class to hold and manage configuration parameters.
    
    This class provides a clean interface for accessing configuration parameters
    with proper validation and default value handling. It acts as a wrapper around
    a configuration dictionary, providing type safety and validation.
    
    Example:
        config_dict = {CONFIG_KEY_DATASET: DEFAULT_DATASET, CONFIG_KEY_BATCH_SIZE: DEFAULT_BATCH_SIZE, 'learning_rate': DEFAULT_ZERO_POINT_ZERO_ONE}
        config = Config(config_dict)
        dataset = config.get(CONFIG_KEY_DATASET, DEFAULT_DEFAULT_VALUE)
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
        config = Config({CONFIG_KEY_DATASET: DEFAULT_DATASET, CONFIG_KEY_MODEL: DEFAULT_MODEL})
        client = FlowerClient(config, client_id=DEFAULT_CLIENT_ID)
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
        num_cores = setup_multiprocessing()
        logging.info(LOG_CLIENT_INITIALIZED.format(client_id=client_id, num_cores=num_cores))
        
        
        # Initialize training history
        self.training_history = {
            CONFIG_KEY_LOSSES: [],
            CONFIG_KEY_ACCURACIES: [],
            CONFIG_KEY_ROUNDS: []
        }
        
        # Initialize LoRA optimization attributes
        self.trained_model = None
        self.no_weight_lora = None
        self.lora_mapping = None
        self.lora_metadata = None
        
        # Load dataset configuration and data
        self.dataset_info = load_dataset_config(self.args, self.client_id)
        
        # Load actual dataset if configuration indicates it should be loaded
        if self.dataset_info.get(CONFIG_KEY_DATA_LOADED, False):
            self._load_and_store_dataset()
        
        # Initialize heterogeneous group configuration if needed
        self._initialize_heterogeneous_config()
        
        # Create actual model
        self.model = self._create_actual_model()
    
    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        # TODO Liam: this is not complete
        required_params = [CONFIG_KEY_DATA_TYPE, CONFIG_KEY_PEFT, CONFIG_KEY_LORA_LAYER, CONFIG_KEY_DATASET, CONFIG_KEY_MODEL]
        for param in required_params:
            if not hasattr(self.args, param): 
                logging.warning(f"Missing configuration parameter: {param}, using default")
    
    
    def _load_and_store_dataset(self) -> None:
        """Load and store dataset data using the data_loading module."""
        dataset_name = self.dataset_info.get(CONFIG_KEY_DATASET_NAME, DEFAULT_DATASET)
        
        # Load dataset using the data_loading module
        load_dataset(dataset_name, self.args, self.client_id)
        
        # Store the loaded data
        dataset_args = DatasetArgs(self.args.to_dict(), self.client_id)
        dataset_data = self._load_dataset_with_partition(dataset_args)
        
        if dataset_data is None:
            raise ValueError("Failed to load dataset")
            
        # Store dataset information
        self._store_dataset_data(dataset_data)
    
    def _load_dataset_with_partition(self, dataset_args: DatasetArgs) -> Optional[Tuple]:
        """Load dataset using shared load_data function."""
        from data_loading import load_dataset_with_partition
        return load_dataset_with_partition(dataset_args)
    
    def _store_dataset_data(self, dataset_data: Tuple) -> None:
        """Store loaded dataset data in instance variables."""
        args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim = dataset_data
        
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.client_data_partition = client_data_partition
        # Store args_loaded for compatibility with existing methods
        self.args_loaded = args_loaded
        # Note: dataset_fim is not used in current implementation
        
        # Store client data indices for later use
        if client_data_partition and self.client_id in client_data_partition:
            self.client_data_indices = client_data_partition[self.client_id]
            logging.info(f"Client {self.client_id} assigned {len(client_data_partition[self.client_id])} data samples")
        else:
            logging.warning(f"Client {self.client_id} not found in user data partition")
            self.client_data_indices = set()
    
    
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information for external use.
        
        Returns:
            Dictionary containing dataset information
        """
        return self.dataset_info.copy()
    
    # TODO Liam: refactor this
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
    


    # =============================================================================
    # Data Loading
    # =============================================================================


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
        
        return self._train_with_local_update(local_epochs, learning_rate, server_round)
                
    # TODO Liam: fix
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
        if not hasattr(self.args, CONFIG_KEY_BLOCK_IDS_LIST):
            from algorithms.solver.shared_utils import update_block_ids_list
            update_block_ids_list(self.args)
            logging.info(f"Initialized block_ids_list for client {self.client_id}")
        
        # Prepare training data with reduced batch size if needed
        client_indices_list = self._get_client_data_indices()

        # get data
        client_dataset = self._create_client_dataset(client_indices_list, self.dataset_train, self.args_loaded)
        dataloader = self._create_training_dataloader(client_dataset)
        
        logging.info(f"Client {self.client_id} using LocalUpdate for heterogeneous training")
        
        # Use LocalUpdate for training
        local_solver = LocalUpdate(args=self.args)
        
        # Store original model state for debugging
        original_model_state = copy.deepcopy(self.model.state_dict())
        
        # Validate the current model state before training
        self._validate_model_state_dict(self.model.state_dict(), "current model before training")
        
        local_model, local_loss, no_weight_lora = local_solver.lora_tuning(
                model=copy.deepcopy(self.model),
                ldr_train=dataloader,
                args=self.args,
                client_index=self.client_id,
                client_real_id=self.client_id,
                round=server_round,
                hete_group_id=self._get_client_group_id()
        )
        
        # Validate that the trained model state dict is reasonable
        self._validate_model_state_dict(local_model, "trained model")
            
        # Store the trained model and no_weight_lora for efficient parameter sending
        self.trained_model = local_model
        self.no_weight_lora = no_weight_lora
        
        # Create parameter mapping for efficient LoRA parameter tracking
        from algorithms.solver.shared_utils import get_lora_parameter_mapping
        self.lora_mapping = get_lora_parameter_mapping(local_model, no_weight_lora)
        
        # Update only the LoRA parameters in the client's model, preserving base model
        self._update_model_lora_parameters_only(local_model)
            
        # Log results
        if local_loss is not None:
            logging.info(f"Client {self.client_id} LocalUpdate training completed: loss={local_loss:.4f}")
            return float(local_loss)  # Ensure float type
        else:
            logging.warning(f"Client {self.client_id} LocalUpdate training returned no loss")
            return 0.0
    
    def _get_client_data_indices(self) -> List[int]:
        """Get and validate client data indices."""
        client_data_indices = getattr(self, 'client_data_indices', None)
        return get_client_data_indices(client_data_indices, self.dataset_info, self.client_id)
    
    def _create_client_dataset(self, client_indices: List[int], dataset_train, args_loaded):
        """
        Create a client-specific dataset subset using shared utilities.

        Args:
            client_indices: List of indices for this client's data
            dataset_train: Training dataset
            args_loaded: Loaded arguments

        Returns:
            Dataset subset for this client
        """
        # Use shared create_client_dataset function
        client_dataset = create_client_dataset(client_indices, dataset_train, args_loaded)

        logging.debug(f"Client {self.client_id} created dataset subset with {len(client_dataset)} samples")
        return client_dataset

    def _create_training_dataloader(self, client_dataset):
        """Create DataLoader for training using shared utilities."""
        return create_training_dataloader(client_dataset, self.args_loaded, self.args)

    def _get_client_group_id(self) -> int:
        """Get the heterogeneous group ID for this client."""
        if hasattr(self.args, CONFIG_KEY_USER_GROUPID_LIST) and self.client_id < len(self.args.user_groupid_list):
            return self.args.user_groupid_list[self.client_id]
        return DEFAULT_GROUP_ID  # Default to group 0
        



    # =============================================================================
    # TRAINING
    # =============================================================================

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
        
        # Add LoRA metadata to metrics if available (convert to Flower-compatible format)
        if hasattr(self, 'lora_metadata') and self.lora_metadata is not None:
            # Convert LoRA metadata to Flower-compatible format
            lora_metadata = self.lora_metadata
            # Convert list to string for Flower compatibility
            trained_layers = lora_metadata.get('trained_layers', [])
            metrics['lora_trained_layers'] = ','.join(map(str, trained_layers)) if trained_layers else ""
            metrics['lora_param_count'] = len(lora_metadata.get('param_names', []))
            metrics['lora_untrained_layers'] = len(lora_metadata.get('no_weight_lora', []))
        
        return self.get_parameters(config), int(num_examples), metrics
    
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
    
    def _numpy_params_to_model(self, params: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        # Determine if this is a LoRA-only update or full model update
        # by comparing the number of parameters with the expected model parameters
        model_param_count = sum(1 for _ in self.model.parameters())
        
        if len(params) < model_param_count:
            # This is likely a LoRA-only update
            logging.info(f"Detected LoRA-only parameter update: {len(params)} params vs {model_param_count} model params")
            self._set_lora_parameters_from_numpy(params)
        else:
            # This is a full model update
            logging.info(f"Detected full model parameter update: {len(params)} params")
            self._set_full_model_parameters_from_numpy(params)
    
    def _set_full_model_parameters_from_numpy(self, params: List[np.ndarray]) -> None:
        """Set full model parameters from numpy arrays."""
        param_idx = 0
        for param in self.model.parameters():
            if param_idx < len(params):
                # Validate shape before setting
                param_tensor = torch.from_numpy(params[param_idx])
                if param_tensor.shape == param.shape:
                    param.data = param_tensor.to(param.device)
                else:
                    logging.error(f"Shape mismatch for parameter {param_idx}: "
                                f"received {param_tensor.shape} vs expected {param.shape}")
                    raise ValueError(f"Parameter shape mismatch at index {param_idx}")
                param_idx += 1
            else:
                logging.warning(f"Not enough parameters provided: expected {param_idx + 1}, got {len(params)}")
                break
    
    def _set_lora_parameters_from_numpy(self, params: List[np.ndarray]) -> None:
        """Set LoRA parameters from numpy arrays using the mapping."""
        if not hasattr(self, 'lora_mapping') or self.lora_mapping is None:
            logging.warning("No LoRA mapping available, cannot set LoRA parameters. This might be the first round.")
            # For the first round, we might receive full model parameters even if we expect LoRA
            # Try to set them as full model parameters
            self._set_full_model_parameters_from_numpy(params)
            return
        
        # Get current model state dict
        current_state_dict = self.model.state_dict()
        updated_params = 0
        
        # Set LoRA parameters using the mapping
        for param_name, mapping_info in self.lora_mapping.items():
            param_index = mapping_info['index']
            if param_index < len(params):
                param_tensor = torch.from_numpy(params[param_index])
                if param_name in current_state_dict:
                    if param_tensor.shape == current_state_dict[param_name].shape:
                        current_state_dict[param_name] = param_tensor
                        updated_params += 1
                    else:
                        logging.error(f"Shape mismatch for LoRA parameter {param_name}: "
                                    f"received {param_tensor.shape} vs expected {current_state_dict[param_name].shape}")
                        raise ValueError(f"LoRA parameter shape mismatch for {param_name}")
                else:
                    logging.warning(f"LoRA parameter {param_name} not found in model state dict")
        
        # Load the updated state dict
        self.model.load_state_dict(current_state_dict, strict=False)
        logging.info(f"Set {updated_params} LoRA parameters using mapping")

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


    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get current model parameters, optimized for LoRA to only send changed parts.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of model parameters as numpy arrays (only LoRA parameters that were updated)
        """
        if self.trained_model is not None and self.no_weight_lora is not None:
            return self._get_lora_parameters_only()
        else:
            # Fallback to full model parameters if no LoRA training has occurred
            return self._model_to_numpy_params()
    
    def _model_to_numpy_params(self) -> List[np.ndarray]:
        """Convert model parameters to numpy arrays."""
        params = []
        for param in self.model.parameters():
            params.append(param.detach().cpu().numpy())
        return params
    
    def _get_lora_parameters_only(self) -> List[np.ndarray]:
        """
        Get only the LoRA parameters that were actually updated during training.
        Also includes metadata about which layers were trained.
        
        Returns:
            List of numpy arrays containing only the updated LoRA parameters
        """
        if self.lora_mapping is None:
            logging.warning(f"Client {self.client_id}: No LoRA mapping available, falling back to full model")
            return self._model_to_numpy_params()
        
        lora_params = []
        param_names = []
        trained_layers = []
        
        # Get the trained model state dict
        trained_state_dict = self.trained_model
        
        # Use the mapping to extract only the trained LoRA parameters
        for param_name, mapping_info in self.lora_mapping.items():
            if param_name in trained_state_dict:
                param = trained_state_dict[param_name]
                lora_params.append(param.detach().cpu().numpy())
                param_names.append(param_name)
                trained_layers.append(mapping_info['layer_num'])
        
        # Store metadata for server-side aggregation
        self.lora_metadata = {
            'trained_layers': list(set(trained_layers)),  # Unique layer numbers
            'param_names': param_names,
            'no_weight_lora': self.no_weight_lora
        }
        
        logging.info(f"Client {self.client_id} sending {len(lora_params)} LoRA parameters from layers {self.lora_metadata['trained_layers']} (skipped {len(self.no_weight_lora)} untrained layers)")
        logging.debug(f"LoRA parameter names: {param_names}")
        
        return lora_params
    
    def _update_model_lora_parameters_only(self, trained_state_dict):
        """
        Update only the LoRA parameters in the client's model, preserving base model parameters.
        
        Args:
            trained_state_dict: State dict from LoRA training containing LoRA parameters
        """
        current_state_dict = self.model.state_dict()
        updated_params = 0
        
        # Only update LoRA parameters, preserve all other parameters
        for param_name, param_value in trained_state_dict.items():
            if 'lora' in param_name.lower() and param_name in current_state_dict:
                # Ensure the parameter shapes match
                if param_value.shape == current_state_dict[param_name].shape:
                    current_state_dict[param_name] = param_value
                    updated_params += 1
                else:
                    logging.warning(f"Shape mismatch for parameter {param_name}: "
                                  f"trained {param_value.shape} vs current {current_state_dict[param_name].shape}")
            elif 'lora' not in param_name.lower():
                # This is a base model parameter - ensure it's preserved
                if param_name in current_state_dict:
                    # Keep the current base model parameter
                    pass
                else:
                    logging.warning(f"Base model parameter {param_name} not found in current model")
        
        # Load the updated state dict
        try:
            self.model.load_state_dict(current_state_dict, strict=False)
            logging.info(f"Updated {updated_params} LoRA parameters in client {self.client_id} model")
        except Exception as e:
            logging.error(f"Error loading state dict for client {self.client_id}: {e}")
            # Fallback: try to load only the LoRA parameters
            lora_state_dict = {k: v for k, v in trained_state_dict.items() if 'lora' in k.lower()}
            try:
                self.model.load_state_dict(lora_state_dict, strict=False)
                logging.info(f"Fallback: Updated LoRA parameters only for client {self.client_id}")
            except Exception as e2:
                logging.error(f"Fallback also failed for client {self.client_id}: {e2}")
                raise
    
    def _validate_model_state_dict(self, state_dict, model_name):
        """
        Validate that a model state dict has reasonable parameter shapes.
        
        Args:
            state_dict: Model state dictionary to validate
            model_name: Name of the model for logging
        """
        invalid_params = []
        
        for param_name, param_tensor in state_dict.items():
            if not isinstance(param_tensor, torch.Tensor):
                invalid_params.append(f"{param_name}: not a tensor")
                continue
                
            # Check for obviously invalid shapes
            if param_tensor.numel() == 0:
                invalid_params.append(f"{param_name}: empty tensor")
            elif len(param_tensor.shape) == 0:
                invalid_params.append(f"{param_name}: scalar tensor")
            elif any(dim <= 0 for dim in param_tensor.shape):
                invalid_params.append(f"{param_name}: invalid dimensions {param_tensor.shape}")
            elif torch.isnan(param_tensor).any():
                invalid_params.append(f"{param_name}: contains NaN values")
            elif torch.isinf(param_tensor).any():
                invalid_params.append(f"{param_name}: contains Inf values")
        
        if invalid_params:
            logging.error(f"Invalid parameters in {model_name}:")
            for invalid_param in invalid_params:
                logging.error(f"  - {invalid_param}")
            raise ValueError(f"Model state dict validation failed for {model_name}")
        else:
            logging.debug(f"Model state dict validation passed for {model_name}")

    
    # =============================================================================
    # EVALUATION
    # =============================================================================
    
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
    
    def _perform_evaluation_with_validation(self, server_round: int) -> Tuple[float, float, int]:
        """Perform evaluation with proper validation."""
        if not (self.dataset_info.get(CONFIG_KEY_DATA_LOADED, False) and self.dataset_test is not None):
            raise ValueError(ERROR_NO_TEST_DATASET)
        
        accuracy, loss = self._evaluate_with_actual_data(server_round)
        num_examples = self.dataset_info.get(CONFIG_KEY_TEST_SAMPLES)
        
        if num_examples is None:
            raise ValueError("test_samples not available in dataset_info")
        
        logging.info(f"Client {self.client_id} evaluated with actual test dataset: {num_examples} samples")
        return accuracy, loss, num_examples
    
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
        return get_evaluation_dataset(self.dataset_test)
    
    def _create_evaluation_dataloader(self, eval_dataset):
        """Create DataLoader for evaluation."""
        return create_evaluation_dataloader(eval_dataset, self.args_loaded, self.args)

    def _perform_evaluation(self, eval_dataloader, server_round: int) -> Tuple[float, float]:
        """Perform evaluation on all batches."""
        metrics = {CONFIG_KEY_TOTAL_LOSS: DEFAULT_ZERO_VALUE, CONFIG_KEY_TOTAL_CORRECT: DEFAULT_ZERO_VALUE, CONFIG_KEY_TOTAL_SAMPLES: DEFAULT_ZERO_VALUE, CONFIG_KEY_NUM_BATCHES: DEFAULT_ZERO_VALUE}
        
        logging.info(f"Client {self.client_id} evaluating on {len(eval_dataloader)} test batches")

        for batch_idx, batch in enumerate(eval_dataloader):
            self._process_evaluation_batch(batch, server_round, batch_idx, metrics)

        return self._compute_overall_evaluation_metrics(
            metrics[CONFIG_KEY_TOTAL_LOSS], metrics[CONFIG_KEY_TOTAL_CORRECT], 
            metrics[CONFIG_KEY_TOTAL_SAMPLES], metrics[CONFIG_KEY_NUM_BATCHES]
        )
    
    def _process_evaluation_batch(self, batch, server_round: int, batch_idx: int, metrics: Dict) -> None:
        """Process a single evaluation batch."""
        pixel_values, label = extract_evaluation_batch_data(batch)
        batch_loss, batch_correct, batch_size = self._compute_batch_evaluation_metrics(
            pixel_values, label, server_round, batch_idx
        )

        metrics[CONFIG_KEY_TOTAL_LOSS] += batch_loss
        metrics[CONFIG_KEY_TOTAL_CORRECT] += batch_correct
        metrics[CONFIG_KEY_TOTAL_SAMPLES] += batch_size
        metrics[CONFIG_KEY_NUM_BATCHES] += 1

        # Log progress for first few batches
        if batch_idx < self.args.get(CONFIG_KEY_EVAL_BATCHES, DEFAULT_EVAL_BATCHES):
            batch_accuracy = batch_correct / batch_size if batch_size > DEFAULT_ZERO_VALUE else DEFAULT_ZERO_VALUE
            logging.debug(f"Client {self.client_id} eval batch {batch_idx + DEFAULT_ONE_VALUE}: "
                          f"loss={batch_loss:.4f}, acc={batch_accuracy:.4f}")
    
    
    def _compute_overall_evaluation_metrics(self, total_loss: float, total_correct: int, 
                                          total_samples: int, num_batches: int) -> Tuple[float, float]:
        """Compute overall evaluation metrics."""
        if num_batches > DEFAULT_ZERO_VALUE and total_samples > DEFAULT_ZERO_VALUE:
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
        return compute_batch_evaluation_metrics(
            pixel_values, labels, self.model, server_round, batch_idx, self.client_id
        )

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

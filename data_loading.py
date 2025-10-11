"""
Data Loading Module for Flower Client

This module provides comprehensive data loading functionality for federated learning
scenarios. It handles dataset configuration, loading, partitioning, and data processing
for various datasets including CIFAR-100 and LEDGAR.

Author: Team1-FL-RHLA
Version: 1.0.0
"""

# Standard library imports
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

# Third-party imports
import numpy as np
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

# Local imports
from algorithms.solver.shared_utils import (
    load_data, vit_collate_fn, create_client_dataset as shared_create_client_dataset, 
    create_client_dataloader
)

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
CONFIG_KEY_OPTIMIZER = 'optimizer'
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
CONFIG_KEY_DATA_COLLATOR = 'data_collator'
CONFIG_KEY_TRAIN_SAMPLES = 'train_samples'
CONFIG_KEY_TEST_SAMPLES = 'test_samples'
CONFIG_KEY_DATA_LOADED = 'data_loaded'
CONFIG_KEY_DATASET_NAME = 'dataset_name'
CONFIG_KEY_MODEL_NAME = 'model_name'
CONFIG_KEY_LABELS = 'labels'
CONFIG_KEY_LOSSES = 'losses'
CONFIG_KEY_ACCURACIES = 'accuracies'
CONFIG_KEY_ROUNDS = 'rounds'
CONFIG_KEY_PIXEL_VALUES = 'pixel_values'
CONFIG_KEY_TOTAL_LOSS = 'total_loss'
CONFIG_KEY_TOTAL_CORRECT = 'total_correct'
CONFIG_KEY_TOTAL_SAMPLES = 'total_samples'
CONFIG_KEY_NUM_BATCHES = 'num_batches'
CONFIG_KEY_TR_LABELS = '_tr_labels'

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
DEFAULT_LOGGING_BATCHES = DEFAULT_THREE_VALUE  # Number of batches to log during training
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
LOG_DATASET_LOADED = "Successfully loaded dataset: {dataset_name}"

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

class DatasetArgs:
    """
    Arguments class for dataset loading compatibility.
    
    This class provides a bridge between the configuration dictionary and the 
    dataset loading functions, ensuring compatibility with the existing data 
    preprocessing pipeline.
    
    Example:
        config_dict = {CONFIG_KEY_DATASET: DEFAULT_DATASET, CONFIG_KEY_BATCH_SIZE: DEFAULT_BATCH_SIZE}
        args = DatasetArgs(config_dict, client_id=DEFAULT_CLIENT_ID)
    """
    
    def __init__(self, config_dict: Dict[str, Any], client_id: int):
        """Initialize dataset args from configuration dictionary."""
        # Store config dict for direct access instead of individual attributes
        self.config_dict = config_dict
        self.client_id = client_id
        
        # Only store essential attributes that are frequently accessed
        self.dataset = config_dict.get(CONFIG_KEY_DATASET, DEFAULT_DATASET)
        self.model = config_dict.get(CONFIG_KEY_MODEL, DEFAULT_MODEL)
        self.data_type = config_dict.get(CONFIG_KEY_DATA_TYPE, DEFAULT_DATA_TYPE)
        self.peft = config_dict.get(CONFIG_KEY_PEFT, DEFAULT_PEFT)
        self.batch_size = config_dict.get(CONFIG_KEY_BATCH_SIZE, DEFAULT_BATCH_SIZE)
        self.num_users = config_dict.get(CONFIG_KEY_NUM_USERS, DEFAULT_NUM_USERS)
        
        # Device configuration
        self.device = torch.device(DEFAULT_CUDA_DEVICE if torch.cuda.is_available() else DEFAULT_CPU_DEVICE)
        
        # Logger
        self.logger = self._create_simple_logger()
        
        # Additional attributes that might be needed
        self.num_classes = config_dict.get(CONFIG_KEY_NUM_CLASSES, DEFAULT_NUM_CLASSES)
        self.labels = None  # Will be set by load_partition
        self.label2id = None  # Will be set by load_partition
        self.id2label = None  # Will be set by load_partition
        
        # Computed attributes
        self.iid = config_dict.get(CONFIG_KEY_IID, DEFAULT_ZERO_VALUE) == DEFAULT_ONE_VALUE
        self.noniid = not self.iid
        
        # Non-IID configuration
        self.noniid_type = config_dict.get(CONFIG_KEY_NONIID_TYPE, DEFAULT_DIRICHLET_TYPE)
        self.dir_cls_alpha = config_dict.get(CONFIG_KEY_DIR_CLS_ALPHA, DEFAULT_ONE_POINT_ZERO)  # Use alpha 1.0 for balanced non-IID
        self.pat_num_cls = config_dict.get(CONFIG_KEY_PAT_NUM_CLS, DEFAULT_PAT_NUM_CLS)
        self.partition_mode = config_dict.get(CONFIG_KEY_PARTITION_MODE, DEFAULT_PARTITION_MODE)
        self.model_heterogeneity = config_dict.get(CONFIG_KEY_MODEL_HETEROGENEITY, DEFAULT_MODEL_HETEROGENEITY)
        self.freeze_datasplit = config_dict.get(CONFIG_KEY_FREEZE_DATASPLIT, False)

    def _create_simple_logger(self):
        """Create a simple logger for compatibility."""
        class SimpleLogger:
            def info(self, msg, main_process_only=False):
                logging.info(msg)
        return SimpleLogger()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self.config_dict.get(key, default)
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute from config_dict if not found as direct attribute."""
        if name in self.config_dict:
            return self.config_dict[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility with existing code."""
        return self.config_dict.copy()


# =============================================================================
# DATASET CONFIGURATION FUNCTIONS
# =============================================================================

def load_dataset_config(args, client_id: int) -> Dict[str, Any]:
    """
    Load dataset configuration and prepare dataset information.
    
    Args:
        args: Configuration object containing dataset parameters
        client_id: Client identifier
        
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
    dataset_name = get_dataset_name(args)
    dataset_info = create_base_dataset_info(dataset_name, args)
    
    validate_dataset_name(dataset_name)
    apply_dataset_specific_config(dataset_info, dataset_name)
    
    # Load actual dataset if available
    dataset_info[CONFIG_KEY_DATA_LOADED] = load_dataset(dataset_name, args, client_id)
    logging.info(f"Dataset loading result: {dataset_info[CONFIG_KEY_DATA_LOADED]}")
    
    # Update with actual data if loaded successfully
    if dataset_info[CONFIG_KEY_DATA_LOADED]:
        update_dataset_info_with_loaded_data(dataset_info, args, client_id)

    logging.info(f"Dataset configuration loaded: {dataset_name} "
                 f"({dataset_info[CONFIG_KEY_NUM_CLASSES]} classes, {dataset_info[CONFIG_KEY_DATA_TYPE]} data)")

    return dataset_info


def get_dataset_name(args) -> str:
    """Get dataset name from configuration."""
    return args.get(CONFIG_KEY_DATASET, DEFAULT_DATASET)


def create_base_dataset_info(dataset_name: str, args) -> Dict[str, Any]:
    """Create base dataset information dictionary."""
    return {
        CONFIG_KEY_DATASET_NAME: dataset_name,
        CONFIG_KEY_DATA_TYPE: args.get(CONFIG_KEY_DATA_TYPE, DEFAULT_IMAGE_DATA_TYPE),
        CONFIG_KEY_MODEL_NAME: args.get(CONFIG_KEY_MODEL, DEFAULT_MODEL),
        CONFIG_KEY_BATCH_SIZE: args.get(CONFIG_KEY_BATCH_SIZE, DEFAULT_BATCH_SIZE),
        CONFIG_KEY_NUM_CLASSES: None,
        CONFIG_KEY_LABELS: None,
        CONFIG_KEY_LABEL2ID: None,
        CONFIG_KEY_ID2LABEL: None,
        CONFIG_KEY_DATA_LOADED: False
    }


def validate_dataset_name(dataset_name: str) -> None:
    """Validate dataset name."""
    if not dataset_name or not isinstance(dataset_name, str):
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    logging.info(f"Loading dataset configuration for: {dataset_name}")


def apply_dataset_specific_config(dataset_info: Dict[str, Any], dataset_name: str) -> None:
    """Apply dataset-specific configuration."""
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        dataset_info[CONFIG_KEY_NUM_CLASSES] = config[CONFIG_KEY_NUM_CLASSES]
        dataset_info[CONFIG_KEY_DATA_TYPE] = config[CONFIG_KEY_DATA_TYPE]
    else:
        raise ValueError(ERROR_INVALID_DATASET.format(dataset_name=dataset_name))


def update_dataset_info_with_loaded_data(dataset_info: Dict[str, Any], args, client_id: int) -> None:
    """Update dataset info with actual loaded data."""
    # Get stored dataset data
    if hasattr(args, 'dataset_data') and args.dataset_data:
        stored_data = args.dataset_data
        dataset_info.update({
            CONFIG_KEY_TRAIN_SAMPLES: len(stored_data['dataset_train']) if hasattr(stored_data['dataset_train'], '__len__') else 0,
            CONFIG_KEY_TEST_SAMPLES: len(stored_data['dataset_test']) if hasattr(stored_data['dataset_test'], '__len__') else 0,
            CONFIG_KEY_NUM_USERS: len(stored_data['client_data_partition']) if stored_data['client_data_partition'] else 0,
            CONFIG_KEY_CLIENT_DATA_INDICES: stored_data['client_data_indices'],
            CONFIG_KEY_NONIID_TYPE: getattr(args, CONFIG_KEY_NONIID_TYPE, DEFAULT_DIRICHLET_TYPE)
        })
    else:
        # Fallback to placeholder values
        dataset_info.update({
            CONFIG_KEY_TRAIN_SAMPLES: 0,
            CONFIG_KEY_TEST_SAMPLES: 0,
            CONFIG_KEY_NUM_USERS: 0,
            CONFIG_KEY_CLIENT_DATA_INDICES: set(),
            CONFIG_KEY_NONIID_TYPE: getattr(args, CONFIG_KEY_NONIID_TYPE, DEFAULT_DIRICHLET_TYPE)
        })


# =============================================================================
# DATASET LOADING FUNCTIONS
# =============================================================================

def load_dataset(dataset_name: str, args, client_id: int) -> bool:
    """
    Attempt to load actual dataset data.
    
    Args:
        dataset_name: Name of the dataset to load
        args: Configuration object
        client_id: Client identifier
        
    Returns:
        True if dataset was loaded successfully, False otherwise
    """
    logging.info(f"Loading dataset: {dataset_name}")

    # Create dataset args and load dataset
    dataset_args = DatasetArgs(args.to_dict(), client_id)
    dataset_data = load_dataset_with_partition(dataset_args)
    
    if dataset_data is None:
        raise ValueError("Failed to load dataset")
    
    # Store dataset data in args for later use
    stored_data = store_dataset_data(dataset_data, client_id)
    args.dataset_data = stored_data
        
    # Log dataset statistics
    log_dataset_statistics(dataset_name, dataset_data, client_id)
    
    return True


def load_dataset_with_partition(dataset_args: DatasetArgs) -> Optional[Tuple]:
    """Load dataset using efficient Flower Datasets partitioning."""
    
    logging.info(f"Loading dataset: {dataset_args.dataset} with model: {dataset_args.model}")
    logging.info(f"Non-IID configuration: {dataset_args.noniid_type}, {dataset_args.pat_num_cls} classes per client")

    # Use efficient Flower Datasets partitioning
    args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim = load_data_efficient(dataset_args)

    # Log dataset loading information
    if hasattr(dataset_train, '__len__'):
        logging.info(f'Dataset train length: {len(dataset_train)}')
    else:
        logging.info(f'Dataset train type: {type(dataset_train)}')
    
    if hasattr(dataset_test, '__len__'):
        logging.info(f'Dataset test length: {len(dataset_test)}')
    else:
        logging.info(f'Dataset test type: {type(dataset_test)}')
    
    logging.info(f'Client data partition length: {len(client_data_partition)}')
    logging.info(f'Dataset FIM length: {len(dataset_fim) if dataset_fim else 0}')
    logging.debug(f'Args loaded: {args_loaded}')

    # Validate that we have the required data
    if not dataset_train:
        raise ValueError("Failed to load training dataset")
        
    if not client_data_partition:
        raise ValueError("Failed to load user data partition")

    return (args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim)


def load_data_efficient(dataset_args: DatasetArgs) -> Tuple:
    """
    Load dataset using efficient Flower Datasets partitioning with DirichletPartitioner.
    
    This function uses the Flower Datasets library to efficiently partition data
    using DirichletPartitioner with alpha 1.0 for balanced non-IID distribution.
    
    Args:
        dataset_args: Configuration arguments for dataset loading
        
    Returns:
        Tuple of (args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim)
    """
    logging.info("Using efficient Flower Datasets partitioning")
    
    # Create partitioner based on configuration
    if dataset_args.iid:
        partitioner = IidPartitioner(num_partitions=dataset_args.num_users)
        logging.info(f"Using IID partitioning for {dataset_args.num_users} clients")
    else:
        # Use DirichletPartitioner with alpha 1.0 for balanced non-IID distribution
        alpha = getattr(dataset_args, 'dir_cls_alpha', 1.0)
        partitioner = DirichletPartitioner(
            num_partitions=dataset_args.num_users,
            alpha=alpha,
            partition_by="fine_label" if dataset_args.dataset == 'cifar100' else "label"
        )
        logging.info(f"Using DirichletPartitioner with alpha={alpha} for {dataset_args.num_users} clients")
    
    # Create federated dataset
    if dataset_args.dataset == 'cifar100':
        fds = FederatedDataset(
            dataset="cifar100",
            partitioners={"train": partitioner}
        )
    elif dataset_args.dataset == 'ledgar':
        fds = FederatedDataset(
            dataset="lex_glue",
            subset="ledgar",
            partitioners={"train": partitioner}
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_args.dataset}")
    
    # Load test dataset (no partitioning needed)
    if dataset_args.dataset == 'cifar100':
        from datasets import load_dataset
        test_dataset = load_dataset("cifar100", split="test")
    elif dataset_args.dataset == 'ledgar':
        from datasets import load_dataset
        test_dataset = load_dataset("lex_glue", "ledgar", split="test")
    
    # Extract all partitions and combine into a single dataset for compatibility
    all_partitions = []
    client_data_partition = {}
    current_index = 0
    
    for client_id in range(dataset_args.num_users):
        try:
            partition = fds.load_partition(partition_id=client_id)
            partition_size = len(partition)
            
            # Add partition data to combined dataset
            for i in range(partition_size):
                all_partitions.append(partition[i])
            
            # Create indices for this client
            client_indices = set(range(current_index, current_index + partition_size))
            client_data_partition[client_id] = client_indices
            current_index += partition_size
            
        except Exception as e:
            logging.warning(f"Failed to load partition for client {client_id}: {e}")
            client_data_partition[client_id] = set()
    
    # Create a combined dataset that can be indexed
    combined_dataset = create_combined_dataset(all_partitions, dataset_args)
    
    # Create args_loaded object for compatibility
    args_loaded = create_compatible_args(dataset_args, combined_dataset, len(all_partitions))
    
    # Handle FIM dataset if needed
    dataset_fim = None
    if hasattr(dataset_args, 'model_heterogeneity') and 'depthffm_fim' in dataset_args.model_heterogeneity:
        dataset_fim = create_fim_dataset(dataset_args)
    
    return (args_loaded, combined_dataset, test_dataset, client_data_partition, dataset_fim)


def create_combined_dataset(all_partitions: List, dataset_args: DatasetArgs):
    """
    Create a combined dataset from all partitions that can be indexed.
    
    Args:
        all_partitions: List of all partition data
        dataset_args: Dataset configuration arguments
        
    Returns:
        Combined dataset that supports indexing
    """
    class CombinedDataset:
        def __init__(self, data, dataset_name, model_name):
            self.data = data
            self.dataset_name = dataset_name
            self.model_name = model_name
            
            # Set up transforms for compatibility
            if dataset_name == 'cifar100':
                from transformers import AutoImageProcessor
                from torchvision.transforms import Compose, Normalize, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Resize
                
                image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
                normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
                
                if model_name == 'google/vit-base-patch16-224-in21k':
                    self.train_transforms = Compose([
                        RandomResizedCrop(image_processor.size["height"]),
                        RandomHorizontalFlip(),
                        ToTensor(),
                        normalize,
                    ])
                else:
                    self.train_transforms = Compose([
                        RandomResizedCrop((32, 32)),
                        RandomHorizontalFlip(),   
                        ToTensor(),
                        normalize,
                    ])
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            
            if self.dataset_name == 'cifar100':
                # Apply transforms to create pixel_values
                image = item['img']
                if hasattr(self, 'train_transforms'):
                    pixel_values = self.train_transforms(image.convert("RGB"))
                else:
                    # Fallback if transforms not available
                    pixel_values = None
                
                return {
                    'img': image,
                    'fine_label': item['fine_label'],
                    'coarse_label': item.get('coarse_label', 0),
                    'pixel_values': pixel_values
                }
            else:
                return item
    
    return CombinedDataset(all_partitions, dataset_args.dataset, dataset_args.model)


def create_compatible_args(dataset_args: DatasetArgs, combined_dataset, total_samples: int = 0):
    """
    Create a compatible args object for existing code.
    
    Args:
        dataset_args: Original dataset arguments
        combined_dataset: Combined dataset object
        total_samples: Total number of samples across all partitions
        
    Returns:
        Compatible args object
    """
    class CompatibleArgs:
        def __init__(self, dataset_args, combined_dataset, total_samples):
            # Copy all attributes from dataset_args
            for attr in dir(dataset_args):
                if not attr.startswith('_'):
                    setattr(self, attr, getattr(dataset_args, attr))
            
            # Add dataset reference
            self.dataset_train = combined_dataset  # For compatibility
            self.total_samples = total_samples
            
            # Set labels and mappings
            if dataset_args.dataset == 'cifar100':
                self.labels = [f"class_{i}" for i in range(100)]
                self.label2id = {f"class_{i}": i for i in range(100)}
                self.id2label = {i: f"class_{i}" for i in range(100)}
                self.num_classes = 100
            elif dataset_args.dataset == 'ledgar':
                self.labels = ["class_0", "class_1"]
                self.label2id = {"class_0": 0, "class_1": 1}
                self.id2label = {0: "class_0", 1: "class_1"}
                self.num_classes = 2
            
            # Extract labels for compatibility with existing code
            self._tr_labels = extract_labels_from_dataset(combined_dataset, dataset_args.dataset)
    
    return CompatibleArgs(dataset_args, combined_dataset, total_samples)


def extract_labels_from_dataset(dataset, dataset_name: str):
    """
    Extract labels from the combined dataset for compatibility.
    
    Args:
        dataset: Combined dataset
        dataset_name: Name of the dataset
        
    Returns:
        Array of labels
    """
    try:
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            if dataset_name == 'cifar100':
                labels.append(item['fine_label'])
            elif dataset_name == 'ledgar':
                labels.append(item['label'])
        return np.array(labels)
    except Exception as e:
        logging.warning(f"Failed to extract labels: {e}")
        return None


def create_fim_dataset(dataset_args: DatasetArgs):
    """
    Create FIM dataset for model heterogeneity.
    
    Args:
        dataset_args: Dataset configuration arguments
        
    Returns:
        FIM dataset or None
    """
    if dataset_args.dataset == 'cifar100':
        from datasets import load_dataset
        dataset = load_dataset("cifar100", split="train")
        # Select random samples for FIM
        import numpy as np
        selected_indices = np.random.choice(len(dataset), 100, replace=False)
        return dataset.select(selected_indices)
    elif dataset_args.dataset == 'ledgar':
        from datasets import load_dataset
        dataset = load_dataset("lex_glue", "ledgar", split="train")
        # Select random samples for FIM
        import numpy as np
        selected_indices = np.random.choice(len(dataset), 50, replace=False)
        return dataset.select(selected_indices)
    
    return None


def store_dataset_data(dataset_data: Tuple, client_id: int) -> Dict[str, Any]:
    """Store loaded dataset data and return for caller to store."""
    args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim = dataset_data
    
    # Store client data indices for later use
    if client_data_partition and client_id in client_data_partition:
        client_data_indices = client_data_partition[client_id]
        logging.info(f"Client {client_id} assigned {len(client_data_partition[client_id])} data samples")
    else:
        logging.warning(f"Client {client_id} not found in user data partition")
        client_data_indices = set()
    
    # Return the data for the caller to store
    return {
        'args_loaded': args_loaded,
        'dataset_train': dataset_train,
        'dataset_test': dataset_test,
        'client_data_partition': client_data_partition,
        'client_data_indices': client_data_indices,
        'dataset_fim': dataset_fim
    }


def log_dataset_statistics(dataset_name: str, dataset_data: Tuple, client_id: int) -> None:
    """Log comprehensive dataset statistics."""
    args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim = dataset_data
    
    logging.info(LOG_DATASET_LOADED.format(dataset_name=dataset_name))
    
    # Handle different dataset types
    if hasattr(dataset_train, '__len__'):
        logging.info(f"Train samples: {len(dataset_train) if dataset_train else 0}")
    else:
        logging.info(f"Train dataset type: {type(dataset_train)}")
    
    if hasattr(dataset_test, '__len__'):
        logging.info(f"Test samples: {len(dataset_test) if dataset_test else 0}")
    else:
        logging.info(f"Test dataset type: {type(dataset_test)}")
    
    logging.info(f"Total clients: {len(client_data_partition) if client_data_partition else 0}")
    logging.info(f"Client {client_id} has {len(client_data_partition.get(client_id, set())) if client_data_partition else 0} data samples")

    # Log non-IID distribution information
    log_client_class_distribution(client_data_partition, args_loaded, client_id)


def log_client_class_distribution(client_data_partition: Dict, args_loaded, client_id: int) -> None:
    """Log client-specific class distribution information."""
    if not client_data_partition or client_id not in client_data_partition:
        return
        
    client_indices = list(client_data_partition[client_id])
    if not hasattr(args_loaded, CONFIG_KEY_TR_LABELS) or args_loaded._tr_labels is None:
        logging.info(f"Client {client_id} has {len(client_indices)} data samples (label distribution not available)")
        return
        
    client_labels = args_loaded._tr_labels[client_indices]
    unique_labels = np.unique(client_labels)
    class_counts = dict(zip(*np.unique(client_labels, return_counts=True)))
    
    logging.info(f"Client {client_id} has classes: {unique_labels.tolist()}")
    logging.info(f"Client {client_id} class distribution: {class_counts}")


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def get_client_data_indices(client_data_indices, dataset_info: Dict[str, Any], client_id: int) -> List[int]:
    """Get and validate client data indices."""
    if client_data_indices:
        client_indices = client_data_indices
    else:
        client_indices = dataset_info.get(CONFIG_KEY_CLIENT_DATA_INDICES, set())
    
    if not client_indices:
        raise ValueError(ERROR_NO_DATA_INDICES.format(client_id=client_id))
    return list(client_indices)


def create_training_dataloader(client_dataset, args_loaded, args):
    """Create DataLoader for training using shared utilities."""
    collate_fn = get_collate_function(args_loaded)
    
    # Use shared create_client_dataloader function
    return create_client_dataloader(client_dataset, args, collate_fn)


def get_collate_function(args_loaded):
    """Get the appropriate collate function based on dataset type."""
    if is_cifar100_dataset(args_loaded):
        return vit_collate_fn
    elif is_ledgar_dataset(args_loaded):
        return getattr(args_loaded, CONFIG_KEY_DATA_COLLATOR, None)
    else:
        raise ValueError(f"Invalid dataset: {args_loaded.dataset}")


def is_cifar100_dataset(args_loaded) -> bool:
    """Check if the dataset is CIFAR-100."""
    return (hasattr(args_loaded, 'dataset') and 
            args_loaded.dataset == DEFAULT_CIFAR100_DATASET)


def is_ledgar_dataset(args_loaded) -> bool:
    """Check if the dataset is LEDGAR."""
    return (hasattr(args_loaded, 'dataset') and 
            DEFAULT_LEDGAR_DATASET in args_loaded.dataset)


def create_client_dataset(client_indices: List[int], dataset_train, args_loaded):
    """
    Create a client-specific dataset subset using shared utilities.

    Args:
        client_indices: List of indices for this client's data
        dataset_train: Training dataset
        args_loaded: Loaded arguments

    Returns:
        Dataset subset for this client
    """
    # Use shared create_client_dataset function from algorithms.solver.shared_utils
    client_dataset = shared_create_client_dataset(dataset_train, client_indices, args_loaded)

    logging.debug(f"Created dataset subset with {len(client_dataset)} samples")
    return client_dataset


def get_evaluation_dataset(dataset_test):
    """Get evaluation dataset (test only)."""
    return dataset_test


def create_evaluation_dataloader(eval_dataset, args_loaded, args):
    """Create DataLoader for evaluation."""
    from torch.utils.data import DataLoader
    batch_size = len(eval_dataset)  # Use full dataset for evaluation
    collate_fn = get_collate_function(args_loaded)
    
    return DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=args.get(CONFIG_KEY_SHUFFLE_EVAL, DEFAULT_SHUFFLE_EVAL),
        drop_last=args.get(CONFIG_KEY_DROP_LAST_EVAL, DEFAULT_DROP_LAST_EVAL),
        num_workers=args.get(CONFIG_KEY_NUM_WORKERS, DEFAULT_NUM_WORKERS),
        collate_fn=collate_fn
    )


# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

def extract_batch_data(batch) -> Tuple:
    """Extract pixel_values and labels from batch."""
    if isinstance(batch, dict):
        return extract_from_dict_batch(batch)
    elif len(batch) == DEFAULT_BATCH_FORMAT_3_ELEMENTS:
        return extract_from_three_element_batch(batch)
    elif len(batch) == DEFAULT_BATCH_FORMAT_2_ELEMENTS:
        return extract_from_two_element_batch(batch)
    else:
        raise ValueError(ERROR_INVALID_BATCH_FORMAT.format(batch_type=type(batch)))


def extract_from_dict_batch(batch: Dict) -> Tuple:
    """Extract data from dictionary format batch."""
    return batch[CONFIG_KEY_PIXEL_VALUES], batch[CONFIG_KEY_LABELS]


def extract_from_three_element_batch(batch) -> Tuple:
    """Extract data from three-element batch (image, label, pixel_values)."""
    image, label, pixel_values = batch
    return pixel_values, label


def extract_from_two_element_batch(batch) -> Tuple:
    """Extract data from two-element batch (pixel_values, labels)."""
    return batch[DEFAULT_ZERO_VALUE], batch[DEFAULT_ONE_VALUE]


def extract_evaluation_batch_data(batch) -> Tuple:
    """Extract batch data for evaluation."""
    if len(batch) == DEFAULT_BATCH_FORMAT_3_ELEMENTS:  # DatasetSplit format
        image, label, pixel_values = batch
        return pixel_values, label
    elif len(batch) == DEFAULT_BATCH_FORMAT_2_ELEMENTS:  # Standard format
        return batch[DEFAULT_ZERO_VALUE], batch[DEFAULT_ONE_VALUE]
    else:
        raise ValueError(f"Invalid batch length for evaluation: {len(batch)}")


def compute_batch_evaluation_metrics(pixel_values, labels, model, server_round: int, batch_idx: int, client_id: int) -> Tuple[float, int, int]:
    """
    Compute actual evaluation metrics for a batch of data.

    Args:
        pixel_values: Batch of image data (tensor or None)
        labels: Batch of labels (tensor or None)
        model: PyTorch model
        server_round: Current server round
        batch_idx: Batch index within evaluation
        client_id: Client identifier

    Returns:
        Tuple of (batch_loss, num_correct, batch_size)
    """
    if pixel_values is None or labels is None:
        raise ValueError(f"Invalid pixel_values or labels: {pixel_values}, {labels}")
    
    batch_size = pixel_values.shape[0] if hasattr(pixel_values, 'shape') else labels.shape[0]
    
    # Move to device
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    labels = labels.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Get predictions
        predictions = torch.argmax(logits, dim=DEFAULT_DIMENSION_1)
        num_correct = int(torch.sum(predictions == labels).item())

    logging.debug(f"Eval batch {batch_idx}: size={batch_size}, "
                  f"loss={loss:.4f}, correct={num_correct}")

    return float(loss.item()), num_correct, batch_size

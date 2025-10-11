"""
Data loading functionality for the Flower client.

This module contains all data loading, dataset configuration, and data management
functionality for the Flower client.

Author: Team1-FL-RHLA
Version: 1.0.0
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from client_constants import *
from client_config import DatasetArgs
from algorithms.solver.shared_utils import load_data


class ClientDataLoadingMixin:
    """
    Mixin class providing data loading functionality for the Flower client.
    
    This class contains all data loading, dataset configuration, and data management
    methods that can be mixed into the main FlowerClient class.
    """
    
    def _load_dataset_config(self) -> Dict[str, Any]:
        """
        Load dataset configuration and prepare dataset information.
        
        Returns:
            Dictionary containing dataset information
        """
        dataset_name = self._get_dataset_name()
        dataset_info = self._create_base_dataset_info(dataset_name)
        
        self._validate_dataset_name(dataset_name)
        self._apply_dataset_specific_config(dataset_info, dataset_name)
        
        # Load actual dataset if available
        dataset_info[CONFIG_KEY_DATA_LOADED] = self._load_dataset(dataset_name)
        logging.info(f"Dataset loading result: {dataset_info[CONFIG_KEY_DATA_LOADED]}")
        
        # Update with actual data if loaded successfully
        if dataset_info[CONFIG_KEY_DATA_LOADED]:
            self._update_dataset_info_with_loaded_data(dataset_info)

        logging.info(f"Dataset configuration loaded: {dataset_name} "
                     f"({dataset_info[CONFIG_KEY_NUM_CLASSES]} classes, {dataset_info[CONFIG_KEY_DATA_TYPE]} data)")

        return dataset_info
    
    def _get_dataset_name(self) -> str:
        """Get dataset name from configuration."""
        return self.args.get(CONFIG_KEY_DATASET, DEFAULT_DATASET)
    
    def _create_base_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Create base dataset information dictionary."""
        return {
            CONFIG_KEY_DATASET_NAME: dataset_name,
            CONFIG_KEY_DATA_TYPE: self.args.get(CONFIG_KEY_DATA_TYPE, DEFAULT_IMAGE_DATA_TYPE),
            CONFIG_KEY_MODEL_NAME: self.args.get(CONFIG_KEY_MODEL, DEFAULT_MODEL),
            CONFIG_KEY_BATCH_SIZE: self.args.get(CONFIG_KEY_BATCH_SIZE, DEFAULT_BATCH_SIZE),
            CONFIG_KEY_NUM_CLASSES: None,
            CONFIG_KEY_LABELS: None,
            CONFIG_KEY_LABEL2ID: None,
            CONFIG_KEY_ID2LABEL: None,
            CONFIG_KEY_DATA_LOADED: False
        }
    
    def _validate_dataset_name(self, dataset_name: str) -> None:
        """Validate dataset name."""
        if not dataset_name or not isinstance(dataset_name, str):
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        logging.info(f"Loading dataset configuration for: {dataset_name}")
    
    def _apply_dataset_specific_config(self, dataset_info: Dict[str, Any], dataset_name: str) -> None:
        """Apply dataset-specific configuration."""
        if dataset_name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_name]
            dataset_info[CONFIG_KEY_NUM_CLASSES] = config[CONFIG_KEY_NUM_CLASSES]
            dataset_info[CONFIG_KEY_DATA_TYPE] = config[CONFIG_KEY_DATA_TYPE]
        else:
            raise ValueError(ERROR_INVALID_DATASET.format(dataset_name=dataset_name))
    
    def _update_dataset_info_with_loaded_data(self, dataset_info: Dict[str, Any]) -> None:
        """Update dataset info with actual loaded data."""
        if hasattr(self, 'dataset_train'):
            dataset_info.update({
                CONFIG_KEY_TRAIN_SAMPLES: len(self.dataset_train) if self.dataset_train else DEFAULT_ZERO_VALUE,
                CONFIG_KEY_TEST_SAMPLES: len(self.dataset_test) if self.dataset_test else DEFAULT_ZERO_VALUE,
                CONFIG_KEY_NUM_USERS: len(self.client_data_partition) if self.client_data_partition else DEFAULT_ZERO_VALUE,
                CONFIG_KEY_CLIENT_DATA_INDICES: getattr(self, CONFIG_KEY_DATASET_INFO, {}).get(CONFIG_KEY_CLIENT_DATA_INDICES, set()),
                CONFIG_KEY_NONIID_TYPE: getattr(self.args_loaded, CONFIG_KEY_NONIID_TYPE, DEFAULT_DIRICHLET_TYPE) if hasattr(self, 'args_loaded') else DEFAULT_DIRICHLET_TYPE
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

        # Log dataset loading information
        logging.info(f'Dataset train length: {len(dataset_train)}')
        logging.info(f'Dataset test length: {len(dataset_test)}')
        logging.info(f'Client data partition length: {len(client_data_partition)}')
        logging.info(f'Dataset FIM length: {len(dataset_fim) if dataset_fim else 0}')
        logging.debug(f'Args loaded: {args_loaded}')

        # Validate that we have the required data
        if not dataset_train:
            raise ValueError("Failed to load training dataset")
            
        if not client_data_partition:
            raise ValueError("Failed to load user data partition")

        return (args_loaded, dataset_train, dataset_test, client_data_partition, dataset_fim)
    
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
        if not hasattr(args_loaded, CONFIG_KEY_TR_LABELS) or args_loaded._tr_labels is None:
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

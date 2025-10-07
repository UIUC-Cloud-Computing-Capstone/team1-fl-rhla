"""
Simplified Flower Client for Testing
This version doesn't require data loading and can be used for basic testing
"""
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
import os
import sys
import yaml
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.solver.fl_utils import (
    compute_model_update,
    compute_update_norm,
    get_optimizer_parameters,
    setup_multiprocessing
)


class SimpleFlowerClient(fl.client.NumPyClient):
    """
    Simplified Flower client for testing without data dependencies.
    """
    
    def __init__(self, args, client_id=0):
        self.args = args
        self.client_id = client_id
        
        # Setup multiprocessing for optimal CPU utilization
        self.num_cores = setup_multiprocessing()
        logging.info(f"Client {client_id} initialized with {self.num_cores} CPU cores")
        
        # Initialize loss function based on data type
        if getattr(args, 'data_type', 'image') == 'image':
            self.loss_func = nn.CrossEntropyLoss()
        elif getattr(args, 'data_type', 'image') == 'text':
            self.loss_func = nn.CrossEntropyLoss()
        elif getattr(args, 'data_type', 'image') == 'sentiment':
            self.loss_func = nn.NLLLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        
        # Initialize training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'rounds': []
        }
        
        # Create dummy model parameters for testing
        self.model_params = self._create_dummy_model_params()
    
    def _create_dummy_model_params(self):
        """Create dummy model parameters for testing"""
        # Create some dummy parameters based on typical model sizes
        params = []
        
        # Add some dummy weight matrices
        params.append(np.random.randn(768, 768).astype(np.float32))  # Hidden layer
        params.append(np.random.randn(768).astype(np.float32))       # Bias
        params.append(np.random.randn(100, 768).astype(np.float32))  # Output layer
        params.append(np.random.randn(100).astype(np.float32))       # Output bias
        
        # Add LoRA parameters if specified
        if getattr(self.args, 'peft', '') == 'lora':
            lora_layers = getattr(self.args, 'lora_layer', 12)
            for i in range(lora_layers):
                params.append(np.random.randn(64, 768).astype(np.float32))  # LoRA A
                params.append(np.random.randn(768, 64).astype(np.float32))  # LoRA B
        
        return params
    
    def get_parameters(self, config):
        """
        Get current model parameters.
        """
        return self.model_params
    
    def set_parameters(self, parameters):
        """
        Set model parameters from server.
        """
        self.model_params = [param.copy() for param in parameters]
    
    def fit(self, parameters, config):
        """
        Train the model on local data (simulated).
        """
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Extract configuration
        server_round = config.get('server_round', 0)
        local_epochs = config.get('local_epochs', getattr(self.args, 'tau', 1))
        learning_rate = config.get('learning_rate', getattr(self.args, 'local_lr', 0.01))
        
        logging.info(f"Client {self.client_id} starting training for round {server_round}")
        
        # Simulate training by adding small random updates to parameters
        total_loss = 0.0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            
            # Simulate training steps
            for step in range(10):  # Simulate 10 training steps per epoch
                # Simulate forward pass and loss calculation
                loss = np.random.exponential(1.0)  # Simulate decreasing loss
                epoch_loss += loss
                
                # Simulate parameter updates
                for i, param in enumerate(self.model_params):
                    # Add small random update
                    update = np.random.normal(0, 0.01, param.shape).astype(param.dtype)
                    self.model_params[i] = param + learning_rate * update
            
            total_loss += epoch_loss / 10  # Average loss per epoch
        
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
        
        # Simulate number of examples (random between 100-1000)
        num_examples = np.random.randint(100, 1000)
        
        return self.get_parameters(config), num_examples, metrics
    
    def evaluate(self, parameters, config):
        """
        Evaluate the model on local test data (simulated).
        """
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Simulate evaluation
        # Simulate accuracy (should improve over rounds)
        base_accuracy = 0.5 + min(0.4, server_round * 0.01)  # Improve over rounds
        accuracy = base_accuracy + np.random.normal(0, 0.05)  # Add some noise
        accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]
        
        # Simulate loss (should decrease over rounds)
        base_loss = 2.0 - min(1.5, server_round * 0.01)  # Decrease over rounds
        loss = base_loss + np.random.normal(0, 0.1)  # Add some noise
        loss = max(0.1, loss)  # Clamp to positive values
        
        # Update training history
        self.training_history['accuracies'].append(accuracy)
        
        logging.info(f"Client {self.client_id} evaluation: "
                    f"Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Return loss, number of examples, and metrics
        metrics = {
            'accuracy': accuracy,
            'client_id': self.client_id
        }
        
        # Simulate number of test examples
        num_examples = np.random.randint(50, 200)
        
        return loss, num_examples, metrics


def create_simple_flower_client(args, client_id=0):
    """
    Create a simplified Flower client instance.
    """
    return SimpleFlowerClient(args, client_id)


def start_simple_flower_client(args, server_address="localhost", server_port=8080, client_id=0):
    """
    Start a simplified Flower client.
    """
    # Create client
    client = create_simple_flower_client(args, client_id)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - Client {client_id} - %(levelname)s - %(message)s'
    )
    
    # Start client
    logging.info(f"Starting simplified Flower client {client_id} connecting to {server_address}:{server_port}")
    
    fl.client.start_numpy_client(
        server_address=f"{server_address}:{server_port}",
        client=client,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified Flower Client for Testing")
    parser.add_argument("--server_address", type=str, default="localhost", help="Server address")
    parser.add_argument("--server_port", type=int, default=8080, help="Server port")
    parser.add_argument("--client_id", type=int, default=0, help="Client ID")
    parser.add_argument("--config_name", type=str, 
                       default="experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml",
                       help="Configuration file")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID (-1 for CPU)")
    
    meta_args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join('config/', meta_args.config_name)
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to namespace-like object
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = Config(config_dict)
    
    # Merge arguments
    for arg in vars(meta_args):
        setattr(config, arg, getattr(meta_args, arg))
    
    # Setup device
    config.device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() and config.gpu != -1 else 'cpu')
    
    # Set random seeds
    torch.manual_seed(config.seed + meta_args.client_id)  # Different seed per client
    np.random.seed(config.seed + meta_args.client_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    start_simple_flower_client(config, meta_args.server_address, meta_args.server_port, meta_args.client_id)

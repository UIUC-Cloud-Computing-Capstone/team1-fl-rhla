"""
Shared utilities for federated learning implementations.

This module contains reusable functions extracted from ffm_fedavg_depthffm_fim.py
to be used by both the original implementation and the Flower client.
"""

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.data_pre_process import load_partition, DatasetSplit
from utils.model_utils import model_setup
from fractions import Fraction
import re


def load_data(args):
    """
    Load dataset and partition data for federated learning.
    
    Args:
        args: Configuration arguments
        
    Returns:
        Tuple of (args, dataset_train, dataset_test, dict_users, dataset_fim)
    """
    args.logger.info("{:<50}".format("-" * 15 + " data setup " + "-" * 50)[0:60], main_process_only=True)
    args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users, dataset_fim = load_partition(args)
    args.logger.info('length of dataset:{}'.format(len(dataset_train) + len(dataset_test)), main_process_only=True)
    args.logger.info('num. of training data:{}'.format(len(dataset_train)), main_process_only=True)
    args.logger.info('num. of testing data:{}'.format(len(dataset_test)), main_process_only=True)
    return args, dataset_train, dataset_test, dict_users, dataset_fim


def get_data_loader_list(args, dataset_train, dict_users):
    """
    Create data loaders for all clients.
    
    Args:
        args: Configuration arguments
        dataset_train: Training dataset
        dict_users: User data partition dictionary
        
    Returns:
        List of DataLoaders for each client
    """
    data_loader_list = []
    for i in range(args.num_users):
        dataset = DatasetSplit(dataset_train, dict_users[i], args)
        if 'vit' in args.model:
            ldr_train = DataLoader(dataset, shuffle=True, collate_fn=vit_collate_fn, batch_size=args.batch_size)
        elif 'ledgar' in args.dataset:
            ldr_train = DataLoader(dataset, shuffle=True, collate_fn=args.data_collator, batch_size=args.batch_size)
        data_loader_list.append(ldr_train)
    return data_loader_list


def get_dataset_fim(args, dataset_fim):
    """
    Create DataLoader for FIM dataset.
    
    Args:
        args: Configuration arguments
        dataset_fim: FIM dataset
        
    Returns:
        DataLoader for FIM dataset
    """
    if 'vit' in args.model:
        dataset_fim = DataLoader(dataset_fim, collate_fn=test_collate_fn, batch_size=args.batch_size)
    elif 'ledgar' in args.dataset:
        dataset_fim = DataLoader(dataset_fim, shuffle=True, collate_fn=args.data_collator, batch_size=args.batch_size)
    return dataset_fim


def vit_collate_fn(examples):
    """
    Collate function for ViT models.
    
    Args:
        examples: List of examples from dataset
        
    Returns:
        Dictionary with pixel_values and labels
    """
    pixel_values = torch.stack([example[2] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def test_collate_fn(examples):
    """
    Collate function for test datasets.
    
    Args:
        examples: List of examples from dataset
        
    Returns:
        Dictionary with pixel_values and labels
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    if 'label' in examples[0]:
        labels = torch.tensor([example["label"] for example in examples])
    else:
        labels = torch.tensor([example["fine_label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def get_group_cnt(args):
    """
    Calculate group counts for heterogeneous federated learning.
    
    Args:
        args: Configuration arguments
        
    Returns:
        List of group counts
    """
    group_num = len(args.heterogeneous_group)
    group_cnt = []
    for g in range(group_num):
        if g == (group_num - 1):
            remind_cnt = args.num_users
            for c in group_cnt:
                remind_cnt -= c
            group_cnt.append(remind_cnt)
        else:    
            group_cnt.append(int(args.num_users * float(Fraction(args.heterogeneous_group[g]))))
    return group_cnt


# TODO Liam: refactor this
def update_user_groupid_list(args, group_cnt):
    """
    Create user group ID list for heterogeneous federated learning.
    
    Args:
        args: Configuration arguments
        group_cnt: List of group counts
    """
    args.user_groupid_list = []
    for id, c in enumerate(group_cnt):
        args.user_groupid_list += [id] * c


def update_block_ids_list(args):
    """
    Update block IDs list for heterogeneous LoRA training.
    
    Args:
        args: Configuration arguments
    """
    args.block_ids_list = []
    for id in args.user_groupid_list:
        layer_list = np.random.choice(range(args.lora_layer),
                                                    p=[float(Fraction(x)) for x in args.layer_prob],
                                                    size=getattr(args, 'heterogeneous_group'+str(id)+'_lora'),
                                                    replace=False)
        args.block_ids_list.append(sorted(layer_list))


def update_block_ids_list_with_observed_probability(args, observed_probability):
    """
    Update block IDs list with observed probability for FIM-based selection.
    
    Args:
        args: Configuration arguments
        observed_probability: Observed probability distribution
    """
    args.block_ids_list = []
    for id in args.user_groupid_list:
        layer_list = np.random.choice(range(args.lora_layer),
                                            p=observed_probability,
                                            size=getattr(args, 'heterogeneous_group'+str(id)+'_lora'),
                                            replace=False)
        args.block_ids_list.append(sorted(layer_list))


def get_observed_prob(cluster_labels):
    """
    Get observed probability from cluster labels.
    
    Args:
        cluster_labels: Cluster labels from FIM analysis
        
    Returns:
        Normalized observed probability array
    """
    observed_probability = get_observed_probability(cluster_labels)
    observed_probability = np.array([float(Fraction(x)) for x in observed_probability])
    observed_probability /= sum(observed_probability)
    return observed_probability


def get_observed_probability(cluster_labels):
    """
    Convert cluster labels to probability fractions.
    
    Args:
        cluster_labels: Cluster labels
        
    Returns:
        List of probability fractions
    """
    observed_probability = []
    for label in cluster_labels:
        if label == 0:
            observed_probability.append('1/27')
        elif label == 1:
            observed_probability.append('2/27')
        elif label == 2:
            observed_probability.append('1/9')
    return observed_probability


def get_model_update(args, global_model, local_model, no_weight_lora):
    """
    Compute model update for federated learning.
    
    Args:
        args: Configuration arguments
        global_model: Global model state dict
        local_model: Local model state dict
        no_weight_lora: List of LoRA layers not trained
        
    Returns:
        Model update dictionary
    """
    model_update = {}
    if args.peft == 'lora':
        for k in global_model.keys():
            if 'lora' in k: # no classifier
                if int(re.findall(r"\d+", k)[0]) not in no_weight_lora:
                    model_update[k] = local_model[k].detach().cpu() - global_model[k].detach().cpu() 
    else:
        model_update = {k: local_model[k].detach().cpu() - global_model[k].detach().cpu() for k in global_model.keys()}
    return model_update


def get_lora_parameter_mapping(model_state_dict, no_weight_lora):
    """
    Create a mapping of LoRA parameters that were actually trained.
    
    Args:
        model_state_dict: Model state dictionary
        no_weight_lora: List of LoRA layers not trained
        
    Returns:
        Dictionary mapping parameter names to their indices and layer numbers
    """
    import re
    
    lora_mapping = {}
    param_index = 0
    
    for name, param in model_state_dict.items():
        if 'lora' in name.lower():
            # Extract layer number from parameter name
            layer_numbers = re.findall(r'\d+', name)
            if layer_numbers:
                layer_num = int(layer_numbers[0])
                # Only include parameters from layers that were trained
                if layer_num not in no_weight_lora:
                    lora_mapping[name] = {
                        'index': param_index,
                        'layer_num': layer_num,
                        'shape': param.shape,
                        'param': param
                    }
                    param_index += 1
    
    return lora_mapping


def reconstruct_full_model_from_lora(lora_params, lora_mapping, base_model_state_dict):
    """
    Reconstruct full model state dict from LoRA parameters and mapping.
    
    Args:
        lora_params: List of LoRA parameters (numpy arrays)
        lora_mapping: Mapping of parameter names to indices
        base_model_state_dict: Base model state dictionary
        
    Returns:
        Full model state dictionary with LoRA parameters updated
    """
    import torch
    
    # Start with base model
    full_state_dict = base_model_state_dict.copy()
    
    # Update with LoRA parameters
    for param_name, mapping_info in lora_mapping.items():
        param_index = mapping_info['index']
        if param_index < len(lora_params):
            # Convert numpy array back to tensor
            lora_tensor = torch.from_numpy(lora_params[param_index])
            full_state_dict[param_name] = lora_tensor
    
    return full_state_dict


def get_norm_updates(model_update):
    """
    Get norm updates for model parameters.
    
    Args:
        model_update: Model update dictionary
        
    Returns:
        List of flattened parameter tensors
    """
    norm_updates = []
    for k in model_update.keys():
        norm_updates.append(torch.flatten(model_update[k]))
    return norm_updates


def get_train_loss(local_losses):
    """
    Calculate average training loss.
    
    Args:
        local_losses: List of local training losses
        
    Returns:
        Average training loss
    """
    if len(local_losses) > 0:
        train_loss = sum(local_losses) / len(local_losses)
    else:
        train_loss = 100
    return train_loss


def get_norm(delta_norms):
    """
    Calculate median norm of model updates.
    
    Args:
        delta_norms: List of delta norms
        
    Returns:
        Median norm
    """
    if len(delta_norms) > 0:
        norm = torch.median(torch.stack(delta_norms)).cpu()
    else:
        norm = 100
    return norm


def update_delta_norms(delta_norms, norm_updates):
    """
    Update delta norms list with new norm updates.
    
    Args:
        delta_norms: List of delta norms to update
        norm_updates: New norm updates to add
    """
    if len(norm_updates) > 0:
        delta_norm = torch.norm(torch.cat(norm_updates))
    else:
        delta_norm = None
    if delta_norm:
        delta_norms.append(delta_norm)


def create_client_dataset(dataset_train, client_indices, args_loaded):
    """
    Create a client-specific dataset subset.
    
    Args:
        dataset_train: Full training dataset
        client_indices: Indices for this client's data
        args_loaded: Loaded arguments
        
    Returns:
        Dataset subset for this client
    """
    return DatasetSplit(dataset_train, client_indices, args_loaded)


def create_client_dataloader(client_dataset, args, collate_fn=None):
    """
    Create DataLoader for a client's dataset with memory management.
    
    Args:
        client_dataset: Client's dataset subset
        args: Configuration arguments
        collate_fn: Collate function to use
        
    Returns:
        DataLoader for the client
    """
    if collate_fn is None:
        if 'vit' in args.model:
            collate_fn = vit_collate_fn
        elif 'ledgar' in args.dataset:
            collate_fn = args.data_collator
    
    # Use memory-safe batch size
    batch_size = getattr(args, 'batch_size', 32)
    
    # Reduce batch size for memory-constrained environments
    if batch_size > 16:
        batch_size = 16
    
    return DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues in federated setting
        pin_memory=False  # Disable pin_memory for memory efficiency
    )

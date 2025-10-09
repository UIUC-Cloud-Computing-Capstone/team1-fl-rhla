"""
Common utilities for federated learning operations.
Refactored from global_aggregator.py and local_solver.py for reuse.
"""
import torch
import numpy as np
import copy
from typing import Dict, List, Tuple, Any
from collections import defaultdict




def aggregate_updates_simple(local_updates: List[Dict[str, torch.Tensor]], 
                           global_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Simple averaging of local updates.
    
    Args:
        local_updates: List of local model updates
        global_model: Current global model
        
    Returns:
        Updated global model
    """
    if not local_updates:
        return global_model
        
    model_update = {k: local_updates[0][k] * 0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        model_update = {k: model_update[k] + local_updates[i][k] for k in global_model.keys()}

    global_model = {k: global_model[k] + model_update[k] / len(local_updates) for k in global_model.keys()}
    return global_model


def aggregate_updates_lora(local_updates: List[Dict[str, torch.Tensor]], 
                          global_model: Dict[str, torch.Tensor], 
                          peft: str = 'lora') -> Dict[str, torch.Tensor]:
    """
    LoRA-aware averaging of local updates.
    
    Args:
        local_updates: List of local model updates
        global_model: Current global model
        peft: Parameter efficient fine-tuning method
        
    Returns:
        Updated global model
    """
    if not local_updates:
        return global_model
        
    model_update = {k: local_updates[0][k] * 0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        if peft == 'lora':
            model_update = {k: model_update[k] + local_updates[i][k] 
                           for k in global_model.keys() if 'lora' in k}
        else:
            model_update = {k: model_update[k] + local_updates[i][k] for k in global_model.keys()}
    
    if peft == 'lora':
        for k in global_model.keys():
            if 'lora' in k:
                global_model[k] = global_model[k].detach().cpu() + model_update[k] / len(local_updates)
    else:
        global_model = {k: global_model[k].detach().cpu() + model_update[k] / len(local_updates) 
                       for k in global_model.keys()}

    return global_model


def aggregate_updates_heterogeneous(local_updates: List[Dict[str, torch.Tensor]], 
                                   global_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Heterogeneous averaging for LoRA layers (DepthFL style).
    
    Args:
        local_updates: List of local model updates
        global_model: Current global model
        
    Returns:
        Updated global model
    """
    if not local_updates:
        return global_model
        
    model_update_avg_dict = {}

    for k in global_model.keys():
        if 'lora' in k or 'classifier' in k:
            for loc_update in local_updates:
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])

    for k in global_model.keys():
        if k in model_update_avg_dict:
            global_model[k] = global_model[k].detach().cpu() + sum(model_update_avg_dict[k]) / len(model_update_avg_dict[k])

    return global_model


def aggregate_updates_weighted_heterogeneous(local_updates: List[Dict[str, torch.Tensor]], 
                                           global_model: Dict[str, torch.Tensor], 
                                           num_samples: List[int]) -> Dict[str, torch.Tensor]:
    """
    Weighted heterogeneous averaging based on number of samples.
    
    Args:
        local_updates: List of local model updates
        global_model: Current global model
        num_samples: List of sample counts for each client
        
    Returns:
        Updated global model
    """
    if not local_updates:
        return global_model
        
    model_update_avg_dict = {}
    model_weights_cnt = {}
    model_weights_list = {}

    for k in global_model.keys():
        if 'lora' in k or 'classifier' in k:
            for client_i, loc_update in enumerate(local_updates):
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                        model_weights_cnt[k] += num_samples[client_i]
                        model_weights_list[k].append(num_samples[client_i])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])
                        model_weights_cnt[k] = 0.0
                        model_weights_cnt[k] += num_samples[client_i]
                        model_weights_list[k] = []
                        model_weights_list[k].append(num_samples[client_i])

    for k in global_model.keys():
        if k in model_update_avg_dict:
            weight_based = model_weights_cnt[k]
            model_weights_list[k] = [item/weight_based for item in model_weights_list[k]]
            global_model[k] = global_model[k].detach().cpu() + sum([model*weight for model, weight in zip(model_update_avg_dict[k], model_weights_list[k])])

    return global_model

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def setup_multiprocessing():
    """
    Setup multiprocessing for optimal CPU utilization.
    """
    import torch
    import os
    
    # Set number of threads for PyTorch
    num_cores = os.cpu_count()
    torch.set_num_threads(num_cores)
    
    # Set environment variables for optimal performance
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ['MKL_NUM_THREADS'] = str(num_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_cores)
    
    return num_cores

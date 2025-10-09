"""
Common utilities for federated learning operations.
Refactored from global_aggregator.py and local_solver.py for reuse.
"""

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

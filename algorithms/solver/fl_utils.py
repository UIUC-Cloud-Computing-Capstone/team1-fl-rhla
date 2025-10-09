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

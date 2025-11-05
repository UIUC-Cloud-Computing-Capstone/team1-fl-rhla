import copy
import numpy as np
import time, math
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_pre_process import load_partition, DatasetSplit
from utils.model_utils import model_setup
from utils.log_utils import set_log_path
from test import test, test_vit, test_ledgar

from ..solver.local_solver import LocalUpdate
from ..solver.global_aggregator import average, average_lora, average_lora_depthfl, weighted_average_lora_depthfl
import gc
from fractions import Fraction
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from dataclasses import dataclass
from utils.fim_calculator import FIMCalculator
import torch.distributed as dist
import threading
import random

gpu_lock = threading.Lock()

def vit_collate_fn(examples):
    """
    Collate function for Vision Transformer (ViT) training data.
    
    This function processes a batch of examples for ViT model training by extracting
    pixel values and labels, then stacking them into proper batch tensors.
    It assumes the dataset format where each example is a tuple with label at index 1
    and pixel values at index 2.
    
    Args:
        examples (list): List of examples, where each example is a tuple containing:
                        - Index 0: (unused in this function)
                        - Index 1: Label (int or tensor)
                        - Index 2: Pixel values (tensor with shape [C, H, W])
    
    Returns:
        dict: Dictionary containing:
            - "pixel_values": Stacked tensor of pixel values with shape [batch_size, C, H, W]
            - "labels": Tensor of labels with shape [batch_size]
    
    Example:
        >>> examples = [
        ...     (None, 0, torch.randn(3, 224, 224)),
        ...     (None, 1, torch.randn(3, 224, 224)),
        ...     (None, 2, torch.randn(3, 224, 224))
        ... ]
        >>> batch = vit_collate_fn(examples)
        >>> print(batch["pixel_values"].shape)  # torch.Size([3, 3, 224, 224])
        >>> print(batch["labels"])  # tensor([0, 1, 2])
    
    Note:
        - Pixel values are stacked along the batch dimension (dimension 0)
        - Labels are converted to tensors if they aren't already
        - This function is specifically designed for ViT model input format
        - Assumes all images have the same dimensions
    """
    pixel_values = torch.stack([example[2] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def test_collate_fn(examples):
    """
    Collate function for Vision Transformer (ViT) testing data.
    
    This function processes a batch of examples for ViT model testing by extracting
    pixel values and labels from dictionary-formatted examples. It handles different
    label field names that may be used in different datasets.
    
    Args:
        examples (list): List of examples, where each example is a dictionary containing:
                        - "pixel_values": Pixel values tensor with shape [C, H, W]
                        - "label" or "fine_label": Label value (int or tensor)
    
    Returns:
        dict: Dictionary containing:
            - "pixel_values": Stacked tensor of pixel values with shape [batch_size, C, H, W]
            - "labels": Tensor of labels with shape [batch_size]
    
    Label Field Handling:
        The function checks for two possible label field names:
        - "label": Standard label field (used in most datasets)
        - "fine_label": Fine-grained label field (used in CIFAR-100 and similar datasets)
    
    Example:
        >>> examples = [
        ...     {"pixel_values": torch.randn(3, 224, 224), "label": 0},
        ...     {"pixel_values": torch.randn(3, 224, 224), "label": 1},
        ...     {"pixel_values": torch.randn(3, 224, 224), "fine_label": 2}
        ... ]
        >>> batch = test_collate_fn(examples)
        >>> print(batch["pixel_values"].shape)  # torch.Size([3, 3, 224, 224])
        >>> print(batch["labels"])  # tensor([0, 1, 2])
    
    Note:
        - Pixel values are stacked along the batch dimension (dimension 0)
        - Labels are converted to tensors if they aren't already
        - Handles mixed label field names within the same batch
        - This function is specifically designed for ViT model testing
        - Assumes all images have the same dimensions
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    if 'label' in examples[0]:
        labels = torch.tensor([example["label"] for example in examples])
    else:
        labels = torch.tensor([example["fine_label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator for multiple choice question answering tasks.
    
    This class handles the collation of multiple choice examples by dynamically padding
    input sequences and reshaping them into the proper format for multiple choice models.
    It processes batches where each example contains multiple choice options and selects
    the correct answer based on the provided label.
    
    Attributes:
        tokenizer (PreTrainedTokenizerBase): Tokenizer used for padding and processing
        padding (Union[bool, str, PaddingStrategy]): Padding strategy for sequences
        max_length (Optional[int]): Maximum sequence length for padding
        pad_to_multiple_of (Optional[int]): Pad sequences to multiples of this value
    
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        >>> collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
        >>> features = [
        ...     {
        ...         "input_ids": [["Hello", "world"], ["Hi", "there"], ["Good", "morning"]],
        ...         "attention_mask": [[1, 1], [1, 1], [1, 1]],
        ...         "correct_answer_num": 2
        ...     }
        ... ]
        >>> batch = collator(features)
        >>> print(batch["input_ids"].shape)  # torch.Size([1, 3, max_length])
        >>> print(batch["labels"])  # tensor([1])  # 0-indexed
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        """
        Collate a batch of multiple choice examples.
        
        Args:
            features (list): List of examples, where each example is a dictionary containing:
                            - "input_ids": List of tokenized sequences for each choice
                            - "token_type_ids": (optional) List of token type IDs for each choice
                            - "attention_mask": List of attention masks for each choice
                            - "correct_answer_num": Integer indicating the correct choice (1-indexed)
        
        Returns:
            dict: Collated batch containing:
                - "input_ids": Tensor with shape [batch_size, num_choices, max_length]
                - "token_type_ids": Tensor with shape [batch_size, num_choices, max_length]
                - "attention_mask": Tensor with shape [batch_size, num_choices, max_length]
                - "labels": Tensor with shape [batch_size] containing correct choice indices (0-indexed)
        
        Algorithm:
            1. Extract labels and convert to 0-indexed format
            2. Flatten all choice sequences for efficient padding
            3. Apply tokenizer padding to flattened sequences
            4. Reshape back to [batch_size, num_choices, max_length] format
            5. Add labels tensor to the batch
        """
        label_name = "correct_answer_num"
        labels = [torch.tensor(int(feature[label_name])-1) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [
            [{k: v[i] for k, v in feature.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def ffm_fedavg_depthffm_fim(args):
    """
    Main federated learning algorithm implementing FIM-based adaptive layer selection with heterogeneous clients.
    
    This function implements a federated learning algorithm that uses Fisher Information Matrix (FIM) analysis
    to adaptively select which LoRA layers each client should train. The algorithm supports heterogeneous
    clients with different computational capabilities and model configurations.
    
    The algorithm follows these key steps:
    1. Initialize data loaders and model setup
    2. Set up heterogeneous client groups with different LoRA layer configurations
    3. For each communication round:
       - Update layer assignments based on FIM analysis (after initial epochs)
       - Select random subset of clients for training
       - Perform local training with LoRA fine-tuning
       - Aggregate model updates using weighted averaging
       - Test global model performance
    
    Args:
        args: Configuration object containing all hyperparameters and settings:
            - num_users (int): Total number of clients in the system
            - num_selected_users (int): Number of clients selected per round
            - round (int): Total number of communication rounds
            - fim_prior_epoch (int): Epochs to wait before starting FIM analysis
            - fim_every_iter (int): Interval for FIM analysis updates
            - heterogeneous_group (list): Proportions for client groups
            - heterogeneous_group{i}_lora (int): Number of LoRA layers for group i
            - lora_layer (int): Total available LoRA layers
            - layer_prob (list): Predefined layer selection probabilities
            - local_lr (float): Local learning rate
            - lr_step_size (int): Learning rate decay step size
            - decay_weight (float): Learning rate decay factor
            - peft (str): Parameter-efficient fine-tuning method ('lora')
            - model (str): Base model identifier
            - dataset (str): Dataset name
            - batch_size (int): Training batch size
            - device (torch.device): Computing device
            - logger: Logging object
            - accelerator: Accelerator for distributed training
            - log_path (str): Path for logging results
    
    Returns:
        tuple: A tuple containing:
            - (best_test_acc, best_test_f1, best_test_macro_f1, best_test_micro_f1): 
              Best performance metrics achieved during training
            - metric_keys (dict): Dictionary tracking which metrics achieved best performance
    
    Algorithm Flow:
        1. Data and Model Setup:
           - Load and partition datasets
           - Initialize global model with LoRA configuration
           - Set up data loaders for each client
        
        2. Heterogeneous Group Setup:
           - Calculate user distribution across groups
           - Create user-to-group mapping
           - Initialize layer assignment lists
        
        3. Training Loop (for each round):
           - Update layer assignments (FIM-based or probability-based)
           - Select random subset of clients
           - Local training with LoRA fine-tuning
           - Model update aggregation
           - Global model testing and logging
        
        4. Performance Tracking:
           - Monitor training loss and model update norms
           - Track best test accuracy, F1 scores
           - Log metrics to TensorBoard
    
    Example:
        >>> args = argparse.Namespace()
        >>> args.num_users = 100
        >>> args.num_selected_users = 10
        >>> args.round = 200
        >>> args.fim_prior_epoch = 50
        >>> args.fim_every_iter = 50
        >>> args.heterogeneous_group = ['1/3', '1/3', '1/3']
        >>> # ... set other required parameters
        >>> best_metrics, metric_keys = ffm_fedavg_depthffm_fim(args)
        >>> print(f"Best accuracy: {best_metrics[0]:.3f}")
    
    Note:
        - Uses GPU lock for thread-safe FIM computation
        - Supports multiple datasets (CIFAR-100, SST-2, QQP, QNLI, LEDGAR, Belebele)
        - Implements adaptive layer selection based on layer importance
        - Supports both IID and non-IID data distributions
        - Uses weighted aggregation based on client data sizes
    """
    ################################### hyperparameter setup ########################################
    args, dataset_train, dataset_test, dict_users, dataset_fim = load_data(args)
    
    args.logger.info('num. of users:{}'.format(len(dict_users)), main_process_only=True)
    sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users))
    args.logger.info('average num. of samples per user:{}'.format(sample_per_users), main_process_only=True)

    args.logger.info("{:<50}".format("-" * 15 + " log path " + "-" * 50)[0:60], main_process_only=True)
    if args.accelerator.is_local_main_process:
        writer = SummaryWriter(args.log_path)
    args.logger.info(args.log_path, main_process_only=True)
    
    args.logger.info("{:<50}".format("-" * 15 + " model setup " + "-" * 50)[0:60], main_process_only=True)
    args, net_glob, global_model, args.dim = model_setup(args)
    
    args.logger.info('model dim: '+str(args.dim), main_process_only=True)

    ###################################### model initialization ###########################
    t1 = time.time()
    args.logger.info("{:<50}".format("-" * 15 + " training... " + "-" * 50)[0:60], main_process_only=True)
    # initialize data loader for training and/or public dataset
    data_loader_list = get_data_loader_list(args, dataset_train, dict_users)
    dataset_fim = update_dataset_fim(args, dataset_fim)
    
    # heterogenity
    update_user_groupid_list(args)

    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_test_macro_f1 = 0.0
    best_test_micro_f1 = 0.0
    metric_keys = {'Accuracy': 0, 'F1': 0, 'Macro_F1': 0, "Micro_F1": 0}
    saved_block_ids_list = list()
    saved_rank_list = list()
    fim_prior_epoch = max(args.fim_prior_epoch,1)

    for t in range(args.round):
        args.logger.info('Round: ' + str(t) + '/' + str(args.round), main_process_only=True)
        
        # block ids for each clients, update every {fim_every_iter} round with pre-defined warm-start
        if t < fim_prior_epoch:
            update_block_ids_list_predefined(args, dataset_fim, net_glob, t)
            saved_block_ids_list = args.block_ids_list
            saved_rank_list = args.rank_list
        elif t >= fim_prior_epoch and t % args.fim_every_iter == 0:
            update_block_ids_list(args, dataset_fim, net_glob, t)
            saved_block_ids_list = args.block_ids_list
            saved_rank_list = args.rank_list
        else:
            args.block_ids_list = saved_block_ids_list
            args.saved_rank_list = saved_rank_list

        # debug list and rank:
        #args.block_ids_list[14] = [0,1]
        #args.rank_list[14] = [3,1]

        ## learning rate decaying
        decay_learning_rate(args, t)

        ## user selection
        selected_idxs = list(np.random.choice(range(args.num_users), args.num_selected_users, replace=False))
        print('selected client idxs: '+str(selected_idxs))
        
        ## local training
        local_losses, local_updates, delta_norms, num_samples = train_selected_clients(args, net_glob, global_model, data_loader_list, t, selected_idxs)

        if len(local_updates) == 0:
            args.logger.info('The number of trainable client is 0, skip the round for average')
            continue
        
        # metrics
        norm, train_loss = log_metrics(args, writer, t, local_losses, delta_norms)

        # global model update
        global_model = update_global_model(args, global_model, local_updates, num_samples)

        # test global model on server side   
        best_test_acc, best_test_f1, best_test_macro_f1, best_test_micro_f1 = test_global_model(args, dataset_test, writer, net_glob, global_model, best_test_acc, best_test_f1, best_test_micro_f1, best_test_macro_f1, metric_keys, t, norm, train_loss)

        args.accelerator.wait_for_everyone()

    return (best_test_acc, best_test_f1, best_test_macro_f1, best_test_micro_f1), metric_keys

def log_metrics(args, writer, t, local_losses, delta_norms):
    norm = get_norm(delta_norms)
    train_loss = get_train_loss(local_losses)
    if args.accelerator.is_local_main_process:
        writer.add_scalar('norm', norm, t)
        writer.add_scalar('train_loss', train_loss, t)
    return norm,train_loss

def test_global_model(args, dataset_test, writer, net_glob, global_model, best_test_acc, best_test_f1, best_test_micro_f1, best_test_macro_f1, metric_keys, t, norm, train_loss):
    net_glob.load_state_dict(global_model)
    net_glob.eval()
    if 'vit' in args.model:
        test_acc, test_loss = test_vit(copy.deepcopy(net_glob), dataset_test, args, t)
            # metrics
        if args.accelerator.is_local_main_process:
            writer.add_scalar('test_acc', test_acc, t)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                metric_keys['Accuracy'] = 1
        args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                format(t, train_loss, norm, test_acc), main_process_only=True)
    elif 'bert' in args.model:
        if 'sst2' in args.dataset:
            test_acc, test_loss = test_sst2(copy.deepcopy(net_glob), dataset_test, args, t)
                # metrics
            if args.accelerator.is_local_main_process:
                writer.add_scalar('test_acc', test_acc, t)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    metric_keys['Accuracy'] = 1
            args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                    format(t, train_loss, norm, test_acc), main_process_only=True)
        elif 'qqp' in args.dataset:
            test_f1, test_acc, test_loss = test_qqp(copy.deepcopy(net_glob), dataset_test, args, t)
                # metrics
            if args.accelerator.is_local_main_process:
                writer.add_scalar('test_acc', test_acc, t)
                writer.add_scalar('test_f1', test_f1, t)

                if test_f1 > best_test_f1:
                    best_test_f1 = test_f1
                        # best performance based on f1
                    best_test_acc = test_acc
                    metric_keys['Accuracy'] = 1
                    metric_keys['F1'] = 1
            args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_f1 = {:.3f}, test_acc = {:.3f}'.
                    format(t, train_loss, norm, test_f1, test_acc), main_process_only=True)
        elif 'qnli' in args.dataset:
            test_acc, test_loss = test_qnli(copy.deepcopy(net_glob), dataset_test, args, t)
                # metrics
            if args.accelerator.is_local_main_process:
                writer.add_scalar('test_acc', test_acc, t)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    metric_keys['Accuracy'] = 1
            args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                    format(t, train_loss, norm, test_acc), main_process_only=True)
        elif 'ledgar' in args.dataset:
            test_macro_f1, test_micro_f1, test_loss = test_ledgar(copy.deepcopy(net_glob), dataset_test, args, t)
                # metrics
            if args.accelerator.is_local_main_process:
                writer.add_scalar('test_macro_f1', test_macro_f1, t)
                writer.add_scalar('test_micro_f1', test_micro_f1, t)
                if test_micro_f1 > best_test_micro_f1:
                    best_test_micro_f1 = test_micro_f1
                    best_test_macro_f1 = test_macro_f1
                    metric_keys['Macro_F1'] = 1
                    metric_keys['Micro_F1'] = 1

            args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_macro_f1 = {:.3f}, test_micro_f1 = {:.3f}'.
                    format(t, train_loss, norm, test_macro_f1, test_micro_f1), main_process_only=True)
        elif 'belebele' in args.dataset:
            test_acc, test_loss = test_belebele(copy.deepcopy(net_glob), dataset_test, args, t)
                # metrics
            if args.accelerator.is_local_main_process:
                writer.add_scalar('test_acc', test_acc, t)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    metric_keys['Accuracy'] = 1
            args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                    format(t, train_loss, norm, test_acc), main_process_only=True)
    else:
        test_acc, test_loss = test(copy.deepcopy(net_glob), dataset_test, args)
            # metrics
        if args.accelerator.is_local_main_process:
            writer.add_scalar('test_acc', test_acc, t)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                metric_keys['Accuracy'] = 1
        args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                format(t, train_loss, norm, test_acc), main_process_only=True)
            
    return best_test_acc,best_test_f1,best_test_macro_f1,best_test_micro_f1

def train_selected_clients(args, net_glob, global_model, data_loader_list, t, selected_idxs):

    # sets the model to training mode. https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train
    net_glob.train()

    local_solver = LocalUpdate(args=args)
    local_models, local_losses, local_updates, delta_norms, num_samples = [], [], [], [], []
    for num_index, i in enumerate(selected_idxs):
        if args.peft == 'lora':

            #net_model = net_glob.state_dict()
            #print('################### before $$$$$$$$$$$$$$$$$$$$$')
            #for k in global_model.keys():
            #    if 'lora_A' in k:
            #        print(f'{k}, {net_model[k].detach() - global_model[k].detach()}')



            local_model, local_loss, no_weight_lora =  local_solver.lora_tuning(model=copy.deepcopy(net_glob),
                                                                                    ldr_train=data_loader_list[i],
                                                                                    args=args,
                                                                                    client_index=num_index,
                                                                                    client_real_id=i,
                                                                                    round=t,
                                                                                    hete_group_id=args.user_groupid_list[i])
            #print('################### after $$$$$$$$$$$$$$$$$$$$$')
            #for k in global_model.keys():
            #    if 'lora_A' in k:
            #        print(f'{k}, {local_model[k].detach() - global_model[k].detach()}')
                    
        if local_loss:
            local_losses.append(local_loss)
            # compute model update
        model_update = get_model_update(args, global_model, local_model, no_weight_lora)
            # compute model update norm
        norm_updates = get_norm_updates(model_update)
        append_delta_norm(delta_norms, norm_updates)

        local_updates.append(model_update)
        num_samples.append(len(data_loader_list[i]))
    return local_losses,local_updates,delta_norms,num_samples

def update_global_model(args, global_model, local_updates, num_samples):
    # TODO Liam: add aggregation function for heterogenous rank
    if hasattr(args, 'aggregation'):
        if args.aggregation ==  'weighted_average':
            global_model = weighted_average_lora_depthfl(args, global_model, local_updates, num_samples)
    else:
        global_model = average_lora_depthfl(args, global_model, local_updates)
    return global_model

def append_delta_norm(delta_norms, norm_updates):
    delta_norm = get_delta_norm(norm_updates)
            # delta_norm = torch.norm(torch.cat([torch.flatten(model_update[k])for k in model_update.keys()]))
    if delta_norm:
        delta_norms.append(delta_norm)

def get_delta_norm(norm_updates):
    if len(norm_updates) > 0:
        delta_norm = torch.norm(torch.cat(norm_updates))
    else:
        delta_norm = None
    return delta_norm

def get_norm_updates(model_update):
    norm_updates = []
    for k in model_update.keys():
        norm_updates.append(torch.flatten(model_update[k]))
    return norm_updates

def get_model_update(args, global_model, local_model, no_weight_lora):
    model_update = {}
    if args.peft == 'lora':
        for k in global_model.keys():
            if 'lora' in k: # no classifier
                if int(re.findall(r"\d+", k)[0]) not in no_weight_lora:
                    model_update[k] = local_model[k].detach().cpu() - global_model[k].detach().cpu() 
            elif args.train_classifier and 'classifier' in k:
                model_update[k] = local_model[k].detach().cpu() - global_model[k].detach().cpu() 
    else:
        model_update = {k: local_model[k].detach().cpu() - global_model[k].detach().cpu() for k in global_model.keys()}
    return model_update

def get_norm(delta_norms):
    """
    Calculate the median norm of model updates from selected clients.
    
    This function computes the median L2 norm of model updates from all participating
    clients in a federated learning round. The median norm is used as a measure of
    the magnitude of model changes and can be used for monitoring training progress
    and detecting potential issues like gradient explosion.
    
    Args:
        delta_norms (list): List of L2 norms of model updates from each participating client.
                           Each element should be a torch.Tensor representing the norm
                           of one client's model update.
    
    Returns:
        torch.Tensor: Median norm of all client updates, computed on CPU.
                     Returns 100.0 if no norms are provided (fallback value).
    
    Algorithm:
        1. Check if delta_norms list is non-empty
        2. If non-empty: Stack all norms and compute median
        3. If empty: Return fallback value of 100.0
        4. Move result to CPU for logging/storage
    
    Example:
        >>> delta_norms = [torch.tensor(2.5), torch.tensor(3.1), torch.tensor(1.8)]
        >>> median_norm = get_norm(delta_norms)
        >>> print(median_norm)  # tensor(2.5)
        
        >>> delta_norms = []
        >>> median_norm = get_norm(delta_norms)
        >>> print(median_norm)  # tensor(100.)
    
    Note:
        - Uses median instead of mean to be robust against outlier clients
        - The fallback value of 100.0 is arbitrary and used when no clients participate
        - Result is moved to CPU to avoid GPU memory issues during logging
        - This metric is typically logged to TensorBoard for monitoring
    """
    if len(delta_norms) > 0:
        norm = torch.median(torch.stack(delta_norms)).cpu()
    else:
        norm = 100
    return norm

def get_train_loss(local_losses):
    """
    Calculate the average training loss across all participating clients.
    
    This function computes the mean training loss from all clients that participated
    in a federated learning round. The average loss is used as a global measure
    of training progress and is typically logged for monitoring purposes.
    
    Args:
        local_losses (list): List of training losses from each participating client.
                            Each element should be a scalar value (float or tensor)
                            representing the final training loss for one client.
    
    Returns:
        float: Average training loss across all participating clients.
               Returns 100.0 if no losses are provided (fallback value).
    
    Algorithm:
        1. Check if local_losses list is non-empty
        2. If non-empty: Sum all losses and divide by count
        3. If empty: Return fallback value of 100.0
    
    Example:
        >>> local_losses = [0.5, 0.7, 0.3, 0.6]
        >>> avg_loss = get_train_loss(local_losses)
        >>> print(avg_loss)  # 0.525
        
        >>> local_losses = []
        >>> avg_loss = get_train_loss(local_losses)
        >>> print(avg_loss)  # 100.0
    
    Note:
        - Uses simple arithmetic mean (not weighted by data size)
        - The fallback value of 100.0 is arbitrary and used when no clients participate
        - This metric is typically logged to TensorBoard for monitoring
        - Loss values should be from the same loss function for meaningful comparison
    """
    if len(local_losses) > 0:
        train_loss = sum(local_losses) / len(local_losses)
    else:
        train_loss = 100
    return train_loss

def decay_learning_rate(args, t):
    """
    Apply learning rate decay based on the current training round.
    
    This function implements step-wise learning rate decay for federated learning.
    The learning rate is reduced by a multiplicative factor at regular intervals
    to improve convergence and final model performance.
    
    Args:
        args: Configuration object containing:
            - lr_step_size (int): Number of rounds between learning rate decay steps
            - decay_weight (float): Multiplicative factor for learning rate decay (typically < 1.0)
            - local_lr (float): Current local learning rate (modified in place)
        t (int): Current training round (0-indexed)
    
    Returns:
        None: The function modifies args.local_lr in place.
    
    Algorithm:
        1. Check if current round (t+1) is divisible by lr_step_size
        2. If yes: Multiply current learning rate by decay_weight
        3. If no: Keep learning rate unchanged
    
    Example:
        >>> args.lr_step_size = 50
        >>> args.decay_weight = 0.9
        >>> args.local_lr = 0.01
        >>> decay_learning_rate(args, 49)  # Round 50 (0-indexed)
        >>> print(args.local_lr)  # 0.009
        
        >>> decay_learning_rate(args, 50)  # Round 51
        >>> print(args.local_lr)  # 0.009 (unchanged)
    
    Note:
        - Uses (t+1) because rounds are 0-indexed but step counting is 1-indexed
        - Learning rate decay is applied to local_lr used by clients
        - Common decay_weight values: 0.1, 0.5, 0.9
        - Step size varies by dataset (e.g., Shakespeare: 50, others: 1)
    """
    if t+1 % args.lr_step_size == 0: # shakespear: 50, femnist: x, other: 1
        args.local_lr = args.local_lr * args.decay_weight

def update_block_ids_list(args, dataset_fim, net_glob, t):
    """
    Update the list of LoRA layer blocks assigned to each user based on FIM analysis or predefined probabilities.
    
    This function determines which LoRA layers each user will train based on either:
    1. Fisher Information Matrix (FIM) analysis (after initial epochs and at specified intervals)
    2. Predefined layer probabilities (during initial epochs or when FIM analysis is not performed)
    
    The function implements adaptive layer selection where layers with lower FIM values
    (indicating less importance for the current task) are more likely to be selected for training.
    
    Args:
        args: Configuration object containing:
            - fim_prior_epoch (int): Number of epochs to wait before starting FIM analysis
            - fim_every_iter (int): Interval (in rounds) for performing FIM analysis
            - lora_layer (int): Total number of LoRA layers available
            - layer_prob (list): Predefined probabilities for each layer (used when FIM is not active)
            - user_groupid_list (list): Mapping of users to their heterogeneous groups
            - heterogeneous_group{i}_lora (int): Number of LoRA layers for group i
        dataset_fim: Dataset used for FIM computation
        net_glob: Global model for FIM analysis
        t (int): Current training round/epoch
    
    Returns:
        None: The function modifies args.block_ids_list in place.
    
    Algorithm:
        1. FIM-based selection (when t > fim_prior_epoch-1 and t % fim_every_iter == 0):
           - Compute Fisher Information Matrix for all layers
           - Rank layers by FIM values and cluster them
           - Generate probability distribution based on layer importance
           - Sample layers for each user group based on these probabilities
        
        2. Probability-based selection (otherwise):
           - Use predefined layer_prob distribution
           - Sample layers for each user group based on these probabilities
    
    Example:
        >>> # FIM-based selection (after epoch 50, every 50 rounds)
        >>> args.fim_prior_epoch = 50
        >>> args.fim_every_iter = 50
        >>> args.lora_layer = 12
        >>> args.user_groupid_list = [0, 0, 1, 1, 2, 2]
        >>> args.heterogeneous_group0_lora = 6
        >>> args.heterogeneous_group1_lora = 9
        >>> args.heterogeneous_group2_lora = 12
        >>> update_block_ids_list(args, dataset_fim, net_glob, 100)
        >>> # args.block_ids_list will contain 6 lists, each with the assigned layer IDs
    
    Note:
        - Uses GPU lock to prevent concurrent FIM computation
        - FIM analysis helps identify which layers are most important for the current task
        - Layers with lower FIM values are more likely to be selected (inverse relationship)
        - The function supports heterogeneous federated learning with different numbers
          of LoRA layers per user group
        - All selected layer lists are sorted for consistency
    """

    gpu_lock.acquire()
    calc = FIMCalculator(args, copy.deepcopy(net_glob), dataset_fim)
    fim = calc.compute_fim(empirical=True, verbose=True, every_n=None)
    gpu_lock.release()
        # select those with lowest FIM layers to freeze
    layers_rank, cluster_labels = calc.bottom_k_layers(fim, k=args.lora_layer)
    observed_probability = get_observed_probability(cluster_labels)
    args.block_ids_list = []
    args.rank_list = []
    for id in args.user_groupid_list:
        layer_list = np.random.choice(range(args.lora_layer),
                                        p=observed_probability,
                                        size=getattr(args, 'heterogeneous_group'+str(id)+'_lora'),
                                        replace=False)
        args.block_ids_list.append(sorted(layer_list))
        if args.enable_rank_var:
            get_rank_list(args, layer_list, fim, id)
        else:
            get_rank_list(args, layer_list, [1]*args.lora_layer, id)

def update_block_ids_list_predefined(args, dataset_fim, net_glob, t):
    if hasattr(args, 'heterogeneous_group0_lora'):
        if isinstance(getattr(args, 'heterogeneous_group0_lora'), int):
            args.block_ids_list = []
            args.rank_list = []
            for id in args.user_groupid_list:
                layer_list = np.random.choice(range(args.lora_layer),
                                                p=[float(Fraction(x)) for x in args.layer_prob],
                                                size=getattr(args, 'heterogeneous_group'+str(id)+'_lora'),
                                                replace=False)
                args.block_ids_list.append(sorted(layer_list))
                get_rank_list(args, layer_list, [1]*args.lora_layer, id)


def get_rank_list(args, layer_list, fim, id):
    # Get the rank list based on the fim value
    sorted_layer_list = sorted(layer_list)
    selected_layer_fim = [fim[x] for x in sorted_layer_list]
    # reserve 1-rank for each selected block
    rank_budget = getattr(args, 'var_rank_group'+str(id)+'_lora') - len(layer_list)
    normalized_selected_layer_fim = [x/sum(selected_layer_fim) for x in selected_layer_fim]
    rank_list = [int(x*rank_budget) for x in normalized_selected_layer_fim]
    left_over = rank_budget - sum(rank_list)
    max_index = normalized_selected_layer_fim.index(max(normalized_selected_layer_fim))
    rank_list[max_index] += left_over

    # add back the reserved rank for each block.
    final_rank_list = [x + 1 for x in rank_list]
    args.rank_list.append(final_rank_list)

    print(f'group {id}: rank_budget = {rank_budget}, fim = {selected_layer_fim}, rank_list = {final_rank_list} ')
    #print(f'args.rank_list = {args.rank_list}')

def get_observed_probability(cluster_labels):
    """
    Convert cluster labels from FIM analysis into probability distribution for layer selection.
    
    This function maps cluster labels (which represent layer importance groups from FIM analysis)
    to specific probability values that determine how likely each layer is to be selected for training.
    The mapping follows a predefined scheme where different cluster labels correspond to different
    selection probabilities.
    
    Args:
        cluster_labels (list): List of cluster labels for each layer, where each label indicates
                              the importance group the layer belongs to based on FIM analysis.
                              Expected values: 0, 1, 2
    
    Returns:
        numpy.ndarray: Normalized probability distribution where each element represents the
                      probability of selecting the corresponding layer for training.
                      The probabilities sum to 1.0.
    
    Cluster Label Mapping:
        - Label 0: Probability = 1/27 (lowest importance, highest selection probability)
        - Label 1: Probability = 2/27 (medium importance, medium selection probability)  
        - Label 2: Probability = 1/9 (highest importance, lowest selection probability)
    
    Algorithm:
        1. Map each cluster label to its corresponding fraction string
        2. Convert fraction strings to float values using Fraction
        3. Normalize the probabilities so they sum to 1.0
    
    Example:
        >>> cluster_labels = [0, 1, 2, 0, 1]
        >>> probs = get_observed_probability(cluster_labels)
        >>> print(probs)
        [0.2, 0.4, 0.2, 0.2, 0.4]  # Normalized probabilities
        
        >>> cluster_labels = [0, 0, 0]
        >>> probs = get_observed_probability(cluster_labels)
        >>> print(probs)
        [0.333, 0.333, 0.333]  # All equal probabilities
    
    Note:
        - The inverse relationship between importance and selection probability is intentional
        - Layers with lower FIM values (less important) are more likely to be selected
        - This encourages training of less critical layers to improve overall model robustness
        - The function assumes cluster labels are in the range [0, 2]
    """
    observed_probability = []
    for label in cluster_labels:
        if label == 0:
            observed_probability.append('1/27')
        elif label == 1:
            observed_probability.append('2/27')
        elif label == 2:
            observed_probability.append('3/27')
    observed_probability = np.array([float(Fraction(x)) for x in observed_probability])
    observed_probability /= sum(observed_probability)
    return observed_probability

def update_user_groupid_list(args):
    """
    Create a user-to-group mapping list based on heterogeneous group configuration.
    
    This function generates a list that maps each user to their assigned heterogeneous group.
    It first calculates the number of users per group using get_group_cnt(), then creates
    the mapping by repeating each group ID according to the calculated counts.
    
    Args:
        args: Configuration object containing:
            - heterogeneous_group (list): List of fractions representing group proportions
            - num_users (int): Total number of users in the system
            The function will modify args by adding:
            - user_groupid_list (list): A list where each element represents the group ID
              assigned to a user. The length equals the total number of users.
    
    Returns:
        None: The function modifies args.user_groupid_list in place.
    
    Algorithm:
        1. Call get_group_cnt(args) to calculate user distribution across groups
        2. Initialize an empty user_groupid_list
        3. For each group ID and its corresponding count:
           - Add the group ID to the list 'count' times
           - This creates a mapping where users are assigned to groups sequentially
    
    Example:
        >>> args = argparse.Namespace()
        >>> args.heterogeneous_group = ['1/3', '1/3', '1/3']
        >>> args.num_users = 9
        >>> update_user_groupid_list(args)
        >>> print(args.user_groupid_list)
        [0, 0, 0, 1, 1, 1, 2, 2, 2]
        
        >>> args.heterogeneous_group = ['1/2', '1/4', '1/4']
        >>> args.num_users = 8
        >>> update_user_groupid_list(args)
        >>> print(args.user_groupid_list)
        [0, 0, 0, 0, 1, 1, 2, 2]
    
    Note:
        - The function internally calls get_group_cnt() to determine group sizes
        - The function modifies the args object in place
        - The resulting list length equals args.num_users
        - Users are assigned to groups in sequential order
        - This list is used later to determine which heterogeneous configuration
          each user should use during federated learning
    """
    group_cnt = get_group_cnt(args)
    args.user_groupid_list = []
    for id, c in enumerate(group_cnt):
        args.user_groupid_list += [id] * c

def get_group_cnt(args):
    """
    Calculate the number of users assigned to each heterogeneous group in federated learning.
    
    This function distributes users across different heterogeneous groups based on the specified
    proportions in `args.heterogeneous_group`. It ensures that the total number of users matches
    `args.num_users` by assigning any remaining users to the last group after proportional
    distribution.
    
    Args:
        args: Configuration object containing:
            - heterogeneous_group (list): List of fractions (as strings) representing the 
              proportion of users for each group. Example: ['1/3', '1/3', '1/3']
            - num_users (int): Total number of users in the federated learning system
    
    Returns:
        list: A list of integers where each element represents the number of users
              assigned to the corresponding heterogeneous group. The sum of all elements
              equals `args.num_users`.
    
    Algorithm:
        1. For each group except the last one:
           - Calculate the number of users as: int(num_users * fraction)
           - This may result in rounding down, leaving some users unassigned
        2. For the last group:
           - Assign all remaining users to ensure the total equals num_users
           - This handles any rounding errors from the proportional distribution
    
    Example:
        >>> args.heterogeneous_group = ['1/3', '1/3', '1/3']
        >>> args.num_users = 100
        >>> group_cnt = get_group_cnt(args)
        >>> print(group_cnt)  # [33, 33, 34]
        
        >>> args.heterogeneous_group = ['1/2', '1/4', '1/4']
        >>> args.num_users = 10
        >>> group_cnt = get_group_cnt(args)
        >>> print(group_cnt)  # [5, 2, 3]
    
    Note:
        - The function uses Fraction to handle fractional proportions accurately
        - The last group always gets any remaining users to ensure exact total
        - This is used in heterogeneous federated learning where different groups
          may have different model configurations or capabilities
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

def load_data(args):
    """
    Load and partition datasets for federated learning training and testing.
    
    This function handles the data loading and partitioning process for federated learning.
    It loads the specified dataset, partitions it among clients according to the data
    distribution strategy (IID or non-IID), and prepares datasets for both training
    and FIM computation.
    
    Args:
        args: Configuration object containing:
            - dataset (str): Name of the dataset to load (e.g., 'cifar100', 'sst2', 'qqp')
            - data_type (str): Type of data ('image' or 'text')
            - num_users (int): Number of clients for data partitioning
            - logger: Logging object for output messages
    
    Returns:
        tuple: A tuple containing:
            - args: Updated configuration object with dataset information
            - dataset_train: Training dataset
            - dataset_test: Testing dataset  
            - dict_users: Dictionary mapping client IDs to their data indices
            - dataset_fim: Dataset subset used for FIM computation
    
    Data Partitioning:
        The function uses load_partition() to create client-specific data splits:
        - For IID distribution: Data is randomly shuffled and evenly distributed
        - For non-IID distribution: Data is partitioned based on labels or other criteria
        - Each client gets a subset of the original dataset
    
    Logging:
        The function logs important dataset statistics:
        - Total dataset size (training + testing)
        - Number of training samples
        - Number of testing samples
    
    Example:
        >>> args.dataset = 'cifar100'
        >>> args.data_type = 'image'
        >>> args.num_users = 100
        >>> args, train_data, test_data, user_dict, fim_data = load_data(args)
        >>> print(f"Training samples: {len(train_data)}")
        >>> print(f"Testing samples: {len(test_data)}")
        >>> print(f"Number of clients: {len(user_dict)}")
    
    Note:
        - The function delegates actual data loading to load_partition()
        - FIM dataset is typically a subset of training data used for layer importance analysis
        - Data partitioning strategy depends on the configuration in args
        - All datasets are returned as PyTorch Dataset objects
    """
    args.logger.info("{:<50}".format("-" * 15 + " data setup " + "-" * 50)[0:60], main_process_only=True)
    args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users, dataset_fim = load_partition(args)
    args.logger.info('length of dataset:{}'.format(len(dataset_train) + len(dataset_test)), main_process_only=True)
    args.logger.info('num. of training data:{}'.format(len(dataset_train)), main_process_only=True)
    args.logger.info('num. of testing data:{}'.format(len(dataset_test)), main_process_only=True)
    return args,dataset_train,dataset_test,dict_users,dataset_fim

def update_dataset_fim(args, dataset_fim):
    """
    Convert FIM dataset to appropriate DataLoader format based on model and dataset type.
    
    This function prepares the FIM (Fisher Information Matrix) dataset for computation by
    converting it to a DataLoader with the appropriate collate function and batch size.
    The collate function and data handling depend on the specific model and dataset being used.
    
    Args:
        args: Configuration object containing:
            - model (str): Model identifier (e.g., 'vit', 'bert')
            - dataset (str): Dataset name (e.g., 'cifar100', 'sst2', 'qqp', 'qnli', 'ledgar', 'belebele')
            - batch_size (int): Batch size for DataLoader
            - data_collator: Collate function for text datasets
            - tokenizer: Tokenizer for multiple choice datasets
        dataset_fim: Raw dataset to be converted to DataLoader
    
    Returns:
        DataLoader: Configured DataLoader for FIM computation with appropriate:
            - Collate function based on data type
            - Batch size from configuration
            - Shuffling enabled for text datasets
    
    DataLoader Configuration by Model/Dataset:
        1. Vision Transformer (ViT) models:
           - Uses test_collate_fn for image data
           - No shuffling (deterministic for FIM computation)
        
        2. Text classification datasets (SST-2, QQP, QNLI, LEDGAR):
           - Uses args.data_collator for text tokenization
           - Shuffling enabled for better FIM estimation
        
        3. Multiple choice datasets (Belebele):
           - Uses DataCollatorForMultipleChoice
           - Shuffling enabled
           - Requires tokenizer for proper formatting
    
    Example:
        >>> args.model = 'google/vit-base-patch16-224-in21k'
        >>> args.dataset = 'cifar100'
        >>> args.batch_size = 32
        >>> fim_loader = update_dataset_fim(args, fim_dataset)
        >>> print(f"FIM DataLoader created with batch size: {fim_loader.batch_size}")
    
    Note:
        - FIM computation requires consistent data format across different model types
        - The function handles the complexity of different data types transparently
        - Shuffling is used for text datasets to improve FIM estimation quality
        - Vision datasets typically don't use shuffling for deterministic results
    """
    if 'vit' in args.model:
        dataset_fim = DataLoader(dataset_fim, collate_fn=test_collate_fn, batch_size=args.batch_size)
    elif 'sst2' in args.dataset or 'qqp' in args.dataset or 'qnli' in args.dataset or 'ledgar' in args.dataset:
        dataset_fim = DataLoader(dataset_fim, shuffle=True, collate_fn=args.data_collator, batch_size=args.batch_size)
    elif 'belebele' in args.dataset:
        dataset_fim = DataLoader(dataset_fim, shuffle=True, collate_fn=DataCollatorForMultipleChoice(tokenizer=args.tokenizer), batch_size=args.batch_size)
    return dataset_fim

def get_data_loader_list(args, dataset_train, dict_users):
    """
    Create a list of DataLoaders for each client's local training data.
    
    This function creates individual DataLoaders for each client by splitting the global
    training dataset according to the client data distribution specified in dict_users.
    Each client gets a DataLoader configured with the appropriate collate function
    based on the model and dataset type.
    
    Args:
        args: Configuration object containing:
            - num_users (int): Total number of clients
            - model (str): Model identifier (e.g., 'vit', 'bert')
            - dataset (str): Dataset name (e.g., 'cifar100', 'sst2', 'qqp', 'qnli', 'ledgar', 'belebele')
            - batch_size (int): Batch size for training
            - data_collator: Collate function for text datasets
            - tokenizer: Tokenizer for multiple choice datasets
        dataset_train: Global training dataset
        dict_users: Dictionary mapping client IDs to their data indices
    
    Returns:
        list: List of DataLoader objects, where each DataLoader corresponds to one client's
              local training data. The list has length equal to args.num_users.
    
    DataLoader Configuration by Model/Dataset:
        1. Vision Transformer (ViT) models:
           - Uses vit_collate_fn for image data processing
           - Shuffling enabled for training
           - Handles pixel values and labels
        
        2. Text classification datasets (SST-2, QQP, QNLI, LEDGAR):
           - Uses args.data_collator for text tokenization
           - Shuffling enabled for training
           - Handles input_ids, attention_mask, labels
        
        3. Multiple choice datasets (Belebele):
           - Uses DataCollatorForMultipleChoice
           - Shuffling enabled for training
           - Handles multiple choice question formatting
    
    Algorithm:
        1. For each client (0 to num_users-1):
           - Create DatasetSplit with client's data indices
           - Configure DataLoader based on model/dataset type
           - Add DataLoader to the list
        2. Return the complete list of client DataLoaders
    
    Example:
        >>> args.num_users = 100
        >>> args.model = 'google/vit-base-patch16-224-in21k'
        >>> args.dataset = 'cifar100'
        >>> args.batch_size = 32
        >>> data_loaders = get_data_loader_list(args, train_dataset, user_dict)
        >>> print(f"Created {len(data_loaders)} DataLoaders")
        >>> print(f"Client 0 has {len(data_loaders[0])} batches")
    
    Note:
        - Each client gets a different subset of the training data
        - DataLoaders are configured for local training (shuffling enabled)
        - The function handles different data types transparently
        - Client data distribution is determined by dict_users partitioning
    """
    data_loader_list = []
    for i in range(args.num_users):
        dataset = DatasetSplit(dataset_train, dict_users[i], args)
        if 'vit' in args.model:
            ldr_train = DataLoader(dataset, shuffle=True, collate_fn=vit_collate_fn, batch_size=args.batch_size)
        elif 'sst2' in args.dataset or 'qqp' in args.dataset or 'qnli' in args.dataset or 'ledgar' in args.dataset:
            ldr_train = DataLoader(dataset, shuffle=True, collate_fn=args.data_collator, batch_size=args.batch_size)
        elif 'belebele' in args.dataset:
            ldr_train = DataLoader(dataset, shuffle=True, collate_fn=DataCollatorForMultipleChoice(tokenizer=args.tokenizer), batch_size=args.batch_size)
        data_loader_list.append(ldr_train)
    return data_loader_list
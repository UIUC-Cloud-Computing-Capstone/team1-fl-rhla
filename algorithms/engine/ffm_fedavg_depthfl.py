"""
Federated learning engine: FedAvg with heterogeneous LoRA depth (DepthFL).

This module implements a federated averaging algorithm where clients may train
different subsets of LoRA layers (heterogeneous depth) and optionally use
alternative aggregation methods (weighted average, SVD average, product average).
Supports ViT, BERT-style models, and LEDGAR; uses load_partition for data and
model_setup for the global model.
"""
import copy
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.data_pre_process import load_partition, DatasetSplit
from utils.model_utils import model_setup
from test import test, test_vit, test_ledgar
from ..solver.local_solver import LocalUpdate
from ..solver.global_aggregator import average_lora_depthfl, weighted_average_lora_depthfl, svd_average, product_average
from fractions import Fraction
import re
import random

VISION_MODEL = 'facebook/deit-small-patch16-224'


def vit_collate_fn(examples):
    """
    Collate a batch of examples for Vision Transformer (ViT) training.

    Expects each example to be a tuple with label at index 1 and pixel values
    at index 2. Returns a dict with "pixel_values" (stacked [N, C, H, W]) and
    "labels" (tensor of length N).

    Args:
        examples: List of (_, label, pixel_values) tuples.

    Returns:
        dict: "pixel_values" and "labels" tensors for the batch.
    """
    pixel_values = torch.stack([example[2] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def ffm_fedavg_depthfl(args):
    """
    Federated averaging with heterogeneous LoRA depth (DepthFL).

    Runs FedAvg over multiple rounds: each round samples clients, performs
    local LoRA (or LoKR) tuning with per-client layer/rank config, aggregates
    updates (simple average, weighted_average, svd_average, or product_average),
    then evaluates the global model. Supports ViT, BERT, and LEDGAR evaluation.

    Args:
        args: Config object with at least:
            - num_users, num_selected_users, round: FL setup
            - heterogeneous_group: list of fraction strings (e.g. ['1/3','1/3','1/3'])
            - heterogeneous_group{i}_lora: int or list of layer indices for group i
            - lora_layer: total number of LoRA layers
            - rank_group{i}_lora: used when LEGEND/HetLoRA/FlexLoRA
            - peft: 'lora' (and LOKR flag for LoKR)
            - lr_step_size, decay_weight, local_lr: learning rate decay
            - model, dataset, batch_size, log_path, device
            - logger, accelerator, (optional) aggregation, data_collator

    Returns:
        tuple: ((best_test_acc, best_test_f1, best_test_macro_f1, best_test_micro_f1), metric_keys).
    """
    ################################### hyperparameter setup ########################################
    args.logger.info("{:<50}".format("-" * 15 + " data setup " + "-" * 50)[0:60], main_process_only=True)
    args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users, dataset_fim = load_partition(args)
    args.logger.info('length of dataset:{}'.format(len(dataset_train) + len(dataset_test)), main_process_only=True)
    args.logger.info('num. of training data:{}'.format(len(dataset_train)), main_process_only=True)
    args.logger.info('num. of testing data:{}'.format(len(dataset_test)), main_process_only=True)
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
    data_loader_list = []
    for i in range(args.num_users):
        dataset = DatasetSplit(dataset_train, dict_users[i], args)
        
        if VISION_MODEL in args.model:
            ldr_train = DataLoader(dataset, shuffle=True, collate_fn=vit_collate_fn, batch_size=args.batch_size)
        elif 'ledgar' in args.dataset:
            ldr_train = DataLoader(dataset, shuffle=True, collate_fn=args.data_collator, batch_size=args.batch_size)

        data_loader_list.append(ldr_train)

    # heterogenity
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

    args.user_groupid_list = []
    for id, c in enumerate(group_cnt):
        args.user_groupid_list += [id] * c
    # block ids for each clients
    if hasattr(args, 'heterogeneous_group0_lora'):
        if isinstance(getattr(args, 'heterogeneous_group0_lora'), int):
            args.block_ids_list = []
            for id in args.user_groupid_list:
                args.block_ids_list.append(sorted(random.sample(range(args.lora_layer), getattr(args, 'heterogeneous_group'+str(id)+'_lora'))))
        elif isinstance(getattr(args, 'heterogeneous_group0_lora'), list):
            args.block_ids_list = []
            '''
            if hasattr(args, 'probability_structure'):
                id_list = set(args.user_groupid_list)
                prob_distribution = []
                for k in id_list:
                    prob_distribution.extend(getattr(args, 'heterogeneous_group'+str(k)+'_lora'))
                total_elements = len(prob_distribution)
                prob_distribution = [count/total_elements for _, count in Counter(sorted(prob_distribution)).items()]
                for id in args.user_groupid_list:
                    args.block_ids_list.append(sorted(np.random.choice(np.arange(len(getattr(args, 'heterogeneous_group'+str(list(id_list)[-1])+'_lora'))),
                                            size=len(getattr(args, 'heterogeneous_group'+str(id)+'_lora')),
                                            p=prob_distribution,
                                            replace=False)))
            else:
            '''
            for id in args.user_groupid_list:
                args.block_ids_list.append(getattr(args, 'heterogeneous_group'+str(id)+'_lora'))

            # for exclusive and straggler tuning, all the rank are the same as the max lora rank
    args.rank_list = []
    if args.LEGEND:
        print('running LEGEND')
        for id in args.user_groupid_list:
            args.rank_list.append(getattr(args, 'rank_group'+str(id)+'_lora'))
    elif args.HetLoRA:
        print('running HetLoRA')
        for id in args.user_groupid_list:
            args.rank_list.append(getattr(args, 'rank_group'+str(id)+'_lora'))
    elif args.FlexLoRA:
        print('running FlexLoRA')
        for id in args.user_groupid_list:
            args.rank_list.append(getattr(args, 'rank_group'+str(id)+'_lora')) 
    else:
        args.rank_list = []
    print(f'args.block_ids_list {args.block_ids_list}, rank list {args.rank_list}')
    
    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_test_macro_f1 = 0.0
    best_test_micro_f1 = 0.0
    metric_keys = {'Accuracy': 0, 'F1': 0, 'Macro_F1': 0, "Micro_F1": 0}

    for t in range(args.round):
        args.logger.info('Round: ' + str(t) + '/' + str(args.round), main_process_only=True)
        ## learning rate decaying
        if t+1 % args.lr_step_size == 0: # shakespear: 50, femnist: x, other: 1
            args.local_lr = args.local_lr * args.decay_weight
        
        ############################################################# FedAvg ##########################################
        ## user selection
        selected_idxs = list(np.random.choice(range(args.num_users), args.num_selected_users, replace=False))
        ## copy global model
        net_glob.train()
        ## local training
        local_solver = LocalUpdate(args=args)
        local_models, local_losses, local_updates, delta_norms, num_samples = [], [], [], [], []
        print('selected client idxs: '+str(selected_idxs))
        for num_index, i in enumerate(selected_idxs):
            if args.peft == 'lora':
                local_model, local_loss, no_weight_lora =  local_solver.lora_tuning(model=copy.deepcopy(net_glob),
                                                                                    ldr_train=data_loader_list[i],
                                                                                    args=args,
                                                                                    client_index=num_index,
                                                                                    client_real_id=i,
                                                                                    round=t,
                                                                                    hete_group_id=args.user_groupid_list[i])
            if local_loss:
                local_losses.append(local_loss)
            # compute model update
            lora_str = 'lora'
            if args.LOKR:
                print('train lokr')
                lora_str = 'lokr'

            model_update = {}
            if args.peft == 'lora':
                for k in global_model.keys():
                    if lora_str in k: # no classifier
                        if int(re.findall(r"\d+", k)[0]) not in no_weight_lora:
                            model_update[k] = local_model[k].detach().cpu() - global_model[k].detach().cpu() 
            else:
                model_update = {k: local_model[k].detach().cpu() - global_model[k].detach().cpu() for k in global_model.keys()}
            # compute model update norm
            norm_updates = []
            for k in model_update.keys():
                norm_updates.append(torch.flatten(model_update[k]))
            if len(norm_updates) > 0:
                delta_norm = torch.norm(torch.cat(norm_updates))
            else:
                delta_norm = None
            if delta_norm:
                delta_norms.append(delta_norm)
            local_updates.append(model_update)
            num_samples.append(len(data_loader_list[i]))

        if len(local_updates) == 0:
            args.logger.info('The number of trainable client is 0, skip the round for average')
            continue
        # metrics
        if len(delta_norms) > 0:
            norm = torch.median(torch.stack(delta_norms)).cpu()
        else:
            norm = 100
        if len(local_losses) > 0:
            train_loss = sum(local_losses) / len(local_losses)
        else:
            train_loss = 100
        if args.accelerator.is_local_main_process:
            writer.add_scalar('norm', norm, t)
            writer.add_scalar('train_loss', train_loss, t)

        if hasattr(args, 'aggregation'):
            if args.aggregation ==  'weighted_average':
                global_model = weighted_average_lora_depthfl(args, global_model, local_updates, num_samples)
            elif args.aggregation == 'svd_average':
                global_model = svd_average(args, global_model, local_updates, num_samples)
            elif args.aggregation == 'product_average':
                global_model = product_average(args, global_model, local_updates, num_samples)
        else:
            global_model = average_lora_depthfl(args, global_model, local_updates)

        # test global model on server side   
        net_glob.load_state_dict(global_model)
        net_glob.eval()
        if VISION_MODEL in args.model:
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
            if 'ledgar' in args.dataset:
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

        args.accelerator.wait_for_everyone()

    return (best_test_acc, best_test_f1, best_test_macro_f1, best_test_micro_f1), metric_keys
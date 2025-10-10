import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.data_pre_process import load_partition, DatasetSplit
from utils.model_utils import model_setup
from test import test, test_vit, test_ledgar
from ..solver.local_solver import LocalUpdate
from ..solver.global_aggregator import average_lora_depthfl, weighted_average_lora_depthfl
from ..solver.shared_utils import (
    load_data, get_data_loader_list, get_dataset_fim, vit_collate_fn, test_collate_fn,
    get_group_cnt, update_user_groupid_list, update_block_ids_list, update_block_ids_list_with_observed_probability,
    get_observed_prob, get_observed_probability, get_model_update, get_norm_updates,
    get_train_loss, get_norm, update_delta_norms
)
from fractions import Fraction
import re
import numpy as np
from utils.fim_calculator import FIMCalculator
import threading


gpu_lock = threading.Lock()

def ffm_fedavg_depthffm_fim(args):
    ################################### hyperparameter setup ########################################
    args, dataset_train, dataset_test, dict_users, dataset_fim, writer, net_glob, global_model = set_up_hyperparameters(args)

    ###################################### model initialization ###########################
    #t1 = time.time()
    args.logger.info("{:<50}".format("-" * 15 + " training... " + "-" * 50)[0:60], main_process_only=True)
    # initialize data loader for training and/or public dataset
    data_loader_list = get_data_loader_list(args, dataset_train, dict_users)
    dataset_fim = get_dataset_fim(args, dataset_fim)
    
    # heterogenity
    group_cnt = get_group_cnt(args)

    update_user_groupid_list(args, group_cnt)

    best_test_acc, best_test_f1, best_test_micro_f1, metric_keys = init_metrics()

    for t in range(args.round):
        # block ids for each clients
        update_block_ids_for_each_client(args, dataset_fim, net_glob, t)

        args.logger.info('Round: ' + str(t) + '/' + str(args.round), main_process_only=True)
        ## learning rate decaying
        if t+1 % args.lr_step_size == 0: # shakespear: 50, femnist: x, other: 1
            args.local_lr = args.local_lr * args.decay_weight

        ############################################################# FedAvg ##########################################
        ## user selection
        selected_idxs = list(np.random.choice(range(args.num_users), args.num_selected_users, replace=False))
        print('selected client idxs: '+str(selected_idxs))
        ## copy global model
        net_glob.train()
        ## local training
        local_losses, local_updates, delta_norms, num_samples = local_training(args, net_glob, global_model, data_loader_list, t, selected_idxs)

        if len(local_updates) == 0:
            args.logger.info('The number of trainable client is 0, skip the round for average')
            continue
        # metrics
        # train_loss.append(sum(local_losses) / args.num_selected_users)
        # median_model_norm.append(torch.median(torch.stack(delta_norms)).cpu())
        norm = get_norm(delta_norms)
        train_loss = get_train_loss(local_losses)
        
        add_to_writer(args, writer, t, norm, train_loss)

        global_model = get_global_model(args, global_model, local_updates, num_samples)

        # test global model on server side   
        best_test_acc, best_test_macro_f1, best_test_micro_f1 = test_global_model_on_server_side(args, dataset_test, writer, net_glob, global_model, best_test_acc, best_test_micro_f1, metric_keys, t, norm, train_loss)

        args.accelerator.wait_for_everyone()

    return (best_test_acc, best_test_f1, best_test_macro_f1, best_test_micro_f1), metric_keys

def add_to_writer(args, writer, t, norm, train_loss):
    if args.accelerator.is_local_main_process:
        writer.add_scalar('norm', norm, t)
        writer.add_scalar('train_loss', train_loss, t)

def init_metrics():
    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_test_macro_f1 = 0.0
    best_test_micro_f1 = 0.0
    metric_keys = {'Accuracy': 0, 'F1': 0, 'Macro_F1': 0, "Micro_F1": 0}
    return best_test_acc,best_test_f1,best_test_micro_f1,metric_keys

def set_up_hyperparameters(args):
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
    # memory calculation for ViT
    # print(model_memory_usage_ViT(net_glob))
    
    args.logger.info('model dim: '+str(args.dim), main_process_only=True)
    return args,dataset_train,dataset_test,dict_users,dataset_fim,writer,net_glob,global_model

def update_block_ids_for_each_client(args, dataset_fim, net_glob, t):
    if t > args.fim_prior_epoch-1 and t % args.fim_every_iter == 0:
        gpu_lock.acquire()
        calc = FIMCalculator(args, copy.deepcopy(net_glob), dataset_fim)
        fim = calc.compute_fim(empirical=True, verbose=True, every_n=None)
        gpu_lock.release()
            # select those with lowest FIM layers to freeze
        cluster_labels = calc.bottom_k_layers(fim, k=args.lora_layer)
        observed_probability = get_observed_prob(cluster_labels)
        update_block_ids_list_with_observed_probability(args, observed_probability)
    else:
        if hasattr(args, 'heterogeneous_group0_lora'):
            if isinstance(getattr(args, 'heterogeneous_group0_lora'), int):
                update_block_ids_list(args)

def local_training(args, net_glob, global_model, data_loader_list, t, selected_idxs):
    local_solver = LocalUpdate(args=args)
    local_losses, local_updates, delta_norms, num_samples = [], [], [], []
    for num_index, i in enumerate(selected_idxs):
        train_each_client(args, net_glob, global_model, data_loader_list, t, local_solver, local_losses, local_updates, delta_norms, num_samples, num_index, i)
    return local_losses,local_updates,delta_norms,num_samples

def train_each_client(args, net_glob, global_model, data_loader_list, t, local_solver, local_losses, local_updates, delta_norms, num_samples, num_index, i):
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
    model_update = get_model_update(args, global_model, local_model, no_weight_lora)
            # compute model update norm
    norm_updates = get_norm_updates(model_update)
    update_delta_norms(delta_norms, norm_updates)

    local_updates.append(model_update)
    num_samples.append(len(data_loader_list[i]))

def test_global_model_on_server_side(args, dataset_test, writer, net_glob, global_model, best_test_acc, best_test_micro_f1, metric_keys, t, norm, train_loss):
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
            
    return best_test_acc,best_test_macro_f1,best_test_micro_f1

def get_global_model(args, global_model, local_updates, num_samples):
    if hasattr(args, 'aggregation'):
        if args.aggregation ==  'weighted_average':
            global_model = weighted_average_lora_depthfl(args, global_model, local_updates, num_samples)
    else:
        global_model = average_lora_depthfl(args, global_model, local_updates)
    return global_model
















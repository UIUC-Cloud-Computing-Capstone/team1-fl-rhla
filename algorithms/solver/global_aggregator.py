import torch
import random
import numpy as np
import copy
from utils.model_utils import model_clip
from collections import defaultdict

def average(args, global_model, local_updates):
    '''
    simple average
    '''
    model_update = {k: local_updates[0][k] *0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys()}

    global_model = {k: global_model[k] +  model_update[k]/ len(local_updates) for k in global_model.keys()}

    return global_model

def average_lora(args, global_model, local_updates):
    '''
    simple average
    '''
    model_update = {k: local_updates[0][k] *0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        if args.peft == 'lora':
            model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys() if 'lora' in k}
        else:
            model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys()}
    if args.peft == 'lora':
        for k in global_model.keys():
            if 'lora' in k:
                global_model[k] = global_model[k].detach().cpu() +  model_update[k]/ len(local_updates)
    else:
        global_model = {k: global_model[k].detach().cpu() +  model_update[k]/ len(local_updates) for k in global_model.keys()}

    return global_model

def average_lora_depthfl(args, global_model, loc_updates):
    '''
    hetero average
    '''
    model_update_avg_dict = {}

    print('############## global aggregation ####################')
    lora_str = 'lora'
    if args.LOKR:
        lora_str = 'lokr'

    for k in global_model.keys():
        if lora_str in k or 'classifier' in k:
            for loc_update in loc_updates:
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])
                        # Print the update content to check the rank variation and update param
                        #print(k)
                        #print(loc_update[k])
                        #print(loc_update[k].shape)

    for k in global_model.keys():
        if k in model_update_avg_dict:
            global_model[k] = global_model[k].detach().cpu() +  sum(model_update_avg_dict[k]) / len(model_update_avg_dict[k])
    return global_model

def weighted_average_lora_depthfl(args, global_model, loc_updates, num_samples):
    '''
    weighted hetero average
    '''
    model_update_avg_dict = {}
    model_weights_cnt = {}
    model_weights_list = {}

    lora_str = 'lora'
    if args.LOKR:
        lora_str = 'lokr'
    for k in global_model.keys():
        if lora_str in k or 'classifier' in k: # classifier is not included
            for client_i, loc_update in enumerate(loc_updates):
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
                        # Print the update content to check the rank variation and update param
                        #print(k)
                        #print(loc_update[k])
                        #print(loc_update[k].shape)

    for k in global_model.keys():
        if k in model_update_avg_dict:
            weight_based = model_weights_cnt[k]
            model_weights_list[k] = [item/weight_based for item in model_weights_list[k]]
            global_model[k] = global_model[k].detach().cpu() + sum([model*weight for model, weight in zip(model_update_avg_dict[k], model_weights_list[k])])



    return global_model


def svd_average(args, global_model, loc_updates, num_samples):
    '''
    hetero average
    '''
    update_weights = []
    for updates in loc_updates:
        svd_weights = []
        keys = list(updates.keys())
        # print('$$$$$$$$$$$$')
        # print(keys)
        # keys from layer 0 -> 11
        for i in range(0, len(keys), 2):
            keyA = keys[i]
            keyB = keys[i+1]
            tensorA = updates[keyA]
            tensorB = updates[keyB]
            tensorSVD = torch.matmul(tensorB, tensorA)
            #print(f'$$$$$$$$$$$$$$ size of tensor_svd = {tensorSVD.shape}')
            svd_weights.append(tensorSVD)
            #print(f'$$$$$$$$$$$$$$ size of svd_weights = {svd_weights.shape}')

        svd_weights = torch.cat(svd_weights, dim=0)
        print(f'$$$$$$$$$$$$$$ after stack size of svd_weights = {svd_weights.shape}')
        frobenius_norm = np.linalg.norm(svd_weights, 'fro')
        update_weights.append(frobenius_norm)
    update_weights = update_weights / np.sum(update_weights)
    args.logger.info(f'the weights of different clients {update_weights}', main_process_only=True)
    for index, (u, w) in enumerate(zip(loc_updates, update_weights)):
        loc_updates[index] = {key: value * w for key, value in u.items()}

    model_update_avg_dict = {}
    for k in global_model.keys():
        if 'lora' in k or 'classifier' in k:
            for loc_update in loc_updates:
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])
    for k in global_model.keys():
        if k in model_update_avg_dict:
            global_model[k] = global_model[k].detach().cpu() +  sum(model_update_avg_dict[k])
    return global_model


def product_average(args, global_model, loc_updates, num_samples):
    '''
    hetero average
    '''
    update_weights = []
    svd_dict = []
    for updates in loc_updates:
        svd_weights = []
        keys = list(updates.keys())
        # print('$$$$$$$$$$$$')
        # print(keys)
        # keys from layer 0 -> 11
        client_svd = {}
        for i in range(0, len(keys), 2):
            keyA = keys[i]
            keyB = keys[i+1]
            tensorA = updates[keyA]
            tensorB = updates[keyB]
            tensorSVD = torch.matmul(tensorB, tensorA)
            client_svd[keyA] = tensorSVD
            client_svd[keyB] = tensorSVD
        svd_dict.append(client_svd) # list of dictionary same size as loc_updates

    model_update_avg_dict = {}
    for k in global_model.keys():
        if 'lora' in k or 'classifier' in k:
            for svd in svd_dict:
                if k in svd:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(svd[k])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(svd[k])

    # average of the product
    for key, value in model_update_avg_dict.items():
        model_update_avg_dict[key] = torch.mean(torch.stack(value), dim=0)

    # svd split
    for k in global_model.keys():
        if k in model_update_avg_dict:
            if 'lora_B' in k:
                B = model_update_avg_dict[k].detach().cpu()
                lora_name = k.replace('lora_B', 'lora_A')
                A = model_update_avg_dict[lora_name].detach().cpu()
                U, S, VT = torch.linalg.svd(B@A, full_matrices=False) 

                # suppress the deficient singular value
                #print(f'smallest singulvar value = {min(S)}')
                tol = 1e-6
                S[S<tol]=0

                diff_B = (U@torch.diag(torch.sqrt(S)))[:,0:args.lora_max_rank]
                diff_A = (torch.diag(torch.sqrt(S))@VT)[0:args.lora_max_rank,:]


                global_model[k] = global_model[k].detach().cpu() +  diff_B
                global_model[lora_name] = global_model[lora_name].detach().cpu() +  diff_A
    return global_model


    if args.privc == 'dp_randk' or args.privc == 'dp_topk' or args.privc == 'randk' or args.privc == 'topk':
        # error compensate, clipping, privately compress and average
        # lr_g = args.global_lr * 1/(t+1)**0.5
        pris_local_updates =[]
        norm_para=torch.zeros(args.num_users).to(args.device)
        for i in idxs_users:
            if (t+1) == args.tau:
                local_models[i] = {k:local_models[i][k] - global_model[k] for k in global_model.keys()}
                local_models[i], norm_para[i] = model_clip(local_models[i], args.clip)
            else:
                local_models[i] = {k:local_models[i][k] - global_model[k] + errors[i][k] for k in global_model.keys()}
                local_models[i], norm_para[i] = model_clip(local_models[i], args.clip)
                            
            pris_update = private_com(local_models[i], args)
            pris_local_updates.append(pris_update)
        print('weight norm', sum(norm_para)/len(idxs_users))
        
        global_update = {k: global_model[k] *0.0 for k in global_model.keys()}
        for i in len(idxs_users):
            global_update = {k: global_update[k] +  pris_local_updates[i][k] for k in global_update.keys()}
        
        global_update = {k: global_update[k] +  pris_local_updates[i][k] for k in global_update.keys()}
        global_model = {k: global_model[k] + global_update[k] for k in global_model.keys()}
    else:
        exit('Error: unrecognized private compressor for pefed.') 
        # global_model = aggregation_avg(local_models, idxs_users)
    return global_model